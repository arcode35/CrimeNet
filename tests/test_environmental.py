from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import dagster as dg
import h3.api.basic_int as h3
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from crimenet_data.assets.environmental.gold import (
    DEFAULT_RAW_WEATHER_READ_WORKERS,
    _write_silver_weather_snapshot,
    _stable_snapshot_id,
    environmental_features,
    published_integration_sampling,
    raw_model_weather_v2,
    silver_weather_features,
)
from crimenet_data.assets.environmental.transformations import (
    ENVIRONMENTAL_FEATURE_SCHEMA,
    REQUIREMENT_KEY_SCHEMA,
    SILVER_WEATHER_SCHEMA,
    EnvironmentalContractError,
    build_environmental_features,
    build_r9_to_r6_mapping,
    classify_lighting,
    compute_lighting_features,
    normalize_weather_envelope,
    validate_environmental_features,
    validate_silver_weather,
)
from crimenet_data.assets.model_table.transformations import prepare_lighting
from crimenet_data.definitions import defs
from crimenet_data.resources.crime_lake import CrimeLakeResources


def _cell(latitude: float = 41.88, longitude: float = -87.63, resolution: int = 6) -> int:
    return h3.latlng_to_cell(latitude, longitude, resolution)


def _weather_envelope(
    *,
    start: date = date(2024, 1, 1),
    end: date = date(2024, 1, 1),
    cell: int | None = None,
) -> dict[str, object]:
    row_count = ((end - start).days + 1) * 24
    first = datetime.combine(start, datetime.min.time())
    hours = [
        (first + timedelta(hours=index)).strftime("%Y-%m-%dT%H:%M")
        for index in range(row_count)
    ]
    return {
        "weather_contract_version": "model_weather_v2",
        "request_id": f"test-{start.isoformat()}-{end.isoformat()}",
        "provider": "open_meteo",
        "model": "best_match",
        "model_selection_policy": "open_meteo_default_best_match",
        "weather_query_cell_id": cell if cell is not None else _cell(),
        "h3_resolution": 6,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly_variables": ["relative_humidity_2m", "temperature_2m"],
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
        },
        "hourly": {
            "time": hours,
            "temperature_2m": [12.5] * row_count,
            "relative_humidity_2m": [67.0] * row_count,
        },
    }


def _normalize(envelope: dict[str, object] | None = None) -> pl.DataFrame:
    return normalize_weather_envelope(
        envelope or _weather_envelope(),
        source_object_uri="memory://weather.json",
    )


def _requirements(hours: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "h3_cell_id": pl.Series([_cell()] * len(hours), dtype=pl.Int64),
            "hour": pl.Series(hours, dtype=pl.Datetime("us", time_zone="UTC")),
            "event_reference_count": pl.Series([2] * len(hours), dtype=pl.Int64),
            "integration_reference_count": pl.Series([3] * len(hours), dtype=pl.Int64),
        },
        schema=REQUIREMENT_KEY_SCHEMA,
    )


def test_model_weather_v2_envelope_normalizes_to_hourly_silver_rows() -> None:
    frame = _normalize()

    assert frame.schema == SILVER_WEATHER_SCHEMA
    assert frame.height == 24
    assert frame["h3_cell_id"].n_unique() == 1
    assert frame["hour"].min() == datetime(2024, 1, 1, tzinfo=UTC)
    assert frame["hour"].max() == datetime(2024, 1, 1, 23, tzinfo=UTC)
    assert frame["weather_temperature_2m_c"].to_list() == [12.5] * 24
    assert validate_silver_weather(frame)["duplicate_key_count"] == 0


def test_leap_year_envelope_has_8784_hours() -> None:
    frame = _normalize(
        _weather_envelope(start=date(2020, 1, 1), end=date(2020, 12, 31))
    )

    assert frame.height == 366 * 24
    assert frame["hour"].max() == datetime(2020, 12, 31, 23, tzinfo=UTC)


def test_duplicate_raw_hour_fails() -> None:
    envelope = _weather_envelope()
    hourly = envelope["hourly"]
    assert isinstance(hourly, dict)
    times = hourly["time"]
    assert isinstance(times, list)
    times[2] = times[1]

    with pytest.raises(EnvironmentalContractError, match="duplicate hourly"):
        _normalize(envelope)


def test_wrong_weather_units_fail() -> None:
    envelope = _weather_envelope()
    units = envelope["hourly_units"]
    assert isinstance(units, dict)
    units["temperature_2m"] = "°F"

    with pytest.raises(EnvironmentalContractError, match="unexpected unit"):
        _normalize(envelope)


def test_invalid_weather_h3_resolution_fails() -> None:
    envelope = _weather_envelope(cell=_cell(resolution=9))
    envelope["h3_resolution"] = 9

    with pytest.raises(EnvironmentalContractError, match="h3_resolution"):
        _normalize(envelope)


def test_silver_duplicate_h3_hour_fails() -> None:
    frame = _normalize()
    duplicate = pl.concat([frame, frame.head(1)])

    with pytest.raises(EnvironmentalContractError, match="duplicate"):
        validate_silver_weather(duplicate)


def test_pvlib_solar_position_near_equinox_noon() -> None:
    equatorial_cell = _cell(0.0, 0.0)
    keys = pl.DataFrame(
        {
            "h3_cell_id": pl.Series([equatorial_cell], dtype=pl.Int64),
            "hour": pl.Series(
                [datetime(2024, 3, 20, 12, tzinfo=UTC)],
                dtype=pl.Datetime("us", time_zone="UTC"),
            ),
        }
    )

    result = compute_lighting_features(keys).row(0, named=True)

    assert result["solar_elevation_deg"] > 87.0
    assert result["solar_zenith_deg"] < 3.0
    assert 0.0 <= result["solar_azimuth_deg"] <= 360.0
    assert result["lighting_condition"] == "day"
    assert result["is_daylight"] is True


def test_lighting_classification_boundaries() -> None:
    assert classify_lighting(
        [0.0, -0.01, -6.0, -6.01, -12.0, -12.01, -18.0, -18.01]
    ) == [
        "day",
        "civil_twilight",
        "civil_twilight",
        "nautical_twilight",
        "nautical_twilight",
        "astronomical_twilight",
        "astronomical_twilight",
        "night",
    ]


def test_weather_null_row_retains_valid_lighting() -> None:
    hour = datetime(2026, 8, 26, 12, tzinfo=UTC)
    empty_weather = pl.DataFrame(schema=SILVER_WEATHER_SCHEMA)

    result = build_environmental_features(
        requirements=_requirements([hour]),
        silver_weather=empty_weather,
    )
    summary = validate_environmental_features(
        result,
        archive_cutoff_hour=datetime(2026, 8, 20, 23, tzinfo=UTC),
    )

    assert result.schema == ENVIRONMENTAL_FEATURE_SCHEMA
    assert result.height == 1
    assert result["weather_available"].to_list() == [False]
    assert result["weather_temperature_2m_c"].null_count() == 1
    assert result["lighting_condition"].null_count() == 0
    assert summary["weather_null_rows"] == 1


def test_integration_r9_cells_map_to_exact_r6_parents() -> None:
    r9_cells = [_cell(41.88, -87.63, 9), _cell(34.05, -118.24, 9)]

    mapping = build_r9_to_r6_mapping([r9_cells[1], r9_cells[0], r9_cells[0]])

    assert mapping.height == 2
    assert dict(mapping.iter_rows()) == {
        cell: h3.cell_to_parent(cell, 6) for cell in r9_cells
    }


def test_gold_left_join_does_not_drop_requirement_keys() -> None:
    hours = [
        datetime(2024, 1, 1, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 1, tzinfo=UTC),
    ]
    weather = _normalize().filter(pl.col("hour") == hours[0])

    result = build_environmental_features(
        requirements=_requirements(hours),
        silver_weather=weather,
    )

    assert result.height == 2
    assert result["hour"].to_list() == hours
    assert result["weather_available"].to_list() == [True, False]
    assert result["lighting_condition"].null_count() == 0


def test_archive_eligible_missing_weather_is_retained_and_reported() -> None:
    first_hour = datetime(2024, 1, 1, tzinfo=UTC)
    hours = [first_hour + timedelta(hours=index) for index in range(100)]
    weather = _normalize(
        _weather_envelope(start=date(2024, 1, 1), end=date(2024, 1, 5))
    ).head(99)

    result = build_environmental_features(
        requirements=_requirements(hours),
        silver_weather=weather,
    )
    summary = validate_environmental_features(
        result,
        archive_cutoff_hour=hours[-1],
    )

    unmatched = result.filter(~pl.col("weather_available"))
    assert result.height == 100
    assert unmatched.height == 1
    assert unmatched["hour"].item() == hours[-1]
    assert unmatched["weather_temperature_2m_c"].item() is None
    assert unmatched["weather_relative_humidity_2m_pct"].item() is None
    assert summary["row_count"] == 100
    assert summary["weather_available_rows"] == 99
    assert summary["weather_null_rows"] == 1
    assert summary["weather_coverage_pct"] == pytest.approx(99.0)
    assert summary["unexpected_archive_eligible_missing_rows"] == 1


def test_missing_weather_tolerance_does_not_weaken_key_uniqueness() -> None:
    hour = datetime(2024, 1, 1, tzinfo=UTC)
    result = build_environmental_features(
        requirements=_requirements([hour]),
        silver_weather=pl.DataFrame(schema=SILVER_WEATHER_SCHEMA),
    )

    with pytest.raises(EnvironmentalContractError, match="duplicate keys"):
        validate_environmental_features(pl.concat([result, result]))


def test_gold_lighting_schema_is_directly_model_table_joinable() -> None:
    hour = datetime(2024, 1, 1, 12, tzinfo=UTC)
    lighting = compute_lighting_features(
        _requirements([hour]).select("h3_cell_id", "hour")
    )

    prepared = prepare_lighting(lighting.lazy()).collect()

    assert prepared["weather_query_cell_id"].to_list() == [_cell()]
    assert prepared["weather_timestamp"].to_list() == [hour]
    assert prepared["_lighting_matched"].to_list() == [True]


def test_identical_input_sets_have_stable_snapshot_identity_and_output() -> None:
    first = _normalize()
    second = _normalize()

    assert _stable_snapshot_id("weather-v2", ["b", "a"]) == _stable_snapshot_id(
        "weather-v2", ["a", "b"]
    )
    assert_frame_equal(first, second)


def test_silver_asset_rerun_reuses_validated_immutable_snapshot(tmp_path: Path) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    raw_uri = f"{lake.model_weather_v2_best_match_root}/year=2024/test.json"
    lake._write_object(
        raw_uri,
        json.dumps(_weather_envelope(), ensure_ascii=False).encode("utf-8"),
        content_type="application/json",
    )

    first = dg.materialize(
        [raw_model_weather_v2, silver_weather_features],
        resources={"crime_lake": lake},
    )
    first_pointer = json.loads(
        Path(lake.silver_weather_latest_pointer_uri).read_text()
    )
    second = dg.materialize(
        [raw_model_weather_v2, silver_weather_features],
        resources={"crime_lake": lake},
    )
    second_pointer = json.loads(
        Path(lake.silver_weather_latest_pointer_uri).read_text()
    )

    assert first.success and second.success
    assert first_pointer == second_pointer
    assert len(list(Path(lake.silver_weather_root).glob("snapshot_id=*"))) == 1
    output = pl.read_parquet(
        lake.silver_weather_year_uri(first_pointer["snapshot_uri"], 2024)
    )
    assert output.height == 24
    assert validate_silver_weather(output)["duplicate_key_count"] == 0


def test_silver_snapshot_reads_objects_concurrently_and_writes_one_year_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    cells = sorted(h3.grid_disk(_cell(), 1))[:4]
    raw_uris = [
        f"{lake.model_weather_v2_best_match_root}/year=2024/object-{index}.json"
        for index in range(len(cells))
    ]
    payloads: dict[str, bytes] = {}
    for index, (uri, cell) in enumerate(zip(raw_uris, cells, strict=True)):
        envelope = _weather_envelope(cell=cell)
        envelope["request_id"] = f"parallel-{index}"
        payloads[uri] = json.dumps(envelope, ensure_ascii=False).encode("utf-8")

    barrier = threading.Barrier(len(raw_uris))
    lock = threading.Lock()
    active_readers = 0
    maximum_active_readers = 0

    def concurrent_read(_lake: CrimeLakeResources, uri: str) -> bytes:
        nonlocal active_readers, maximum_active_readers
        with lock:
            active_readers += 1
            maximum_active_readers = max(maximum_active_readers, active_readers)
        try:
            barrier.wait(timeout=5)
            return payloads[uri]
        finally:
            with lock:
                active_readers -= 1

    monkeypatch.setattr(CrimeLakeResources, "_read_object", concurrent_read)
    progress: list[str] = []
    snapshot_uri = lake.silver_weather_snapshot_uri("parallel-smoke")

    years, summary = _write_silver_weather_snapshot(
        lake=lake,
        raw_uris=raw_uris,
        snapshot_uri=snapshot_uri,
        max_workers=4,
        progress_interval=2,
        progress_log=progress.append,
    )

    assert DEFAULT_RAW_WEATHER_READ_WORKERS == 16
    assert maximum_active_readers == 4
    assert summary["row_count"] == 4 * 24
    assert summary["raw_reader_workers"] == 4
    assert years[0]["raw_object_count"] == 4
    assert len(list(Path(snapshot_uri).rglob("*.parquet"))) == 1
    assert any("year=2024 objects=2/4 rows_written=48" in line for line in progress)
    assert any("year=2024 objects=4/4 rows_written=96" in line for line in progress)


def test_concurrent_silver_failure_includes_source_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    bad_uri = f"{lake.model_weather_v2_best_match_root}/year=2024/bad.json"
    envelope = _weather_envelope()
    units = envelope["hourly_units"]
    assert isinstance(units, dict)
    units["temperature_2m"] = "°F"

    def invalid_read(_lake: CrimeLakeResources, uri: str) -> bytes:
        assert uri == bad_uri
        return json.dumps(envelope, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(CrimeLakeResources, "_read_object", invalid_read)

    with pytest.raises(RuntimeError, match=rf"raw weather object {bad_uri}"):
        _write_silver_weather_snapshot(
            lake=lake,
            raw_uris=[bad_uri],
            snapshot_uri=lake.silver_weather_snapshot_uri("failed-smoke"),
            max_workers=2,
        )


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        ("cell_year", "Duplicate raw weather logical object"),
        ("request_id", "Duplicate raw weather request_id"),
        ("uri_year", "Raw weather URI year disagrees with payload"),
    ],
)
def test_concurrent_silver_preserves_object_level_coordinator_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error_match: str,
) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    cells = sorted(h3.grid_disk(_cell(), 1))[:2]
    raw_uris = [
        f"{lake.model_weather_v2_best_match_root}/year=2024/a.json",
        f"{lake.model_weather_v2_best_match_root}/year=2024/b.json",
    ]
    first = _weather_envelope(cell=cells[0])
    first["request_id"] = "first"
    second = _weather_envelope(cell=cells[1])
    second["request_id"] = "second"
    if case == "cell_year":
        second["weather_query_cell_id"] = cells[0]
    elif case == "request_id":
        second["request_id"] = "first"
    else:
        second = _weather_envelope(
            start=date(2023, 1, 1),
            end=date(2023, 1, 1),
            cell=cells[1],
        )
        second["request_id"] = "second"
    payloads = {
        raw_uris[0]: json.dumps(first, ensure_ascii=False).encode("utf-8"),
        raw_uris[1]: json.dumps(second, ensure_ascii=False).encode("utf-8"),
    }

    def object_read(_lake: CrimeLakeResources, uri: str) -> bytes:
        return payloads[uri]

    monkeypatch.setattr(CrimeLakeResources, "_read_object", object_read)

    with pytest.raises(RuntimeError, match=error_match) as captured:
        _write_silver_weather_snapshot(
            lake=lake,
            raw_uris=raw_uris,
            snapshot_uri=lake.silver_weather_snapshot_uri(f"failed-{case}"),
            max_workers=2,
        )

    assert raw_uris[1] in str(captured.value)


def test_local_gold_asset_smoke_materialization(tmp_path: Path) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    raw_uri = f"{lake.model_weather_v2_best_match_root}/year=2024/test.json"
    lake._write_object(
        raw_uri,
        json.dumps(
            _weather_envelope(start=date(2024, 1, 1), end=date(2024, 1, 5)),
            ensure_ascii=False,
        ).encode("utf-8"),
        content_type="application/json",
    )

    event_snapshot_id = "event-smoke"
    event_snapshot_uri = lake.event_spine_snapshot_uri(event_snapshot_id)
    event_part = Path(
        event_snapshot_uri,
        "source_city=city",
        "occurrence_year=2024",
        "part-00000.parquet",
    )
    event_part.parent.mkdir(parents=True)
    event_hours = [
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index)
        for index in range(121)
    ]
    pl.DataFrame(
        {
            "weather_query_cell_id": pl.Series(
                [_cell()] * len(event_hours), dtype=pl.Int64
            ),
            "occurrence_timestamp_utc": pl.Series(
                event_hours,
                dtype=pl.Datetime("us", time_zone="UTC"),
            ),
        }
    ).write_parquet(event_part)
    lake._write_object(
        lake.event_spine_manifest_uri(event_snapshot_uri),
        json.dumps(
            {
                "snapshot_id": event_snapshot_id,
                "snapshot_uri": event_snapshot_uri,
            }
        ).encode(),
        content_type="application/json",
    )
    lake._write_object(
        lake.event_spine_success_uri(event_snapshot_uri),
        b"",
        content_type="application/octet-stream",
    )
    lake._write_object(
        lake.event_spine_latest_pointer_uri,
        json.dumps(
            {
                "snapshot_id": event_snapshot_id,
                "snapshot_uri": event_snapshot_uri,
            }
        ).encode(),
        content_type="application/json",
    )

    integration_snapshot_id = "integration-smoke"
    integration_snapshot_uri = lake.integration_snapshot_uri(integration_snapshot_id)
    integration_part = Path(
        lake.integration_sample_part_uri(integration_snapshot_uri, "city", 0)
    )
    integration_part.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "osm_h3_cell_id": pl.Series(
                [_cell(41.88, -87.63, resolution=9)], dtype=pl.Int64
            ),
            "integration_timestamp_utc": pl.Series(
                [datetime(2024, 1, 1, 1, 45, tzinfo=UTC)],
                dtype=pl.Datetime("us", time_zone="UTC"),
            ),
        }
    ).write_parquet(integration_part)
    integration_manifest = {
        "snapshot_id": integration_snapshot_id,
        "snapshot_root": integration_snapshot_uri,
        "sources": [{"source_city": "city", "sample_part_count": 1}],
    }
    lake._write_object(
        lake.integration_manifest_uri(integration_snapshot_uri),
        json.dumps(integration_manifest).encode(),
        content_type="application/json",
    )
    lake._write_object(
        lake.integration_success_uri(integration_snapshot_uri),
        b"",
        content_type="application/octet-stream",
    )
    lake._write_object(
        lake.integration_latest_pointer_uri,
        json.dumps(
            {
                "snapshot_id": integration_snapshot_id,
                "snapshot_uri": integration_snapshot_uri,
            }
        ).encode(),
        content_type="application/json",
    )

    result = dg.materialize(
        [
            raw_model_weather_v2,
            published_integration_sampling,
            dg.AssetSpec("gold_event_spine"),
            silver_weather_features,
            environmental_features,
        ],
        resources={"crime_lake": lake},
    )

    assert result.success
    _, manifest = lake.resolve_current_environmental_features_snapshot()
    assert manifest["event_spine_snapshot_id"] == event_snapshot_id
    assert manifest["integration_sampling_snapshot_id"] == integration_snapshot_id
    assert manifest["row_count"] == 121
    assert manifest["unique_h3_cells"] == 1
    assert manifest["weather_available_rows"] == 120
    assert manifest["weather_null_rows"] == 1
    assert manifest["weather_coverage_pct"] == pytest.approx(100.0 * 120 / 121)
    assert manifest["event_weighted_weather_coverage_pct"] == pytest.approx(
        100.0 * 120 / 121
    )
    assert manifest["integration_weighted_weather_coverage_pct"] == 100.0
    assert manifest["unexpected_archive_eligible_missing_rows"] == 1
    assert manifest["duplicate_key_count"] == 0

    output = pl.read_parquet(
        lake.environmental_features_year_uri(manifest["snapshot_uri"], 2024)
    )
    unmatched = output.filter(~pl.col("weather_available"))
    assert output.height == 121
    assert unmatched.height == 1
    assert unmatched["hour"].item() == event_hours[-1]
    assert unmatched["weather_temperature_2m_c"].item() is None
    assert unmatched["weather_relative_humidity_2m_pct"].item() is None


def test_environmental_assets_and_centralized_paths_are_registered(tmp_path) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    keys = {
        key.to_user_string() for key in defs.resolve_asset_graph().get_all_asset_keys()
    }

    assert {
        "raw/model_weather_v2",
        "silver_weather_features",
        "environmental_features",
    } <= keys
    assert lake.model_weather_v2_best_match_root.endswith(
        "/raw_files/landing/weather/model_weather_v2/open_meteo/best_match"
    )
    assert lake.silver_weather_root.endswith("/silver/environmental/weather")
    assert lake.environmental_features_root.endswith("/gold/environmental_features")
