from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import dagster as dg
import h3.api.basic_int as h3
import polars as pl
import pytest

from crimenet_data.assets.final_model_table.transformations import (
    ENVIRONMENTAL_FEATURE_COLUMNS,
    SOCIOECONOMIC_FEATURE_COLUMNS,
    STATIC_FEATURE_COLUMNS,
    FinalModelContractError,
    assign_model_split,
    attach_environmental_features,
    attach_temporal_features,
    build_final_model_table,
    normalize_event_rows,
    normalize_integration_rows,
    validate_normalized_rows,
    validate_source_temporal_coverage,
)
from crimenet_data.assets.final_model_table.gold import final_model_table
from crimenet_data.resources.crime_lake import CrimeLakeResources
from machine_learning.model_table_io import scan_model_table
from crimenet_data.assets.integration.transforms import TemporalCoverageInterval


CELL_R9 = h3.latlng_to_cell(41.88, -87.63, 9)
CELL_R6 = h3.cell_to_parent(CELL_R9, 6)


def _event(timestamp: datetime, *, crime_id: str = "crime-1") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "crime_id": [crime_id],
            "source_city": ["chicago"],
            "occurrence_timestamp_utc": pl.Series(
                [timestamp], dtype=pl.Datetime("us", time_zone="UTC")
            ),
            "osm_h3_cell_id": pl.Series([CELL_R9], dtype=pl.Int64),
            "weather_query_cell_id": pl.Series([CELL_R6], dtype=pl.Int64),
            "latitude": [41.88],
            "longitude": [-87.63],
            "canonical_family_code": ["property"],
            "canonical_offense_family": ["Property"],
            "canonical_subtype_code": ["burglary"],
            "canonical_offense_subtype": ["Burglary"],
            "is_violent": [False],
            "is_property": [True],
        }
    )


def _integration(timestamp: datetime, *, split: str = "validation") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_city": ["chicago"],
            "integration_sample_id": ["sample-1"],
            "sample_index": pl.Series([1], dtype=pl.Int64),
            "integration_timestamp_utc": pl.Series(
                [timestamp], dtype=pl.Datetime("us", time_zone="UTC")
            ),
            "osm_h3_cell_id": pl.Series([CELL_R9], dtype=pl.Int64),
            "latitude": [41.88],
            "longitude": [-87.63],
            "integration_weight_cell_seconds": [123.5],
            "split": [split],
        }
    )


def _history() -> pl.DataFrame:
    values: dict[str, pl.Series | list[object]] = {
        "osm_h3_cell_id": pl.Series([CELL_R9], dtype=pl.Int64),
        "feature_available_at": pl.Series(
            [datetime(2023, 12, 1, tzinfo=UTC)],
            dtype=pl.Datetime("us", time_zone="UTC"),
        ),
    }
    for column in [*SOCIOECONOMIC_FEATURE_COLUMNS, *STATIC_FEATURE_COLUMNS]:
        values[column] = [1.0]
    return pl.DataFrame(values)


def _environmental(
    timestamps: list[datetime], *, unavailable_first: bool = False
) -> pl.DataFrame:
    available = [not unavailable_first, *([True] * (len(timestamps) - 1))]
    values: dict[str, pl.Series | list[object]] = {
        "h3_cell_id": pl.Series([CELL_R6] * len(timestamps), dtype=pl.Int64),
        "hour": pl.Series(
            [value.replace(minute=0, second=0, microsecond=0) for value in timestamps],
            dtype=pl.Datetime("us", time_zone="UTC"),
        ),
        "weather_temperature_2m_c": [None if not flag else 12.0 for flag in available],
        "weather_relative_humidity_2m_pct": [
            None if not flag else 65.0 for flag in available
        ],
        "weather_available": available,
        "solar_elevation_deg": [20.0] * len(timestamps),
        "solar_zenith_deg": [70.0] * len(timestamps),
        "solar_azimuth_deg": [180.0] * len(timestamps),
        "lighting_condition": ["day"] * len(timestamps),
        "is_daylight": [True] * len(timestamps),
    }
    return pl.DataFrame(values).select("h3_cell_id", "hour", *ENVIRONMENTAL_FEATURE_COLUMNS)


def _coverage(
    start: datetime, end: datetime
) -> TemporalCoverageInterval:
    return TemporalCoverageInterval(
        source_city="chicago",
        source_timezone="America/Chicago",
        start_utc=start,
        end_utc=end,
        coverage_basis="test",
        coverage_reference="fixture",
    )


def test_canonical_split_boundaries_are_half_open() -> None:
    timestamps = [
        datetime(2023, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
    ]
    rows = pl.DataFrame(
        {
            "model_timestamp_utc": pl.Series(
                timestamps, dtype=pl.Datetime("us", time_zone="UTC")
            )
        }
    )

    result = assign_model_split(
        rows, dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC)
    ).collect()

    assert result["split"].to_list() == [
        "train",
        "validation",
        "validation",
        "test",
    ]


def test_source_non_calendar_start_and_gap_fail_closed() -> None:
    intervals = [
        _coverage(
            datetime(2023, 3, 15, 6, tzinfo=UTC),
            datetime(2023, 6, 1, tzinfo=UTC),
        ),
        _coverage(
            datetime(2023, 7, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
    ]
    valid = pl.DataFrame(
        {
            "source_city": ["chicago"],
            "model_timestamp_utc": [datetime(2023, 3, 15, 6, tzinfo=UTC)],
        }
    )
    gap = pl.DataFrame(
        {
            "source_city": ["chicago"],
            "model_timestamp_utc": [datetime(2023, 6, 15, tzinfo=UTC)],
        }
    )

    assert validate_source_temporal_coverage(valid, intervals=intervals) == {
        "rows_outside_source_temporal_coverage": 0
    }
    with pytest.raises(FinalModelContractError, match="outside authoritative"):
        validate_source_temporal_coverage(gap, intervals=intervals)


def test_weather_unavailable_row_is_retained_with_null_features() -> None:
    timestamp = datetime(2024, 2, 1, 12, 30, tzinfo=UTC)
    rows = normalize_event_rows(
        _event(timestamp), dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC)
    )

    result = attach_environmental_features(
        rows, _environmental([timestamp], unavailable_first=True)
    ).collect()

    assert result.height == 1
    assert result["_environmental_row_exists"].item() is True
    assert result["weather_available"].item() is False
    assert result["weather_temperature_2m_c"].item() is None
    assert result["weather_relative_humidity_2m_pct"].item() is None


def test_structural_environmental_miss_is_distinct_from_weather_unavailable() -> None:
    timestamp = datetime(2024, 2, 1, 12, 30, tzinfo=UTC)
    rows = normalize_event_rows(
        _event(timestamp), dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC)
    )
    empty = _environmental([timestamp]).head(0)

    result = attach_environmental_features(rows, empty).collect()

    assert result.height == 1
    assert result["_environmental_row_exists"].item() is False
    assert result["weather_available"].item() is False


def test_integration_weight_is_preserved_and_event_weight_is_null() -> None:
    timestamp = datetime(2024, 2, 1, tzinfo=UTC)
    integration = normalize_integration_rows(
        _integration(timestamp), dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC)
    ).collect()
    event = normalize_event_rows(
        _event(timestamp), dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC)
    ).collect()

    assert integration["integration_weight_cell_seconds"].item() == 123.5
    assert event["integration_weight_cell_seconds"].item() is None


def test_integration_split_mismatch_fails() -> None:
    timestamp = datetime(2024, 2, 1, tzinfo=UTC)
    rows = normalize_integration_rows(
        _integration(timestamp, split="train"),
        dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with pytest.raises(FinalModelContractError, match="split_mismatch"):
        validate_normalized_rows(rows)


def test_duplicate_row_id_and_null_structural_key_fail() -> None:
    timestamp = datetime(2024, 2, 1, tzinfo=UTC)
    rows = normalize_event_rows(
        pl.concat([_event(timestamp), _event(timestamp)]),
        dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC),
    ).with_columns(pl.lit(None, dtype=pl.Int64).alias("weather_query_cell_id"))

    with pytest.raises(FinalModelContractError, match="duplicate_row_ids"):
        validate_normalized_rows(rows)

    null_only = normalize_event_rows(
        _event(timestamp, crime_id="unique"),
        dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC),
    ).with_columns(pl.lit(None, dtype=pl.Int64).alias("weather_query_cell_id"))
    with pytest.raises(FinalModelContractError, match="null_structural_rows"):
        validate_normalized_rows(null_only)


def test_feature_joins_cannot_multiply_rows() -> None:
    timestamp = datetime(2024, 2, 1, tzinfo=UTC)
    rows = normalize_event_rows(
        _event(timestamp), dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC)
    )
    duplicate_history = pl.concat([_history(), _history()])

    with pytest.raises(FinalModelContractError, match="duplicate keys"):
        attach_temporal_features(rows, duplicate_history)


def test_temporal_feature_join_selects_only_information_available_at_query() -> None:
    timestamp = datetime(2024, 2, 1, tzinfo=UTC)
    rows = normalize_event_rows(
        _event(timestamp), dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC)
    )
    older = _history()
    future = _history().with_columns(
        pl.lit(datetime(2024, 3, 1, tzinfo=UTC)).alias("feature_available_at"),
        pl.lit(999.0).alias("socio_population"),
    )

    result = attach_temporal_features(rows, pl.concat([older, future])).collect()

    assert result["socio_population"].item() == 1.0
    assert result["_feature_available_at"].item() < timestamp


def test_small_full_table_preserves_nullable_weather() -> None:
    event_time = datetime(2024, 1, 1, 1, tzinfo=UTC)
    integration_time = event_time + timedelta(hours=1)
    table, summary = build_final_model_table(
        events=_event(event_time),
        integration=_integration(integration_time),
        environmental=_environmental(
            [event_time, integration_time], unavailable_first=True
        ),
        temporal_history=_history(),
        coverage_intervals=[
            _coverage(
                datetime(2023, 3, 15, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
        dataset_end_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    result = table.collect()

    assert result.height == 2
    assert summary["structural_environmental_missing_rows"] == 0
    assert summary["weather_unavailable_rows"] == 1
    assert result.filter(pl.col("row_type") == "event")[
        "weather_temperature_2m_c"
    ].item() is None


def _publish_pointer(
    lake: CrimeLakeResources,
    *,
    pointer_uri: str,
    snapshot_id: str,
    snapshot_uri: str,
) -> None:
    lake._write_object(
        pointer_uri,
        json.dumps(
            {"snapshot_id": snapshot_id, "snapshot_uri": snapshot_uri}
        ).encode(),
        content_type="application/json",
    )


def test_final_model_asset_publishes_immutable_partitioned_snapshot(
    tmp_path: Path,
) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    event_time = datetime(2023, 12, 1, 1, tzinfo=UTC)
    integration_time = event_time + timedelta(hours=1)

    event_id = "event-test"
    event_uri = lake.event_spine_snapshot_uri(event_id)
    event_part = (
        Path(event_uri)
        / "source_city=chicago"
        / "occurrence_year=2023"
        / "part-00000.parquet"
    )
    event_part.parent.mkdir(parents=True)
    _event(event_time).drop("source_city").write_parquet(event_part)
    lake._write_object(
        lake.event_spine_manifest_uri(event_uri),
        json.dumps(
            {
                "snapshot_id": event_id,
                "snapshot_uri": event_uri,
                "row_count": 1,
            }
        ).encode(),
        content_type="application/json",
    )
    lake._write_object(
        lake.event_spine_success_uri(event_uri),
        b"",
        content_type="application/octet-stream",
    )
    _publish_pointer(
        lake,
        pointer_uri=lake.event_spine_latest_pointer_uri,
        snapshot_id=event_id,
        snapshot_uri=event_uri,
    )

    integration_id = "integration-test"
    integration_uri = lake.integration_snapshot_uri(integration_id)
    integration_part = Path(
        lake.integration_sample_part_uri(integration_uri, "chicago", 0)
    )
    integration_part.parent.mkdir(parents=True)
    _integration(integration_time, split="train").drop(
        "integration_sample_id", "split", "integration_weight_cell_seconds"
    ).with_columns(pl.lit(1.5).alias("mc_weight_cell_hours")).write_parquet(
        integration_part
    )
    integration_manifest = {
        "snapshot_id": integration_id,
        "snapshot_root": integration_uri,
        "train_end_year": 2023,
        "sources": [
            {"source_city": "chicago", "sample_part_count": 1}
        ],
    }
    lake._write_object(
        lake.integration_manifest_uri(integration_uri),
        json.dumps(integration_manifest).encode(),
        content_type="application/json",
    )
    lake._write_object(
        lake.integration_success_uri(integration_uri),
        b"",
        content_type="application/octet-stream",
    )
    _publish_pointer(
        lake,
        pointer_uri=lake.integration_latest_pointer_uri,
        snapshot_id=integration_id,
        snapshot_uri=integration_uri,
    )

    environmental_id = "environmental-test"
    environmental_uri = lake.environmental_features_snapshot_uri(environmental_id)
    environmental_part = Path(
        lake.environmental_features_year_uri(environmental_uri, 2023)
    )
    environmental_part.parent.mkdir(parents=True)
    _environmental([event_time, integration_time], unavailable_first=True).write_parquet(
        environmental_part
    )
    lake._write_object(
        lake.snapshot_manifest_uri(environmental_uri),
        json.dumps(
            {
                "snapshot_id": environmental_id,
                "snapshot_uri": environmental_uri,
            }
        ).encode(),
        content_type="application/json",
    )
    lake._write_object(
        lake.snapshot_success_uri(environmental_uri),
        b"",
        content_type="application/octet-stream",
    )
    _publish_pointer(
        lake,
        pointer_uri=lake.environmental_features_latest_pointer_uri,
        snapshot_id=environmental_id,
        snapshot_uri=environmental_uri,
    )

    history_part = (
        Path(lake.national_temporal_history_root)
        / "feature_available_date=2023-01-01"
        / "version_id=test"
        / "part-00000.parquet"
    )
    history_part.parent.mkdir(parents=True)
    _history().write_parquet(history_part)
    coverage_path = Path(lake.temporal_coverage_uri)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "source_city": ["chicago"],
            "source_timezone": ["America/Chicago"],
            "coverage_start_utc": ["2023-03-15T00:00:00Z"],
            "coverage_end_utc": ["2024-01-01T00:00:00Z"],
            "coverage_basis": ["test"],
            "coverage_reference": ["fixture"],
        }
    ).write_csv(coverage_path)

    result = dg.materialize(
        [
            dg.AssetSpec("gold_event_spine"),
            dg.AssetSpec("published_integration_sampling"),
            dg.AssetSpec("environmental_features"),
            final_model_table,
        ],
        resources={"crime_lake": lake},
    )

    assert result.success
    snapshot_uri, manifest = lake.resolve_current_final_model_table_snapshot()
    assert manifest["row_count"] == 2
    assert manifest["event_rows"] == 1
    assert manifest["integration_rows"] == 1
    assert manifest["weather_unavailable_rows"] == 1
    assert manifest["duplicate_row_ids"] == 0
    assert manifest["event_spine_snapshot_id"] == event_id
    assert manifest["integration_snapshot_id"] == integration_id
    assert manifest["environmental_snapshot_id"] == environmental_id
    assert list(Path(snapshot_uri).glob("split=train/source_city=chicago/*.parquet"))
    assert scan_model_table(snapshot_uri).select(pl.len()).collect().item() == 2
