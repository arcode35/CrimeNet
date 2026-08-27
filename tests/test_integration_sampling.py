from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import h3
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from crimenet_data.assets.integration.transforms import (
    INTEGRATION_DOMAIN_SCHEMA,
    INTEGRATION_SAMPLE_SCHEMA,
    build_frozen_source_domain,
    effective_coverage_local_years,
    integration_sample_count,
    monte_carlo_cell_hour_weight,
    prepare_h3_sampling_geometry,
    resolve_temporal_coverage,
    sample_integration_chunk,
    sample_latlon_within_h3,
    select_training_events,
    source_seed,
    validate_authoritative_boundary_years,
)
from crimenet_data.assets.integration import integration_sampling_job
from crimenet_data.assets.integration.build import _event_spine_dropped_rows
from crimenet_data.resources.crime_lake import CrimeLakeResources


def _cell(latitude: float, longitude: float) -> int:
    return h3.str_to_int(h3.latlng_to_cell(latitude, longitude, 9))


def _utc_us(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000)


def _coverage_row(
    source: str,
    start: str,
    end: str,
    *,
    timezone: str = "UTC",
    reference: str = "publisher metadata fixture",
) -> dict[str, str]:
    return {
        "source_city": source,
        "source_timezone": timezone,
        "coverage_start_utc": start,
        "coverage_end_utc": end,
        "coverage_basis": "publisher_documentation",
        "coverage_reference": reference,
    }


def test_frozen_domain_union_records_each_origin() -> None:
    official_only = _cell(40.70, -74.02)
    overlap = _cell(40.71, -74.01)
    observed_only = _cell(40.72, -74.00)

    cells, domain = build_frozen_source_domain(
        source="new_york",
        official_cells=np.asarray([official_only, overlap], dtype=np.int64),
        event_cells=np.asarray([overlap, observed_only], dtype=np.int64),
    )

    assert set(cells.tolist()) == {official_only, overlap, observed_only}
    assert domain.schema == INTEGRATION_DOMAIN_SCHEMA
    assert {
        row["osm_h3_cell_id"]: row["domain_origin"]
        for row in domain.iter_rows(named=True)
    } == {
        official_only: "official_only",
        overlap: "official_and_observed",
        observed_only: "training_event_extension",
    }


def test_monte_carlo_weight_and_sample_count_contract() -> None:
    sample_count = integration_sample_count(
        observed_event_count=11,
        samples_per_event=5,
    )

    assert sample_count == 55
    assert monte_carlo_cell_hour_weight(
        domain_cell_count=7,
        temporal_support_hours=87_648.0,
        sample_count=sample_count,
    ) == pytest.approx(7 * 87_648.0 / 55)


def test_temporal_coverage_is_declared_clipped_and_can_have_gaps() -> None:
    rows = [
        _coverage_row("source", "2013-12-01T00:00:00Z", "2014-03-01T00:00:00Z"),
        _coverage_row("source", "2014-06-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    ]
    intervals, starts_us, durations_us = resolve_temporal_coverage(
        rows,
        source="source",
        start_year=2014,
        end_year=2014,
    )

    assert [interval.start_utc.isoformat() for interval in intervals] == [
        "2014-01-01T00:00:00+00:00",
        "2014-06-01T00:00:00+00:00",
    ]
    assert [interval.end_utc.isoformat() for interval in intervals] == [
        "2014-03-01T00:00:00+00:00",
        "2015-01-01T00:00:00+00:00",
    ]
    assert starts_us.tolist() == [
        _utc_us("2014-01-01T00:00:00Z"),
        _utc_us("2014-06-01T00:00:00Z"),
    ]
    assert durations_us.sum() / 3_600_000_000 == (59 + 214) * 24


def test_temporal_coverage_fails_closed_when_missing_or_overlapping() -> None:
    with pytest.raises(RuntimeError, match="no outcome-independent"):
        resolve_temporal_coverage([], source="missing", start_year=2014, end_year=2023)

    rows = [
        _coverage_row("source", "2014-01-01T00:00:00Z", "2014-07-01T00:00:00Z"),
        _coverage_row("source", "2014-06-01T00:00:00Z", "2015-01-01T00:00:00Z"),
    ]
    with pytest.raises(ValueError, match="overlap"):
        resolve_temporal_coverage(rows, source="source", start_year=2014, end_year=2014)


def test_effective_coverage_local_years_respects_half_open_local_year_end() -> None:
    intervals, _, _ = resolve_temporal_coverage(
        [
            _coverage_row(
                "source",
                "2022-12-31T05:00:00Z",
                "2023-01-01T05:00:00Z",
                timezone="America/New_York",
            )
        ],
        source="source",
        start_year=2022,
        end_year=2023,
    )

    assert effective_coverage_local_years(intervals) == [2022]


def test_missing_authoritative_boundary_vintage_fails_closed() -> None:
    with pytest.raises(
        RuntimeError, match=r"missing authoritative boundary vintages: \[2019\]"
    ):
        validate_authoritative_boundary_years(
            source="source",
            effective_local_coverage_years=[2018, 2019, 2020],
            authoritative_boundary_years=[2018, 2020],
        )


@pytest.mark.parametrize("manifest", [{}, {"dropped_rows": 3}])
def test_event_spine_losslessness_guard_rejects_unsafe_manifests(
    manifest: dict[str, int],
) -> None:
    with pytest.raises(RuntimeError, match="observed-event H3 footprint"):
        _event_spine_dropped_rows(manifest, require_zero=True)

    assert _event_spine_dropped_rows(manifest, require_zero=False) == int(
        manifest.get("dropped_rows", 0)
    )


def test_event_spine_losslessness_guard_requires_retain_policy() -> None:
    with pytest.raises(RuntimeError, match="does not retain feature-unmatched"):
        _event_spine_dropped_rows(
            {"dropped_rows": 0, "unmatched_history_policy": "drop"},
            require_zero=True,
        )

    assert (
        _event_spine_dropped_rows(
            {
                "dropped_rows": 0,
                "unmatched_history_policy": "retain_event_with_null_features",
            },
            require_zero=True,
        )
        == 0
    )


def test_random_subcell_points_map_back_and_are_not_centroids() -> None:
    cell = _cell(38.9072, -77.0369)
    domain_cells = np.asarray([cell], dtype=np.int64)
    geometry = prepare_h3_sampling_geometry(domain_cells)
    latitude, longitude = sample_latlon_within_h3(
        selected_cell_indices=np.zeros(256, dtype=np.int64),
        geometry=geometry,
        rng=np.random.default_rng(42),
    )

    cell_hex = h3.int_to_str(cell)
    assert {
        h3.latlng_to_cell(float(lat), float(lon), 9)
        for lat, lon in zip(latitude, longitude, strict=True)
    } == {cell_hex}
    center_latitude, center_longitude = h3.cell_to_latlng(cell_hex)
    assert np.any(np.abs(latitude - center_latitude) > 1e-10)
    assert np.any(np.abs(longitude - center_longitude) > 1e-10)


def test_same_source_seed_produces_identical_sample_output() -> None:
    domain_cells = np.asarray(
        [_cell(47.60, -122.34), _cell(47.61, -122.33)],
        dtype=np.int64,
    )
    geometry = prepare_h3_sampling_geometry(domain_cells)
    starts_us = np.asarray(
        [_utc_us("2014-01-01T00:00:00Z"), _utc_us("2015-07-01T00:00:00Z")],
        dtype=np.int64,
    )
    durations_us = np.asarray(
        [31 * 24 * 3_600_000_000, 31 * 24 * 3_600_000_000],
        dtype=np.int64,
    )

    def sample() -> pl.DataFrame:
        return sample_integration_chunk(
            source="seattle",
            start_row=0,
            n=50,
            domain_cells=domain_cells,
            geometry=geometry,
            starts_us=starts_us,
            durations_us=durations_us,
            mc_weight_cell_hours=123.0,
            rng=np.random.default_rng(source_seed(2026, "seattle")),
        )

    first = sample()
    second = sample()
    assert_frame_equal(first, second)
    assert first.schema == INTEGRATION_SAMPLE_SCHEMA
    assert first.height == 50
    assert first["source_city"].unique().to_list() == ["seattle"]
    sampled_us = first["integration_timestamp_utc"].dt.epoch("us")
    in_first = sampled_us.is_between(
        starts_us[0], starts_us[0] + durations_us[0] - 1, closed="both"
    )
    in_second = sampled_us.is_between(
        starts_us[1], starts_us[1] + durations_us[1] - 1, closed="both"
    )
    assert (in_first | in_second).all()


def test_training_event_selection_is_source_specific_and_excludes_2024_plus() -> None:
    starts_us = np.asarray([_utc_us("2014-03-01T00:00:00Z")], dtype=np.int64)
    durations_us = np.asarray(
        [_utc_us("2024-01-01T00:00:00Z") - starts_us[0]], dtype=np.int64
    )
    selected = select_training_events(
        pl.LazyFrame(
            {
                "source_city": ["a", "a", "a", "a", "b"],
                "occurrence_timestamp_utc": [
                    datetime(2014, 2, 28, tzinfo=UTC),
                    datetime(2014, 3, 1, tzinfo=UTC),
                    datetime(2023, 12, 31, 23, 59, tzinfo=UTC),
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2020, 1, 1, tzinfo=UTC),
                ],
                "osm_h3_cell_id": [1, 2, 3, 4, 5],
            }
        ),
        source="a",
        source_timezone="UTC",
        starts_us=starts_us,
        durations_us=durations_us,
    ).collect()

    assert selected.to_dict(as_series=False) == {
        "event_year": [2014, 2023],
        "osm_h3_cell_id": [2, 3],
    }


def test_training_event_selection_fails_on_invalid_required_spine_fields() -> None:
    with pytest.raises(RuntimeError, match="invalid required fields"):
        select_training_events(
            pl.LazyFrame(
                {
                    "source_city": ["a"],
                    "occurrence_timestamp_utc": pl.Series(
                        [None], dtype=pl.Datetime("us", time_zone="UTC")
                    ),
                    "osm_h3_cell_id": pl.Series([None], dtype=pl.Int64),
                }
            ),
            source="a",
            source_timezone="UTC",
            starts_us=np.asarray(
                [_utc_us("2014-01-01T00:00:00Z")], dtype=np.int64
            ),
            durations_us=np.asarray([24 * 3_600_000_000], dtype=np.int64),
        )


def test_job_publishes_source_partitions_and_training_only_sample_counts(
    tmp_path: Path,
) -> None:
    lake = CrimeLakeResources(bucket=str(tmp_path / "lake"))
    snapshot_uri = lake.event_spine_snapshot_uri("spine-test")
    source_cells = {
        "alpha": _cell(40.71, -74.01),
        "beta": _cell(47.61, -122.33),
    }
    alpha_outside_coverage_boundary_cell = _cell(40.73, -73.99)

    event_rows = {
        "alpha": {
            2014: [source_cells["alpha"], source_cells["alpha"]],
            2024: [_cell(40.72, -74.00)],
        },
        "beta": {2023: [source_cells["beta"]]},
    }
    for source, years in event_rows.items():
        for year, cells in years.items():
            path = Path(
                snapshot_uri,
                f"source_city={source}",
                f"occurrence_year={year}",
                "part-00000.parquet",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "osm_h3_cell_id": cells,
                    "occurrence_timestamp_utc": [
                        datetime(year, 6, 1, tzinfo=UTC)
                    ]
                    * len(cells),
                }
            ).write_parquet(path)

    Path(lake.event_spine_manifest_uri(snapshot_uri)).write_text(
        json.dumps(
            {
                "snapshot_id": "spine-test",
                "snapshot_uri": snapshot_uri,
                "dropped_rows": 0,
                "unmatched_history_policy": "retain_event_with_null_features",
            }
        )
    )
    Path(lake.event_spine_success_uri(snapshot_uri)).write_bytes(b"")
    Path(lake.event_spine_latest_pointer_uri).write_text(
        json.dumps({"snapshot_id": "spine-test", "snapshot_uri": snapshot_uri})
    )

    base_domain_uri = Path(lake.base_domain_uri)
    base_domain_uri.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "source_city": ["alpha", "alpha", "beta"],
            "event_year": [2014, 2023, 2023],
            "h3_r9": [
                h3.int_to_str(source_cells["alpha"]),
                h3.int_to_str(alpha_outside_coverage_boundary_cell),
                h3.int_to_str(source_cells["beta"]),
            ],
        }
    ).write_csv(base_domain_uri)

    temporal_coverage_uri = Path(lake.temporal_coverage_uri)
    pl.DataFrame(
        [
            _coverage_row(
                "alpha", "2014-01-01T00:00:00Z", "2014-07-01T00:00:00Z"
            ),
            _coverage_row(
                "beta", "2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"
            ),
        ]
    ).write_csv(temporal_coverage_uri)

    result = integration_sampling_job.execute_in_process(
        resources={"crime_lake": lake},
        run_config={
            "ops": {
                "emit_integration_source_plans": {
                    "config": {
                        "train_start_year": 2014,
                        "train_end_year": 2023,
                        "samples_per_event": 2,
                        "chunk_rows": 3,
                        "seed": 2026,
                        "sources": [],
                        "require_zero_dropped_spine_rows": True,
                    }
                }
            }
        },
    )

    assert result.success
    output_root = Path(lake.integration_root)
    snapshot_root = output_root / f"snapshot_id={result.run_id}"
    manifest = json.loads((snapshot_root / "manifest.json").read_text())
    assert manifest["integration_sample_rows"] == 6
    assert manifest["source_count"] == 2
    assert manifest["train_start_year"] == 2014
    assert manifest["train_end_year"] == 2023
    assert manifest["temporal_coverage_uri"] == str(temporal_coverage_uri)
    assert manifest["base_domain_uri"] == str(base_domain_uri)
    assert "crime timestamps" in manifest["temporal_coverage_policy"]
    assert manifest["chunk_rows"] == 3
    assert (snapshot_root / "_SUCCESS").is_file()
    assert json.loads((output_root / "_latest.json").read_text())["snapshot_id"] == (
        result.run_id
    )

    expected_counts = {"alpha": 4, "beta": 2}
    for source, expected_count in expected_counts.items():
        domain = pl.read_parquet(
            snapshot_root / "domain" / f"source_city={source}" / "*.parquet"
        )
        samples = pl.read_parquet(
            snapshot_root / "samples" / f"source_city={source}" / "*.parquet"
        )
        assert domain["source_city"].unique().to_list() == [source]
        assert samples["source_city"].unique().to_list() == [source]
        assert samples.height == expected_count
        assert samples["integration_timestamp_utc"].dt.year().max() <= 2023
        if source == "alpha":
            assert alpha_outside_coverage_boundary_cell not in set(
                domain["osm_h3_cell_id"].to_list()
            )

    sources = {row["source_city"]: row for row in manifest["sources"]}
    assert sources["alpha"]["temporal_support_hours"] == 181 * 24
    assert sources["beta"]["temporal_support_hours"] == 214 * 24
    assert sources["alpha"]["temporal_coverage_intervals"] == [
        {
            "coverage_basis": "publisher_documentation",
            "coverage_end_utc": "2014-07-01T00:00:00+00:00",
            "coverage_reference": "publisher metadata fixture",
            "coverage_start_utc": "2014-01-01T00:00:00+00:00",
            "source_timezone": "UTC",
        }
    ]
    assert sources["alpha"]["effective_local_coverage_years"] == [2014]
    assert sources["alpha"]["authoritative_boundary_years"] == [2014]
    assert sources["alpha"]["chunk_rows"] == 3
    assert sources["alpha"]["domain_readback_rows"] == (
        sources["alpha"]["integration_domain_h3_cells"]
    )
    assert sources["alpha"]["sample_readback_rows"] == 4
    assert sources["alpha"]["mc_weight_cell_hours"] == pytest.approx(
        sources["alpha"]["integration_domain_h3_cells"] * 181 * 24 / 4
    )


def test_integration_job_has_no_required_canonical_storage_path_config() -> None:
    config_type = integration_sampling_job.get_node_named(
        "emit_integration_source_plans"
    ).definition.config_schema.config_type
    fields = config_type.fields

    assert "base_domain_uri" not in fields
    assert "temporal_coverage_uri" not in fields
    assert "output_root" not in fields
    assert "event_spine_snapshot_uri" not in fields
    assert not fields["event_spine_snapshot_override_uri"].is_required
