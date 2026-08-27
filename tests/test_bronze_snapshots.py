from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from crimenet_data.assets.crime.canonical import (
    CANONICAL_CRIME_SCHEMA,
    CANONICAL_MAPPING_VERSION,
)
from crimenet_data.resources.crime_lake import (
    CrimeLakeResources,
    SilverSnapshotPointer,
)


def _lake(tmp_path: Path) -> CrimeLakeResources:
    return CrimeLakeResources(bucket=str(tmp_path / "object-store"))


def _frame(*years: int) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "occurrence_year": list(years),
            "source_record_id": [f"record-{index}" for index in range(len(years))],
        }
    )


def _write_snapshot(
    lake: CrimeLakeResources,
    snapshot_id: str,
    *years: int,
) -> Path:
    uri = lake.write_bronze_snapshot(
        _frame(*years),
        source_key="new_york",
        snapshot_id=snapshot_id,
        partitioning_columns=["occurrence_year"],
    )
    return Path(uri)


def test_bronze_snapshot_path_is_versioned_under_s3_root() -> None:
    lake = CrimeLakeResources()

    assert lake.bronze_snapshot_uri("new_york", "abc123") == (
        "s3://crimenet-data/bronze/crime/new_york/snapshot_id=abc123"
    )
    assert lake.silver_crime_offenses_uri == (
        "s3://crimenet-data/silver/crime_offenses"
    )
    assert lake.silver_snapshot_uri("snapshot-1") == (
        "s3://crimenet-data/silver/crime_offenses/snapshot_id=snapshot-1"
    )


def test_crime_lake_owns_gold_and_integration_reference_topology() -> None:
    lake = CrimeLakeResources()

    assert lake.raw_root == "s3://crimenet-data/raw_files"
    assert lake.reference_root == (
        "s3://crimenet-data/raw_files/landing/reference"
    )
    assert lake.base_domain_uri == (
        "s3://crimenet-data/raw_files/landing/reference/"
        "integration_sampling/base_domain_h3.csv"
    )
    assert lake.temporal_coverage_uri == (
        "s3://crimenet-data/raw_files/landing/reference/"
        "integration_sampling/source_temporal_coverage.csv"
    )
    assert lake.national_temporal_history_root == (
        "s3://crimenet-data/gold/national_feature_store/temporal/h3_r9/history"
    )
    assert lake.event_spine_snapshot_uri("spine-1") == (
        "s3://crimenet-data/gold/event_spine/snapshot_id=spine-1"
    )
    assert lake.integration_snapshot_uri("run-1") == (
        "s3://crimenet-data/gold/integration_sampling/snapshot_id=run-1"
    )


def test_polars_s3_retries_are_centralized_in_storage_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B2_ENDPOINT_URL", "https://s3.example.invalid")
    monkeypatch.setenv("B2_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "application-key")
    monkeypatch.setenv("B2_REGION", "us-east-005")

    options = CrimeLakeResources().storage_options

    assert options["aws_endpoint_url"] == "https://s3.example.invalid"
    assert options["aws_region"] == "us-east-005"
    assert options["max_retries"] == 15
    assert options["retry_timeout_ms"] == 300_000
    assert options["retry_init_backoff_ms"] == 400
    assert options["retry_max_backoff_ms"] == 20_000
    assert options["retry_base_multiplier"] == 2.0


def _silver_frame(
    *records: tuple[str, int],
    mapped: bool = True,
) -> pl.LazyFrame:
    rows: list[dict[str, object]] = []
    for record_id, year in records:
        row = {name: None for name in CANONICAL_CRIME_SCHEMA}
        row.update(
            {
                "crime_id": f"new_york:{record_id}",
                "source_city": "new_york",
                "source_record_id": record_id,
                "occurrence_timestamp": datetime(year, 1, 2, 3, 4),  # noqa: DTZ001 - source-local Silver time
                "report_timestamp": datetime(year, 1, 3),  # noqa: DTZ001 - source-local Silver time
                "occurrence_year": year,
                "source_timezone": "America/New_York",
                "source_offense_description": "ARSON",
                "mapping_version": CANONICAL_MAPPING_VERSION if mapped else None,
                "canonical_family_code": "F10" if mapped else None,
                "canonical_offense_family": "arson" if mapped else None,
                "canonical_subtype_code": "F10.01" if mapped else None,
                "canonical_offense_subtype": "arson" if mapped else None,
                "canonical_domain": "property" if mapped else None,
                "canonical_target": "property" if mapped else None,
                "is_criminal_event": True if mapped else None,
                "is_violent": False if mapped else None,
                "is_property": True if mapped else None,
                "canonical_mapping_found": mapped,
                "mapping_confidence": "high" if mapped else None,
                "review_required": False,
                "mapping_action": "map" if mapped else None,
                "include_in_model": mapped,
                "source_coordinate_bounds_valid": True,
                "latitude": 40.71,
                "longitude": -74.0,
                "location_label": "Main",
                "location_type": "Street",
                "police_district": "1",
                "local_area": "Manhattan",
                "source_file_uri": "s3://example/source.parquet",
                "ingestion_run_id": "bronze-run",
                "ingested_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
            }
        )
        rows.append(row)
    return pl.DataFrame(rows).cast(CANONICAL_CRIME_SCHEMA).lazy()


def _publish_silver(
    lake: CrimeLakeResources,
    frame: pl.LazyFrame,
    snapshot_id: str,
) -> dict[str, object]:
    return lake.publish_silver_snapshot(
        frame,
        snapshot_id=snapshot_id,
        created_at_utc=datetime(2026, 8, 25, 12, tzinfo=UTC),
        mapping_version=CANONICAL_MAPPING_VERSION,
        schema_version="crime_silver_v1",
        crosswalk_sha256="abc123",
        source_snapshots={"new_york": "s3://bronze/new_york/snapshot_id=input"},
        per_source=[
            {
                "source_city": "new_york",
                "input_rows": len(frame.collect()),
                "output_rows": len(frame.collect()),
                "mapped_rows": len(frame.collect()),
                "unmapped_rows": 0,
                "unexpected_unmapped_rows": 0,
                "include_in_model_rows": len(frame.collect()),
                "drop_rows": 0,
                "excluded_rows": 0,
                "review_required_rows": 0,
            }
        ],
    )


def test_silver_snapshot_publication_and_exact_schema(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    frame = _silver_frame(("one", 2023), ("two", 2024))

    manifest = _publish_silver(lake, frame, "silver-one")
    snapshot = Path(lake.silver_snapshot_uri("silver-one"))

    assert (snapshot / "source_city=new_york" / "occurrence_year=2023").is_dir()
    assert not (snapshot / "source_city%3Dnew_york").exists()
    assert (snapshot / "manifest.json").is_file()
    assert (snapshot / "_SUCCESS").is_file()
    assert Path(lake.silver_crime_offenses_root, "_latest.json").is_file()
    assert manifest["partition_columns"] == ["source_city", "occurrence_year"]
    assert manifest["parquet_file_count"] == 2
    assert manifest["canonical_schema_sha256"] == lake.canonical_schema_sha256()
    assert {
        "snapshot_id",
        "snapshot_uri",
        "created_at_utc",
        "schema_version",
        "mapping_version",
        "crosswalk_sha256",
        "row_count",
        "include_in_model_rows",
        "source_count",
        "partition_columns",
        "parquet_file_count",
        "min_occurrence_timestamp",
        "max_occurrence_timestamp",
        "unexpected_populated_unmapped_rows",
        "review_required_rows",
        "outside_source_bounds_rows",
        "source_snapshots",
        "per_source",
        "canonical_schema",
        "canonical_schema_sha256",
    } <= set(manifest)
    pointer = json.loads(
        Path(lake.silver_crime_offenses_root, "_latest.json").read_text()
    )
    assert set(pointer) == {
        "snapshot_id",
        "snapshot_uri",
        "created_at_utc",
        "mapping_version",
    }

    resolved = lake.resolve_current_silver_snapshot()
    assert resolved == str(snapshot)
    result = lake.scan_silver_snapshot().collect().sort("source_record_id")
    assert result.schema == CANONICAL_CRIME_SCHEMA
    assert result.schema["occurrence_year"] == pl.Int16
    assert result["occurrence_year"].to_list() == [2023, 2024]


def test_silver_snapshot_retains_mapped_out_of_bounds_row(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    frame = _silver_frame(("outside", 2024)).with_columns(
        pl.lit(False).alias("source_coordinate_bounds_valid"),
        pl.lit(False).alias("include_in_model"),
    )

    manifest = _publish_silver(lake, frame, "outside-bounds")
    result = lake.scan_silver_snapshot().collect()

    assert manifest["row_count"] == 1
    assert manifest["include_in_model_rows"] == 0
    assert manifest["outside_source_bounds_rows"] == 1
    assert result["crime_id"].to_list() == ["new_york:outside"]
    assert result["source_coordinate_bounds_valid"].to_list() == [False]
    assert result["include_in_model"].to_list() == [False]


def test_silver_pointer_requires_success_marker(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    snapshot_uri = lake.silver_snapshot_uri("incomplete")
    pointer = SilverSnapshotPointer(
        snapshot_id="incomplete",
        snapshot_uri=snapshot_uri,
        created_at_utc=datetime(2026, 8, 25, 12, tzinfo=UTC),
        mapping_version=CANONICAL_MAPPING_VERSION,
    )
    pointer_path = Path(lake.silver_crime_offenses_root, "_latest.json")
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_bytes(pointer.to_json())

    with pytest.raises(RuntimeError, match="incomplete snapshot"):
        lake.resolve_current_silver_snapshot()


def test_silver_snapshot_reader_isolates_history_and_delta_garbage(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    _publish_silver(lake, _silver_frame(("old", 2023)), "old")
    _publish_silver(lake, _silver_frame(("current", 2024)), "current")
    garbage = Path(lake.silver_crime_offenses_root, "_delta_log")
    garbage.mkdir(parents=True)
    (garbage / "00000000000000000000.json").write_text("invalid legacy delta")

    assert lake.resolve_current_silver_snapshot().endswith("snapshot_id=current")
    result = lake.scan_silver_snapshot().collect()
    assert result["source_record_id"].to_list() == ["current"]


def test_failed_silver_quality_gate_does_not_advance_pointer(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    _publish_silver(lake, _silver_frame(("valid", 2024)), "valid")
    current = lake.resolve_current_silver_snapshot()

    with pytest.raises(RuntimeError, match="quality gate failed"):
        _publish_silver(
            lake,
            _silver_frame(("unmapped", 2024), mapped=False),
            "invalid-unmapped",
        )
    assert lake.resolve_current_silver_snapshot() == current
    assert not Path(lake.silver_snapshot_uri("invalid-unmapped")).exists()


def test_silver_quality_gate_rejects_included_out_of_bounds_row(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    _publish_silver(lake, _silver_frame(("valid", 2024)), "valid")
    current = lake.resolve_current_silver_snapshot()
    invalid = _silver_frame(("outside", 2024)).with_columns(
        pl.lit(False).alias("source_coordinate_bounds_valid")
    )

    with pytest.raises(RuntimeError, match="included_outside_source_bounds_rows"):
        _publish_silver(lake, invalid, "included-outside-bounds")

    assert lake.resolve_current_silver_snapshot() == current
    assert not Path(lake.silver_snapshot_uri("included-outside-bounds")).exists()


def test_failed_silver_schema_gate_does_not_advance_pointer(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    _publish_silver(lake, _silver_frame(("valid", 2024)), "valid")
    current = lake.resolve_current_silver_snapshot()

    invalid = _silver_frame(("bad-schema", 2024)).drop("source_auxiliary")
    with pytest.raises(ValueError, match="schema mismatch"):
        _publish_silver(lake, invalid, "invalid-schema")
    assert lake.resolve_current_silver_snapshot() == current
    assert not Path(lake.silver_snapshot_uri("invalid-schema")).exists()


def test_blank_taxonomy_unmapped_rows_remain_publishable(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    frame = _silver_frame(("blank", 2024), mapped=False).with_columns(
        pl.lit(None, dtype=pl.String).alias("source_offense_description")
    )

    manifest = _publish_silver(lake, frame, "blank-taxonomy")

    assert manifest["unexpected_populated_unmapped_rows"] == 0
    assert lake.resolve_current_silver_snapshot().endswith("snapshot_id=blank-taxonomy")


def test_bronze_parquet_snapshot_round_trip_is_hive_partitioned(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    snapshot = _write_snapshot(lake, "abc123", 2024, 2025, 2024)

    assert (snapshot / "occurrence_year=2024").is_dir()
    assert (snapshot / "occurrence_year=2025").is_dir()
    assert not (snapshot / "_SUCCESS").exists()

    completed_at = datetime(2026, 8, 24, 12, tzinfo=UTC)
    lake.complete_bronze_snapshot(
        source_key="new_york",
        snapshot_id="abc123",
        created_at=completed_at,
    )

    assert (snapshot / "_SUCCESS").is_file()
    assert lake.resolve_current_bronze_snapshot("new_york") == str(snapshot)
    result = lake.scan_bronze_snapshot("new_york").collect().sort("source_record_id")
    assert result["occurrence_year"].to_list() == [2024, 2025, 2024]
    assert result.schema["occurrence_year"] == pl.Int64


def test_newer_incomplete_snapshot_does_not_replace_completed_snapshot(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    first = _write_snapshot(lake, "complete-run", 2024)
    lake.complete_bronze_snapshot(
        source_key="new_york",
        snapshot_id="complete-run",
        created_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
    )

    incomplete = _write_snapshot(lake, "failed-run", 2025)

    assert not (incomplete / "_SUCCESS").exists()
    assert lake.resolve_current_bronze_snapshot("new_york") == str(first)
    assert lake.scan_bronze_snapshot("new_york").collect()[
        "occurrence_year"
    ].to_list() == [2024]


def test_completed_snapshot_is_not_overwritten_by_same_run_id(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    snapshot = _write_snapshot(lake, "stable-run", 2024)
    completed_at = datetime(2026, 8, 24, 12, tzinfo=UTC)
    lake.complete_bronze_snapshot(
        source_key="new_york",
        snapshot_id="stable-run",
        created_at=completed_at,
    )

    _write_snapshot(lake, "stable-run", 2025)

    assert lake.scan_bronze_snapshot("new_york").collect()[
        "occurrence_year"
    ].to_list() == [2024]
    assert (snapshot / "occurrence_year=2024").is_dir()
    assert not (snapshot / "occurrence_year=2025").exists()


def test_older_completion_cannot_rewind_latest_pointer(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    first_time = datetime(2026, 8, 24, 12, tzinfo=UTC)
    second_time = first_time + timedelta(minutes=1)
    _write_snapshot(lake, "older-run", 2024)
    newer = _write_snapshot(lake, "newer-run", 2025)
    lake.complete_bronze_snapshot(
        source_key="new_york",
        snapshot_id="newer-run",
        created_at=second_time,
    )

    lake.complete_bronze_snapshot(
        source_key="new_york",
        snapshot_id="older-run",
        created_at=first_time,
    )

    assert lake.resolve_current_bronze_snapshot("new_york") == str(newer)


def test_missing_or_malformed_pointer_fails_clearly(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    with pytest.raises(FileNotFoundError, match="metadata object not found"):
        lake.resolve_current_bronze_snapshot("new_york")

    pointer_path = Path(lake.resolve_source_path("new_york", "bronze")) / "_latest.json"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_text("not-json")
    with pytest.raises(ValueError, match="Malformed Bronze snapshot pointer"):
        lake.resolve_current_bronze_snapshot("new_york")


def test_pointer_to_snapshot_without_success_marker_is_rejected(
    tmp_path: Path,
) -> None:
    lake = _lake(tmp_path)
    _write_snapshot(lake, "failed-run", 2025)
    pointer_path = Path(lake.resolve_source_path("new_york", "bronze")) / "_latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "source_key": "new_york",
                "snapshot_id": "failed-run",
                "created_at": "2026-08-24T12:00:00+00:00",
            }
        )
    )

    with pytest.raises(RuntimeError, match="incomplete snapshot"):
        lake.resolve_current_bronze_snapshot("new_york")


def test_snapshot_without_parquet_cannot_advance_pointer(tmp_path: Path) -> None:
    lake = _lake(tmp_path)

    with pytest.raises(RuntimeError, match="without Parquet files"):
        lake.complete_bronze_snapshot(
            source_key="new_york",
            snapshot_id="empty-run",
            created_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
        )

    with pytest.raises(FileNotFoundError):
        lake.resolve_current_bronze_snapshot("new_york")
