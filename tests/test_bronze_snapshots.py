from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from crimenet_data.resources.crime_lake import CrimeLakeResources


def _lake(tmp_path: Path) -> CrimeLakeResources:
    return CrimeLakeResources(
        bucket=str(tmp_path / "object-store"),
        delta_bucket=str(tmp_path / "delta-store"),
    )


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
    assert lake.resolve_source_path("new_york", "silver").startswith("b2://")


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
    assert options["max_retries"] == "5"


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
