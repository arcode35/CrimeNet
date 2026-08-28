from datetime import UTC, datetime

import numpy as np

from crimenet_data.assets.integration.transforms import resolve_temporal_coverage


def _row(source: str, timezone: str, start: str, end: str) -> dict[str, object]:
    return {
        "source_city": source,
        "source_timezone": timezone,
        "coverage_start_utc": start,
        "coverage_end_utc": end,
        "coverage_basis": "test_full_frozen_support",
        "coverage_reference": "test_manifest",
    }


def test_atlanta_full_support_clips_into_all_three_splits() -> None:
    rows = [
        _row(
            "atlanta",
            "America/New_York",
            "2014-01-01T05:00:00Z",
            "2026-08-27T18:00:00Z",
        )
    ]

    train, train_starts, train_durations = resolve_temporal_coverage(
        rows, source="atlanta", start_year=2014, end_year=2023
    )
    validation, val_starts, val_durations = resolve_temporal_coverage(
        rows, source="atlanta", start_year=2024, end_year=2024
    )
    test, test_starts, test_durations = resolve_temporal_coverage(
        rows, source="atlanta", start_year=2025, end_year=2199
    )

    assert train[0].start_utc == datetime(2014, 1, 1, 5, tzinfo=UTC)
    assert train[-1].end_utc == datetime(2024, 1, 1, 5, tzinfo=UTC)
    assert validation[0].start_utc == datetime(2024, 1, 1, 5, tzinfo=UTC)
    assert validation[-1].end_utc == datetime(2025, 1, 1, 5, tzinfo=UTC)
    assert test[0].start_utc == datetime(2025, 1, 1, 5, tzinfo=UTC)
    assert test[-1].end_utc == datetime(2026, 8, 27, 18, tzinfo=UTC)

    assert train_starts.size == train_durations.size == 1
    assert val_starts.size == val_durations.size == 1
    assert test_starts.size == test_durations.size == 1
    assert np.all(train_durations > 0)
    assert np.all(val_durations > 0)
    assert np.all(test_durations > 0)


def test_source_local_boundary_not_utc_calendar_boundary() -> None:
    rows = [
        _row(
            "san_francisco",
            "America/Los_Angeles",
            "2018-01-01T08:00:00Z",
            "2026-08-27T18:00:00Z",
        )
    ]
    train, _, _ = resolve_temporal_coverage(
        rows, source="san_francisco", start_year=2014, end_year=2023
    )
    assert train[-1].end_utc == datetime(2024, 1, 1, 8, tzinfo=UTC)


def test_explicit_gap_is_preserved() -> None:
    rows = [
        _row(
            "atlanta",
            "America/New_York",
            "2024-01-01T05:00:00Z",
            "2024-06-01T04:00:00Z",
        ),
        _row(
            "atlanta",
            "America/New_York",
            "2024-07-01T04:00:00Z",
            "2025-01-01T05:00:00Z",
        ),
    ]
    intervals, starts, durations = resolve_temporal_coverage(
        rows, source="atlanta", start_year=2024, end_year=2024
    )
    assert len(intervals) == 2
    assert starts.size == durations.size == 2
    assert intervals[0].end_utc < intervals[1].start_utc
