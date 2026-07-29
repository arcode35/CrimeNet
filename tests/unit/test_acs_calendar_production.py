from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from crimenet.socioeconomic.acs_calendar import (
    ACS_CALENDAR_DEFINITION_VERSION,
    ACS_VINTAGE_RELEASES,
    AcsVintageRelease,
    calendar_record_id,
    create_calendar_dataframe,
    select_vintage_releases,
    validate_calendar_dataframe,
    validate_vintage_releases,
)


def test_checked_in_calendar_is_contiguous_and_current() -> None:
    releases = validate_vintage_releases(ACS_VINTAGE_RELEASES)

    assert releases[0] == AcsVintageRelease(
        2012,
        date(2013, 12, 17),
        2012,
        2010,
    )
    assert releases[-1] == AcsVintageRelease(
        2024,
        date(2026, 1, 29),
        2024,
        2020,
    )
    assert [release.acs_vintage for release in releases] == list(range(2012, 2025))


def test_calendar_record_id_is_stable_and_version_aware() -> None:
    release = ACS_VINTAGE_RELEASES[-1]

    first = calendar_record_id(release)
    second = calendar_record_id(release)
    revised = calendar_record_id(
        release,
        definition_version="acs5_release_calendar_v2",
    )

    assert first == second
    assert len(first) == 64
    assert revised != first


def test_select_vintage_releases_requires_a_complete_known_range() -> None:
    selected = select_vintage_releases(
        start_vintage=2021,
        end_vintage=2023,
    )
    assert [release.acs_vintage for release in selected] == [
        2021,
        2022,
        2023,
    ]

    with pytest.raises(ValueError, match="unknown vintages"):
        select_vintage_releases(
            start_vintage=2023,
            end_vintage=2025,
        )


@pytest.mark.parametrize(
    "releases, message",
    [
        (
            (
                ACS_VINTAGE_RELEASES[0],
                ACS_VINTAGE_RELEASES[0],
            ),
            "duplicate vintages",
        ),
        (
            (
                ACS_VINTAGE_RELEASES[0],
                ACS_VINTAGE_RELEASES[2],
            ),
            "contiguous",
        ),
        (
            (
                replace(
                    ACS_VINTAGE_RELEASES[0],
                    tiger_line_year=2013,
                ),
            ),
            "TIGER/Line year",
        ),
    ],
)
def test_invalid_calendar_contracts_are_rejected(
    releases: tuple[AcsVintageRelease, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_vintage_releases(releases)


def test_calendar_spark_candidate_has_stable_schema_and_ids() -> None:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-acs-calendar")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        selected = select_vintage_releases(
            start_vintage=2022,
            end_vintage=2024,
        )
        dataframe = create_calendar_dataframe(
            spark,
            selected,
            definition_version=ACS_CALENDAR_DEFINITION_VERSION,
        )

        validate_calendar_dataframe(dataframe)
        rows = dataframe.orderBy("acs_vintage").collect()

        assert [row["acs_vintage"] for row in rows] == [
            2022,
            2023,
            2024,
        ]
        assert all(len(row["calendar_record_id"]) == 64 for row in rows)
        assert dataframe.schema["acs_vintage"].nullable is False
        assert dataframe.schema["calendar_record_id"].nullable is False
    finally:
        spark.stop()
