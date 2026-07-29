"""Versioned ACS 5-year release calendar used for point-in-time joins.

The dates in this module are public release dates, not survey period end
dates.  A vintage becomes eligible on the day after its public release so a
crime record can never be enriched with information that was not yet public.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


ACS_CALENDAR_DEFINITION_VERSION = "acs5_release_calendar_v1"
ACS_RELEASE_SOURCE_URL = (
    "https://www.census.gov/programs-surveys/acs/news/data-releases.html"
)


@dataclass(frozen=True, order=True)
class AcsVintageRelease:
    """One published ACS 5-year vintage and its compatible geography."""

    acs_vintage: int
    acs_release_date: date
    tiger_line_year: int
    tract_definition_vintage: int


# Public release dates transcribed from the Census ACS release schedules.
# 2020 was delayed until 2022 and 2024 was delayed until January 2026.
ACS_VINTAGE_RELEASES: tuple[AcsVintageRelease, ...] = (
    AcsVintageRelease(2012, date(2013, 12, 17), 2012, 2010),
    AcsVintageRelease(2013, date(2014, 12, 4), 2013, 2010),
    AcsVintageRelease(2014, date(2015, 12, 3), 2014, 2010),
    AcsVintageRelease(2015, date(2016, 12, 8), 2015, 2010),
    AcsVintageRelease(2016, date(2017, 12, 7), 2016, 2010),
    AcsVintageRelease(2017, date(2018, 12, 6), 2017, 2010),
    AcsVintageRelease(2018, date(2019, 12, 19), 2018, 2010),
    AcsVintageRelease(2019, date(2020, 12, 10), 2019, 2010),
    AcsVintageRelease(2020, date(2022, 3, 17), 2020, 2020),
    AcsVintageRelease(2021, date(2022, 12, 8), 2021, 2020),
    AcsVintageRelease(2022, date(2023, 12, 7), 2022, 2020),
    AcsVintageRelease(2023, date(2024, 12, 12), 2023, 2020),
    AcsVintageRelease(2024, date(2026, 1, 29), 2024, 2020),
)


def _stable_digest(parts: Iterable[object]) -> str:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calendar_record_id(
    release: AcsVintageRelease,
    *,
    definition_version: str = ACS_CALENDAR_DEFINITION_VERSION,
) -> str:
    """Return a deterministic identity for one versioned calendar record."""

    if not definition_version.strip():
        raise ValueError("definition_version cannot be blank.")

    return _stable_digest(
        (
            definition_version,
            release.acs_vintage,
            release.acs_release_date.isoformat(),
            release.tiger_line_year,
            release.tract_definition_vintage,
        )
    )


def select_vintage_releases(
    *,
    start_vintage: int,
    end_vintage: int,
    releases: Iterable[AcsVintageRelease] = ACS_VINTAGE_RELEASES,
) -> tuple[AcsVintageRelease, ...]:
    """Select a complete inclusive range of known ACS vintages."""

    if start_vintage > end_vintage:
        raise ValueError("start_vintage cannot be greater than end_vintage.")

    selected = tuple(
        release
        for release in releases
        if start_vintage <= release.acs_vintage <= end_vintage
    )
    expected_vintages = set(range(start_vintage, end_vintage + 1))
    actual_vintages = {release.acs_vintage for release in selected}

    if actual_vintages != expected_vintages:
        missing = sorted(expected_vintages - actual_vintages)
        raise ValueError(
            f"The requested ACS release range contains unknown vintages: {missing}."
        )

    return tuple(sorted(selected))


def validate_vintage_releases(
    releases: Iterable[AcsVintageRelease],
) -> tuple[AcsVintageRelease, ...]:
    """Validate uniqueness, chronology, and geography compatibility."""

    ordered = tuple(sorted(releases))
    if not ordered:
        raise ValueError("The ACS release calendar cannot be empty.")

    vintages = [release.acs_vintage for release in ordered]
    if len(vintages) != len(set(vintages)):
        raise ValueError("The ACS release calendar contains duplicate vintages.")

    expected = list(range(vintages[0], vintages[-1] + 1))
    if vintages != expected:
        missing = sorted(set(expected) - set(vintages))
        raise ValueError(
            f"The ACS release calendar must be contiguous; missing vintages={missing}."
        )

    release_dates = [release.acs_release_date for release in ordered]
    if release_dates != sorted(release_dates) or len(release_dates) != len(
        set(release_dates)
    ):
        raise ValueError(
            "ACS public release dates must be unique and strictly increasing."
        )

    for release in ordered:
        if release.acs_release_date <= date(release.acs_vintage, 12, 31):
            raise ValueError(
                f"An ACS 5-year release must occur after its vintage year: {release!r}."
            )
        if release.tiger_line_year != release.acs_vintage:
            raise ValueError(
                "CrimeNet requires the TIGER/Line year to match the ACS "
                f"vintage; received {release!r}."
            )

        expected_tract_definition = 2010 if release.acs_vintage <= 2019 else 2020
        if release.tract_definition_vintage != expected_tract_definition:
            raise ValueError(
                "Unexpected decennial tract definition for ACS vintage "
                f"{release.acs_vintage}: expected "
                f"{expected_tract_definition}, received "
                f"{release.tract_definition_vintage}."
            )

    return ordered


def create_calendar_dataframe(
    spark: SparkSession,
    releases: Iterable[AcsVintageRelease],
    *,
    definition_version: str = ACS_CALENDAR_DEFINITION_VERSION,
) -> DataFrame:
    """Create the deterministic Spark representation used by Silver/Gold."""

    from pyspark.sql.types import (
        DateType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    validated = validate_vintage_releases(releases)
    if not definition_version.strip():
        raise ValueError("definition_version cannot be blank.")

    rows = [
        (
            release.acs_vintage,
            release.acs_release_date,
            release.tiger_line_year,
            release.tract_definition_vintage,
            definition_version,
            calendar_record_id(
                release,
                definition_version=definition_version,
            ),
            ACS_RELEASE_SOURCE_URL,
        )
        for release in validated
    ]

    schema = StructType(
        [
            StructField("acs_vintage", IntegerType(), False),
            StructField("acs_release_date", DateType(), False),
            StructField("tiger_line_year", IntegerType(), False),
            StructField("tract_definition_vintage", IntegerType(), False),
            StructField(
                "calendar_definition_version",
                StringType(),
                False,
            ),
            StructField("calendar_record_id", StringType(), False),
            StructField("source_reference", StringType(), False),
        ]
    )
    return spark.createDataFrame(rows, schema=schema)


def validate_calendar_dataframe(dataframe: DataFrame) -> None:
    """Run blocking checks on a staged release-calendar candidate."""

    from pyspark.sql import functions as F

    required_columns = {
        "acs_vintage",
        "acs_release_date",
        "tiger_line_year",
        "tract_definition_vintage",
        "calendar_definition_version",
        "calendar_record_id",
        "source_reference",
    }
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"ACS release calendar is missing required columns: {missing_columns}."
        )

    if dataframe.isEmpty():
        raise ValueError("ACS release calendar candidate is empty.")

    duplicate_vintages = (
        dataframe.groupBy("acs_vintage")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    duplicate_ids = (
        dataframe.groupBy("calendar_record_id")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    invalid_rows = (
        dataframe.filter(
            F.col("acs_vintage").isNull()
            | F.col("acs_release_date").isNull()
            | (F.length("calendar_record_id") != 64)
            | (F.col("tiger_line_year") != F.col("acs_vintage"))
        )
        .limit(1)
        .count()
    )

    if duplicate_vintages:
        raise ValueError("ACS release calendar has duplicate vintages.")
    if duplicate_ids:
        raise ValueError("ACS release calendar has duplicate record IDs.")
    if invalid_rows:
        raise ValueError("ACS release calendar contains invalid records.")
