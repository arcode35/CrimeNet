import json
from pathlib import Path

import polars as pl


EVENT_SPINE_ROOT = (
    "gs://crimenet/gold/event_spine"
)

OUTPUT_JSON = Path(
    "observation_ranges_candidates.json"
)


CITY_TIMEZONES = {
    "baltimore":
        "America/New_York",

    "chicago":
        "America/Chicago",

    "dallas":
        "America/Chicago",

    "fort_worth":
        "America/Chicago",

    "new_york":
        "America/New_York",

    "san_francisco":
        "America/Los_Angeles",

    "seattle":
        "America/Los_Angeles",

    "washington_dc":
        "America/New_York",
}


# If there are no events for this many consecutive days, surface the
# interval for manual investigation.
SUSPICIOUS_GAP_DAYS = 30


def scan_event_spine() -> pl.LazyFrame:
    credentials = (
        pl.CredentialProviderGCP()
    )

    return pl.scan_delta(
        EVENT_SPINE_ROOT,
        credential_provider=credentials,
    )


def get_modeled_events(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        events
        .filter(
            pl.col("include_in_model")
            .fill_null(False)
            &
            pl.col("is_criminal_event")
            .fill_null(False)
            &
            pl.col(
                "occurrence_timestamp_utc"
            )
            .is_not_null()
        )
        .select(
            "crime_id",
            "source_city",
            "occurrence_timestamp_utc",
        )
    )

def build_local_events(
    modeled_events: pl.LazyFrame,
) -> pl.LazyFrame:
    parts: list[pl.LazyFrame] = []

    for city, timezone in (
        CITY_TIMEZONES.items()
    ):
        part = (
            modeled_events
            .filter(
                pl.col("source_city")
                == city
            )
            .with_columns(
                # Convert UTC instant into the city's local wall clock,
                # then remove timezone metadata so every city's column
                # has the same Polars dtype: Datetime("us").
                pl.col(
                    "occurrence_timestamp_utc"
                )
                .dt.convert_time_zone(
                    timezone
                )
                .dt.replace_time_zone(
                    None
                )
                .alias(
                    "occurrence_timestamp_local"
                )
            )
            .with_columns(
                pl.col(
                    "occurrence_timestamp_local"
                )
                .dt.date()
                .alias(
                    "occurrence_date_local"
                ),

                pl.col(
                    "occurrence_timestamp_local"
                )
                .dt.year()
                .cast(pl.Int32)
                .alias(
                    "occurrence_year_local"
                ),
            )
        )

        parts.append(part)

    return pl.concat(
        parts,
        how="vertical",
    )

def summarize_city_ranges(
    events: pl.LazyFrame,
) -> pl.DataFrame:
    return (
        events
        .group_by(
            "source_city"
        )
        .agg(
            pl.len()
            .alias(
                "event_count"
            ),

            pl.col(
                "occurrence_timestamp_local"
            )
            .min()
            .alias(
                "first_event_local"
            ),

            pl.col(
                "occurrence_timestamp_local"
            )
            .max()
            .alias(
                "last_event_local"
            ),

            pl.col(
                "occurrence_date_local"
            )
            .n_unique()
            .alias(
                "active_days"
            ),
        )
        .sort(
            "source_city"
        )
        .collect()
    )


def build_daily_activity(
    events: pl.LazyFrame,
) -> pl.DataFrame:
    return (
        events
        .group_by(
            [
                "source_city",
                "occurrence_date_local",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "events"
            )
        )
        .sort(
            [
                "source_city",
                "occurrence_date_local",
            ]
        )
        .collect()
    )


def build_monthly_activity(
    events: pl.LazyFrame,
) -> pl.DataFrame:
    return (
        events
        .with_columns(
            pl.col(
                "occurrence_timestamp_local"
            )
            .dt.truncate("1mo")
            .alias("month")
        )
        .group_by(
            [
                "source_city",
                "month",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "events"
            ),

            pl.col(
                "occurrence_date_local"
            )
            .n_unique()
            .alias(
                "active_days"
            ),
        )
        .sort(
            [
                "source_city",
                "month",
            ]
        )
        .collect()
    )


def detect_event_gaps(
    daily: pl.DataFrame,
) -> pl.DataFrame:
    """
    Detect long gaps between dates that contain events.

    IMPORTANT:
        This does NOT prove that the source was unavailable.

    A crime dataset can legitimately have days with zero events.
    These are merely suspicious gaps worth verifying.
    """

    return (
        daily
        .sort(
            [
                "source_city",
                "occurrence_date_local",
            ]
        )
        .with_columns(
            pl.col(
                "occurrence_date_local"
            )
            .shift(1)
            .over(
                "source_city"
            )
            .alias(
                "previous_active_date"
            )
        )
        .with_columns(
            (
                pl.col(
                    "occurrence_date_local"
                )
                -
                pl.col(
                    "previous_active_date"
                )
            )
            .dt.total_days()
            .alias(
                "gap_days"
            )
        )
        .filter(
            pl.col("gap_days")
            > SUSPICIOUS_GAP_DAYS
        )
        .select(
            "source_city",
            "previous_active_date",
            "occurrence_date_local",
            "gap_days",
        )
        .sort(
            [
                "source_city",
                "previous_active_date",
            ]
        )
    )


def summarize_years(
    events: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Useful for spotting weak edge years.

    Example:
        2014 has 12 records while 2015 has 100,000

    That is strong evidence that 2014 may only contain partial source
    coverage.
    """

    return (
        events
        .group_by(
            [
                "source_city",
                "occurrence_year_local",
            ]
        )
        .agg(
            pl.len()
            .alias(
                "events"
            ),

            pl.col(
                "occurrence_date_local"
            )
            .min()
            .alias(
                "first_active_date"
            ),

            pl.col(
                "occurrence_date_local"
            )
            .max()
            .alias(
                "last_active_date"
            ),

            pl.col(
                "occurrence_date_local"
            )
            .n_unique()
            .alias(
                "active_days"
            ),
        )
        .sort(
            [
                "source_city",
                "occurrence_year_local",
            ]
        )
        .collect()
    )


def make_candidate_ranges(
    city_summary: pl.DataFrame,
) -> dict[str, list[tuple[str, str]]]:
    from datetime import timedelta

    result = {}

    for row in city_summary.iter_rows(
        named=True
    ):
        city = row["source_city"]
        first = row["first_event_local"]
        last = row["last_event_local"]

        if first is None or last is None:
            continue

        start_date = first.date()

        end_date = (
            last.date()
            + timedelta(days=1)
        )

        result[city] = [
            (
                f"{start_date.isoformat()}T00:00:00",
                f"{end_date.isoformat()}T00:00:00",
            )
        ]

    return result


def print_python_config(
    ranges: dict[
        str,
        list[tuple[str, str]],
    ],
) -> None:
    print()
    print(
        "Candidate OBSERVATION_RANGES_LOCAL:"
    )
    print()
    print(
        "OBSERVATION_RANGES_LOCAL = {"
    )

    for city in sorted(ranges):
        print(
            f'    "{city}": ['
        )

        for start, end in (
            ranges[city]
        ):
            print(
                "        ("
            )
            print(
                f'            "{start}",'
            )
            print(
                f'            "{end}",'
            )
            print(
                "        ),"
            )

        print(
            "    ],"
        )

    print(
        "}"
    )


def main() -> None:
    events = scan_event_spine()

    modeled = (
        get_modeled_events(
            events
        )
    )

    local_events = (
        build_local_events(
            modeled
        )
    )

    city_summary = (
        summarize_city_ranges(
            local_events
        )
    )

    daily = (
        build_daily_activity(
            local_events
        )
    )

    monthly = (
        build_monthly_activity(
            local_events
        )
    )

    yearly = (
        summarize_years(
            local_events
        )
    )

    suspicious_gaps = (
        detect_event_gaps(
            daily
        )
    )

    candidate_ranges = (
        make_candidate_ranges(
            city_summary
        )
    )

    # -----------------------------------------------------------------
    # Human-readable diagnostics
    # -----------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )
    print(
        "CITY COVERAGE SUMMARY"
    )
    print(
        "============================================================"
    )

    print(
        city_summary
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        "YEARLY ACTIVITY"
    )
    print(
        "============================================================"
    )

    print(
        yearly
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        f"SUSPICIOUS GAPS > "
        f"{SUSPICIOUS_GAP_DAYS} DAYS"
    )
    print(
        "============================================================"
    )

    if suspicious_gaps.height:
        print(
            suspicious_gaps
        )
    else:
        print(
            "None detected."
        )

    print_python_config(
        candidate_ranges
    )

    # -----------------------------------------------------------------
    # Persist complete diagnostics
    # -----------------------------------------------------------------

    result = {
        "warning": (
            "These ranges are inferred from observed events and are "
            "candidates only. Verify source coverage before using them "
            "as the statistical observation domain."
        ),

        "ranges": {
            city: [
                {
                    "start_local":
                        start,

                    "end_local_exclusive":
                        end,
                }
                for start, end
                in ranges
            ]
            for city, ranges
            in candidate_ranges.items()
        },

        "suspicious_gaps": [
            {
                key:
                    (
                        value.isoformat()
                        if hasattr(
                            value,
                            "isoformat",
                        )
                        else value
                    )
                for key, value
                in row.items()
            }
            for row in (
                suspicious_gaps
                .iter_rows(
                    named=True
                )
            )
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    monthly.write_csv(
        "observation_monthly_activity.csv"
    )

    yearly.write_csv(
        "observation_yearly_activity.csv"
    )

    suspicious_gaps.write_csv(
        "observation_suspicious_gaps.csv"
    )

    print()
    print(
        f"Wrote {OUTPUT_JSON}"
    )
    print(
        "Wrote observation_monthly_activity.csv"
    )
    print(
        "Wrote observation_yearly_activity.csv"
    )
    print(
        "Wrote observation_suspicious_gaps.csv"
    )


if __name__ == "__main__":
    main()