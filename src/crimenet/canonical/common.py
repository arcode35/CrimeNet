"""Shared PySpark canonical expressions."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


CITY_BOUNDS = {
    "dallas": (
        32.40,
        33.10,
        -97.10,
        -96.40,
    ),
    "fort_worth": (
        32.50,
        33.05,
        -97.65,
        -96.95,
    ),
    "new_york": (
        40.40,
        41.00,
        -74.30,
        -73.60,
    ),
    "chicago": (
        41.60,
        42.10,
        -88.00,
        -87.45,
    ),
    "san_francisco": (
        37.60,
        37.95,
        -122.60,
        -122.25,
    ),
    "seattle": (
        47.40,
        47.85,
        -122.50,
        -122.15,
    ),
    "baltimore": (
        39.15,
        39.45,
        -76.80,
        -76.45,
    ),
    "washington_dc": (
        38.75,
        39.05,
        -77.20,
        -76.85,
    ),
}


LOCAL_TIMEZONES = {
    "dallas": "America/Chicago",
    "fort_worth": "America/Chicago",
    "new_york": "America/New_York",
    "chicago": "America/Chicago",
    "san_francisco": (
        "America/Los_Angeles"
    ),
    "seattle": "America/Los_Angeles",
    "baltimore": "America/New_York",
    "washington_dc": "America/New_York",
}


NULL_TEXT_VALUES = (
    "null",
    "(null)",
    "none",
    "nan",
    "n/a",
    "na",
)


ISO_LOCAL_PATTERNS = (
    "yyyy-MM-dd'T'HH:mm:ss.SSSSSSS",
    "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ss",
)


SQL_LOCAL_PATTERNS = (
    "yyyy-MM-dd HH:mm:ss.SSSSSSS",
    "yyyy-MM-dd HH:mm:ss.SSSSSS",
    "yyyy-MM-dd HH:mm:ss.SSS",
    "yyyy-MM-dd HH:mm:ss",
)


def _column(
    expression: Column | str,
) -> Column:
    if isinstance(expression, str):
        return F.col(expression)

    return expression


def existing_column(
    dataframe: DataFrame,
    *names: str,
    data_type: str = "string",
) -> Column:
    """Return the first existing column or a typed null."""

    for name in names:
        if name in dataframe.columns:
            return F.col(name)

    return F.lit(None).cast(data_type)


def clean_string(
    expression: Column | str,
) -> Column:
    text = F.trim(
        _column(expression).cast("string")
    )

    lowered = F.lower(text)

    return (
        F.when(
            text.isNull()
            | (text == "")
            | lowered.isin(
                *NULL_TEXT_VALUES
            ),
            F.lit(None).cast("string"),
        )
        .otherwise(text)
    )


def clean_column(
    name: str,
) -> Column:
    return clean_string(
        F.col(name)
    )


def coalesce_strings(
    *expressions: Column | str,
) -> Column:
    return F.coalesce(
        *[
            clean_string(expression)
            for expression in expressions
        ]
    )


def integer_or_default(
    expression: Column | str,
    *,
    default: int,
) -> Column:
    return F.coalesce(
        clean_string(
            expression
        ).try_cast("bigint"),
        F.lit(default).cast("bigint"),
    )


def float_value(
    expression: Column | str,
) -> Column:
    value = clean_string(
        expression
    ).try_cast("double")

    valid = (
        value.isNotNull()
        & ~F.isnan(value)
        & (
            F.abs(value)
            != F.lit(float("inf"))
        )
    )

    return (
        F.when(valid, value)
        .otherwise(
            F.lit(None).cast("double")
        )
    )


def nullable_boolean(
    expression: Column | str,
    *,
    true_values: Sequence[str] = (
        "1",
        "true",
        "t",
        "yes",
        "y",
    ),
    false_values: Sequence[str] = (
        "0",
        "false",
        "f",
        "no",
        "n",
    ),
) -> Column:
    normalized = F.lower(
        clean_string(expression)
    )

    return (
        F.when(
            normalized.isin(*true_values),
            F.lit(True),
        )
        .when(
            normalized.isin(*false_values),
            F.lit(False),
        )
        .otherwise(
            F.lit(None).cast("boolean")
        )
    )


def nonblank_is_true(
    expression: Column | str,
) -> Column:
    return clean_string(
        expression
    ).isNotNull()


def find_dst_transition_dates(
    *,
    timezone: str,
    start_year: int = 1900,
    end_year: int = 2035,
) -> tuple[list[date], list[date]]:
    zone = ZoneInfo(timezone)

    spring_dates: list[date] = []
    fall_dates: list[date] = []

    current = date(start_year, 1, 1)
    end = date(end_year + 1, 1, 1)

    while current < end:
        midnight = datetime.combine(
            current,
            time.min,
            tzinfo=zone,
        )

        noon = datetime.combine(
            current,
            time(hour=12),
            tzinfo=zone,
        )

        midnight_offset = (
            midnight.utcoffset()
        )
        noon_offset = noon.utcoffset()

        if (
            midnight_offset is not None
            and noon_offset is not None
        ):
            if noon_offset > midnight_offset:
                spring_dates.append(current)

            elif noon_offset < midnight_offset:
                fall_dates.append(current)

        current += timedelta(days=1)

    return spring_dates, fall_dates


DST_TRANSITION_DATES = {
    timezone: find_dst_transition_dates(
        timezone=timezone
    )
    for timezone in set(
        LOCAL_TIMEZONES.values()
    )
}


def parse_naive_timestamp(
    expression: Column | str,
    *,
    patterns: Sequence[str],
) -> Column:
    text = clean_string(expression)

    candidates = [
        F.try_to_timestamp(
            text,
            F.lit(pattern),
        )
        for pattern in patterns
    ]

    return F.coalesce(
        *candidates
    )


def _add_localized_timestamp(
    dataframe: DataFrame,
    *,
    naive_column: str,
    output_prefix: str,
    timezone: str,
) -> DataFrame:
    (
        spring_dates,
        fall_dates,
    ) = DST_TRANSITION_DATES[
        timezone
    ]

    gap_column = (
        f"_{output_prefix}_is_dst_gap"
    )
    fold_column = (
        f"_{output_prefix}_is_dst_fold"
    )
    adjusted_column = (
        f"_{output_prefix}_adjusted"
    )
    utc_column = (
        f"_{output_prefix}_utc"
    )
    status_column = (
        f"_{output_prefix}_status"
    )

    local_date = F.to_date(
        F.col(naive_column)
    )

    local_hour = F.hour(
        F.col(naive_column)
    )

    dataframe = (
        dataframe
        .withColumn(
            gap_column,
            local_date.isin(
                *[
                    value.isoformat()
                    for value in spring_dates
                ]
            )
            & (local_hour == 2),
        )
        .withColumn(
            fold_column,
            local_date.isin(
                *[
                    value.isoformat()
                    for value in fall_dates
                ]
            )
            & (local_hour == 1),
        )
        .withColumn(
            adjusted_column,
            F.when(
                F.col(gap_column),
                F.col(naive_column)
                + F.expr(
                    "INTERVAL 1 HOUR"
                ),
            )
            .otherwise(
                F.col(naive_column)
            ),
        )
        .withColumn(
            utc_column,
            F.to_utc_timestamp(
                F.col(adjusted_column),
                timezone,
            ),
        )
        .withColumn(
            status_column,
            F.when(
                F.col(naive_column).isNull(),
                F.lit("parse_failed"),
            )
            .when(
                F.col(gap_column),
                F.lit(
                    "nonexistent_shifted_forward"
                ),
            )
            .when(
                F.col(fold_column),
                F.lit(
                    "ambiguous_earliest"
                ),
            )
            .otherwise(F.lit("exact")),
        )
    )

    return dataframe


def add_fast_local_datetime(
    dataframe: DataFrame,
    *,
    source_column: str,
    output_prefix: str,
    timezone: str,
    patterns: Sequence[str],
) -> DataFrame:
    naive_column = (
        f"_{output_prefix}_naive"
    )

    dataframe = dataframe.withColumn(
        naive_column,
        parse_naive_timestamp(
            F.col(source_column),
            patterns=patterns,
        ),
    )

    return _add_localized_timestamp(
        dataframe,
        naive_column=naive_column,
        output_prefix=output_prefix,
        timezone=timezone,
    )


def add_fast_local_datetime_pair(
    dataframe: DataFrame,
    *,
    date_column: str,
    time_column: str,
    output_prefix: str,
    timezone: str,
    time_width: int,
    patterns: Sequence[str],
) -> DataFrame:
    naive_column = (
        f"_{output_prefix}_naive"
    )

    date_text = F.substring(
        clean_column(date_column),
        1,
        10,
    )

    time_text = F.lpad(
        F.substring(
            clean_column(time_column),
            1,
            time_width,
        ),
        time_width,
        "0",
    )

    combined = (
        F.when(
            date_text.isNotNull()
            & time_text.isNotNull(),
            F.concat_ws(
                " ",
                date_text,
                time_text,
            ),
        )
        .otherwise(
            F.lit(None).cast("string")
        )
    )

    dataframe = dataframe.withColumn(
        naive_column,
        parse_naive_timestamp(
            combined,
            patterns=patterns,
        ),
    )

    return _add_localized_timestamp(
        dataframe,
        naive_column=naive_column,
        output_prefix=output_prefix,
        timezone=timezone,
    )


def parse_exact_utc_download_timestamp(
    expression: Column | str,
) -> Column:
    text = clean_string(expression)

    return F.coalesce(
        F.try_to_timestamp(
            text,
            F.lit(
                "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"
            ),
        ),
        F.try_to_timestamp(
            text,
            F.lit(
                "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"
            ),
        ),
        F.try_to_timestamp(
            text,
            F.lit(
                "yyyy-MM-dd'T'HH:mm:ssXXX"
            ),
        ),
        text.try_cast("timestamp"),
    )


def parse_epoch_milliseconds(
    expression: Column | str,
) -> Column:
    milliseconds = clean_string(
        expression
    ).try_cast("bigint")

    return F.timestamp_millis(
        milliseconds
    )


def naive_datetime_as_utc(
    expression: Column | str,
) -> Column:
    return clean_string(
        expression
    ).try_cast("timestamp")


def observed_time_precision(
    timestamp_expression: Column | str,
) -> Column:
    timestamp = _column(
        timestamp_expression
    )

    microseconds = (
        F.date_format(
            timestamp,
            "SSSSSS",
        )
        .try_cast("int")
    )

    return (
        F.when(
            timestamp.isNull(),
            F.lit("unknown"),
        )
        .when(
            (microseconds != 0)
            | (F.second(timestamp) != 0),
            F.lit("second"),
        )
        .when(
            F.minute(timestamp) != 0,
            F.lit("minute"),
        )
        .otherwise(F.lit("hour"))
    )


def valid_interval_expression(
    start: Column | str,
    end: Column | str,
) -> Column:
    start_column = _column(start)
    end_column = _column(end)

    return (
        start_column.isNotNull()
        & end_column.isNotNull()
        & (end_column >= start_column)
    )


def occurrence_interval_minutes(
    start: Column | str,
    end: Column | str,
) -> Column:
    start_column = _column(start)
    end_column = _column(end)

    difference = (
        F.unix_micros(end_column)
        - F.unix_micros(start_column)
    ) / F.lit(60_000_000)

    return (
        F.when(
            valid_interval_expression(
                start_column,
                end_column,
            ),
            difference.cast("bigint"),
        )
        .otherwise(
            F.lit(None).cast("bigint")
        )
    )


def canonical_coordinate_expressions(
    *,
    city: str,
    latitude_expression: Column | str,
    longitude_expression: Column | str,
) -> dict[str, Column]:
    (
        minimum_latitude,
        maximum_latitude,
        minimum_longitude,
        maximum_longitude,
    ) = CITY_BOUNDS[city]

    latitude_raw = float_value(
        latitude_expression
    )

    longitude_raw = float_value(
        longitude_expression
    )

    both_missing = (
        latitude_raw.isNull()
        & longitude_raw.isNull()
    )

    pair_mismatch = (
        (
            latitude_raw.isNull()
            & longitude_raw.isNotNull()
        )
        | (
            latitude_raw.isNotNull()
            & longitude_raw.isNull()
        )
    )

    globally_valid = (
        latitude_raw.isNotNull()
        & longitude_raw.isNotNull()
        & latitude_raw.between(
            -90.0,
            90.0,
        )
        & longitude_raw.between(
            -180.0,
            180.0,
        )
    )

    within_city = (
        globally_valid
        & latitude_raw.between(
            minimum_latitude,
            maximum_latitude,
        )
        & longitude_raw.between(
            minimum_longitude,
            maximum_longitude,
        )
    )

    latitude = (
        F.when(
            globally_valid,
            latitude_raw,
        )
        .otherwise(
            F.lit(None).cast("double")
        )
    )

    longitude = (
        F.when(
            globally_valid,
            longitude_raw,
        )
        .otherwise(
            F.lit(None).cast("double")
        )
    )

    status = (
        F.when(
            both_missing,
            F.lit("missing"),
        )
        .when(
            pair_mismatch,
            F.lit(
                "coordinate_pair_mismatch"
            ),
        )
        .when(
            ~globally_valid,
            F.lit(
                "invalid_global_bounds"
            ),
        )
        .when(
            ~within_city,
            F.lit(
                "outside_city_bounds"
            ),
        )
        .otherwise(F.lit("valid"))
    )

    within_city_value = (
        F.when(
            globally_valid,
            within_city,
        )
        .otherwise(
            F.lit(None).cast("boolean")
        )
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "status": status,
        "within_city": (
            within_city_value
        ),
    }


def namespaced_record_id(
    *,
    city: str,
    local_key: Column | str,
) -> Column:
    key = clean_string(local_key)

    return (
        F.when(
            key.isNotNull(),
            F.concat(
                F.lit(f"{city}:"),
                key,
            ),
        )
        .otherwise(
            F.lit(None).cast("string")
        )
    )


def stable_composite_sha256(
    *,
    namespace: str,
    components: Sequence[
        Column | str
    ],
) -> Column:
    encoded = [
        F.lit(
            f"{len(namespace.encode('utf-8'))}:"
            f"{namespace}"
        )
    ]

    for component in components:
        value = clean_string(component)

        byte_length = F.length(
            F.encode(
                value,
                "UTF-8",
            )
        ).cast("string")

        encoded.append(
            F.when(
                value.isNull(),
                F.lit("-1:"),
            )
            .otherwise(
                F.concat(
                    byte_length,
                    F.lit(":"),
                    value,
                )
            )
        )

    payload = F.concat_ws(
        "|",
        *encoded,
    )

    return F.sha2(
        payload,
        256,
    )


def with_coordinate_metadata(
    *,
    latitude: Column | str,
    method: str,
    precision: str,
    source_crs: str | None,
) -> list[Column]:
    latitude_column = _column(latitude)
    present = latitude_column.isNotNull()

    return [
        F.when(
            present,
            F.lit(method),
        )
        .otherwise(
            F.lit(None).cast("string")
        )
        .alias("coordinate_method"),

        F.when(
            present,
            F.lit(precision),
        )
        .otherwise(
            F.lit(None).cast("string")
        )
        .alias("coordinate_precision"),

        F.when(
            present
            & F.lit(
                source_crs is not None
            ),
            F.lit(source_crs),
        )
        .otherwise(
            F.lit(None).cast("string")
        )
        .alias("source_crs"),

        F.when(
            present,
            F.lit("EPSG:4326"),
        )
        .otherwise(
            F.lit(None).cast("string")
        )
        .alias("target_crs"),
    ]