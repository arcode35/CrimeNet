import polars as pl
import polars_h3 as plh3


EVENT_KEY = [
    "integration_sample_id",
]

WEATHER_KEY = [
    "weather_query_cell_id",
    "weather_timestamp",
]

TRACT_KEY = [
    "source_city",
    "tiger_line_year",
    "osm_h3_cell_id",
]


# =============================================================================
# Helpers
# =============================================================================


def prefix_except(
    frame: pl.LazyFrame,
    *,
    prefix: str,
    exclude: set[str],
) -> pl.LazyFrame:
    schema = frame.collect_schema()

    mapping = {
        column: f"{prefix}{column}"
        for column in schema.names()
        if (
            column not in exclude
            and not column.startswith(prefix)
        )
    }

    return frame.rename(mapping)


# =============================================================================
# Sample spatial/time representation
# =============================================================================


def add_sample_spatial_time(
    samples: pl.LazyFrame,
    *,
    city_timezones: dict[str, str],
) -> pl.LazyFrame:
    """
    Add:

        sample_latitude
        sample_longitude
        sample_timestamp_local
        sample_date_local
        sample_year_local
        weather_timestamp

    H3 center is the canonical representative point for the discrete H3 cell.
    """

    parts: list[pl.LazyFrame] = []

    for city, timezone in (
        city_timezones.items()
    ):
        part = (
            samples
            .filter(
                pl.col("source_city")
                == city
            )
            .with_columns(
                plh3.cell_to_lat(
                    "osm_h3_cell_id"
                )
                .cast(pl.Float64)
                .alias(
                    "sample_latitude"
                ),

                plh3.cell_to_lng(
                    "osm_h3_cell_id"
                )
                .cast(pl.Float64)
                .alias(
                    "sample_longitude"
                ),

                pl.col(
                    "sample_timestamp_utc"
                )
                .dt.convert_time_zone(
                    timezone
                )
                .dt.replace_time_zone(
                    None
                )
                .alias(
                    "sample_timestamp_local"
                ),

                pl.col(
                    "sample_timestamp_utc"
                )
                .dt.truncate("1h")
                .alias(
                    "weather_timestamp"
                ),
            )
            .with_columns(
                pl.col(
                    "sample_timestamp_local"
                )
                .dt.date()
                .alias(
                    "sample_date_local"
                ),

                pl.col(
                    "sample_timestamp_local"
                )
                .dt.year()
                .cast(pl.Int32)
                .alias(
                    "sample_year_local"
                ),
            )
            .with_columns(
                (
                    pl.col(
                        "sample_year_local"
                    )
                    !=
                    pl.col(
                        "support_year"
                    )
                )
                .alias(
                    "_sample_local_year_mismatch"
                )
            )
        )

        parts.append(part)

    return pl.concat(
        parts,
        how="vertical",
    )


# =============================================================================
# ACS calendar
# =============================================================================


def prepare_acs_calendar(
    calendar: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        calendar
        .select(
            "acs_vintage",
            "acs_release_date",
            "tiger_line_year",
            "tract_definition_vintage",
        )
        .with_columns(
            pl.col(
                "acs_vintage"
            )
            .cast(pl.Int32),

            pl.col(
                "acs_release_date"
            )
            .cast(pl.Date),

            pl.col(
                "tiger_line_year"
            )
            .cast(pl.Int32),

            pl.col(
                "tract_definition_vintage"
            )
            .cast(pl.Int32),
        )
        .with_columns(
            pl.col(
                "acs_release_date"
            )
            .alias(
                "acs_available_date"
            )
        )
        .sort(
            "acs_available_date"
        )
    )


def join_acs_calendar(
    samples: pl.LazyFrame,
    calendar: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Latest ACS release available as of the sample's LOCAL calendar date.
    """

    return (
        samples
        .sort(
            "sample_date_local"
        )
        .join_asof(
            calendar,
            left_on=
                "sample_date_local",
            right_on=
                "acs_available_date",
            strategy="backward",
        )
        .with_columns(
            pl.col(
                "acs_vintage"
            )
            .is_not_null()
            .alias(
                "_acs_calendar_matched"
            )
        )
    )


# =============================================================================
# OSM
# =============================================================================


def prepare_osm(
    osm: pl.LazyFrame,
) -> pl.LazyFrame:
    osm = (
        osm
        .with_columns(
            pl.col(
                "snapshot_year"
            )
            .cast(pl.Int32),

            plh3.str_to_int(
                "osm_h3_cell_id"
            )
            .alias(
                "osm_h3_cell_id"
            ),
        )
        .with_columns(
            pl.col(
                "snapshot_year"
            )
            .alias(
                "osm_snapshot_year"
            ),

            pl.lit(True)
            .alias(
                "_matched"
            ),
        )
        .rename(
            {
                "snapshot_year":
                    "support_year",
            }
        )
    )

    osm = prefix_except(
        osm,
        prefix="osm_",
        exclude={
            "source_city",
            "support_year",
            "osm_h3_cell_id",
            "osm_snapshot_year",
            "_matched",
        },
    )

    return osm.rename(
        {
            "_matched":
                "_osm_matched",
        }
    )


def join_osm(
    samples: pl.LazyFrame,
    osm: pl.LazyFrame,
) -> pl.LazyFrame:
    return samples.join(
        osm,
        on=[
            "source_city",
            "support_year",
            "osm_h3_cell_id",
        ],
        how="left",
        validate="m:1",
    )


# =============================================================================
# Weather
# =============================================================================


def prepare_land_weather(
    weather: pl.LazyFrame,
) -> pl.LazyFrame:
    weather = (
        weather
        .with_columns(
            pl.lit(True)
            .alias("_matched")
        )
    )

    weather = prefix_except(
        weather,
        prefix=
            "weather_land_",
        exclude={
            "weather_query_cell_id",
            "weather_timestamp",
            "_matched",
        },
    )

    return weather.rename(
        {
            "_matched":
                "_weather_land_matched",
        }
    )


def join_land_weather(
    samples: pl.LazyFrame,
    weather: pl.LazyFrame,
) -> pl.LazyFrame:
    return samples.join(
        weather,
        on=WEATHER_KEY,
        how="left",
        validate="m:1",
    )


def prepare_coastal_weather(
    weather: pl.LazyFrame,
) -> pl.LazyFrame:
    schema = set(
        weather.collect_schema().names()
    )

    wanted = [
        column
        for column in [
            "weather_query_cell_id",
            "weather_timestamp",
            "temperature_2m_c",
            "provider",
            "model",
            "request_id",
        ]
        if column in schema
    ]

    result = (
        weather
        .select(
            wanted
        )
        .with_columns(
            pl.lit(True)
            .alias(
                "_weather_coastal_matched"
            )
        )
    )

    rename = {}

    for column in wanted:
        if column not in WEATHER_KEY:
            rename[column] = (
                f"weather_coastal_{column}"
            )

    return result.rename(
        rename
    )


def join_coastal_weather(
    samples: pl.LazyFrame,
    coastal: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Query coastal source only for samples whose land temperature is missing.
    """

    missing = (
        samples
        .filter(
            pl.col(
                "weather_land_temperature_2m_c"
            )
            .is_null()
        )
        .select(
            "integration_sample_id",
            "weather_query_cell_id",
            "weather_timestamp",
        )
        .join(
            coastal,
            on=WEATHER_KEY,
            how="left",
            validate="m:1",
        )
        .select(
            "integration_sample_id",
            *[
                column
                for column in (
                    coastal
                    .collect_schema()
                    .names()
                )
                if column
                not in WEATHER_KEY
            ],
        )
    )

    return samples.join(
        missing,
        on=
            "integration_sample_id",
        how="left",
        validate="1:1",
    )


def add_canonical_weather(
    samples: pl.LazyFrame,
) -> pl.LazyFrame:
    schema = set(
        samples.collect_schema().names()
    )

    coastal_temp = (
        "weather_coastal_temperature_2m_c"
    )

    # Coastal temperature may not exist depending on source schema.
    if coastal_temp not in schema:
        samples = (
            samples
            .with_columns(
                pl.lit(
                    None,
                    dtype=pl.Float64,
                )
                .alias(
                    coastal_temp
                )
            )
        )

        schema.add(
            coastal_temp
        )

    expressions: list[pl.Expr] = [
        # -----------------------------------------------------------------
        # Effective temperature used by the model.
        # -----------------------------------------------------------------
        pl.coalesce(
            [
                pl.col(
                    "weather_land_temperature_2m_c"
                ),
                pl.col(
                    coastal_temp
                ),
            ]
        )
        .alias(
            "weather_temperature_2m_c"
        ),

        # -----------------------------------------------------------------
        # Which spatial weather source supplied the effective temperature.
        #
        # This is intentionally NOT called weather_temperature_source,
        # because the upstream land table already has temperature_source
        # provenance (e.g. original / patched).
        # -----------------------------------------------------------------
        (
            pl.when(
                pl.col(
                    "weather_land_temperature_2m_c"
                )
                .is_not_null()
            )
            .then(
                pl.lit("land")
            )
            .when(
                pl.col(
                    coastal_temp
                )
                .is_not_null()
            )
            .then(
                pl.lit(
                    "coastal_fallback"
                )
            )
            .otherwise(
                pl.lit("missing")
            )
            .alias(
                "weather_temperature_spatial_source"
            )
        ),
    ]

    # ---------------------------------------------------------------------
    # Mirror ordinary land-weather variables into canonical weather_*
    # aliases.
    #
    # temperature_2m_c is handled explicitly above because it can fall
    # back to coastal weather.
    #
    # Provider/model/request_id remain source-specific metadata.
    # ---------------------------------------------------------------------

    for column in schema:
        if not column.startswith(
            "weather_land_"
        ):
            continue

        suffix = (
            column.removeprefix(
                "weather_land_"
            )
        )

        if suffix in {
            "temperature_2m_c",
            "provider",
            "model",
            "request_id",
        }:
            continue

        canonical = (
            f"weather_{suffix}"
        )

        if canonical in schema:
            continue

        expressions.append(
            pl.col(column)
            .alias(canonical)
        )

    return (
        samples
        .with_columns(
            expressions
        )
    )

# =============================================================================
# Integration H3 -> tract mapping
# =============================================================================


def join_tract_mapping(
    samples: pl.LazyFrame,
    mapping: pl.LazyFrame,
) -> pl.LazyFrame:
    return samples.join(
        mapping,
        on=TRACT_KEY,
        how="left",
        validate="m:1",
    )


# =============================================================================
# Socioeconomic
# =============================================================================


def prepare_socioeconomic(
    socioeconomic: pl.LazyFrame,
) -> pl.LazyFrame:
    socioeconomic = (
        socioeconomic
        .with_columns(
            pl.col(
                "acs_vintage"
            )
            .cast(pl.Int32),

            pl.lit(True)
            .alias("_matched"),
        )
        .rename(
            {
                "geoid":
                    "tract_geoid",
            }
        )
    )

    socioeconomic = prefix_except(
        socioeconomic,
        prefix="socio_",
        exclude={
            "acs_vintage",
            "tract_geoid",
            "_matched",
        },
    )

    return socioeconomic.rename(
        {
            "_matched":
                "_socioeconomic_matched",
        }
    )


def join_socioeconomic(
    samples: pl.LazyFrame,
    socioeconomic: pl.LazyFrame,
) -> pl.LazyFrame:
    return samples.join(
        socioeconomic,
        on=[
            "acs_vintage",
            "tract_geoid",
        ],
        how="left",
        validate="m:1",
    )


# =============================================================================
# Leakage / alignment flags
# =============================================================================


def add_context_validation_flags(
    samples: pl.LazyFrame,
) -> pl.LazyFrame:
    return samples.with_columns(
        (
            pl.col(
                "acs_available_date"
            )
            >
            pl.col(
                "sample_date_local"
            )
        )
        .fill_null(False)
        .alias(
            "_leak_acs_future_release"
        ),

        (
            pl.col(
                "acs_vintage"
            )
            >
            pl.col(
                "sample_year_local"
            )
        )
        .fill_null(False)
        .alias(
            "_leak_future_acs_vintage"
        ),

        (
            pl.col(
                "tiger_line_year"
            )
            >
            pl.col(
                "sample_year_local"
            )
        )
        .fill_null(False)
        .alias(
            "_leak_future_tiger_year"
        ),

        (
            pl.col(
                "osm_snapshot_year"
            )
            >
            pl.col(
                "sample_year_local"
            )
        )
        .fill_null(False)
        .alias(
            "_leak_future_osm_snapshot"
        ),

        (
            pl.col(
                "weather_timestamp"
            )
            >
            pl.col(
                "sample_timestamp_utc"
            )
        )
        .fill_null(False)
        .alias(
            "_leak_future_weather"
        ),

        (
            pl.col(
                "weather_timestamp"
            )
            !=
            pl.col(
                "sample_timestamp_utc"
            )
            .dt.truncate("1h")
        )
        .fill_null(True)
        .alias(
            "_weather_hour_alignment_violation"
        ),
    )


# =============================================================================
# Base context
# =============================================================================


def build_integration_base_context(
    *,
    samples: pl.LazyFrame,
    acs_calendar: pl.LazyFrame,
    osm: pl.LazyFrame,
    land_weather: pl.LazyFrame,
    coastal_weather: pl.LazyFrame,
    city_timezones: dict[str, str],
) -> pl.LazyFrame:
    result = add_sample_spatial_time(
        samples,
        city_timezones=
            city_timezones,
    )

    result = join_acs_calendar(
        result,
        prepare_acs_calendar(
            acs_calendar
        ),
    )

    result = join_osm(
        result,
        prepare_osm(
            osm
        ),
    )

    result = join_land_weather(
        result,
        prepare_land_weather(
            land_weather
        ),
    )

    result = join_coastal_weather(
        result,
        prepare_coastal_weather(
            coastal_weather
        ),
    )

    result = add_canonical_weather(
        result
    )

    return result


# =============================================================================
# Complete context
# =============================================================================


def build_integration_context(
    *,
    base: pl.LazyFrame,
    tract_mapping: pl.LazyFrame,
    socioeconomic: pl.LazyFrame,
) -> pl.LazyFrame:
    result = join_tract_mapping(
        base,
        tract_mapping,
    )

    result = join_socioeconomic(
        result,
        prepare_socioeconomic(
            socioeconomic
        ),
    )

    result = add_context_validation_flags(
        result
    )

    return result