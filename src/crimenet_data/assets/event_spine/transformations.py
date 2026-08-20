from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import polars_h3 as plh3


EVENT_KEY = [
    "source_city",
    "crime_id",
]

WEATHER_KEY = [
    "weather_query_cell_id",
    "weather_timestamp",
]


# =============================================================================
# Generic helpers
# =============================================================================


def prefix_except(
    frame: pl.LazyFrame,
    *,
    prefix: str,
    keep: set[str],
) -> pl.LazyFrame:
    mapping = {
        column: f"{prefix}{column}"
        for column in frame.collect_schema().names()
        if column not in keep
    }

    return frame.rename(mapping)


def prefix_osm_columns(
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    keys = {
        "source_city",
        "snapshot_year",
        "osm_h3_cell_id",
    }

    mapping = {}

    for column in frame.collect_schema().names():
        if column in keys:
            continue

        if column.startswith("osm_"):
            continue

        mapping[column] = f"osm_{column}"

    return frame.rename(mapping)


# =============================================================================
# Crime base
# =============================================================================


def build_crime_base(
    crime_by_city: Mapping[str, pl.LazyFrame],
    *,
    city_timezones: Mapping[str, str],
) -> pl.LazyFrame:
    parts: list[pl.LazyFrame] = []

    for city, frame in crime_by_city.items():
        timezone = city_timezones[city]

        frame = frame.with_columns(
            pl.col("occurrence_timestamp")
            .dt.year()
            .cast(pl.Int32)
            .alias("occurrence_year"),

            pl.col("occurrence_timestamp")
            .dt.date()
            .alias("occurrence_date"),

            pl.col("occurrence_timestamp")
            .dt.replace_time_zone(
                timezone,
                ambiguous="null",
                non_existent="null",
            )
            .dt.convert_time_zone("UTC")
            .alias("occurrence_timestamp_utc"),

            plh3.latlng_to_cell(
                "latitude",
                "longitude",
                resolution=6,
                return_dtype=pl.Int64,
            )
            .alias("weather_query_cell_id"),

            plh3.latlng_to_cell(
                "latitude",
                "longitude",
                resolution=9,
                return_dtype=pl.UInt64,
            )
            .alias("osm_h3_cell_id"),
        )

        frame = frame.with_columns(
            pl.col("occurrence_timestamp_utc")
            .dt.truncate("1h")
            .alias("weather_timestamp"),

            pl.when(
                pl.col("occurrence_year") <= 2023
            )
            .then(pl.lit("train"))
            .when(
                pl.col("occurrence_year") == 2024
            )
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("split"),
        )

        parts.append(frame)

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
            pl.col("acs_vintage")
            .cast(pl.Int32),

            pl.col("acs_release_date")
            .cast(pl.Date),

            pl.col("tiger_line_year")
            .cast(pl.Int32),

            pl.col("tract_definition_vintage")
            .cast(pl.Int32),
        )
        .with_columns(
            pl.col("acs_release_date")
            .alias("acs_available_date"),
        )
        .sort("acs_available_date")
    )


def join_acs_calendar(
    events: pl.LazyFrame,
    calendar: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        events
        .sort("occurrence_date")
        .join_asof(
            calendar,
            left_on="occurrence_date",
            right_on="acs_available_date",
            strategy="backward",
        )
    )


# =============================================================================
# Tract mapping
# =============================================================================


def prepare_tract_mapping(
    tract_mapping: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        tract_mapping
        .select(
            "tiger_line_year",
            "latitude",
            "longitude",
            "tract_geoid",
        )
        .with_columns(
            pl.col("tiger_line_year")
            .cast(pl.Int32),

            pl.lit(True)
            .alias("_tract_mapping_matched"),
        )
    )


def join_tract_mapping(
    events: pl.LazyFrame,
    tract_mapping: pl.LazyFrame,
) -> pl.LazyFrame:
    return events.join(
        tract_mapping,
        on=[
            "tiger_line_year",
            "latitude",
            "longitude",
        ],
        how="left",
        validate="m:1",
    )


# =============================================================================
# Socioeconomic
# =============================================================================


def prepare_socioeconomic(
    socioeconomic: pl.LazyFrame,
) -> pl.LazyFrame:
    socioeconomic = socioeconomic.with_columns(
        pl.col("acs_vintage")
        .cast(pl.Int32),

        pl.lit(True)
        .alias("_matched"),
    )

    socioeconomic = prefix_except(
        socioeconomic,
        prefix="socio_",
        keep={
            "acs_vintage",
            "geoid",
        },
    )

    return socioeconomic.rename(
        {
            "geoid":
                "tract_geoid",

            "socio__matched":
                "_socioeconomic_matched",
        }
    )


def join_socioeconomic(
    events: pl.LazyFrame,
    socioeconomic: pl.LazyFrame,
) -> pl.LazyFrame:
    return events.join(
        socioeconomic,
        on=[
            "acs_vintage",
            "tract_geoid",
        ],
        how="left",
        validate="m:1",
    )


# =============================================================================
# OSM H3-9
# =============================================================================


def prepare_osm(
    osm: pl.LazyFrame,
) -> pl.LazyFrame:
    osm = osm.with_columns(
        pl.col("snapshot_year")
        .cast(pl.Int32),

        # Retain source-side temporal provenance.
        pl.col("snapshot_year")
        .cast(pl.Int32)
        .alias("osm_snapshot_year"),

        plh3.str_to_int(
            "osm_h3_cell_id"
        )
        .cast(pl.UInt64)
        .alias("osm_h3_cell_id"),

        pl.lit(True)
        .alias("_matched"),
    )

    osm = prefix_osm_columns(
        osm
    )

    return osm.rename(
        {
            "snapshot_year":
                "occurrence_year",

            "osm__matched":
                "_osm_matched",
        }
    )


def join_osm(
    events: pl.LazyFrame,
    osm: pl.LazyFrame,
) -> pl.LazyFrame:
    return events.join(
        osm,
        on=[
            "source_city",
            "occurrence_year",
            "osm_h3_cell_id",
        ],
        how="left",
        validate="m:1",
    )


# =============================================================================
# Land weather
# =============================================================================


def prepare_land_weather(
    weather: pl.LazyFrame,
) -> pl.LazyFrame:
    weather = weather.with_columns(
        pl.lit(True)
        .alias("_matched")
    )

    weather = prefix_except(
        weather,
        prefix="weather_land_",
        keep=set(WEATHER_KEY),
    )

    return weather.rename(
        {
            "weather_land__matched":
                "_weather_land_matched",
        }
    )


def join_land_weather(
    events: pl.LazyFrame,
    weather: pl.LazyFrame,
) -> pl.LazyFrame:
    return events.join(
        weather,
        on=WEATHER_KEY,
        how="left",
        validate="m:1",
    )


# =============================================================================
# Coastal weather fallback
# =============================================================================


def prepare_coastal_weather(
    weather: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        weather
        .select(
            "weather_query_cell_id",
            "weather_timestamp",
            "temperature_2m_c",
            "provider",
            "model",
            "request_id",
        )
        .rename(
            {
                "temperature_2m_c":
                    "weather_coastal_temperature_2m_c",

                "provider":
                    "weather_coastal_provider",

                "model":
                    "weather_coastal_model",

                "request_id":
                    "weather_coastal_request_id",
            }
        )
        .with_columns(
            pl.lit(True)
            .alias("_weather_coastal_matched")
        )
    )


def join_coastal_weather(
    events: pl.LazyFrame,
    coastal_weather: pl.LazyFrame,
) -> pl.LazyFrame:
    fallback = (
        events
        .filter(
            pl.col(
                "weather_land_temperature_2m_c"
            ).is_null()
        )
        .select(
            *EVENT_KEY,
            *WEATHER_KEY,
        )
        .join(
            coastal_weather,
            on=WEATHER_KEY,
            how="left",
            validate="m:1",
        )
        .select(
            *EVENT_KEY,
            "weather_coastal_temperature_2m_c",
            "weather_coastal_provider",
            "weather_coastal_model",
            "weather_coastal_request_id",
            "_weather_coastal_matched",
        )
    )

    return (
        events
        .join(
            fallback,
            on=EVENT_KEY,
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.coalesce(
                "weather_land_temperature_2m_c",
                "weather_coastal_temperature_2m_c",
            )
            .alias(
                "weather_temperature_2m_c"
            ),

            pl.when(
                pl.col(
                    "weather_land_temperature_2m_c"
                ).is_not_null()
            )
            .then(
                pl.lit("land")
            )
            .when(
                pl.col(
                    "weather_coastal_temperature_2m_c"
                ).is_not_null()
            )
            .then(
                pl.lit("coastal_fallback")
            )
            .otherwise(
                pl.lit("missing")
            )
            .alias(
                "weather_temperature_source"
            ),
        )
    )


# =============================================================================
# Canonical weather projection
# =============================================================================


def add_canonical_weather(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    return events.with_columns(
        pl.col(
            "weather_land_relative_humidity_2m_pct"
        )
        .alias("weather_relative_humidity_2m_pct"),

        pl.col(
            "weather_land_precipitation_mm"
        )
        .alias("weather_precipitation_mm"),

        pl.col(
            "weather_land_rain_mm"
        )
        .alias("weather_rain_mm"),

        pl.col(
            "weather_land_snowfall_cm"
        )
        .alias("weather_snowfall_cm"),

        pl.col(
            "weather_land_cloud_cover_pct"
        )
        .alias("weather_cloud_cover_pct"),

        pl.col(
            "weather_land_surface_pressure_hpa"
        )
        .alias("weather_surface_pressure_hpa"),

        pl.col(
            "weather_land_weather_code"
        )
        .alias("weather_code"),

        pl.col(
            "weather_land_wind_speed_10m_kmh"
        )
        .alias("weather_wind_speed_10m_kmh"),

        pl.col(
            "weather_land_wind_direction_10m_deg"
        )
        .alias("weather_wind_direction_10m_deg"),

        pl.col(
            "weather_land_wind_gusts_10m_kmh"
        )
        .alias("weather_wind_gusts_10m_kmh"),
    )


# =============================================================================
# Leakage contracts
# =============================================================================


def add_leakage_flags(
    events: pl.LazyFrame,
) -> pl.LazyFrame:
    occurrence = pl.col(
        "occurrence_timestamp_utc"
    )

    weather_hour = pl.col(
        "weather_timestamp"
    )

    return events.with_columns(
        (
            pl.col("acs_release_date")
            > pl.col("occurrence_date")
        )
        .fill_null(False)
        .alias("_leak_acs_future_release"),

        (
            pl.col("acs_vintage")
            > pl.col("occurrence_year")
        )
        .fill_null(False)
        .alias("_leak_future_acs_vintage"),

        (
            pl.col("tiger_line_year")
            > pl.col("occurrence_year")
        )
        .fill_null(False)
        .alias("_leak_future_tiger_year"),

        (
            pl.col("osm_snapshot_year")
            > pl.col("occurrence_year")
        )
        .fill_null(False)
        .alias("_leak_future_osm_snapshot"),

        (
            weather_hour
            > occurrence
        )
        .fill_null(False)
        .alias("_leak_future_weather"),

        (
            pl.col("_weather_land_matched")
            .fill_null(False)
            & (
                occurrence.is_null()
                | weather_hour.is_null()
                | (
                    weather_hour
                    > occurrence
                )
                | (
                    occurrence
                    - weather_hour
                    >= pl.duration(hours=1)
                )
            )
        )
        .fill_null(False)
        .alias(
            "_weather_hour_alignment_violation"
        ),
    )


# =============================================================================
# Full transformation graph
# =============================================================================


def build_event_spine(
    *,
    crime_by_city: Mapping[str, pl.LazyFrame],
    city_timezones: Mapping[str, str],
    acs_calendar: pl.LazyFrame,
    tract_mapping: pl.LazyFrame,
    socioeconomic: pl.LazyFrame,
    osm: pl.LazyFrame,
    land_weather: pl.LazyFrame,
    coastal_weather: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Pure composition function.

    No I/O.
    No collect().
    No writes.
    """

    events = build_crime_base(
        crime_by_city,
        city_timezones=city_timezones,
    )

    events = join_acs_calendar(
        events,
        prepare_acs_calendar(
            acs_calendar
        ),
    )

    events = join_tract_mapping(
        events,
        prepare_tract_mapping(
            tract_mapping
        ),
    )

    events = join_socioeconomic(
        events,
        prepare_socioeconomic(
            socioeconomic
        ),
    )

    events = join_osm(
        events,
        prepare_osm(
            osm
        ),
    )

    events = join_land_weather(
        events,
        prepare_land_weather(
            land_weather
        ),
    )

    events = join_coastal_weather(
        events,
        prepare_coastal_weather(
            coastal_weather
        ),
    )

    events = add_canonical_weather(
        events
    )

    events = add_leakage_flags(
        events
    )

    return events