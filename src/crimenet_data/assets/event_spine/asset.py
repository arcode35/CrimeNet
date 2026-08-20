import polars as pl
import dagster as dg
from dagster import (
    AssetExecutionContext,
    AssetSelection,
    MaterializeResult,
    MetadataValue,
    asset,
    define_asset_job,
)

from crimenet_data.resources.crime_lake import CITIES

from .transformations import (
    build_event_spine,
)


# =============================================================================
# Storage
# =============================================================================


SILVER_CRIME_ROOT = (
    "gs://crimenet/silver/crime"
)

TRACT_CALENDAR_ROOT = (
    "gs://crimenet/raw_files/landing/"
    "tract_resources/acs_vintage_calendar"
)

TRACT_MAPPING_ROOT = (
    "gs://crimenet/silver/tract_resources/"
    "crime_location_tract_mapping"
)

SOCIOECONOMIC_ROOT = (
    "gs://crimenet/silver/socioeconomic/tract"
)

OSM_ROOT = (
    "gs://crimenet/silver/osm_h3_features"
)

WEATHER_LAND_ROOT = (
    "gs://crimenet/silver/weather/land_hourly"
)

WEATHER_COASTAL_ROOT = (
    "gs://crimenet/silver/weather/coastal_hourly"
)

EVENT_SPINE_STAGING_ROOT = (
    "gs://crimenet/gold_staging/event_spine"
)

EVENT_SPINE_ROOT = (
    "gs://crimenet/gold/event_spine"
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


EXPECTED_WIDTH = 226

LEAK_FLAGS = [
    "_leak_acs_future_release",
    "_leak_future_acs_vintage",
    "_leak_future_tiger_year",
    "_leak_future_osm_snapshot",
    "_leak_future_weather",
    "_weather_hour_alignment_violation",
]


# =============================================================================
# I/O
# =============================================================================


def scan_delta(
    path: str,
    *,
    credentials: pl.CredentialProviderGCP,
) -> pl.LazyFrame:
    return pl.scan_delta(
        path,
        credential_provider=credentials,
    )


def write_delta(
    frame: pl.LazyFrame,
    *,
    path: str,
    credentials: pl.CredentialProviderGCP,
) -> None:
    frame.sink_delta(
        path,
        mode="overwrite",
        credential_provider=credentials,
        delta_write_options={
            "partition_by": [
                "split",
                "source_city",
            ],
            "schema_mode":
                "overwrite",
        },
    )


# =============================================================================
# Validation
# =============================================================================


def validate_event_spine(
    frame: pl.LazyFrame,
    *,
    expected_source_rows: int,
) -> dict[str, int | float]:
    schema = frame.collect_schema()

    width = len(schema)

    if width != EXPECTED_WIDTH:
        raise ValueError(
            "Unexpected event-spine width: "
            f"expected={EXPECTED_WIDTH}, "
            f"actual={width}"
        )

    metrics = (
        frame
        .select(
            pl.len()
            .cast(pl.Int64)
            .alias("rows"),

            pl.col("crime_id")
            .n_unique()
            .cast(pl.Int64)
            .alias("unique_crime_ids"),

            pl.col("_tract_mapping_matched")
            .fill_null(False)
            .sum()
            .alias("tract_matches"),

            pl.col("_socioeconomic_matched")
            .fill_null(False)
            .sum()
            .alias("socioeconomic_matches"),

            pl.col("_osm_matched")
            .fill_null(False)
            .sum()
            .alias("osm_matches"),

            pl.col("weather_temperature_2m_c")
            .is_not_null()
            .sum()
            .alias("weather_matches"),

            *[
                pl.col(column)
                .fill_null(False)
                .sum()
                .alias(column)
                for column in LEAK_FLAGS
            ],
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    rows = int(
        metrics["rows"]
    )

    unique_ids = int(
        metrics["unique_crime_ids"]
    )

    if rows != expected_source_rows:
        raise ValueError(
            "Event-spine cardinality changed: "
            f"source={expected_source_rows:,}, "
            f"spine={rows:,}"
        )

    if unique_ids != rows:
        raise ValueError(
            "crime_id uniqueness violated: "
            f"rows={rows:,}, "
            f"unique={unique_ids:,}"
        )

    leakage_count = sum(
        int(metrics[column])
        for column in LEAK_FLAGS
    )

    if leakage_count != 0:
        raise ValueError(
            "Temporal leakage detected: "
            f"{leakage_count:,} violations"
        )

    return {
        "rows":
            rows,

        "columns":
            width,

        "tract_match_rate":
            metrics["tract_matches"] / rows,

        "socioeconomic_match_rate":
            metrics["socioeconomic_matches"] / rows,

        "osm_match_rate":
            metrics["osm_matches"] / rows,

        "weather_match_rate":
            metrics["weather_matches"] / rows,

        "leakage_violations":
            leakage_count,
    }


# =============================================================================
# Asset
# =============================================================================


@dg.asset(
    name="event_spine",
    group_name="gold",
    compute_kind="polars",
)
def event_spine(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """
    Build the canonical enriched event spine.

    Grain:
        one row per canonical crime event

    Current contract:
        13,232,409 rows
        226 columns
    """

    credentials = (
        pl.CredentialProviderGCP()
    )

    # -------------------------------------------------------------------------
    # Load upstream tables lazily
    # -------------------------------------------------------------------------

    crime_by_city = {
        city: scan_delta(
            f"{SILVER_CRIME_ROOT}/{city}",
            credentials=credentials,
        )
        for city in CITIES
    }

    acs_calendar = scan_delta(
        TRACT_CALENDAR_ROOT,
        credentials=credentials,
    )

    tract_mapping = scan_delta(
        TRACT_MAPPING_ROOT,
        credentials=credentials,
    )

    socioeconomic = scan_delta(
        SOCIOECONOMIC_ROOT,
        credentials=credentials,
    )

    osm = scan_delta(
        OSM_ROOT,
        credentials=credentials,
    )

    land_weather = scan_delta(
        WEATHER_LAND_ROOT,
        credentials=credentials,
    )

    coastal_weather = scan_delta(
        WEATHER_COASTAL_ROOT,
        credentials=credentials,
    )

    # -------------------------------------------------------------------------
    # Structural source count
    # -------------------------------------------------------------------------

    source_rows = (
        pl.concat(
            [
                frame.select(
                    pl.len().alias("rows")
                )
                for frame
                in crime_by_city.values()
            ],
            how="vertical",
        )
        .select(
            pl.col("rows")
            .sum()
            .alias("rows")
        )
        .collect()
        .item()
    )

    context.log.info(
        "Building event spine "
        f"from {source_rows:,} canonical crime rows"
    )

    # -------------------------------------------------------------------------
    # Build the logical transformation graph
    # -------------------------------------------------------------------------

    final = build_event_spine(
        crime_by_city=crime_by_city,
        city_timezones=CITY_TIMEZONES,
        acs_calendar=acs_calendar,
        tract_mapping=tract_mapping,
        socioeconomic=socioeconomic,
        osm=osm,
        land_weather=land_weather,
        coastal_weather=coastal_weather,
    )

    planned_width = len(
        final.collect_schema()
    )

    if planned_width != EXPECTED_WIDTH:
        raise ValueError(
            "Event-spine schema contract changed: "
            f"expected={EXPECTED_WIDTH}, "
            f"actual={planned_width}"
        )

    context.log.info(
        "Event-spine logical plan validated: "
        f"{planned_width} columns"
    )

    # -------------------------------------------------------------------------
    # Materialize staging
    #
    # Expensive joins execute exactly here.
    # -------------------------------------------------------------------------

    context.log.info(
        "Writing event-spine staging table"
    )

    write_delta(
        final,
        path=EVENT_SPINE_STAGING_ROOT,
        credentials=credentials,
    )

    # -------------------------------------------------------------------------
    # Validate actual persisted output
    # -------------------------------------------------------------------------

    staged = scan_delta(
        EVENT_SPINE_STAGING_ROOT,
        credentials=credentials,
    )

    metrics = validate_event_spine(
        staged,
        expected_source_rows=source_rows,
    )

    context.log.info(
        "Event spine passed validation: "
        f"rows={metrics['rows']:,}, "
        f"columns={metrics['columns']}, "
        f"osm={metrics['osm_match_rate']:.6%}, "
        f"weather={metrics['weather_match_rate']:.6%}"
    )

    # -------------------------------------------------------------------------
    # Publish
    #
    # Important:
    # source joins DO NOT execute again.
    # This is staging Delta -> final Delta.
    # -------------------------------------------------------------------------

    context.log.info(
        "Publishing validated event spine"
    )

    write_delta(
        staged,
        path=EVENT_SPINE_ROOT,
        credentials=credentials,
    )

    return MaterializeResult(
        metadata={
            "rows":
                metrics["rows"],

            "columns":
                metrics["columns"],

            "tract_match_rate":
                metrics["tract_match_rate"],

            "socioeconomic_match_rate":
                metrics[
                    "socioeconomic_match_rate"
                ],

            "osm_match_rate":
                metrics["osm_match_rate"],

            "weather_match_rate":
                metrics["weather_match_rate"],

            "leakage_violations":
                metrics["leakage_violations"],

            "output":
                MetadataValue.text(
                    EVENT_SPINE_ROOT
                ),
        }
    )