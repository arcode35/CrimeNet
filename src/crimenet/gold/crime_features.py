"""Python-wheel entry point for building CrimeNet Gold crime features."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from delta.tables import DeltaTable
from pyspark.databricks.sql import functions as dbf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

LOGGER = logging.getLogger(__name__)

LOCATION_KEYS = (
    "tiger_line_year",
    "latitude",
    "longitude",
)

WEATHER_KEYS = (
    "weather_query_cell_id",
    "weather_timestamp",
)


@dataclass(frozen=True)
class FeatureTables:
    crime: str
    calendar: str
    boundaries: str
    socioeconomic: str
    weather: str
    lighting: str
    location_tract_mapping: str
    features: str

    @classmethod
    def from_schemas(
        cls,
        *,
        catalog: str,
        silver_schema: str,
        gold_schema: str,
    ) -> "FeatureTables":
        silver = f"{catalog}.{silver_schema}"
        gold = f"{catalog}.{gold_schema}"

        return cls(
            crime=f"{silver}.crime_offenses",
            calendar=f"{silver}.acs_vintage_calendar",
            boundaries=f"{silver}.census_tract_boundaries",
            socioeconomic=f"{silver}.tract_socioeconomic",
            weather=f"{silver}.weather_hourly",
            lighting=f"{silver}.solar_lighting_conditions",
            location_tract_mapping=(
                f"{silver}.crime_location_tract_mapping"
            ),
            features=f"{gold}.crime_features",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe crime features from crime, ACS, "
            "Census tract, and hourly weather tables."
        )
    )

    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", default="silver")
    parser.add_argument("--gold-schema", default="gold")
    parser.add_argument("--weather-provider", default="open_meteo")
    parser.add_argument("--weather-model", default="era5_land")
    parser.add_argument(
        "--weather-h3-resolution",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--rebuild-location-tract-mapping",
        action="store_true",
        help=(
            "Replace the complete coordinate/year-to-tract mapping. "
            "When omitted, only previously unseen keys are mapped."
        ),
    )

    return parser.parse_args()


def _has_rows(dataframe: DataFrame) -> bool:
    return not dataframe.isEmpty()


def _raise_on_duplicate_keys(
    dataframe: DataFrame,
    *,
    keys: tuple[str, ...],
    dataset_name: str,
) -> None:
    duplicates = (
        dataframe
        .groupBy(*keys)
        .count()
        .filter(F.col("count") > 1)
    )

    if _has_rows(duplicates):
        examples = [
            row.asDict()
            for row in (
                duplicates
                .orderBy(F.col("count").desc())
                .limit(20)
                .collect()
            )
        ]

        raise RuntimeError(
            f"{dataset_name} contains duplicate keys {keys}. "
            f"Examples: {examples}"
        )


def validate_boundary_inputs(
    calendar_dataframe: DataFrame,
    boundary_dataframe: DataFrame,
) -> None:
    missing_boundary_years = (
        calendar_dataframe
        .select(
            F.col("tiger_line_year").alias(
                "boundary_vintage"
            )
        )
        .distinct()
        .join(
            boundary_dataframe
            .select("boundary_vintage")
            .distinct(),
            on="boundary_vintage",
            how="left_anti",
        )
        .orderBy("boundary_vintage")
        .collect()
    )

    if missing_boundary_years:
        missing_years = [
            row["boundary_vintage"]
            for row in missing_boundary_years
        ]

        raise ValueError(
            "Missing TIGER/Line boundary vintages: "
            f"{missing_years}"
        )

    boundary_srids = {
        row["srid"]
        for row in (
            boundary_dataframe
            .select(
                dbf.st_srid(
                    "tract_geometry"
                ).alias("srid")
            )
            .distinct()
            .collect()
        )
    }

    if boundary_srids != {4326}:
        raise ValueError(
            "Expected all tract geometries to use SRID 4326; "
            f"found {sorted(boundary_srids)}"
        )

    _raise_on_duplicate_keys(
        boundary_dataframe,
        keys=("boundary_vintage", "geoid"),
        dataset_name="Census tract boundaries",
    )


def build_calendar_ranges(
    calendar_dataframe: DataFrame,
) -> DataFrame:
    calendar_window = Window.orderBy("acs_release_date")

    return (
        calendar_dataframe
        .withColumn(
            "eligible_start_date",
            F.date_add(
                F.col("acs_release_date"),
                1,
            ),
        )
        .withColumn(
            "_next_eligible_start_date",
            F.lead(
                "eligible_start_date"
            ).over(calendar_window),
        )
        .withColumn(
            "eligible_end_date",
            F.date_sub(
                F.col("_next_eligible_start_date"),
                1,
            ),
        )
        .drop("_next_eligible_start_date")
    )


def prepare_crimes(
    crime_dataframe: DataFrame,
) -> DataFrame:
    return (
        crime_dataframe
        .withColumn(
            "crime_offense_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("source_city"), F.lit("")),
                    F.coalesce(F.col("source_file"), F.lit("")),
                    F.coalesce(F.col("source_row_hash"), F.lit("")),
                    F.coalesce(F.col("source_record_id"), F.lit("")),
                    F.coalesce(F.col("offense_code"), F.lit("")),
                    F.coalesce(
                        F.col("occurred_at").cast("string"),
                        F.lit(""),
                    ),
                ),
                256,
            ),
        )
        .withColumn(
            "occurred_date",
            F.to_date("occurred_at"),
        )
        .withColumn(
            "weather_timestamp",
            F.date_trunc(
                "hour",
                F.col("occurred_at"),
            ),
        )
    )


def attach_eligible_acs_vintage(
    crime_dataframe: DataFrame,
    calendar_ranges: DataFrame,
) -> DataFrame:
    return (
        crime_dataframe.alias("crime")
        .join(
            F.broadcast(calendar_ranges).alias("calendar"),
            (
                F.col("crime.occurred_date")
                >= F.col("calendar.eligible_start_date")
            )
            & (
                F.col("calendar.eligible_end_date").isNull()
                | (
                    F.col("crime.occurred_date")
                    <= F.col("calendar.eligible_end_date")
                )
            ),
            "left",
        )
        .select(
            "crime.*",
            F.col("calendar.acs_vintage").alias(
                "selected_acs_vintage"
            ),
            F.col("calendar.acs_release_date").alias(
                "selected_acs_release_date"
            ),
            F.col("calendar.tiger_line_year"),
            F.col("calendar.tract_definition_vintage"),
        )
    )


def extract_unique_locations(
    crime_with_calendar: DataFrame,
) -> DataFrame:
    valid_location_condition = (
        F.col("latitude").isNotNull()
        & F.col("longitude").isNotNull()
        & ~F.isnan("latitude")
        & ~F.isnan("longitude")
        & F.col("latitude").between(-90.0, 90.0)
        & F.col("longitude").between(-180.0, 180.0)
        & F.col("tiger_line_year").isNotNull()
    )

    return (
        crime_with_calendar
        .filter(valid_location_condition)
        .select(*LOCATION_KEYS)
        .dropDuplicates(list(LOCATION_KEYS))
    )


def spatially_map_locations(
    location_dataframe: DataFrame,
    boundary_dataframe: DataFrame,
) -> DataFrame:
    locations = location_dataframe.withColumn(
        "crime_point",
        dbf.st_point(
            F.col("longitude"),
            F.col("latitude"),
            4326,
        ),
    )

    tracts = (
        boundary_dataframe
        .filter(F.col("tract_geometry").isNotNull())
        .select(
            "boundary_vintage",
            "geoid",
            "tract_geometry",
        )
    )

    contains_matches = (
        locations.alias("location")
        .join(
            tracts.alias("tract"),
            (
                F.col("location.tiger_line_year")
                == F.col("tract.boundary_vintage")
            )
            & dbf.st_contains(
                F.col("tract.tract_geometry"),
                F.col("location.crime_point"),
            ),
            "inner",
        )
        .select(
            *[
                F.col(f"location.{column}")
                for column in LOCATION_KEYS
            ],
            F.col("tract.geoid").alias("tract_geoid"),
        )
    )

    _raise_on_duplicate_keys(
        contains_matches,
        keys=LOCATION_KEYS,
        dataset_name="ST_Contains tract mapping",
    )

    unmatched_locations = (
        locations
        .select(*LOCATION_KEYS, "crime_point")
        .join(
            contains_matches.select(*LOCATION_KEYS),
            on=list(LOCATION_KEYS),
            how="left_anti",
        )
    )

    covers_candidates = (
        unmatched_locations.alias("location")
        .join(
            tracts.alias("tract"),
            (
                F.col("location.tiger_line_year")
                == F.col("tract.boundary_vintage")
            )
            & dbf.st_covers(
                F.col("tract.tract_geometry"),
                F.col("location.crime_point"),
            ),
            "inner",
        )
        .select(
            *[
                F.col(f"location.{column}")
                for column in LOCATION_KEYS
            ],
            F.col("tract.geoid").alias("candidate_geoid"),
        )
    )

    unique_covers_matches = (
        covers_candidates
        .groupBy(*LOCATION_KEYS)
        .agg(
            F.countDistinct("candidate_geoid").alias(
                "_candidate_count"
            ),
            F.first(
                "candidate_geoid",
                ignorenulls=True,
            ).alias("tract_geoid"),
        )
        .filter(F.col("_candidate_count") == 1)
        .drop("_candidate_count")
    )

    resolved_locations = contains_matches.unionByName(
        unique_covers_matches
    )

    mapping = (
        location_dataframe.alias("location")
        .join(
            resolved_locations.alias("resolved"),
            on=list(LOCATION_KEYS),
            how="left",
        )
        .select(*LOCATION_KEYS, "tract_geoid")
    )

    _raise_on_duplicate_keys(
        mapping,
        keys=LOCATION_KEYS,
        dataset_name="Location-to-tract mapping",
    )

    return mapping


def materialize_location_mapping(
    spark: SparkSession,
    *,
    candidate_locations: DataFrame,
    boundary_dataframe: DataFrame,
    target_table: str,
    rebuild: bool,
) -> DataFrame:
    table_exists = spark.catalog.tableExists(target_table)

    if rebuild or not table_exists:
        LOGGER.info(
            "Building complete location-to-tract mapping: %s",
            target_table,
        )

        mapping = spatially_map_locations(
            candidate_locations,
            boundary_dataframe,
        )

        (
            mapping.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(target_table)
        )

        return spark.table(target_table)

    existing_mapping = spark.table(target_table)

    _raise_on_duplicate_keys(
        existing_mapping,
        keys=LOCATION_KEYS,
        dataset_name="Existing location-to-tract mapping",
    )

    missing_locations = (
        candidate_locations
        .join(
            existing_mapping.select(*LOCATION_KEYS),
            on=list(LOCATION_KEYS),
            how="left_anti",
        )
    )

    if not _has_rows(missing_locations):
        LOGGER.info(
            "No new location/year keys require tract mapping."
        )
        return existing_mapping

    LOGGER.info(
        "Mapping previously unseen location/year keys."
    )

    new_mapping = spatially_map_locations(
        missing_locations,
        boundary_dataframe,
    )

    merge_condition = """
        target.tiger_line_year = source.tiger_line_year
        AND target.latitude = source.latitude
        AND target.longitude = source.longitude
    """

    (
        DeltaTable.forName(spark, target_table)
        .alias("target")
        .merge(
            new_mapping.alias("source"),
            merge_condition,
        )
        .whenNotMatchedInsertAll()
        .execute()
    )

    updated_mapping = spark.table(target_table)

    _raise_on_duplicate_keys(
        updated_mapping,
        keys=LOCATION_KEYS,
        dataset_name="Updated location-to-tract mapping",
    )

    return updated_mapping


def attach_tracts(
    crime_dataframe: DataFrame,
    location_mapping: DataFrame,
) -> DataFrame:
    return (
        crime_dataframe.alias("crime")
        .join(
            location_mapping.alias("mapping"),
            (
                F.col("crime.tiger_line_year")
                == F.col("mapping.tiger_line_year")
            )
            & (
                F.col("crime.latitude")
                == F.col("mapping.latitude")
            )
            & (
                F.col("crime.longitude")
                == F.col("mapping.longitude")
            ),
            "left",
        )
        .select(
            "crime.*",
            F.col("mapping.tract_geoid").alias(
                "tract_geoid"
            ),
        )
    )


def attach_socioeconomic_features(
    crime_dataframe: DataFrame,
    socioeconomic_dataframe: DataFrame,
) -> DataFrame:
    _raise_on_duplicate_keys(
        socioeconomic_dataframe,
        keys=("geoid", "acs_vintage"),
        dataset_name="ACS socioeconomic data",
    )

    acs_lookup = (
        socioeconomic_dataframe
        .select(
            "geoid",
            "acs_vintage",
            "geography_name",
            "population",
            "population_moe",
            "median_age",
            "median_age_moe",
            "median_household_income",
            "median_household_income_moe",
            "poverty_rate",
            "unemployment_rate",
            "vacancy_rate",
            "renter_occupied_rate",
            "no_vehicle_rate",
        )
        .withColumn(
            "_socioeconomic_row_found",
            F.lit(True),
        )
    )

    return (
        crime_dataframe.alias("crime")
        .join(
            F.broadcast(acs_lookup).alias("acs"),
            (
                F.col("crime.tract_geoid")
                == F.col("acs.geoid")
            )
            & (
                F.col("crime.selected_acs_vintage")
                == F.col("acs.acs_vintage")
            ),
            "left",
        )
        .select(
            "crime.*",
            F.col("acs.geography_name"),
            F.col("acs.population"),
            F.col("acs.population_moe"),
            F.col("acs.median_age"),
            F.col("acs.median_age_moe"),
            F.col("acs.median_household_income"),
            F.col("acs.median_household_income_moe"),
            F.col("acs.poverty_rate"),
            F.col("acs.unemployment_rate"),
            F.col("acs.vacancy_rate"),
            F.col("acs.renter_occupied_rate"),
            F.col("acs.no_vehicle_rate"),
            F.coalesce(
                F.col("acs._socioeconomic_row_found"),
                F.lit(False),
            ).alias("socioeconomic_match_found"),
        )
    )


def build_weather_lookup(
    crime_dataframe: DataFrame,
    weather_dataframe: DataFrame,
    *,
    provider: str,
    model: str,
    h3_resolution: int,
) -> DataFrame:
    crime_date_bounds = (
        crime_dataframe
        .agg(
            F.min(F.to_date("occurred_at")).alias(
                "minimum_crime_date"
            ),
            F.max(F.to_date("occurred_at")).alias(
                "maximum_crime_date"
            ),
        )
        .first()
    )

    minimum_crime_date = crime_date_bounds[
        "minimum_crime_date"
    ]
    maximum_crime_date = crime_date_bounds[
        "maximum_crime_date"
    ]

    if minimum_crime_date is None or maximum_crime_date is None:
        raise ValueError(
            "Could not determine crime date boundaries."
        )

    relevant_weather_cells = (
        crime_dataframe
        .filter(
            F.col("weather_query_cell_id").isNotNull()
        )
        .select("weather_query_cell_id")
        .distinct()
    )

    lookup = (
        weather_dataframe
        .filter(
            (F.col("provider") == provider)
            & (F.col("model") == model)
            & (F.col("h3_resolution") == h3_resolution)
            & F.col("weather_date").between(
                F.lit(minimum_crime_date),
                F.lit(maximum_crime_date),
            )
        )
        .join(
            F.broadcast(relevant_weather_cells),
            on="weather_query_cell_id",
            how="left_semi",
        )
        .select(
            "weather_query_cell_id",
            "weather_timestamp",
            "temperature_2m_c",
            "grid_elevation",
        )
        .withColumn(
            "_weather_row_found",
            F.lit(True),
        )
    )

    _raise_on_duplicate_keys(
        lookup,
        keys=WEATHER_KEYS,
        dataset_name="Filtered hourly weather",
    )

    return lookup


def attach_weather_features(
    crime_dataframe: DataFrame,
    weather_lookup: DataFrame,
) -> DataFrame:
    return (
        crime_dataframe.alias("crime")
        .join(
            weather_lookup.alias("weather"),
            (
                F.col("crime.weather_query_cell_id")
                == F.col("weather.weather_query_cell_id")
            )
            & (
                F.col("crime.weather_timestamp")
                == F.col("weather.weather_timestamp")
            ),
            "left",
        )
        .select(
            "crime.*",
            F.col("weather.temperature_2m_c"),
            F.col("weather.grid_elevation"),
            F.coalesce(
                F.col("weather._weather_row_found"),
                F.lit(False),
            ).alias("weather_match_found"),
        )
    )

LIGHTING_KEYS = (
    "weather_query_cell_id",
    "solar_timestamp",
)


def build_lighting_lookup(
    lighting_dataframe: DataFrame,
) -> DataFrame:
    lookup = (
        lighting_dataframe
        .select(
            "weather_query_cell_id",
            "solar_timestamp",
            "solar_elevation_deg",
            "apparent_solar_elevation_deg",
            "solar_zenith_deg",
            "solar_azimuth_deg",
            "lighting_condition",
            "is_daylight",
        )
        .withColumn(
            "_lighting_row_found",
            F.lit(True),
        )
    )

    _raise_on_duplicate_keys(
        lookup,
        keys=LIGHTING_KEYS,
        dataset_name="Solar lighting conditions",
    )

    return lookup

def attach_lighting_features(
    crime_dataframe: DataFrame,
    lighting_lookup: DataFrame,
) -> DataFrame:
    return (
        crime_dataframe.alias("crime")
        .join(
            lighting_lookup.alias("light"),
            (
                F.col("crime.weather_query_cell_id")
                == F.col("light.weather_query_cell_id")
            )
            & (
                F.col("crime.weather_timestamp")
                == F.col("light.solar_timestamp")
            ),
            "left",
        )
        .select(
            "crime.*",
            F.col("light.solar_elevation_deg"),
            F.col(
                "light.apparent_solar_elevation_deg"
            ),
            F.col("light.solar_zenith_deg"),
            F.col("light.solar_azimuth_deg"),
            F.col("light.lighting_condition"),
            F.col("light.is_daylight"),
            F.coalesce(
                F.col("light._lighting_row_found"),
                F.lit(False),
            ).alias("lighting_match_found"),
        )
    )

def log_coverage_metrics(
    feature_dataframe: DataFrame,
) -> dict[str, object]:
    metrics = (
        feature_dataframe
        .agg(
            F.count("*").alias("final_rows"),
            F.count("selected_acs_vintage").alias(
                "rows_with_eligible_acs_vintage"
            ),
            F.count("tract_geoid").alias("rows_with_tract"),
            F.sum(
                F.col("socioeconomic_match_found").cast("long")
            ).alias("rows_with_socioeconomic_record"),
            F.count("median_household_income").alias(
                "rows_with_non_null_income"
            ),
            F.sum(
                F.col("weather_match_found").cast("long")
            ).alias("rows_with_weather_record"),
            F.count("temperature_2m_c").alias(
                "rows_with_non_null_temperature"
            ),
            F.round(
                F.avg(
                    F.col("tract_geoid")
                    .isNotNull()
                    .cast("double")
                ),
                8,
            ).alias("tract_match_rate"),
            F.round(
                F.avg(
                    F.col("socioeconomic_match_found")
                    .cast("double")
                ),
                8,
            ).alias("socioeconomic_match_rate"),
            F.round(
                F.avg(
                    F.col("weather_match_found").cast("double")
                ),
                8,
            ).alias("weather_match_rate"),
        )
        .first()
        .asDict()
    )

    LOGGER.info(
        "Crime feature coverage metrics: %s",
        metrics,
    )

    return metrics