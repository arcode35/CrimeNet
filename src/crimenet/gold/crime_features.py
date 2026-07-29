"""Build and validate CrimeNet Gold crime features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from crimenet.contracts.gold import (
    GOLD_IDENTITY_VERSION,
    GoldCoverageThresholds,
    coverage_failures,
)
from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
    LIGHTING_KEYS,
    VALID_LIGHTING_CONDITIONS,
)
from crimenet.observability.logging import get_logger
from crimenet.quality.checks import (
    QualityCheck,
    merge_quality_results,
    quality_results_dataframe,
)
from crimenet.utils.promotion import (
    drop_staging_table,
    normalize_pipeline_run_id,
    promote_staged_delta_table,
    staging_table_name,
)

LOGGER = get_logger(__name__)

LOCATION_KEYS = (
    "tiger_line_year",
    "latitude",
    "longitude",
)

WEATHER_KEYS = (
    "weather_query_cell_id",
    "weather_timestamp",
)

PLAUSIBLE_FEATURE_RANGES = {
    "solar_elevation_deg": (-90.0, 90.0),
    "apparent_solar_elevation_deg": (-90.0, 90.0),
    "solar_zenith_deg": (0.0, 180.0),
    "solar_azimuth_deg": (0.0, 360.0),
    "temperature_2m_c": (-100.0, 70.0),
    "grid_elevation": (-500.0, 9_000.0),
    "median_age": (0.0, 120.0),
    "median_household_income": (0.0, 1_000_000_000.0),
    "population": (0.0, 100_000_000.0),
    "poverty_rate": (0.0, 1.0),
    "unemployment_rate": (0.0, 1.0),
    "vacancy_rate": (0.0, 1.0),
    "renter_occupied_rate": (0.0, 1.0),
    "no_vehicle_rate": (0.0, 1.0),
}

LIGHTING_SOLAR_FEATURES = (
    "solar_elevation_deg",
    "apparent_solar_elevation_deg",
    "solar_zenith_deg",
    "solar_azimuth_deg",
)


@dataclass(frozen=True)
class FeatureTables:
    crime: str
    calendar: str
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
    ) -> FeatureTables:
        silver = f"{catalog}.{silver_schema}"
        gold = f"{catalog}.{gold_schema}"

        return cls(
            crime=f"{silver}.crime_offenses",
            calendar=f"{silver}.acs_vintage_calendar",
            socioeconomic=f"{silver}.tract_socioeconomic",
            weather=f"{silver}.weather_hourly",
            lighting=f"{silver}.solar_lighting_conditions",
            location_tract_mapping=(f"{silver}.crime_location_tract_mapping"),
            features=f"{gold}.crime_features",
        )


def _has_rows(dataframe: DataFrame) -> bool:
    return not dataframe.isEmpty()


def _usable_numeric_feature(column_name: str) -> F.Column:
    minimum, maximum = PLAUSIBLE_FEATURE_RANGES[column_name]
    condition = (
        F.col(column_name).isNotNull()
        & ~F.isnan(column_name)
        & F.col(column_name).between(minimum, maximum)
    )
    return F.coalesce(condition, F.lit(False))


def _usable_weather_feature() -> F.Column:
    return F.col("weather_match_found").eqNullSafe(True) & _usable_numeric_feature(
        "temperature_2m_c"
    )


def _usable_lighting_feature(
    *,
    definition_version: str,
) -> F.Column:
    solar_values_are_usable = F.lit(True)
    for column_name in LIGHTING_SOLAR_FEATURES:
        solar_values_are_usable &= _usable_numeric_feature(column_name)

    condition = (
        F.col("lighting_match_found").eqNullSafe(True)
        & F.col("lighting_query_cell_id").isNotNull()
        & F.col("lighting_definition_version").isNotNull()
        & (F.trim(F.col("lighting_definition_version")) != "")
        & (F.col("lighting_definition_version") == definition_version)
        & F.col("lighting_condition").isNotNull()
        & F.col("lighting_condition").isin(*VALID_LIGHTING_CONDITIONS)
        & F.col("is_daylight").isNotNull()
        & solar_values_are_usable
        & (F.col("is_daylight") == (F.col("solar_elevation_deg") >= 0.0))
    )
    return F.coalesce(condition, F.lit(False))


def _usable_socioeconomic_feature() -> F.Column:
    condition = (
        F.col("socioeconomic_match_found").eqNullSafe(True)
        & F.col("selected_acs_vintage").isNotNull()
        & F.col("tract_geoid").isNotNull()
        & (F.trim(F.col("tract_geoid")) != "")
        & _usable_numeric_feature("median_household_income")
    )
    return F.coalesce(condition, F.lit(False))


def _raise_on_duplicate_keys(
    dataframe: DataFrame,
    *,
    keys: tuple[str, ...],
    dataset_name: str,
) -> None:
    duplicates = dataframe.groupBy(*keys).count().filter(F.col("count") > 1)

    if _has_rows(duplicates):
        examples = [
            row.asDict()
            for row in (duplicates.orderBy(F.col("count").desc()).limit(20).collect())
        ]

        raise RuntimeError(
            f"{dataset_name} contains duplicate keys {keys}. Examples: {examples}"
        )


def _raise_on_key_set_mismatch(
    expected_dataframe: DataFrame,
    actual_dataframe: DataFrame,
    *,
    keys: tuple[str, ...],
    dataset_name: str,
) -> None:
    expected_keys = expected_dataframe.select(*keys).dropDuplicates(list(keys))
    actual_keys = actual_dataframe.select(*keys)

    missing_keys = expected_keys.join(
        actual_keys,
        on=list(keys),
        how="left_anti",
    )
    unexpected_keys = actual_keys.join(
        expected_keys,
        on=list(keys),
        how="left_anti",
    )

    if _has_rows(missing_keys) or _has_rows(unexpected_keys):
        raise RuntimeError(
            f"{dataset_name} has a mismatched business-key set "
            f"for {keys}: missing={missing_keys.count()}, "
            f"unexpected={unexpected_keys.count()}."
        )


def build_calendar_ranges(
    calendar_dataframe: DataFrame,
) -> DataFrame:
    calendar_window = Window.orderBy("acs_release_date")

    return (
        calendar_dataframe.withColumn(
            "eligible_start_date",
            F.date_add(
                F.col("acs_release_date"),
                1,
            ),
        )
        .withColumn(
            "_next_eligible_start_date",
            F.lead("eligible_start_date").over(calendar_window),
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
    def optional_column(
        name: str,
        *,
        fallback_name: str | None = None,
    ) -> F.Column:
        if name in crime_dataframe.columns:
            return F.col(name)
        if fallback_name is not None and fallback_name in crime_dataframe.columns:
            return F.col(fallback_name)
        return F.lit(None)

    def non_blank(column: F.Column) -> F.Column:
        value = F.trim(column.cast("string"))
        return F.when(
            value.isNotNull() & (F.length(value) > 0),
            value,
        )

    source_system_identity = non_blank(
        optional_column(
            "source_system",
            fallback_name="source_city",
        )
    )
    source_incident_identity = non_blank(optional_column("source_incident_id"))
    source_offense_identity = non_blank(
        optional_column(
            "source_offense_id",
            fallback_name="source_record_id",
        )
    )
    complete_source_identity = (
        source_system_identity.isNotNull()
        & source_incident_identity.isNotNull()
        & source_offense_identity.isNotNull()
    )
    stable_source_identity = F.when(
        complete_source_identity,
        F.sha2(
            F.to_json(
                F.struct(
                    source_system_identity.alias("source_system"),
                    source_incident_identity.alias("source_incident_id"),
                    source_offense_identity.alias("source_offense_id"),
                ),
                options={"ignoreNullFields": "false"},
            ),
            256,
        ),
    )

    stable_fallback_payload = F.to_json(
        F.struct(
            *[
                optional_column(
                    name,
                    fallback_name=fallback_name,
                ).alias(name)
                for name, fallback_name in (
                    ("source_system", "source_city"),
                    ("source_incident_id", None),
                    ("source_offense_id", "source_record_id"),
                    ("offense_code", None),
                    ("offense_name", None),
                    ("offense_description", None),
                    ("occurred_at", None),
                    ("reported_at", None),
                    ("updated_at", None),
                    ("offense_count", None),
                    ("address", None),
                    ("city", None),
                    ("state", None),
                    ("postal_code", None),
                    ("latitude", None),
                    ("longitude", None),
                )
            ]
        ),
        options={"ignoreNullFields": "false"},
    )

    logical_identity = F.coalesce(
        non_blank(optional_column("business_identity")),
        stable_source_identity,
        non_blank(optional_column("source_row_hash")),
        F.sha2(stable_fallback_payload, 256),
    )

    return (
        crime_dataframe.withColumn(
            "crime_offense_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.lit(GOLD_IDENTITY_VERSION),
                    logical_identity,
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
        .withColumn(
            "lighting_query_cell_id",
            F.col("weather_query_cell_id").cast("long"),
        )
        .withColumn(
            "solar_timestamp_hour",
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
            (F.col("crime.occurred_date") >= F.col("calendar.eligible_start_date"))
            & (
                F.col("calendar.eligible_end_date").isNull()
                | (F.col("crime.occurred_date") <= F.col("calendar.eligible_end_date"))
            ),
            "left",
        )
        .select(
            "crime.*",
            F.col("calendar.acs_vintage").alias("selected_acs_vintage"),
            F.col("calendar.acs_release_date").alias("selected_acs_release_date"),
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
        crime_with_calendar.filter(valid_location_condition)
        .select(*LOCATION_KEYS)
        .dropDuplicates(list(LOCATION_KEYS))
    )


def attach_tracts(
    crime_dataframe: DataFrame,
    location_mapping: DataFrame,
) -> DataFrame:
    audit_columns = {
        "boundary_definition_version": ("boundary_definition_version"),
        "source_archive_sha256": ("boundary_source_archive_sha256"),
        "mapping_definition_version": ("mapping_definition_version"),
        "location_tract_key": "location_tract_key",
        "match_status": "tract_match_status",
        "candidate_match_count": ("tract_candidate_match_count"),
        "pipeline_run_id": ("tract_mapping_pipeline_run_id"),
        "mapped_at": "tract_mapped_at",
    }
    selected_audit_columns = [
        F.col(f"mapping.{source_name}").alias(target_name)
        for source_name, target_name in audit_columns.items()
        if source_name in location_mapping.columns
    ]

    return (
        crime_dataframe.alias("crime")
        .join(
            location_mapping.alias("mapping"),
            (F.col("crime.tiger_line_year") == F.col("mapping.tiger_line_year"))
            & (F.col("crime.latitude") == F.col("mapping.latitude"))
            & (F.col("crime.longitude") == F.col("mapping.longitude")),
            "left",
        )
        .select(
            "crime.*",
            F.col("mapping.tract_geoid").alias("tract_geoid"),
            *selected_audit_columns,
        )
    )


def build_location_mapping_lookup(
    candidate_locations: DataFrame,
    mapping_dataframe: DataFrame,
) -> DataFrame:
    """Validate and scope the independently materialized mapping."""

    required_columns = {
        *LOCATION_KEYS,
        "tract_geoid",
    }
    missing_columns = sorted(required_columns - set(mapping_dataframe.columns))
    if missing_columns:
        raise RuntimeError(
            f"Location-to-tract mapping is missing required columns: {missing_columns}."
        )

    _raise_on_duplicate_keys(
        mapping_dataframe,
        keys=LOCATION_KEYS,
        dataset_name="Prebuilt location-to-tract mapping",
    )

    lookup = mapping_dataframe.join(
        candidate_locations.select(*LOCATION_KEYS).dropDuplicates(list(LOCATION_KEYS)),
        on=list(LOCATION_KEYS),
        how="left_semi",
    )
    _raise_on_key_set_mismatch(
        candidate_locations,
        lookup,
        keys=LOCATION_KEYS,
        dataset_name="Prebuilt location-to-tract mapping",
    )

    ambiguity_condition = F.lit(False)
    if "match_status" in lookup.columns:
        ambiguity_condition |= F.col("match_status") == "ambiguous"
    if "candidate_match_count" in lookup.columns:
        ambiguity_condition |= F.col("candidate_match_count") > 1

    ambiguous_rows = lookup.filter(ambiguity_condition)
    if _has_rows(ambiguous_rows):
        raise RuntimeError(
            "Prebuilt location-to-tract mapping contains "
            f"ambiguous matches; rows={ambiguous_rows.count()}."
        )

    return lookup


def attach_socioeconomic_features(
    crime_dataframe: DataFrame,
    socioeconomic_dataframe: DataFrame,
) -> DataFrame:
    _raise_on_duplicate_keys(
        socioeconomic_dataframe,
        keys=("geoid", "acs_vintage"),
        dataset_name="ACS socioeconomic data",
    )

    acs_lookup = socioeconomic_dataframe.select(
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
    ).withColumn(
        "_socioeconomic_row_found",
        F.lit(True),
    )

    return (
        crime_dataframe.alias("crime")
        .join(
            F.broadcast(acs_lookup).alias("acs"),
            (F.col("crime.tract_geoid") == F.col("acs.geoid"))
            & (F.col("crime.selected_acs_vintage") == F.col("acs.acs_vintage")),
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
    crime_date_bounds = crime_dataframe.agg(
        F.min(F.to_date("occurred_at")).alias("minimum_crime_date"),
        F.max(F.to_date("occurred_at")).alias("maximum_crime_date"),
    ).first()

    if crime_date_bounds is None:
        raise ValueError("Could not determine crime date boundaries.")

    minimum_crime_date = crime_date_bounds["minimum_crime_date"]
    maximum_crime_date = crime_date_bounds["maximum_crime_date"]

    if minimum_crime_date is None or maximum_crime_date is None:
        raise ValueError("Could not determine crime date boundaries.")

    relevant_weather_cells = (
        crime_dataframe.filter(F.col("weather_query_cell_id").isNotNull())
        .select("weather_query_cell_id")
        .distinct()
    )

    lookup = (
        weather_dataframe.filter(
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
            & (F.col("crime.weather_timestamp") == F.col("weather.weather_timestamp")),
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


def build_lighting_lookup(
    lighting_dataframe: DataFrame,
    *,
    definition_version: str = LIGHTING_DEFINITION_VERSION,
) -> DataFrame:
    lookup = (
        lighting_dataframe.filter(
            F.col("lighting_definition_version") == definition_version
        )
        .select(
            "lighting_query_cell_id",
            "solar_timestamp_hour",
            "lighting_definition_version",
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
    *,
    definition_version: str = LIGHTING_DEFINITION_VERSION,
) -> DataFrame:
    return (
        crime_dataframe.alias("crime")
        .join(
            lighting_lookup.alias("light"),
            (
                F.col("crime.lighting_query_cell_id")
                == F.col("light.lighting_query_cell_id")
            )
            & (
                F.col("crime.solar_timestamp_hour")
                == F.col("light.solar_timestamp_hour")
            )
            & (F.col("light.lighting_definition_version") == F.lit(definition_version)),
            "left",
        )
        .select(
            "crime.*",
            F.col("light.solar_elevation_deg"),
            F.col("light.apparent_solar_elevation_deg"),
            F.col("light.solar_zenith_deg"),
            F.col("light.solar_azimuth_deg"),
            F.col("light.lighting_condition"),
            F.col("light.is_daylight"),
            F.col("light.lighting_definition_version").alias(
                "lighting_definition_version"
            ),
            F.coalesce(
                F.col("light._lighting_row_found"),
                F.lit(False),
            ).alias("lighting_match_found"),
        )
    )


def log_coverage_metrics(
    feature_dataframe: DataFrame,
    *,
    lighting_definition_version: str = LIGHTING_DEFINITION_VERSION,
) -> dict[str, object]:
    usable_socioeconomic = _usable_socioeconomic_feature()
    usable_weather = _usable_weather_feature()
    usable_lighting = _usable_lighting_feature(
        definition_version=lighting_definition_version,
    )
    metric_row = feature_dataframe.agg(
        F.count("*").alias("final_rows"),
        F.count("selected_acs_vintage").alias("rows_with_eligible_acs_vintage"),
        F.count("tract_geoid").alias("rows_with_tract"),
        F.sum(usable_socioeconomic.cast("long")).alias(
            "rows_with_socioeconomic_record"
        ),
        F.count("median_household_income").alias("rows_with_non_null_income"),
        F.sum(usable_weather.cast("long")).alias(
            "rows_with_weather_record"
        ),
        F.sum(usable_lighting.cast("long")).alias(
            "rows_with_lighting_record"
        ),
        F.count("temperature_2m_c").alias("rows_with_non_null_temperature"),
        F.round(
            F.avg(F.col("tract_geoid").isNotNull().cast("double")),
            8,
        ).alias("tract_match_rate"),
        F.round(
            F.avg(usable_socioeconomic.cast("double")),
            8,
        ).alias("socioeconomic_match_rate"),
        F.round(
            F.avg(usable_weather.cast("double")),
            8,
        ).alias("weather_match_rate"),
        F.round(
            F.avg(usable_lighting.cast("double")),
            8,
        ).alias("lighting_match_rate"),
    ).first()
    if metric_row is None:
        raise RuntimeError("Gold coverage aggregation returned no result.")
    metrics = metric_row.asDict()

    LOGGER.info(
        "Crime feature coverage metrics: %s",
        metrics,
    )

    return metrics


def source_coverage_metrics(
    feature_dataframe: DataFrame,
    *,
    lighting_definition_version: str = LIGHTING_DEFINITION_VERSION,
) -> dict[str, dict[str, float]]:
    """Measure the four enrichment rates independently for every source."""
    usable_socioeconomic = _usable_socioeconomic_feature()
    usable_weather = _usable_weather_feature()
    usable_lighting = _usable_lighting_feature(
        definition_version=lighting_definition_version,
    )
    rows = (
        feature_dataframe.groupBy("source_system")
        .agg(
            F.round(
                F.avg(F.col("tract_geoid").isNotNull().cast("double")),
                8,
            ).alias("tract_match_rate"),
            F.round(
                F.avg(usable_socioeconomic.cast("double")),
                8,
            ).alias("socioeconomic_match_rate"),
            F.round(
                F.avg(usable_weather.cast("double")),
                8,
            ).alias("weather_match_rate"),
            F.round(
                F.avg(usable_lighting.cast("double")),
                8,
            ).alias("lighting_match_rate"),
        )
        .collect()
    )
    return {
        str(row["source_system"]): {
            metric_name: float(row[metric_name])
            for metric_name in (
                "weather_match_rate",
                "lighting_match_rate",
                "socioeconomic_match_rate",
                "tract_match_rate",
            )
        }
        for row in rows
    }


def validate_gold_candidate(
    source_dataframe: DataFrame,
    candidate_dataframe: DataFrame,
    *,
    thresholds: GoldCoverageThresholds,
    lighting_definition_version: str = (LIGHTING_DEFINITION_VERSION),
) -> dict[str, object]:
    """Validate candidate content before the final Gold table is replaced."""

    required_types = {
        "crime_offense_id": {"string"},
        "source_system": {"string"},
        "occurred_at": {"timestamp", "timestamp_ntz"},
        "weather_timestamp": {"timestamp", "timestamp_ntz"},
        "solar_timestamp_hour": {"timestamp", "timestamp_ntz"},
        "lighting_query_cell_id": {"bigint"},
        "latitude": {"double"},
        "longitude": {"double"},
        "lighting_definition_version": {"string"},
        "lighting_condition": {"string"},
        "is_daylight": {"boolean"},
        "selected_acs_vintage": {"int"},
        "tract_geoid": {"string"},
        "median_household_income": {"double"},
        "temperature_2m_c": {"double"},
        "solar_elevation_deg": {"double"},
        "apparent_solar_elevation_deg": {"double"},
        "solar_zenith_deg": {"double"},
        "solar_azimuth_deg": {"double"},
        "weather_match_found": {"boolean"},
        "lighting_match_found": {"boolean"},
        "socioeconomic_match_found": {"boolean"},
    }
    required_columns = set(required_types)
    actual_types = {
        field.name: field.dataType.simpleString()
        for field in candidate_dataframe.schema.fields
    }
    missing_columns = sorted(required_columns - set(actual_types))
    type_errors = [
        (
            f"{column_name} expected one of "
            f"{sorted(expected_types)}, "
            f"found {actual_types[column_name]}"
        )
        for column_name, expected_types in required_types.items()
        if column_name in actual_types
        and actual_types[column_name] not in expected_types
    ]

    if missing_columns or type_errors:
        raise RuntimeError(
            "Gold candidate schema is incompatible: "
            + "; ".join(
                [
                    *(
                        [f"missing columns: {missing_columns}"]
                        if missing_columns
                        else []
                    ),
                    *type_errors,
                ]
            )
        )

    _raise_on_duplicate_keys(
        source_dataframe,
        keys=("crime_offense_id",),
        dataset_name="Prepared Silver crime source",
    )
    _raise_on_duplicate_keys(
        candidate_dataframe,
        keys=("crime_offense_id",),
        dataset_name="Gold candidate",
    )
    _raise_on_key_set_mismatch(
        source_dataframe,
        candidate_dataframe,
        keys=("crime_offense_id",),
        dataset_name="Gold candidate",
    )

    invalid_condition = (
        F.col("crime_offense_id").isNull()
        | (F.trim(F.col("crime_offense_id")) == "")
        | F.col("source_system").isNull()
        | (F.trim(F.col("source_system")) == "")
        | F.col("occurred_at").isNull()
        | F.col("weather_timestamp").isNull()
        | F.col("solar_timestamp_hour").isNull()
        | (F.col("weather_timestamp") != F.date_trunc("hour", F.col("occurred_at")))
        | (F.col("solar_timestamp_hour") != F.date_trunc("hour", F.col("occurred_at")))
        | (F.col("latitude").isNull() != F.col("longitude").isNull())
        | (
            F.col("latitude").isNotNull()
            & (F.isnan("latitude") | ~F.col("latitude").between(-90.0, 90.0))
        )
        | (
            F.col("longitude").isNotNull()
            & (F.isnan("longitude") | ~F.col("longitude").between(-180.0, 180.0))
        )
        | F.col("weather_match_found").isNull()
        | F.col("lighting_match_found").isNull()
        | F.col("socioeconomic_match_found").isNull()
        | (
            F.col("weather_match_found")
            & ~_usable_weather_feature()
        )
        | (
            F.col("lighting_match_found")
            & ~_usable_lighting_feature(
                definition_version=lighting_definition_version,
            )
        )
        | (
            F.col("socioeconomic_match_found")
            & ~_usable_socioeconomic_feature()
        )
        | (
            F.col("lighting_match_found")
            & (
                F.col("lighting_definition_version").isNull()
                | (F.col("lighting_definition_version") != lighting_definition_version)
            )
        )
        | (
            F.col("lighting_condition").isNotNull()
            & ~F.col("lighting_condition").isin(*VALID_LIGHTING_CONDITIONS)
        )
        | (
            F.col("lighting_match_found")
            & (
                F.col("is_daylight").isNull()
                | (F.col("is_daylight") != (F.col("solar_elevation_deg") >= 0.0))
            )
        )
    )

    for column_name, (minimum, maximum) in PLAUSIBLE_FEATURE_RANGES.items():
        if column_name in candidate_dataframe.columns:
            invalid_condition |= F.col(column_name).isNotNull() & (
                F.isnan(column_name)
                | ~F.col(column_name).between(
                    minimum,
                    maximum,
                )
            )

    invalid_rows = candidate_dataframe.filter(invalid_condition)
    if _has_rows(invalid_rows):
        raise RuntimeError(
            "Gold candidate contains invalid identifiers, "
            "timestamps, coordinates, join metadata, or feature "
            f"ranges; invalid_rows={invalid_rows.count()}."
        )

    metrics = log_coverage_metrics(
        candidate_dataframe,
        lighting_definition_version=lighting_definition_version,
    )
    failures = coverage_failures(metrics, thresholds)
    if failures:
        raise RuntimeError(
            "Gold candidate failed enrichment coverage checks: " + "; ".join(failures)
        )
    per_source_coverage = source_coverage_metrics(
        candidate_dataframe,
        lighting_definition_version=lighting_definition_version,
    )
    per_source_failures = [
        f"{source_system}: {failure}"
        for source_system, source_metrics in sorted(
            per_source_coverage.items()
        )
        for failure in coverage_failures(source_metrics, thresholds)
    ]
    if per_source_failures:
        raise RuntimeError(
            "Gold candidate failed source-level enrichment coverage checks: "
            + "; ".join(per_source_failures)
        )

    source_count = source_dataframe.count()
    final_count_value = metrics.get("final_rows")
    if not isinstance(final_count_value, int):
        raise RuntimeError(
            f"Gold final row count is missing or non-integral: {final_count_value!r}."
        )
    final_count = final_count_value
    if source_count != final_count:
        raise RuntimeError(
            "Gold candidate changed row cardinality despite "
            "business-key validation: "
            f"source={source_count}, candidate={final_count}."
        )

    metrics["source_rows"] = source_count
    metrics["source_coverage"] = per_source_coverage
    return metrics


def build_gold_quality_checks(
    metrics: dict[str, object],
    thresholds: GoldCoverageThresholds,
) -> list[QualityCheck]:
    """Convert successful Gold validation evidence to audit records."""

    metric_thresholds = thresholds.as_metric_thresholds()
    raw_source_coverage = metrics.get("source_coverage")
    source_checks: list[QualityCheck] = []
    if isinstance(raw_source_coverage, Mapping):
        for source_system, source_metrics in sorted(
            raw_source_coverage.items()
        ):
            if not isinstance(source_metrics, Mapping):
                continue
            for metric_name, minimum in metric_thresholds.items():
                source_checks.append(
                    QualityCheck(
                        check_name=f"gold_{metric_name}",
                        severity="BLOCKING",
                        passed=True,
                        observed_value=source_metrics[metric_name],
                        expected_threshold=f">={minimum:.8f}",
                        source_system=str(source_system),
                    )
                )
    return [
        QualityCheck(
            check_name="gold_candidate_validation",
            severity="BLOCKING",
            passed=True,
            observed_value="passed",
            expected_threshold=("all pre-promotion Gold invariants pass"),
        ),
        QualityCheck(
            check_name="gold_exact_business_key_equality",
            severity="BLOCKING",
            passed=True,
            observed_value="equal",
            expected_threshold="exact equality",
        ),
        QualityCheck(
            check_name="gold_unique_crime_offense_id",
            severity="BLOCKING",
            passed=True,
            observed_value=0,
            expected_threshold="0 duplicates",
        ),
        QualityCheck(
            check_name="gold_row_cardinality",
            severity="BLOCKING",
            passed=True,
            observed_value=(
                f"source={metrics['source_rows']}, candidate={metrics['final_rows']}"
            ),
            expected_threshold=("source rows equal candidate rows"),
        ),
        QualityCheck(
            check_name=("gold_schema_ranges_and_join_cardinality"),
            severity="BLOCKING",
            passed=True,
            observed_value="valid",
            expected_threshold="valid",
        ),
        *[
            QualityCheck(
                check_name=f"gold_{metric_name}",
                severity="BLOCKING",
                passed=True,
                observed_value=metrics[metric_name],
                expected_threshold=f">={minimum:.8f}",
            )
            for metric_name, minimum in (metric_thresholds.items())
        ],
        *source_checks,
    ]


def materialize_gold_features(
    spark: SparkSession,
    *,
    source_dataframe: DataFrame,
    candidate_dataframe: DataFrame,
    target_table: str,
    thresholds: GoldCoverageThresholds,
    lighting_definition_version: str = (LIGHTING_DEFINITION_VERSION),
    pipeline_run_id: str | None = None,
    quality_results_table: str | None = None,
) -> dict[str, object]:
    """Stage, validate, and promote a complete Gold rebuild."""

    resolved_pipeline_run_id = normalize_pipeline_run_id(pipeline_run_id)
    staging_table = staging_table_name(
        target_table,
        resolved_pipeline_run_id,
    )

    try:
        (
            candidate_dataframe.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(staging_table)
        )
        staged_candidate = spark.table(staging_table)
        try:
            metrics = validate_gold_candidate(
                source_dataframe,
                staged_candidate,
                thresholds=thresholds,
                lighting_definition_version=(lighting_definition_version),
            )
        except Exception as exc:
            if quality_results_table is not None:
                failed_results = quality_results_dataframe(
                    spark,
                    checks=[
                        QualityCheck(
                            check_name=("gold_candidate_validation"),
                            severity="BLOCKING",
                            passed=False,
                            observed_value=(f"{type(exc).__name__}: {exc}"),
                            expected_threshold=(
                                "all pre-promotion Gold invariants pass"
                            ),
                        )
                    ],
                    pipeline_run_id=(resolved_pipeline_run_id),
                    table_name=target_table,
                )
                merge_quality_results(
                    spark,
                    results=failed_results,
                    target_table=quality_results_table,
                )
            raise

        if quality_results_table is not None:
            passed_results = quality_results_dataframe(
                spark,
                checks=build_gold_quality_checks(
                    metrics,
                    thresholds,
                ),
                pipeline_run_id=resolved_pipeline_run_id,
                table_name=target_table,
            )
            merge_quality_results(
                spark,
                results=passed_results,
                target_table=quality_results_table,
            )

        promote_staged_delta_table(
            spark,
            staging_table=staging_table,
            target_table=target_table,
        )
    finally:
        drop_staging_table(
            spark,
            staging_table,
        )

    return metrics
