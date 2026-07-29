"""Transform raw ACS tract data into socioeconomic features."""

from __future__ import annotations

from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

SOCIOECONOMIC_DEFINITION_VERSION = "acs5_tract_features_v1"

ACS_QUARANTINE_MESSAGES = {
    "RESCUED_SCHEMA_DATA": "Auto Loader rescued unexpected ACS fields.",
    "MISSING_TRACT_KEY": "GEOID and ACS vintage are required.",
    "INVALID_TRACT_GEOID": "The tract GEOID must contain exactly 11 digits.",
    "INVALID_ACS_VINTAGE": "The ACS vintage is outside the supported range.",
    "INVALID_NUMERIC_VALUE": (
        "A non-sentinel ACS numeric value is malformed or outside its domain."
    ),
}

ACS_COLUMN_NAMES = {
    "b01003_001e": "population",
    "b01003_001m": "population_moe",

    "b01002_001e": "median_age",
    "b01002_001m": "median_age_moe",

    "b19013_001e": "median_household_income",
    "b19013_001m": "median_household_income_moe",

    "b17001_001e": "poverty_universe",
    "b17001_001m": "poverty_universe_moe",
    "b17001_002e": "population_below_poverty",
    "b17001_002m": "population_below_poverty_moe",

    "b23025_003e": "civilian_labor_force",
    "b23025_003m": "civilian_labor_force_moe",
    "b23025_005e": "unemployed_population",
    "b23025_005m": "unemployed_population_moe",

    "b25001_001e": "housing_units",
    "b25001_001m": "housing_units_moe",

    "b25002_002e": "occupied_housing_units",
    "b25002_002m": "occupied_housing_units_moe",
    "b25002_003e": "vacant_housing_units",
    "b25002_003m": "vacant_housing_units_moe",

    "b25003_001e": "occupied_units_tenure_universe",
    "b25003_001m": "occupied_units_tenure_universe_moe",
    "b25003_003e": "renter_occupied_units",
    "b25003_003m": "renter_occupied_units_moe",

    "b08201_001e": "households_vehicle_universe",
    "b08201_001m": "households_vehicle_universe_moe",
    "b08201_002e": "households_no_vehicle",
    "b08201_002m": "households_no_vehicle_moe",
}


INTEGER_COLUMNS = (
    "population",
    "population_moe",
    "poverty_universe",
    "poverty_universe_moe",
    "population_below_poverty",
    "population_below_poverty_moe",
    "civilian_labor_force",
    "civilian_labor_force_moe",
    "unemployed_population",
    "unemployed_population_moe",
    "housing_units",
    "housing_units_moe",
    "occupied_housing_units",
    "occupied_housing_units_moe",
    "vacant_housing_units",
    "vacant_housing_units_moe",
    "occupied_units_tenure_universe",
    "occupied_units_tenure_universe_moe",
    "renter_occupied_units",
    "renter_occupied_units_moe",
    "households_vehicle_universe",
    "households_vehicle_universe_moe",
    "households_no_vehicle",
    "households_no_vehicle_moe",
)


DOUBLE_COLUMNS = (
    "median_age",
    "median_age_moe",
    "median_household_income",
    "median_household_income_moe",
)


CENSUS_SENTINEL_VALUES = (
    "-666666666",
    "-888888888",
    "-999999999",
    "-222222222",
    "-333333333",
    "-555555555",
)

NONNEGATIVE_DOUBLE_COLUMNS = (
    "median_age_moe",
    "median_household_income",
    "median_household_income_moe",
)


def annotate_acs_validation(dataframe: DataFrame) -> DataFrame:
    """Attach source-level ACS validation reasons before normalization."""
    invalid_numeric_conditions: list[Column] = []
    integer_sources = {
        source
        for source, target in ACS_COLUMN_NAMES.items()
        if target in INTEGER_COLUMNS
    }
    for source_name, target_name in ACS_COLUMN_NAMES.items():
        source = F.col(source_name)
        cast_type = "long" if source_name in integer_sources else "double"
        cast_value = F.expr(f"try_cast(`{source_name}` AS {cast_type})")
        non_sentinel = ~source.cast("string").isin(*CENSUS_SENTINEL_VALUES)
        invalid = (
            source.isNotNull()
            & non_sentinel
            & (
                cast_value.isNull()
                | (cast_value < 0)
                | (
                    F.lit(target_name == "median_age")
                    & (cast_value > 120)
                )
            )
        )
        invalid_numeric_conditions.append(invalid)

    invalid_numeric = reduce(
        lambda left, right: left | right,
        invalid_numeric_conditions,
        F.lit(False),
    )
    reasons = F.array_compact(
        F.array(
            F.when(
                F.col("rescued_data").isNotNull(),
                F.lit("RESCUED_SCHEMA_DATA"),
            ),
            F.when(
                F.col("geoid").isNull() | F.col("acs_vintage").isNull(),
                F.lit("MISSING_TRACT_KEY"),
            ),
            F.when(
                F.col("geoid").isNotNull()
                & ~F.col("geoid").cast("string").rlike(r"^[0-9]{11}$"),
                F.lit("INVALID_TRACT_GEOID"),
            ),
            F.when(
                F.expr("try_cast(`acs_vintage` AS INT)").isNull()
                | ~F.expr("try_cast(`acs_vintage` AS INT)").between(
                    2009,
                    2100,
                ),
                F.lit("INVALID_ACS_VINTAGE"),
            ),
            F.when(invalid_numeric, F.lit("INVALID_NUMERIC_VALUE")),
        )
    )
    return dataframe.withColumn("_quarantine_reason_codes", reasons)


def enforce_acs_numeric_domains(
    dataframe: DataFrame,
) -> DataFrame:
    result = dataframe

    # All selected ACS count and MOE fields should be nonnegative.
    # This also catches sentinel values that were represented as
    # "-666666666.0" or otherwise missed by the string comparison.
    for column_name in INTEGER_COLUMNS:
        result = result.withColumn(
            column_name,
            F.when(
                F.col(column_name) >= 0,
                F.col(column_name),
            ).otherwise(
                F.lit(None).cast("long")
            ),
        )

    for column_name in NONNEGATIVE_DOUBLE_COLUMNS:
        result = result.withColumn(
            column_name,
            F.when(
                F.col(column_name) >= 0.0,
                F.col(column_name),
            ).otherwise(
                F.lit(None).cast("double")
            ),
        )

    # Median age has a narrower valid domain.
    result = result.withColumn(
        "median_age",
        F.when(
            F.col("median_age").between(
                0.0,
                120.0,
            ),
            F.col("median_age"),
        ).otherwise(
            F.lit(None).cast("double")
        ),
    )

    return result
def safe_rate(
    numerator: str,
    denominator: str,
) -> Column:
    return F.when(
        F.col(denominator) > 0,
        F.col(numerator) / F.col(denominator),
    )


def rename_acs_columns(
    dataframe: DataFrame,
) -> DataFrame:
    renamed_dataframe = dataframe

    for source_name, target_name in ACS_COLUMN_NAMES.items():
        renamed_dataframe = (
            renamed_dataframe.withColumnRenamed(
                source_name,
                target_name,
            )
        )

    return renamed_dataframe


def nullify_census_sentinels(
    dataframe: DataFrame,
) -> DataFrame:
    result = dataframe

    for column_name in ACS_COLUMN_NAMES.values():
        result = result.withColumn(
            column_name,
            F.when(
                F.col(column_name).isin(
                    *CENSUS_SENTINEL_VALUES
                ),
                F.lit(None),
            ).otherwise(
                F.col(column_name)
            ),
        )

    return result


def cast_acs_columns(
    dataframe: DataFrame,
) -> DataFrame:
    result = dataframe

    for column_name in INTEGER_COLUMNS:
        result = result.withColumn(
            column_name,
            F.col(column_name).cast("long"),
        )

    for column_name in DOUBLE_COLUMNS:
        result = result.withColumn(
            column_name,
            F.col(column_name).cast("double"),
        )

    return result


def transform_acs5_tracts(
    bronze_dataframe: DataFrame,
) -> DataFrame:
    valid_bronze = annotate_acs_validation(bronze_dataframe).filter(
        F.size("_quarantine_reason_codes") == 0
    )
    renamed_dataframe = rename_acs_columns(
        valid_bronze
    )

    cleaned_dataframe = nullify_census_sentinels(
        renamed_dataframe
    )

    typed_dataframe = cast_acs_columns(
        cleaned_dataframe
    )

    validated_dataframe = enforce_acs_numeric_domains(
        typed_dataframe
    )

    return (
        validated_dataframe
        .select(
            F.col("geoid"),
            F.col("name").alias("geography_name"),
            F.col("state").alias("state_fips"),
            F.col("county").alias("county_fips"),
            F.col("tract").alias("tract_code"),
            F.col("acs_vintage").cast("int"),
            F.col("period_start_year").cast("int"),
            F.col("period_end_year").cast("int"),
            F.col("geography_type"),
            F.col("population"),
            F.col("population_moe"),
            F.col("median_age"),
            F.col("median_age_moe"),
            F.col("median_household_income"),
            F.col("median_household_income_moe"),
            F.col("poverty_universe"),
            F.col("poverty_universe_moe"),
            F.col("population_below_poverty"),
            F.col("population_below_poverty_moe"),
            F.col("civilian_labor_force"),
            F.col("civilian_labor_force_moe"),
            F.col("unemployed_population"),
            F.col("unemployed_population_moe"),
            F.col("housing_units"),
            F.col("housing_units_moe"),
            F.col("occupied_housing_units"),
            F.col("occupied_housing_units_moe"),
            F.col("vacant_housing_units"),
            F.col("vacant_housing_units_moe"),
            F.col("occupied_units_tenure_universe"),
            F.col("occupied_units_tenure_universe_moe"),
            F.col("renter_occupied_units"),
            F.col("renter_occupied_units_moe"),
            F.col("households_vehicle_universe"),
            F.col("households_vehicle_universe_moe"),
            F.col("households_no_vehicle"),
            F.col("households_no_vehicle_moe"),
            F.col("source_file"),
            F.col("source_row_hash"),
            F.col("source_contract_version"),
            F.col("ingested_at").alias(
                "bronze_ingested_at"
            ),
        )
        .withColumn(
            "poverty_rate",
            safe_rate(
                "population_below_poverty",
                "poverty_universe",
            ),
        )
        .withColumn(
            "unemployment_rate",
            safe_rate(
                "unemployed_population",
                "civilian_labor_force",
            ),
        )
        .withColumn(
            "vacancy_rate",
            safe_rate(
                "vacant_housing_units",
                "housing_units",
            ),
        )
        .withColumn(
            "renter_occupied_rate",
            safe_rate(
                "renter_occupied_units",
                "occupied_units_tenure_universe",
            ),
        )
        .withColumn(
            "no_vehicle_rate",
            safe_rate(
                "households_no_vehicle",
                "households_vehicle_universe",
            ),
        )
        .withColumn(
            "silver_processed_at",
            F.current_timestamp(),
        )
        .withColumn(
            "socioeconomic_definition_version",
            F.lit(SOCIOECONOMIC_DEFINITION_VERSION),
        )
    )
