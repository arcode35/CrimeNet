"""Transform raw ACS tract data into socioeconomic features."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SOCIOECONOMIC_KEYS = (
    "geoid",
    "acs_vintage",
)

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
            F.col(column_name).try_cast("long"),
        )

    for column_name in DOUBLE_COLUMNS:
        result = result.withColumn(
            column_name,
            F.col(column_name).try_cast("double"),
        )

    return result


def transform_acs5_tracts(
    bronze_dataframe: DataFrame,
) -> DataFrame:
    renamed_dataframe = rename_acs_columns(
        bronze_dataframe
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
            F.col("acs_vintage").try_cast("int"),
            F.col("period_start_year").try_cast("int"),
            F.col("period_end_year").try_cast("int"),
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
        .filter(
            F.col("geoid").isNotNull()
            & F.col("acs_vintage").isNotNull()
        )
    )


def deduplicate_socioeconomic_records(
    dataframe: DataFrame,
) -> DataFrame:
    """Select one deterministic row for each tract/vintage key."""
    missing_columns = [
        column_name
        for column_name in SOCIOECONOMIC_KEYS
        if column_name not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Socioeconomic deduplication is missing key columns: "
            f"{missing_columns}"
        )

    stable_columns = sorted(
        column_name
        for column_name in dataframe.columns
        if column_name
        not in {
            "bronze_ingested_at",
            "silver_processed_at",
        }
    )

    fingerprint = F.sha2(
        F.to_json(
            F.struct(
                *[
                    F.col(column_name)
                    for column_name in stable_columns
                ]
            ),
            options={"ignoreNullFields": "false"},
        ),
        256,
    )

    latest_record_window = (
        Window
        .partitionBy(
            *SOCIOECONOMIC_KEYS
        )
        .orderBy(
            F.col("bronze_ingested_at")
            .desc_nulls_last(),
            fingerprint.desc(),
            F.col("source_file")
            .desc_nulls_last(),
        )
    )

    return (
        dataframe
        .withColumn(
            "_socioeconomic_deduplication_rank",
            F.row_number().over(
                latest_record_window
            ),
        )
        .filter(
            F.col(
                "_socioeconomic_deduplication_rank"
            )
            == 1
        )
        .drop(
            "_socioeconomic_deduplication_rank"
        )
    )
