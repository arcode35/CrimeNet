from datetime import datetime

import polars as pl


SOCIOECONOMIC_KEY = [
    "acs_vintage",
    "geoid",
]

RATE_COLUMNS = [
    "poverty_rate",
    "unemployment_rate",
    "vacancy_rate",
    "renter_occupied_rate",
    "no_vehicle_rate",
]


def build_tract_socioeconomic_silver(
    bronze_lf: pl.LazyFrame,
    *,
    processed_at: datetime,
) -> pl.LazyFrame:
    """
    Produce the canonical tract-level ACS socioeconomic table.

    Grain:
        (acs_vintage, geoid)

    This is intentionally a thin Silver transformation. The Bronze
    dataset is already substantially normalized, so Silver primarily
    establishes a validated downstream contract.
    """

    return (
        bronze_lf
        .select(
            # ---------------------------------------------------------
            # Identity / geography
            # ---------------------------------------------------------
            pl.col("acs_vintage")
            .cast(pl.Int32),

            pl.col("geoid")
            .cast(pl.String),

            pl.col("state_fips")
            .cast(pl.String),

            pl.col("county_fips")
            .cast(pl.String),

            pl.col("tract_code")
            .cast(pl.String),

            pl.col("geography_name")
            .cast(pl.String),

            pl.col("geography_type")
            .cast(pl.String),

            pl.col("metro")
            .cast(pl.String),

            # ---------------------------------------------------------
            # ACS period
            # ---------------------------------------------------------
            pl.col("period_start_year")
            .cast(pl.Int32),

            pl.col("period_end_year")
            .cast(pl.Int32),

            # ---------------------------------------------------------
            # Population / age / income
            # ---------------------------------------------------------
            pl.col("population")
            .cast(pl.Int64),

            pl.col("population_moe")
            .cast(pl.Int64),

            pl.col("median_age")
            .cast(pl.Float64),

            pl.col("median_age_moe")
            .cast(pl.Float64),

            pl.col("median_household_income")
            .cast(pl.Float64),

            pl.col("median_household_income_moe")
            .cast(pl.Float64),

            # ---------------------------------------------------------
            # Poverty
            # ---------------------------------------------------------
            pl.col("poverty_universe")
            .cast(pl.Int64),

            pl.col("poverty_universe_moe")
            .cast(pl.Int64),

            pl.col("population_below_poverty")
            .cast(pl.Int64),

            pl.col("population_below_poverty_moe")
            .cast(pl.Int64),

            pl.col("poverty_rate")
            .cast(pl.Float64),

            # ---------------------------------------------------------
            # Employment
            # ---------------------------------------------------------
            pl.col("civilian_labor_force")
            .cast(pl.Int64),

            pl.col("civilian_labor_force_moe")
            .cast(pl.Int64),

            pl.col("unemployed_population")
            .cast(pl.Int64),

            pl.col("unemployed_population_moe")
            .cast(pl.Int64),

            pl.col("unemployment_rate")
            .cast(pl.Float64),

            # ---------------------------------------------------------
            # Housing
            # ---------------------------------------------------------
            pl.col("housing_units")
            .cast(pl.Int64),

            pl.col("housing_units_moe")
            .cast(pl.Int64),

            pl.col("occupied_housing_units")
            .cast(pl.Int64),

            pl.col("occupied_housing_units_moe")
            .cast(pl.Int64),

            pl.col("vacant_housing_units")
            .cast(pl.Int64),

            pl.col("vacant_housing_units_moe")
            .cast(pl.Int64),

            pl.col("vacancy_rate")
            .cast(pl.Float64),

            # ---------------------------------------------------------
            # Tenure
            # ---------------------------------------------------------
            pl.col("occupied_units_tenure_universe")
            .cast(pl.Int64),

            pl.col("occupied_units_tenure_universe_moe")
            .cast(pl.Int64),

            pl.col("renter_occupied_units")
            .cast(pl.Int64),

            pl.col("renter_occupied_units_moe")
            .cast(pl.Int64),

            pl.col("renter_occupied_rate")
            .cast(pl.Float64),

            # ---------------------------------------------------------
            # Vehicle access
            # ---------------------------------------------------------
            pl.col("households_vehicle_universe")
            .cast(pl.Int64),

            pl.col("households_vehicle_universe_moe")
            .cast(pl.Int64),

            pl.col("households_no_vehicle")
            .cast(pl.Int64),

            pl.col("households_no_vehicle_moe")
            .cast(pl.Int64),

            pl.col("no_vehicle_rate")
            .cast(pl.Float64),

            # ---------------------------------------------------------
            # Bronze lineage worth preserving
            # ---------------------------------------------------------
            pl.col("source_file")
            .cast(pl.String),

            pl.col("_source_file_uri")
            .cast(pl.String),

            # ---------------------------------------------------------
            # Silver lineage
            # ---------------------------------------------------------
            pl.lit(processed_at)
            .alias("_silver_processed_at_utc"),
        )
    )


def count_duplicate_socioeconomic_keys(
    lf: pl.LazyFrame,
) -> int:
    return (
        lf
        .group_by(SOCIOECONOMIC_KEY)
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.len().alias("duplicate_keys"))
        .collect()
        .item()
    )


def count_invalid_geoids(
    lf: pl.LazyFrame,
) -> int:
    """
    Census tract GEOIDs should be 11 numeric characters and their
    first two characters should agree with state_fips.
    """

    return (
        lf
        .filter(
            pl.col("geoid").is_null()
            | (pl.col("geoid").str.len_chars() != 11)
            | ~pl.col("geoid").str.contains(r"^\d{11}$")
            | (
                pl.col("geoid").str.slice(0, 2)
                != pl.col("state_fips")
            )
        )
        .select(pl.len().alias("invalid_geoids"))
        .collect()
        .item()
    )


def count_invalid_geography_types(
    lf: pl.LazyFrame,
) -> int:
    return (
        lf
        .filter(
            pl.col("geography_type").is_null()
            | (
                pl.col("geography_type")
                .str.to_lowercase()
                != "tract"
            )
        )
        .select(pl.len().alias("invalid_geography_types"))
        .collect()
        .item()
    )


def count_invalid_acs_periods(
    lf: pl.LazyFrame,
) -> int:
    """
    ACS 5-year products should end in the ACS vintage year and span
    five calendar years.
    """

    return (
        lf
        .filter(
            pl.col("period_end_year").is_null()
            | pl.col("period_start_year").is_null()
            | (
                pl.col("period_end_year")
                != pl.col("acs_vintage")
            )
            | (
                pl.col("period_start_year")
                != pl.col("acs_vintage") - 4
            )
        )
        .select(pl.len().alias("invalid_acs_periods"))
        .collect()
        .item()
    )


def count_invalid_rates(
    lf: pl.LazyFrame,
) -> int:
    """
    Null rates are allowed. Non-null rates must be probabilities.
    """

    invalid = pl.lit(False)

    for column in RATE_COLUMNS:
        invalid = invalid | (
            pl.col(column).is_not_null()
            & (
                (pl.col(column) < 0.0)
                | (pl.col(column) > 1.0)
            )
        )

    return (
        lf
        .filter(invalid)
        .select(pl.len().alias("invalid_rates"))
        .collect()
        .item()
    )