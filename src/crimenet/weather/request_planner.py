"""request_planner.py"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.spatial.h3 import (
    add_weather_query_cell,
    extract_h3_centers,
)
from functools import reduce
from operator import or_

from pyspark.sql import Column


CITY_BOUNDS = {
    "dallas": {
        "minimum_latitude": 32.40,
        "maximum_latitude": 33.10,
        "minimum_longitude": -97.10,
        "maximum_longitude": -96.40,
    },
    "fort_worth": {
        "minimum_latitude": 32.50,
        "maximum_latitude": 33.05,
        "minimum_longitude": -97.65,
        "maximum_longitude": -96.95,
    },
    "new_york": {
        "minimum_latitude": 40.40,
        "maximum_latitude": 41.00,
        "minimum_longitude": -74.30,
        "maximum_longitude": -73.60,
    },
    "chicago": {
        "minimum_latitude": 41.60,
        "maximum_latitude": 42.10,
        "minimum_longitude": -88.00,
        "maximum_longitude": -87.45,
    },
    "san_francisco": {
        "minimum_latitude": 37.60,
        "maximum_latitude": 37.95,
        "minimum_longitude": -122.60,
        "maximum_longitude": -122.25,
    },
    "seattle": {
        "minimum_latitude": 47.40,
        "maximum_latitude": 47.85,
        "minimum_longitude": -122.50,
        "maximum_longitude": -122.15,
    },
    "baltimore": {
        "minimum_latitude": 39.15,
        "maximum_latitude": 39.45,
        "minimum_longitude": -76.80,
        "maximum_longitude": -76.45,
    },
    "washington_dc": {
        "minimum_latitude": 38.75,
        "maximum_latitude": 39.05,
        "minimum_longitude": -77.20,
        "maximum_longitude": -76.85,
    },
}

CITY_TIMEZONES = {
    "dallas": "America/Chicago",
    "fort_worth": "America/Chicago",
    "new_york": "America/New_York",
    "chicago": "America/Chicago",
    "san_francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "baltimore": "America/New_York",
    "washington_dc": "America/New_York",
}

PROVIDER = "open_meteo"
TIMEZONE = "GMT"
CELL_SELECTION = "nearest"

# ERA5 and ERA5-Land normally trail real time by
# approximately five days. Seven days gives a small
# availability buffer.
DEFAULT_AVAILABILITY_LAG_DAYS = 7

VALID_MODELS = frozenset(
    {
        "era5",
        "era5_land",
    }
)
MODEL_START_DATES = {
    "era5": date(1940, 1, 1),
    "era5_land": date(1950, 1, 1),
}



def _city_bounds_filter(
    cities: Sequence[str],
) -> Column:
    filters = []

    for city in cities:
        bounds = CITY_BOUNDS[city]

        filters.append(
            (F.col("source_city") == city)
            & F.col("latitude").between(
                bounds["minimum_latitude"],
                bounds["maximum_latitude"],
            )
            & F.col("longitude").between(
                bounds["minimum_longitude"],
                bounds["maximum_longitude"],
            )
        )

    return reduce(or_, filters)


def _city_local_year(
    cities: Sequence[str],
) -> Column:
    cases = [
        F.when(
            F.col("source_city") == city,
            F.year(
                F.from_utc_timestamp(
                    F.col("occurred_at"),
                    CITY_TIMEZONES[city],
                )
            ),
        )
        for city in cities
    ]

    return F.coalesce(*cases).cast("int")

def build_weather_request_manifest(
    crime_df: DataFrame,
    *,
    cities: Sequence[str],
    start_year: int,
    end_year: int,
    model: str,
    hourly_variables: Sequence[str],
    h3_resolution: int = 6,
    availability_cutoff: date | None = None,
) -> DataFrame:
    """Build one deterministic Open-Meteo request per H3 cell and year.

    The input DataFrame must contain:
        - source_city
        - latitude
        - longitude
        - occurred_at

    `occurred_at` is assumed to be stored in UTC. The weather year is
    calculated after converting each timestamp into the source city's
    local timezone.
    """
    normalized_model = model.strip().lower()

    if normalized_model not in VALID_MODELS:
        supported = ", ".join(
            sorted(VALID_MODELS)
        )
        raise ValueError(
            f"Unsupported model {model!r}. "
            f"Supported models: {supported}"
        )

    model_start_date = MODEL_START_DATES[
        normalized_model
    ]

    if start_year > end_year:
        raise ValueError(
            "start_year cannot exceed end_year"
        )

    if date(start_year, 1, 1) < model_start_date:
        raise ValueError(
            f"{normalized_model} begins on "
            f"{model_start_date.isoformat()}; "
            f"start_year={start_year} is too early"
        )

    if not 0 <= h3_resolution <= 15:
        raise ValueError(
            "h3_resolution must be between 0 and 15"
        )

    normalized_cities = tuple(
        dict.fromkeys(
            city.strip().lower()
            for city in cities
            if city.strip()
        )
    )

    if not normalized_cities:
        raise ValueError(
            "At least one city is required"
        )

    unsupported_cities = (
        set(normalized_cities)
        - set(CITY_BOUNDS)
    )

    if unsupported_cities:
        raise ValueError(
            "Unsupported cities: "
            + ", ".join(
                sorted(unsupported_cities)
            )
        )

    normalized_variables = tuple(
        sorted(
            {
                variable.strip()
                for variable in hourly_variables
                if variable.strip()
            }
        )
    )

    if not normalized_variables:
        raise ValueError(
            "At least one hourly variable is required"
        )

    effective_cutoff = (
        availability_cutoff
        if availability_cutoff is not None
        else date.today()
        - timedelta(
            days=DEFAULT_AVAILABILITY_LAG_DAYS
        )
    )

    required_columns = {
        "source_city",
        "latitude",
        "longitude",
        "occurred_at",
    }

    missing_columns = (
        required_columns - set(crime_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    valid_crimes = (
        crime_df
        .select(
            F.lower(
                F.trim(
                    F.col("source_city")
                    .cast("string")
                )
            ).alias("source_city"),

            F.col("latitude")
            .cast("double")
            .alias("latitude"),

            F.col("longitude")
            .cast("double")
            .alias("longitude"),

            F.col("occurred_at")
            .cast("timestamp")
            .alias("occurred_at"),
        )
        .filter(
            F.col("source_city").isin(
                *normalized_cities
            )
        )
        .filter(
            F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
            & F.col("occurred_at").isNotNull()
            & ~F.isnan("latitude")
            & ~F.isnan("longitude")
        )
        .filter(
            _city_bounds_filter(
                normalized_cities
            )
        )
        .withColumn(
            "weather_year",
            _city_local_year(
                normalized_cities
            ),
        )
        .filter(
            F.col("weather_year").between(
                start_year,
                end_year,
            )
        )
    )

    crimes_with_cells = (
        add_weather_query_cell(
            valid_crimes,
            resolution=h3_resolution,
        )
    )

    cell_years = (
        crimes_with_cells
        .groupBy(
            "weather_query_cell_id",
            "weather_year",
        )
        .agg(
            F.count(F.lit(1)).alias(
                "crime_record_count"
            ),

            F.sort_array(
                F.collect_set(
                    "source_city"
                )
            ).alias(
                "source_cities"
            ),

            F.countDistinct(
                F.struct(
                    F.col("latitude"),
                    F.col("longitude"),
                )
            ).alias(
                "unique_coordinate_pairs"
            ),
        )
    )

    unique_cells = (
        cell_years
        .select(
            "weather_query_cell_id"
        )
        .distinct()
    )

    cell_centers = extract_h3_centers(
        unique_cells
    )

    manifest = (
        cell_years
        .join(
            cell_centers,
            on="weather_query_cell_id",
            how="inner",
        )
        .withColumn(
            "provider",
            F.lit(PROVIDER),
        )
        .withColumn(
            "model",
            F.lit(normalized_model),
        )
        .withColumn(
            "h3_resolution",
            F.lit(h3_resolution)
            .cast("byte"),
        )
        .withColumn(
            "start_date",
            F.make_date(
                F.col("weather_year"),
                F.lit(1),
                F.lit(1),
            ),
        )
        .withColumn(
            "end_date",
            F.least(
                F.make_date(
                    F.col("weather_year"),
                    F.lit(12),
                    F.lit(31),
                ),
                F.lit(effective_cutoff),
            ),
        )
        .filter(
            F.col("start_date")
            <= F.col("end_date")
        )
        .withColumn(
            "timezone",
            F.lit(TIMEZONE),
        )
        .withColumn(
            "cell_selection",
            F.lit(CELL_SELECTION),
        )
        .withColumn(
            "hourly_variables",
            F.array(
                *[
                    F.lit(variable)
                    for variable
                    in normalized_variables
                ]
            ),
        )
        .withColumn(
            "request_id",
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("provider"),
                    F.col("model"),
                    F.col(
                        "weather_query_cell_id"
                    ).cast("string"),
                    F.col(
                        "start_date"
                    ).cast("string"),
                    F.col(
                        "end_date"
                    ).cast("string"),
                    F.concat_ws(
                        ",",
                        F.col(
                            "hourly_variables"
                        ),
                    ),
                    F.col("timezone"),
                    F.col(
                        "cell_selection"
                    ),
                    F.col(
                        "h3_resolution"
                    ).cast("string"),
                ),
                256,
            ),
        )
        .select(
            "request_id",
            "provider",
            "model",
            "weather_query_cell_id",
            "h3_resolution",
            "weather_year",
            "query_latitude",
            "query_longitude",
            "start_date",
            "end_date",
            "timezone",
            "cell_selection",
            "hourly_variables",
            "crime_record_count",
            "source_cities",
            "unique_coordinate_pairs",
        )
        .orderBy(
            "start_date",
            "weather_query_cell_id",
        )
    )

    return manifest