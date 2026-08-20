import polars as pl
from crimenet_data.resources import CrimeLakeResources, DuckDBResource
from crimenet_data.resources.canonical import (
    CITY_COORDINATE_BOUNDS, 
    CITY_OFFENSE_MAPPING,
    CITY_RECORD_KEYS, 
    CANONICAL_COLUMNS, 
    CITY_CANONICAL_MAPPING, 
    COMMON_CANONICAL_MAPPING,
    CITY_TIMEZONES,
    CANONICAL_CRIME_SCHEMA,
)
import duckdb

resources = CrimeLakeResources()
def normalize_dc_timestamps(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.with_columns(
            pl.from_epoch(
                pl.col("START_DATE").cast(pl.Int64, strict=False),
                time_unit="ms",
            ).alias("occurrence_timestamp"),
        )
        .with_columns(
            pl.col("occurrence_timestamp")
            .dt.year()
            .alias("occurrence_year"),
        )
    )

def project_canonical_crime_schema(
    lf: pl.LazyFrame,
    city: str,
) -> pl.LazyFrame:

    city_mapping = CITY_CANONICAL_MAPPING[city]

    mapping = {
        **COMMON_CANONICAL_MAPPING,
        **city_mapping,
    }

    source_record_id = city_mapping[
        "source_record_id"
    ].cast(pl.String, strict=False)

    mapping.update(
        {
            "crime_id": pl.concat_str(
                [
                    pl.lit(city),
                    source_record_id,
                ],
                separator=":",
                ignore_nulls=False,
            ),
            "source_city": pl.lit(city),
            "source_timezone": pl.lit(
                CITY_TIMEZONES[city]
            ),
        }
    )

    expressions: list[pl.Expr] = []

    for column_name, dtype in (
        CANONICAL_CRIME_SCHEMA.items()
    ):
        expression = mapping.get(
            column_name,
            pl.lit(None),
        )

        expressions.append(
            expression.cast(
                dtype,
                strict=False,
            ).alias(column_name)
        )

    return lf.select(expressions)



def add_canonical_crime(
    lf: pl.LazyFrame,
    city: str,
) -> pl.LazyFrame:
    if city not in CITY_OFFENSE_MAPPING:
        raise KeyError(f"Unsupported city: {city!r}")

    column_mapping = CITY_OFFENSE_MAPPING[city]
    join_keys = list(column_mapping.values())

    selected_columns = list(
        dict.fromkeys(join_keys + CANONICAL_COLUMNS)
    )

    crosswalk_lf = (
        resources.resolve_crosswalk()
        .filter(pl.col("source_city") == city)
        .rename(column_mapping)
        .select(selected_columns)
    )

    mapped_lf = lf.join(
        crosswalk_lf,
        on=join_keys,
        how="left",
        validate="m:1",
    )

    return mapped_lf

def convert_dallas_coordinates(
    con: duckdb.DuckDBPyConnection,
    lf: pl.LazyFrame,
) -> pl.LazyFrame:
    return con.sql(
        """
        WITH transformed AS (
            SELECT
                *,
                ST_Transform(
                    ST_Point(
                        TRY_CAST(x_coordinate AS DOUBLE),
                        TRY_CAST(y_cordinate AS DOUBLE)
                    ),
                    'EPSG:2276',
                    'EPSG:4326',
                    always_xy := true
                ) AS wgs84_point
            FROM lf
        )

        SELECT
            * EXCLUDE (wgs84_point),
            ST_Y(wgs84_point) AS latitude,
            ST_X(wgs84_point) AS longitude
        FROM transformed
        """
    ).pl(lazy=True)

def cleanse_data(lf: pl.LazyFrame, city: str) -> pl.LazyFrame:
    rename_mapping = {
        "Latitude": "latitude",
        "LATITUDE": "latitude",
        "Longitude": "longitude",
        "LONGITUDE": "longitude",
    }

    lf = lf.rename(rename_mapping, strict=False)

    required_columns = {"latitude", "longitude"}

    columns = set(lf.collect_schema().names())

    missing_columns = required_columns - columns 

    if missing_columns:
        raise KeyError(
            f"Latitude or longitude is missing from the schema: Missing: {sorted(missing_columns)}"
        )

    bounds = CITY_COORDINATE_BOUNDS[city]

    lf = lf.cast({
        "latitude": pl.Float64,
        "longitude": pl.Float64,
    }, strict=False)

    filtering_mask = (
       pl.col("latitude").is_finite() & 
       pl.col("longitude").is_finite() &
       pl.col("latitude").is_between(bounds["min_latitude"], bounds["max_latitude"]) & 
       pl.col("longitude").is_between(bounds["min_longitude"], bounds["max_longitude"]) & 
       pl.col("occurrence_year").is_between(2014, 2026)
    ).fill_null(False)

    cleaned_lf = lf.filter(filtering_mask)
    
    return cleaned_lf

def deduplicate_city(lf: pl.LazyFrame, city: str) -> pl.LazyFrame:
    # most cities don't have duplicates. only dallas and new york had a few.
    return lf.unique(subset=CITY_RECORD_KEYS[city], keep="last")



