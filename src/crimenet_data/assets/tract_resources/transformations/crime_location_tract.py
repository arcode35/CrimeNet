import duckdb
import polars as pl


def build_tract_lookup_workload(
    crime_lf: pl.LazyFrame,
    calendar_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Resolve each crime timestamp to the latest available ACS/TIGER
    configuration, then reduce to unique spatial lookup keys.

    Output grain:
        tiger_line_year
        tract_definition_vintage
        latitude
        longitude
    """

    calendar_lf = (
        calendar_lf
        .with_columns(
            pl.col("acs_release_date")
            .cast(pl.Datetime("us"))
            .alias("acs_release_timestamp")
        )
        .sort("acs_release_timestamp")
    )

    crime_lf = (
        crime_lf
        .filter(
            pl.col("occurrence_timestamp").is_not_null()
            & pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
            & pl.col("latitude").is_finite()
            & pl.col("longitude").is_finite()
        )
        .sort("occurrence_timestamp")
    )

    return (
        crime_lf
        .join_asof(
            calendar_lf,
            left_on="occurrence_timestamp",
            right_on="acs_release_timestamp",
            strategy="backward",
        )
        .select(
            "tiger_line_year",
            "tract_definition_vintage",
            "latitude",
            "longitude",
        )
        .filter(
            pl.col("tiger_line_year").is_not_null()
            & pl.col("tract_definition_vintage").is_not_null()
        )
        .unique()
    )

import duckdb
import polars as pl


def build_tract_lookup_workload(
    crime_lf: pl.LazyFrame,
    calendar_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Resolve each crime timestamp to the latest available ACS/TIGER
    configuration, then reduce to unique spatial lookup keys.

    Output grain:
        tiger_line_year
        latitude
        longitude
    """

    calendar_lf = (
        calendar_lf
        .with_columns(
            pl.col("acs_release_date")
            .cast(pl.Datetime("us"))
            .alias("acs_release_timestamp")
        )
        .sort("acs_release_timestamp")
    )

    crime_lf = (
        crime_lf
        .filter(
            pl.col("occurrence_timestamp").is_not_null()
            & pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
            & pl.col("latitude").is_finite()
            & pl.col("longitude").is_finite()
        )
        .sort("occurrence_timestamp")
    )

    return (
        crime_lf
        .join_asof(
            calendar_lf,
            left_on="occurrence_timestamp",
            right_on="acs_release_timestamp",
            strategy="backward",
        )
        .select(
            "tiger_line_year",
            "latitude",
            "longitude",
        )
        .filter(
            pl.col("tiger_line_year").is_not_null()
        )
        .unique()
    )


def resolve_tract_mappings(
    con: duckdb.DuckDBPyConnection,
    lookup_lf: pl.LazyFrame,
    boundaries_lf: pl.LazyFrame,
) -> tuple[pl.LazyFrame, int]:
    """
    Resolve unique crime coordinates to Census tracts using the
    annual TIGER/Line boundary release selected for each lookup.

    Boundary relationship:
        lookup.tiger_line_year
            ==
        boundaries.boundary_vintage

    Returns:
        mapping_lf
        ambiguous_key_count
    """

    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lookup_points AS
        SELECT
            tiger_line_year,
            latitude,
            longitude,
            ST_Point(
                longitude,
                latitude
            ) AS point_geometry
        FROM lookup_lf
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE tract_boundaries AS
        SELECT
            geoid,
            boundary_vintage,
            ST_GeomFromWKB(
                tract_geometry_wkb
            ) AS tract_geometry
        FROM boundaries_lf
        """
    )

    tiger_years = [
        row[0]
        for row in con.execute(
            """
            SELECT DISTINCT
                tiger_line_year
            FROM lookup_points
            ORDER BY tiger_line_year
            """
        ).fetchall()
    ]

    if not tiger_years:
        raise ValueError(
            "No TIGER/Line years were found "
            "in the crime lookup workload."
        )

    spatial_queries = [
        f"""
        SELECT
            p.tiger_line_year,
            p.latitude,
            p.longitude,
            b.geoid AS tract_geoid
        FROM (
            SELECT *
            FROM lookup_points
            WHERE tiger_line_year = {int(year)}
        ) AS p
        JOIN (
            SELECT *
            FROM tract_boundaries
            WHERE boundary_vintage = {int(year)}
        ) AS b
            ON ST_ContainsProperly(
                b.tract_geometry,
                p.point_geometry
            )
        """
        for year in tiger_years
    ]

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE resolved_tract_mapping_raw AS
        {" UNION ALL ".join(spatial_queries)}
        """
    )

    ambiguous_keys = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                tiger_line_year,
                latitude,
                longitude
            FROM resolved_tract_mapping_raw
            GROUP BY
                tiger_line_year,
                latitude,
                longitude
            HAVING COUNT(DISTINCT tract_geoid) > 1
        )
        """
    ).fetchone()[0]

    if ambiguous_keys:
        raise ValueError(
            "Some lookup coordinates resolved to multiple Census "
            f"tracts. Ambiguous keys: {ambiguous_keys}"
        )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE resolved_tract_mapping AS
        SELECT DISTINCT
            tiger_line_year,
            latitude,
            longitude,
            tract_geoid
        FROM resolved_tract_mapping_raw
        """
    )

    mapping_lf = con.sql(
        """
        SELECT
            CAST(tiger_line_year AS INTEGER)
                AS tiger_line_year,
            latitude,
            longitude,
            CAST(tract_geoid AS VARCHAR)
                AS tract_geoid
        FROM resolved_tract_mapping
        """
    ).pl(lazy=True)

    return mapping_lf, ambiguous_keys