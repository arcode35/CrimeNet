from __future__ import annotations

import duckdb
import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    date_time_expr,
    datetime_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.legacy.base import project_adapter_fields


def _shared_fields(lf: pl.LazyFrame) -> dict[str, pl.Expr]:
    return {
        "source_file_uri": text_expr(
            lf,
            "_source_file_uri",
            "source_file",
            "source_url",
        ),
        "ingestion_run_id": text_expr(lf, "_ingestion_run_id"),
        "ingested_at_utc": pl.col("_ingested_at_utc"),
    }


def convert_dallas_coordinates(
    lf: pl.LazyFrame,
    connection: duckdb.DuckDBPyConnection,
) -> pl.LazyFrame:
    return connection.sql(
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


def adapt_dallas(
    lf: pl.LazyFrame,
    connection: duckdb.DuckDBPyConnection | None,
) -> pl.LazyFrame:
    if connection is None:
        raise ValueError("Dallas normalization requires a spatial DuckDB connection")
    lf = convert_dallas_coordinates(lf, connection)
    occurred = pl.coalesce(
        [
            datetime_expr(lf, "occurrence_timestamp"),
            date_time_expr(
                lf,
                ("date1_of_occurrence",),
                ("time1_of_occurrence",),
            ),
        ]
    )
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "service_number_id"),
            "report_timestamp": datetime_expr(lf, "date_of_report"),
            "source_offense_code": text_expr(lf, "nibrs_code", "ucr_code", "rms_code"),
            "source_offense_category": text_expr(lf, "nibrs_crime"),
            "source_offense_description": text_expr(lf, "type_of_incident"),
            "latitude": numeric_expr(lf, "latitude"),
            "longitude": numeric_expr(lf, "longitude"),
            "location_label": text_expr(lf, "incident_address", "location1"),
            "location_type": text_expr(lf, "type_location"),
            "police_district": text_expr(lf, "division"),
            "local_area": text_expr(lf, "community"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_new_york(lf: pl.LazyFrame) -> pl.LazyFrame:
    occurred = date_time_expr(lf, ("cmplnt_fr_dt",), ("cmplnt_fr_tm",))
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "cmplnt_num"),
            "report_timestamp": datetime_expr(lf, "rpt_dt"),
            "source_offense_code": text_expr(lf, "pd_cd", "ky_cd"),
            "source_offense_category": text_expr(lf, "ofns_desc"),
            "source_offense_description": text_expr(lf, "pd_desc"),
            "latitude": numeric_expr(lf, "latitude"),
            "longitude": numeric_expr(lf, "longitude"),
            "location_label": text_expr(
                lf, "station_name", "parks_nm", "hadevelopt", "boro_nm"
            ),
            "location_type": text_expr(lf, "prem_typ_desc"),
            "police_district": text_expr(lf, "addr_pct_cd"),
            "local_area": text_expr(lf, "boro_nm"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_chicago(lf: pl.LazyFrame) -> pl.LazyFrame:
    occurred = datetime_expr(lf, "date")
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "id"),
            "report_timestamp": pl.lit(None, dtype=pl.Datetime("us")),
            "source_offense_code": text_expr(lf, "iucr"),
            "source_offense_category": text_expr(lf, "primary_type"),
            "source_offense_description": text_expr(lf, "description"),
            "latitude": numeric_expr(lf, "latitude"),
            "longitude": numeric_expr(lf, "longitude"),
            "location_label": text_expr(lf, "block"),
            "location_type": text_expr(lf, "location_description"),
            "police_district": text_expr(lf, "district"),
            "local_area": text_expr(lf, "community_area"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_baltimore(lf: pl.LazyFrame) -> pl.LazyFrame:
    occurred = datetime_expr(lf, "CrimeDateTime", "occurred_at_raw")
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "RowID"),
            "report_timestamp": pl.lit(None, dtype=pl.Datetime("us")),
            "source_offense_code": text_expr(lf, "CrimeCode"),
            "source_offense_category": text_expr(lf, "Description"),
            "source_offense_description": text_expr(lf, "Description"),
            "latitude": numeric_expr(lf, "latitude", "Latitude"),
            "longitude": numeric_expr(lf, "longitude", "Longitude"),
            "location_label": text_expr(lf, "Location"),
            "location_type": text_expr(lf, "PremiseType"),
            "police_district": text_expr(lf, "New_District"),
            "local_area": text_expr(lf, "Neighborhood"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_seattle(lf: pl.LazyFrame) -> pl.LazyFrame:
    occurred = datetime_expr(lf, "offense_date", "occurred_at_raw")
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "offense_id"),
            "report_timestamp": datetime_expr(lf, "report_date_time"),
            "source_offense_code": text_expr(lf, "nibrs_offense_code"),
            "source_offense_category": text_expr(lf, "offense_category"),
            "source_offense_description": text_expr(
                lf, "nibrs_offense_code_description"
            ),
            "latitude": numeric_expr(lf, "latitude"),
            "longitude": numeric_expr(lf, "longitude"),
            "location_label": text_expr(lf, "block_address"),
            "location_type": pl.lit(None, dtype=pl.String),
            "police_district": text_expr(lf, "precinct"),
            "local_area": text_expr(lf, "neighborhood"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_san_francisco(lf: pl.LazyFrame) -> pl.LazyFrame:
    occurred = datetime_expr(lf, "incident_datetime", "occurred_at_raw")
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "row_id"),
            "report_timestamp": datetime_expr(lf, "report_datetime"),
            "source_offense_code": text_expr(lf, "incident_code"),
            "source_offense_category": text_expr(lf, "incident_category"),
            "source_offense_description": text_expr(lf, "incident_description"),
            "latitude": numeric_expr(lf, "latitude"),
            "longitude": numeric_expr(lf, "longitude"),
            "location_label": text_expr(lf, "intersection"),
            "location_type": pl.lit(None, dtype=pl.String),
            "police_district": text_expr(lf, "police_district"),
            "local_area": text_expr(lf, "analysis_neighborhood"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_washington_dc(lf: pl.LazyFrame) -> pl.LazyFrame:
    occurred = datetime_expr(lf, "START_DATE", "occurred_at_raw")
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "OBJECTID"),
            "report_timestamp": datetime_expr(lf, "REPORT_DAT"),
            "source_offense_code": pl.lit(None, dtype=pl.String),
            "source_offense_category": text_expr(lf, "OFFENSE"),
            "source_offense_description": text_expr(lf, "OFFENSE"),
            "latitude": numeric_expr(lf, "latitude", "LATITUDE"),
            "longitude": numeric_expr(lf, "longitude", "LONGITUDE"),
            "location_label": text_expr(lf, "BLOCK"),
            "location_type": pl.lit(None, dtype=pl.String),
            "police_district": text_expr(lf, "DISTRICT"),
            "local_area": text_expr(lf, "NEIGHBORHOOD_CLUSTER"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_fort_worth(lf: pl.LazyFrame) -> pl.LazyFrame:
    occurred = datetime_expr(lf, "occurrence_timestamp", "from_date")
    return project_adapter_fields(
        lf,
        {
            "source_record_id": text_expr(lf, "case_no_offense"),
            "report_timestamp": datetime_expr(lf, "reported_date"),
            "source_offense_code": text_expr(lf, "offense"),
            "source_offense_category": text_expr(lf, "offense"),
            "source_offense_description": text_expr(lf, "offense_desc"),
            "latitude": numeric_expr(lf, "latitude"),
            "longitude": numeric_expr(lf, "longitude"),
            "location_label": text_expr(lf, "block_address", "address", "location_1"),
            "location_type": text_expr(lf, "locationtypedescription"),
            "police_district": text_expr(lf, "division"),
            "local_area": pl.lit(None, dtype=pl.String),
            **_shared_fields(lf),
        },
        occurred,
    )


LEGACY_ADAPTERS = {
    "dallas": adapt_dallas,
    "new_york": adapt_new_york,
    "chicago": adapt_chicago,
    "baltimore": adapt_baltimore,
    "seattle": adapt_seattle,
    "san_francisco": adapt_san_francisco,
    "washington_dc": adapt_washington_dc,
    "fort_worth": adapt_fort_worth,
}
