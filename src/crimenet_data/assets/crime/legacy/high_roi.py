from __future__ import annotations

import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    composite_identifier,
    date_time_expr,
    datetime_expr,
    numeric_expr,
    snake_case_columns,
    text_expr,
)
from crimenet_data.assets.crime.legacy.base import project_adapter_fields


def _prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
    return snake_case_columns(lf)


def _shared_fields(lf: pl.LazyFrame) -> dict[str, pl.Expr]:
    return {
        "source_file_uri": text_expr(
            lf,
            "source_file_uri",
            "source_file",
            "source_url",
        ),
        "ingestion_run_id": text_expr(lf, "ingestion_run_id"),
        "ingested_at_utc": pl.col("ingested_at_utc"),
    }


def _philadelphia_occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return pl.coalesce(
        [
            datetime_expr(
                lf,
                "dispatch_date_time",
                "dispatch_datetime",
                "occurred_at",
            ),
            date_time_expr(lf, ("dispatch_date",), ("dispatch_time",)),
        ]
    )


def _atlanta_occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return pl.coalesce(
        [
            datetime_expr(lf, "occur_date_time", "occurrence_datetime", "occurred_at"),
            date_time_expr(
                lf,
                ("occur_date", "occurrence_date"),
                ("occur_time", "occurrence_time"),
            ),
        ]
    )


def _los_angeles_occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return pl.coalesce(
        [
            date_time_expr(
                lf,
                ("date_occ", "occurrence_date"),
                ("time_occ", "occurrence_time"),
            ),
            datetime_expr(lf, "occurred_on", "occurrence_datetime", "date_occ"),
        ]
    )


def _austin_occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return pl.coalesce(
        [
            datetime_expr(lf, "occurred_date_time", "occurred_datetime"),
            date_time_expr(lf, ("occurred_date",), ("occurred_time",)),
        ]
    )


def _boston_occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(
        lf,
        "occurred_on_date",
        "occurred_at",
        "occurrence_datetime",
    )


HIGH_ROI_OCCURRENCE_EXPRESSIONS = {
    "philadelphia": _philadelphia_occurrence,
    "atlanta": _atlanta_occurrence,
    "los_angeles": _los_angeles_occurrence,
    "austin": _austin_occurrence,
    "boston": _boston_occurrence,
}


def prepare_high_roi_bronze(lf: pl.LazyFrame, city: str) -> pl.LazyFrame:
    """Stabilize CSV headings and derive the technical bronze year partition."""

    try:
        occurrence_expression = HIGH_ROI_OCCURRENCE_EXPRESSIONS[city]
    except KeyError as error:
        raise KeyError(f"No CSV bronze parser registered for {city!r}") from error

    lf = _prepare(lf)
    occurred = occurrence_expression(lf)
    source_year = numeric_expr(lf, "year", "occurrence_year", dtype=pl.Int16)
    year = (
        pl.coalesce([source_year, occurred.dt.year()])
        if city == "boston"
        else pl.coalesce([occurred.dt.year(), source_year])
    )
    return lf.with_columns(year.cast(pl.Int16, strict=False).alias("occurrence_year"))


def adapt_philadelphia(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = _prepare(lf)
    occurred = _philadelphia_occurrence(lf)
    return project_adapter_fields(
        lf,
        {
            "source_record_id": composite_identifier(
                lf, "objectid", "dc_key", "ucr_general"
            ),
            "report_timestamp": datetime_expr(
                lf, "dispatch_date_time", "dispatch_datetime"
            ),
            "source_offense_code": text_expr(lf, "ucr_general", "ucr_code"),
            "source_offense_category": text_expr(lf, "ucr_general", "crime_type"),
            "source_offense_description": text_expr(
                lf, "text_general_code", "ucr_general_desc"
            ),
            "latitude": numeric_expr(lf, "latitude", "lat", "point_y"),
            "longitude": numeric_expr(lf, "longitude", "lng", "lon", "point_x"),
            "location_label": text_expr(lf, "location_block", "location", "block"),
            "location_type": text_expr(lf, "location_type"),
            "police_district": text_expr(lf, "dc_dist", "district"),
            "local_area": text_expr(lf, "psa"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_atlanta(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = _prepare(lf)
    occurred = _atlanta_occurrence(lf)
    return project_adapter_fields(
        lf,
        {
            "source_record_id": composite_identifier(
                lf, "offense_id", "report_number", "report_num", "case_number"
            ),
            "report_timestamp": datetime_expr(
                lf, "report_date_time", "report_datetime", "report_date"
            ),
            "source_offense_code": text_expr(
                lf, "nibrs_code", "ucr_code", "offense_code"
            ),
            "source_offense_category": text_expr(lf, "ucr_literal", "offense"),
            "source_offense_description": text_expr(
                lf, "offense_description", "offense_desc", "ucr_literal"
            ),
            "latitude": numeric_expr(lf, "latitude", "lat"),
            "longitude": numeric_expr(lf, "longitude", "lon", "lng"),
            "location_label": text_expr(lf, "location", "address", "block"),
            "location_type": text_expr(lf, "location_type"),
            "police_district": text_expr(lf, "beat", "zone", "district"),
            "local_area": text_expr(lf, "neighborhood", "npu"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_los_angeles(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = _prepare(lf)
    occurred = _los_angeles_occurrence(lf)
    return project_adapter_fields(
        lf,
        {
            # NIBRS rows are offense-grain. Including the offense identifier/code
            # prevents incident-level collapsing across the legacy/NIBRS eras.
            "source_record_id": composite_identifier(
                lf,
                "dr_no",
                "case_no",
                "caseno",
                "incident_number",
                "offense_id",
                "offense_code",
                "crm_cd",
            ),
            "report_timestamp": datetime_expr(
                lf, "date_rptd", "reported_on", "report_date"
            ),
            "source_offense_code": text_expr(lf, "offense_code", "crm_cd", "ucr_code"),
            "source_offense_category": text_expr(lf, "offense_category", "part_1_2"),
            "source_offense_description": text_expr(
                lf, "offense_description", "crm_cd_desc"
            ),
            "latitude": numeric_expr(lf, "lat", "latitude"),
            "longitude": numeric_expr(lf, "lon", "longitude"),
            "location_label": text_expr(lf, "location", "cross_street"),
            "location_type": text_expr(lf, "premise_description", "premis_desc"),
            "police_district": text_expr(lf, "area_name", "area"),
            "local_area": text_expr(lf, "reporting_district", "rpt_dist_no"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_austin(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = _prepare(lf)
    occurred = _austin_occurrence(lf)
    return project_adapter_fields(
        lf,
        {
            "source_record_id": composite_identifier(
                lf, "incident_number", "incident_report_number", "case_number"
            ),
            "report_timestamp": datetime_expr(
                lf, "reported_date_time", "reported_datetime", "reported_date"
            ),
            "source_offense_code": text_expr(lf, "highest_offense_code", "ucr_code"),
            "source_offense_category": text_expr(
                lf, "ucr_category", "highest_offense_description"
            ),
            "source_offense_description": text_expr(
                lf, "highest_offense_description", "offense_description"
            ),
            "latitude": numeric_expr(lf, "latitude", "lat"),
            "longitude": numeric_expr(lf, "longitude", "lon", "lng"),
            "location_label": text_expr(lf, "address", "location"),
            "location_type": text_expr(lf, "location_type"),
            "police_district": text_expr(lf, "district", "sector"),
            "local_area": text_expr(lf, "council_district"),
            **_shared_fields(lf),
        },
        occurred,
    )


def adapt_boston(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = _prepare(lf)
    occurred = _boston_occurrence(lf)
    return project_adapter_fields(
        lf,
        {
            # INCIDENT_NUMBER alone is not offense-grain in Boston.
            "source_record_id": composite_identifier(
                lf,
                "incident_number",
                "offense_number",
                "offense_code",
                "offense_description",
            ),
            "report_timestamp": datetime_expr(lf, "reported_date", "reporting_date"),
            "source_offense_code": text_expr(lf, "offense_code"),
            "source_offense_category": text_expr(lf, "offense_code_group"),
            "source_offense_description": text_expr(lf, "offense_description"),
            "latitude": numeric_expr(lf, "lat", "latitude"),
            "longitude": numeric_expr(lf, "long", "longitude", "lon"),
            "location_label": text_expr(lf, "street", "location"),
            "location_type": text_expr(lf, "location_type"),
            "police_district": text_expr(lf, "district"),
            "local_area": text_expr(lf, "reporting_area"),
            **_shared_fields(lf),
        },
        occurred,
    )


HIGH_ROI_ADAPTERS = {
    "philadelphia": adapt_philadelphia,
    "atlanta": adapt_atlanta,
    "los_angeles": adapt_los_angeles,
    "austin": adapt_austin,
    "boston": adapt_boston,
}
