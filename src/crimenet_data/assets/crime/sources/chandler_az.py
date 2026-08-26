import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    datetime_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    adapt_standard,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
    SourcePattern,
)

EXPECTED_COLUMNS = (
    "id",
    "event_type",
    "agency",
    "report_id",
    "report_number",
    "report_year",
    "reported_date",
    "report_event_date",
    "report_event_time",
    "report_event_date_time",
    "report_event_day",
    "report_event_end_date",
    "report_event_end_time",
    "report_event_end_date_time",
    "report_place_name",
    "report_address",
    "report_city",
    "report_district",
    "report_beat",
    "report_location_type",
    "report_latitude",
    "report_longitude",
    "report_status",
    "report_primary_offense_code",
    "report_primary_offense_description",
    "report_summary_offense_code",
    "report_summary_offense_description",
)


def prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
    return prepare_snake_case(lf).with_columns(
        pl.col("id").cast(pl.Int64, strict=False),
        pl.col("report_id").cast(pl.Int64, strict=False),
        pl.col("report_latitude").cast(pl.Float64, strict=False),
        pl.col("report_longitude").cast(pl.Float64, strict=False),
    )


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(lf, "report_event_date_time")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "report_id"),
        report_timestamp=datetime_expr(lf, "reported_date"),
        source_offense_code=text_expr(lf, "report_primary_offense_code"),
        source_offense_category=text_expr(lf, "report_summary_offense_description"),
        source_offense_description=text_expr(lf, "report_primary_offense_description"),
        latitude=numeric_expr(lf, "report_latitude"),
        longitude=numeric_expr(lf, "report_longitude"),
        location_label=text_expr(lf, "report_address", "report_place_name"),
        location_type=text_expr(lf, "report_location_type"),
        police_district=text_expr(lf, "report_district"),
        local_area=text_expr(lf, "report_beat"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="chandler_az",
        source_system="chandler_police_open_data",
        patterns=(
            SourcePattern(
                "**/*.csv",
                "csv",
                {
                    "strategy": "python_tolerant",
                    "encoding": "cp1252",
                    "expected_columns": EXPECTED_COLUMNS,
                },
            ),
        ),
        timezone="America/Phoenix",
        crosswalk_keys=("source_offense_code",),
    ),
    prepare_bronze=prepare,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
