import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    datetime_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    PARQUET,
    adapt_standard,
    nullable_string,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
)


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(lf, "incident_datetime", "occurred_at_raw")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "row_id"),
        report_timestamp=datetime_expr(lf, "report_datetime"),
        source_offense_code=text_expr(lf, "incident_code"),
        source_offense_category=text_expr(lf, "incident_category"),
        source_offense_description=text_expr(lf, "incident_description"),
        latitude=numeric_expr(lf, "latitude", "latitude_raw"),
        longitude=numeric_expr(lf, "longitude", "longitude_raw"),
        location_label=text_expr(lf, "intersection"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "police_district"),
        local_area=text_expr(lf, "analysis_neighborhood"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="san_francisco",
        source_system="san_francisco_open_data",
        patterns=PARQUET,
        timezone="America/Los_Angeles",
        crosswalk_keys=("source_offense_category", "source_offense_description"),
        deduplication_keys=("row_id",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
