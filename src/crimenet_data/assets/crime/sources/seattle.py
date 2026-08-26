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
    return datetime_expr(lf, "offense_date", "occurred_at_raw")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "offense_id"),
        report_timestamp=datetime_expr(lf, "report_date_time"),
        source_offense_code=text_expr(lf, "nibrs_offense_code"),
        source_offense_category=text_expr(lf, "offense_category"),
        source_offense_description=text_expr(lf, "nibrs_offense_code_description"),
        latitude=numeric_expr(lf, "latitude", "latitude_raw"),
        longitude=numeric_expr(lf, "longitude", "longitude_raw"),
        location_label=text_expr(lf, "block_address"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "precinct"),
        local_area=text_expr(lf, "neighborhood"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="seattle",
        source_system="seattle_open_data",
        patterns=PARQUET,
        timezone="America/Los_Angeles",
        crosswalk_keys=("source_offense_description",),
        deduplication_keys=("offense_id",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
