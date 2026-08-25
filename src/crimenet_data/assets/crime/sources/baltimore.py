import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    datetime_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    PARQUET,
    adapt_standard,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
)


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(
        lf,
        "occurrence_timestamp",
        "crime_date_time",
        "occurred_at_raw",
    )


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "row_id"),
        report_timestamp=pl.lit(None, dtype=pl.Datetime("us")),
        source_offense_code=text_expr(lf, "crime_code"),
        source_offense_category=text_expr(lf, "description"),
        source_offense_description=text_expr(lf, "description"),
        latitude=numeric_expr(lf, "latitude", "latitude_raw"),
        longitude=numeric_expr(lf, "longitude", "longitude_raw"),
        location_label=text_expr(lf, "location"),
        location_type=text_expr(lf, "premise_type"),
        police_district=text_expr(lf, "new_district"),
        local_area=text_expr(lf, "neighborhood"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="baltimore",
        source_system="baltimore_open_data",
        patterns=PARQUET,
        timezone="America/New_York",
        crosswalk_keys=("source_offense_code", "source_offense_description"),
        coordinate_bounds=(39.19713949, 39.3727399, -76.71219868, -76.5285938),
        deduplication_keys=("row_id",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
