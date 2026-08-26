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
    return datetime_expr(
        lf,
        "occurrence_timestamp",
        "occurred_at_raw",
        "start_date",
    )


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "objectid"),
        report_timestamp=datetime_expr(lf, "report_dat"),
        source_offense_code=nullable_string(),
        source_offense_category=text_expr(lf, "offense"),
        source_offense_description=text_expr(lf, "offense"),
        latitude=numeric_expr(lf, "latitude", "latitude_raw"),
        longitude=numeric_expr(lf, "longitude", "longitude_raw"),
        location_label=text_expr(lf, "block"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "district"),
        local_area=text_expr(lf, "neighborhood_cluster"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="washington_dc",
        source_system="dc_crime_cards",
        patterns=PARQUET,
        timezone="America/New_York",
        crosswalk_keys=("source_offense_category",),
        deduplication_keys=("objectid",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
