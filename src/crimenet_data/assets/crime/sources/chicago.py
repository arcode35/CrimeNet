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
    return datetime_expr(lf, "date")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "id"),
        report_timestamp=pl.lit(None, dtype=pl.Datetime("us")),
        source_offense_code=text_expr(lf, "iucr"),
        source_offense_category=text_expr(lf, "primary_type"),
        source_offense_description=text_expr(lf, "description"),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "block"),
        location_type=text_expr(lf, "location_description"),
        police_district=text_expr(lf, "district"),
        local_area=text_expr(lf, "community_area"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="chicago",
        source_system="chicago_data_portal",
        patterns=PARQUET,
        timezone="America/Chicago",
        crosswalk_keys=("source_offense_category", "source_offense_description"),
        coordinate_bounds=(41.64614955, 42.02203444, -87.94227797, -87.52018407),
        deduplication_keys=("id",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
