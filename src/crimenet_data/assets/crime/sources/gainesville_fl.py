import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    date_hour_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    CSV,
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
    return date_hour_expr(lf, "offense_date", "offense_hour_of_day")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "id"),
        report_timestamp=date_hour_expr(lf, "report_date", "report_hour_of_day"),
        source_offense_code=nullable_string(),
        source_offense_category=nullable_string(),
        source_offense_description=text_expr(lf, "narrative"),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "address", "location"),
        location_type=nullable_string(),
        police_district=nullable_string(),
        local_area=nullable_string(),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="gainesville_fl",
        source_system="gainesville_open_data",
        patterns=CSV,
        timezone="America/New_York",
        crosswalk_keys=("source_offense_description",),
        coordinate_bounds=None,
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
