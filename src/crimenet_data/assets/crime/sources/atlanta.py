import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    date_time_expr,
    datetime_expr,
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
    return date_time_expr(lf, ("occurdate",), ("occurtime",))


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "reportnumber"),
        report_timestamp=datetime_expr(lf, "reportdate"),
        source_offense_code=nullable_string(),
        source_offense_category=text_expr(lf, "ucrliteral"),
        source_offense_description=text_expr(lf, "ucrliteral"),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "location"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "beat"),
        local_area=text_expr(lf, "neighborhood", "npu"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="atlanta",
        source_system="atlanta_police_open_data",
        patterns=CSV,
        timezone="America/New_York",
        crosswalk_keys=("source_offense_description",),
        coordinate_bounds=(33.60, 33.95, -84.60, -84.20),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
