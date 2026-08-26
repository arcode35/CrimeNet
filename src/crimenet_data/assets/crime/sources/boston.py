import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    composite_identifier,
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
    return datetime_expr(lf, "occurred_on_date")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=composite_identifier(
            lf, "incident_number", "offense_code", "offense_description"
        ),
        report_timestamp=pl.lit(None, dtype=pl.Datetime("us")),
        source_offense_code=text_expr(lf, "offense_code"),
        source_offense_category=text_expr(lf, "offense_code_group"),
        source_offense_description=text_expr(lf, "offense_description"),
        latitude=numeric_expr(lf, "lat"),
        longitude=numeric_expr(lf, "long"),
        location_label=text_expr(lf, "street"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "district"),
        local_area=text_expr(lf, "reporting_area"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="boston",
        source_system="analyze_boston",
        patterns=CSV,
        timezone="America/New_York",
        crosswalk_keys=("source_offense_code", "source_offense_description"),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
