import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    datetime_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    CSV,
    adapt_standard,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
)


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(lf, "charge_date")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "charge_id", "incident_number"),
        report_timestamp=datetime_expr(lf, "report_date"),
        source_offense_code=text_expr(lf, "nibrs_code"),
        source_offense_category=text_expr(lf, "statute_category"),
        source_offense_description=text_expr(
            lf, "offense_description", "statute_description"
        ),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "street", "street_2"),
        location_type=pl.lit(None, dtype=pl.String),
        police_district=text_expr(lf, "district"),
        local_area=text_expr(lf, "neighborhood"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="east_baton_rouge_parish_sheriff_la",
        source_system="east_baton_rouge_parish_sheriff_open_data",
        patterns=CSV,
        timezone="America/Chicago",
        crosswalk_keys=("source_offense_code", "source_offense_description"),
        coordinate_bounds=None,
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
