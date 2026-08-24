import polars as pl

from crimenet_data.assets.crime.common.expressions import datetime_expr, text_expr
from crimenet_data.assets.crime.sources._shared import (
    CSV,
    adapt_standard,
    nullable_float,
    nullable_string,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
)


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(lf, "date_time")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "id", "incident_number"),
        report_timestamp=pl.lit(None, dtype=pl.Datetime("us")),
        source_offense_code=nullable_string(),
        source_offense_category=text_expr(lf, "incident_type"),
        source_offense_description=text_expr(lf, "incident_type"),
        latitude=nullable_float(),
        longitude=nullable_float(),
        location_label=text_expr(lf, "location", "intersection"),
        location_type=text_expr(lf, "location_type"),
        police_district=text_expr(lf, "agency"),
        local_area=text_expr(lf, "city"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="sonoma_county_sheriff_ca",
        source_system="sonoma_county_sheriff_open_data",
        patterns=CSV,
        timezone="America/Los_Angeles",
        crosswalk_keys=("source_offense_description",),
        coordinate_bounds=None,
        coordinates_required=False,
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
