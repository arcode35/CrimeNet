import polars as pl

from crimenet_data.assets.crime.common.expressions import (
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
    return datetime_expr(lf, "incident_date_time")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "unique_id"),
        report_timestamp=pl.lit(None, dtype=pl.Datetime("us")),
        source_offense_code=nullable_string(),
        source_offense_category=text_expr(lf, "crime_class"),
        source_offense_description=text_expr(lf, "crime"),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "incident_street_address", "location"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "jurisdiction"),
        local_area=text_expr(lf, "incident_city_town", "incident_city_town_mapping"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="marin_county_sheriff_ca",
        source_system="marin_county_sheriff_open_data",
        patterns=CSV,
        timezone="America/Los_Angeles",
        crosswalk_keys=("source_offense_category", "source_offense_description"),
        coordinate_bounds=None,
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
