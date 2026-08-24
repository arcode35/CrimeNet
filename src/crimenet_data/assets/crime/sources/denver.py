import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    epoch_milliseconds_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    GEOJSON,
    adapt_standard,
    nullable_string,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
)


def prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
    return prepare_snake_case(lf).with_columns(
        pl.col("last_occurrence_date").cast(pl.Int64, strict=False),
        pl.col("first_occurrence_date").cast(pl.Int64, strict=False),
        pl.col("reported_date").cast(pl.Int64, strict=False),
        pl.col("geo_lat").cast(pl.Float64, strict=False),
        pl.col("geo_lon").cast(pl.Float64, strict=False),
    )


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return epoch_milliseconds_expr(lf, "first_occurrence_date")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "offense_id"),
        report_timestamp=epoch_milliseconds_expr(lf, "reported_date"),
        source_offense_code=text_expr(lf, "offense_code"),
        source_offense_category=text_expr(lf, "offense_category_id"),
        source_offense_description=text_expr(lf, "offense_type_id"),
        latitude=numeric_expr(lf, "geo_lat"),
        longitude=numeric_expr(lf, "geo_lon"),
        location_label=text_expr(lf, "incident_address"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "district_id"),
        local_area=text_expr(lf, "neighborhood_id"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="denver",
        source_system="denver_open_data",
        patterns=GEOJSON,
        timezone="America/Denver",
        crosswalk_keys=("source_offense_code", "source_offense_description"),
        coordinate_bounds=None,
    ),
    prepare_bronze=prepare,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
