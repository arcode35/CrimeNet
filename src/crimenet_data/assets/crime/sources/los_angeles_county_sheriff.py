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


def deduplicate_exports(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.with_columns(
            pl.col("source_file_uri")
            .str.extract(r"lasd_part_i_ii_(\d{4})\.csv", 1)
            .cast(pl.Int16, strict=False)
            .alias("_export_year")
        )
        .sort(
            [
                "incident_id",
                "lurn_sak",
                "stat",
                "_export_year",
            ]
        )
        .unique(
            subset=[
                "incident_id",
                "lurn_sak",
                "stat",
            ],
            keep="last",
            maintain_order=False,
        )
        .drop("_export_year")
    )


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(lf, "incident_date")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    lf = deduplicate_exports(lf)
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=composite_identifier(lf, "incident_id", "lurn_sak", "stat"),
        report_timestamp=datetime_expr(lf, "incident_reported_date"),
        source_offense_code=text_expr(lf, "stat"),
        source_offense_category=text_expr(lf, "category"),
        source_offense_description=text_expr(lf, "stat_desc"),
        source_auxiliary=text_expr(lf, "part_category"),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "address", "street"),
        location_type=nullable_string(),
        police_district=text_expr(lf, "reporting_district"),
        local_area=text_expr(lf, "unit_name"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="los_angeles_county_sheriff",
        source_system="los_angeles_county_sheriff_open_data",
        patterns=CSV,
        timezone="America/Los_Angeles",
        crosswalk_keys=("source_offense_description",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
