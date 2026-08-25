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
    return datetime_expr(lf, "occurrence_timestamp", "from_date")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "case_no_offense"),
        report_timestamp=datetime_expr(lf, "reported_date"),
        # Preserve the exact profiler/crosswalk key. text_expr intentionally
        # trims strings, which changes Fort Worth's whitespace-bearing labels.
        source_offense_code=pl.col("offense").cast(pl.String, strict=False),
        source_offense_category=nullable_string(),
        source_offense_description=pl.col("offense_desc").cast(
            pl.String, strict=False
        ),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "block_address", "address", "location_1"),
        location_type=text_expr(lf, "locationtypedescription"),
        police_district=text_expr(lf, "division"),
        local_area=nullable_string(),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="fort_worth",
        source_system="fort_worth_open_data",
        patterns=PARQUET,
        timezone="America/Chicago",
        crosswalk_keys=("source_offense_code", "source_offense_description"),
        coordinate_bounds=(32.49684288, 33.04686194, -97.61320601, -96.95624817),
        deduplication_keys=("case_no_offense",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
