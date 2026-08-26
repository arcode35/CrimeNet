import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    date_time_expr,
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
    return date_time_expr(lf, ("cmplnt_fr_dt",), ("cmplnt_fr_tm",))


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "cmplnt_num"),
        report_timestamp=datetime_expr(lf, "rpt_dt"),
        source_offense_code=text_expr(lf, "pd_cd", "ky_cd"),
        source_offense_category=text_expr(lf, "ofns_desc"),
        source_offense_description=text_expr(lf, "pd_desc"),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=nullable_string(),
        location_type=text_expr(lf, "prem_typ_desc"),
        police_district=text_expr(lf, "addr_pct_cd"),
        local_area=text_expr(lf, "boro_nm"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="new_york",
        source_system="nypd_complaint_data",
        patterns=PARQUET,
        timezone="America/New_York",
        crosswalk_keys=("source_offense_category", "source_offense_description"),
        deduplication_keys=("cmplnt_num",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
