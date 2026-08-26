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
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
)


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return pl.coalesce(
        [
            datetime_expr(lf, "occurrence_timestamp"),
            date_time_expr(lf, ("date1_of_occurrence",), ("time1_of_occurrence",)),
        ]
    )


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=text_expr(lf, "service_number_id"),
        report_timestamp=datetime_expr(lf, "date_of_report"),
        source_offense_code=text_expr(lf, "nibrs_code", "ucr_code", "rms_code"),
        source_offense_category=text_expr(lf, "nibrs_crime"),
        source_offense_description=text_expr(lf, "type_of_incident"),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "incident_address", "location1"),
        location_type=text_expr(lf, "type_location"),
        police_district=text_expr(lf, "division"),
        local_area=text_expr(lf, "community"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="dallas",
        source_system="dallas_open_data",
        patterns=PARQUET,
        timezone="America/Chicago",
        crosswalk_keys=("source_offense_category", "source_offense_description"),
        deduplication_keys=("service_number_id",),
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
