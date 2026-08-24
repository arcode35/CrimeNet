import polars as pl

from crimenet_data.assets.crime.common.expressions import (
    composite_identifier,
    datetime_expr,
    numeric_expr,
    text_expr,
)
from crimenet_data.assets.crime.sources._shared import (
    adapt_standard,
    prepare_snake_case,
)
from crimenet_data.assets.crime.sources.base import (
    AdapterContext,
    CrimeSourceConfig,
    SourceDefinition,
    SourcePattern,
)

EXPECTED_COLUMNS = (
    "incident_id",
    "offence_code",
    "case_number",
    "date",
    "start_date",
    "end_date",
    "nibrs_code",
    "victims",
    "crimename1",
    "crimename2",
    "crimename3",
    "district",
    "location",
    "city",
    "state",
    "zip_code",
    "agency",
    "place",
    "sector",
    "beat",
    "pra",
    "address_number",
    "street_prefix_dir",
    "address_street",
    "street_suffix_dir",
    "street_type",
    "latitude",
    "longitude",
    "police_district_number",
    "geolocation",
)


def occurrence(lf: pl.LazyFrame) -> pl.Expr:
    return datetime_expr(lf, "start_date")


def adapt(lf: pl.LazyFrame, _context: AdapterContext) -> pl.LazyFrame:
    return adapt_standard(
        lf,
        occurrence(lf),
        source_record_id=composite_identifier(
            lf, "incident_id", "case_number", "offence_code"
        ),
        report_timestamp=datetime_expr(lf, "date"),
        source_offense_code=text_expr(lf, "nibrs_code", "offence_code"),
        source_offense_category=text_expr(lf, "crimename1", "crimename2"),
        source_offense_description=text_expr(
            lf, "crimename3", "crimename2", "crimename1"
        ),
        latitude=numeric_expr(lf, "latitude"),
        longitude=numeric_expr(lf, "longitude"),
        location_label=text_expr(lf, "location", "address_street"),
        location_type=text_expr(lf, "place"),
        police_district=text_expr(lf, "district", "police_district_number"),
        local_area=text_expr(lf, "sector", "beat", "pra"),
    )


SOURCE = SourceDefinition(
    config=CrimeSourceConfig(
        key="montgomery_county_md",
        source_system="montgomery_county_police_open_data",
        patterns=(
            SourcePattern(
                "**/*.csv",
                "csv",
                {
                    "strategy": "python_tolerant",
                    "encoding": "utf-8-sig",
                    "expected_columns": EXPECTED_COLUMNS,
                    "overflow_column": "crimename1",
                },
            ),
        ),
        timezone="America/New_York",
        crosswalk_keys=("source_offense_code", "source_offense_description"),
        coordinate_bounds=None,
    ),
    prepare_bronze=prepare_snake_case,
    occurrence_timestamp=occurrence,
    adapt_to_silver=adapt,
)
