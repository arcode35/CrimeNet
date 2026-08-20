from __future__ import annotations

import polars as pl

def text(column: str) -> pl.Expr:
    value = (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
    )

    return (
        pl.when(value == "")
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(value)
    )


def coalesce_text(*columns: str) -> pl.Expr:
    return pl.coalesce(
        [text(column) for column in columns]
    )


def parse_datetime_expr(expr: pl.Expr) -> pl.Expr:
    value = (
        expr
        .cast(pl.String, strict=False)
        .str.strip_chars()
    )

    return pl.coalesce(
        [
            value.str.to_datetime(strict=False),
            pl.from_epoch(
                value.cast(pl.Int64, strict=False),
                time_unit="ms",
            ),
        ]
    ).cast(pl.Datetime("us"))


def parse_datetime(column: str) -> pl.Expr:
    return parse_datetime_expr(pl.col(column))


def parse_date_and_time(
    date_column: str,
    time_column: str,
) -> pl.Expr:
    combined = pl.concat_str(
        [
            pl.col(date_column),
            pl.col(time_column),
        ],
        separator=" ",
        ignore_nulls=False,
    )

    # Fall back to the date alone when time is absent.
    return pl.coalesce(
        [
            parse_datetime_expr(combined),
            parse_datetime(date_column),
        ]
    )

def parse_nyc_occurrence_timestamp() -> pl.Expr:
    date_text = (
        pl.col("cmplnt_fr_dt")
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.slice(0, 10)
    )

    time_text = (
        pl.col("cmplnt_fr_tm")
        .cast(pl.String, strict=False)
        .str.strip_chars()
    )

    return (
        pl.concat_str(
            [
                date_text,
                time_text,
            ],
            separator=" ",
            ignore_nulls=False,
        )
        .str.to_datetime(
            format="%Y-%m-%d %H:%M:%S",
            strict=False,
        )
        .cast(pl.Datetime("us"))
    )
def null_value() -> pl.Expr:
    return pl.lit(None)

CANONICAL_COLUMNS = [
    "mapping_version",
    "canonical_family_code",
    "canonical_offense_family",
    "canonical_subtype_code",
    "canonical_offense_subtype",
    "canonical_domain",
    "canonical_target",
    "is_criminal_event",
    "is_violent",
    "is_property",
    "mapping_confidence",
    "review_required",
    "mapping_action",
    "include_in_model",
]

CITY_RECORD_KEYS = {
    "dallas": ["service_number_id"],
    "chicago": ["id"],
    "baltimore": ["RowID"],
    "seattle": ["offense_id"],
    "san_francisco": ["row_id"],
    "washington_dc": ["OBJECTID"],
    "fort_worth": ["case_no_offense"],
    "new_york": ["cmplnt_num"],
}
CITY_OFFENSE_MAPPING = {
    "chicago": {
        "source_offense_category": "primary_type",
        "source_offense_description": "description",
    },
    "dallas": {
        "source_offense_category": "nibrs_crime",
        "source_offense_description": "type_of_incident",
    },
    "fort_worth": {
        "source_offense_code": "offense",
        "source_offense_description": "offense_desc",
    },
    "new_york": {
        "source_offense_category": "ofns_desc",
        "source_offense_description": "pd_desc",
    },
    "san_francisco": {
        "source_offense_category": "incident_category",
        "source_offense_description": "incident_description",
    },
    "seattle": {
        "source_offense_category": "offense_category",
        "source_offense_description": "nibrs_offense_code_description",
    },
    "washington_dc": {
        "source_offense_category": "OFFENSE"
    },
    "baltimore": {
        "source_offense_code": "CrimeCode", 
        "source_offense_description": "Description"
    },
}

CANONICAL_MAPPING_VERSION = "crime_canonical_v1_3"


CANONICAL_CRIME_SCHEMA = pl.Schema(
    {
        # Stable identity
        "crime_id": pl.String,
        "source_city": pl.String,
        "source_record_id": pl.String,

        # Event time
        "occurrence_timestamp": pl.Datetime("us"),
        "report_timestamp": pl.Datetime("us"),
        "occurrence_year": pl.Int16,
        "source_timezone": pl.String,

        # Original source classification
        "source_offense_code": pl.String,
        "source_offense_category": pl.String,
        "source_offense_description": pl.String,

        # Canonical classification
        "mapping_version": pl.String,
        "canonical_family_code": pl.String,
        "canonical_offense_family": pl.String,
        "canonical_subtype_code": pl.String,
        "canonical_offense_subtype": pl.String,
        "canonical_domain": pl.String,
        "canonical_target": pl.String,
        "is_criminal_event": pl.Boolean,
        "is_violent": pl.Boolean,
        "is_property": pl.Boolean,

        # Mapping diagnostics
        "canonical_mapping_found": pl.Boolean,
        "mapping_confidence": pl.String,
        "review_required": pl.Boolean,
        "mapping_action": pl.String,
        "include_in_model": pl.Boolean,

        # Location
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "location_label": pl.String,
        "location_type": pl.String,
        "police_district": pl.String,
        "local_area": pl.String,

        # Provenance
        "source_file_uri": pl.String,
        "ingestion_run_id": pl.String,
        "ingested_at_utc": pl.Datetime(
            time_unit="us",
            time_zone="UTC",
        ),
    }
)

COMMON_CANONICAL_MAPPING: dict[str, pl.Expr] = {
    "occurrence_year": pl.col("occurrence_year"),

    "mapping_version": (
        pl.col("mapping_version")
        .fill_null(CANONICAL_MAPPING_VERSION)
    ),
    "canonical_family_code": pl.col(
        "canonical_family_code"
    ),
    "canonical_offense_family": pl.col(
        "canonical_offense_family"
    ),
    "canonical_subtype_code": pl.col(
        "canonical_subtype_code"
    ),
    "canonical_offense_subtype": pl.col(
        "canonical_offense_subtype"
    ),
    "canonical_domain": pl.col("canonical_domain"),
    "canonical_target": pl.col("canonical_target"),
    "is_criminal_event": pl.col("is_criminal_event"),
    "is_violent": pl.col("is_violent"),
    "is_property": pl.col("is_property"),

    "canonical_mapping_found": (
        pl.col("canonical_family_code").is_not_null()
    ),
    "mapping_confidence": (
        pl.col("mapping_confidence")
        .fill_null("unmatched")
    ),
    "review_required": (
        pl.col("review_required")
        .fill_null(True)
    ),
    "mapping_action": (
        pl.col("mapping_action")
        .fill_null("unmatched")
    ),
    "include_in_model": (
        pl.col("include_in_model")
        .fill_null(False)
    ),

    "ingestion_run_id": pl.col("_ingestion_run_id"),
    "ingested_at_utc": pl.col("_ingested_at_utc"),
}


CITY_COORDINATE_BOUNDS = {
    "dallas": {
        "min_latitude": 32.61341314,
        "max_latitude": 33.02346757,
        "min_longitude": -97.00063702,
        "max_longitude": -96.46270666,
    },
    "new_york": {
        "min_latitude": 40.49601000,
        "max_latitude": 40.91556800,
        "min_longitude": -74.25715900,
        "max_longitude": -73.69921500,
    },
    "chicago": {
        "min_latitude": 41.64614955,
        "max_latitude": 42.02203444,
        "min_longitude": -87.94227797,
        "max_longitude": -87.52018407,
    },
    "baltimore": {
        "min_latitude": 39.19713949,
        "max_latitude": 39.37273990,
        "min_longitude": -76.71219868,
        "max_longitude": -76.52859380,
    },
    "seattle": {
        "min_latitude": 47.49334795,
        "max_latitude": 47.73569416,
        "min_longitude": -122.43079310,
        "max_longitude": -122.24213716,
    },
    "san_francisco": {
        "min_latitude": 37.70452857,
        "max_latitude": 37.83329764,
        "min_longitude": -122.51494758,
        "max_longitude": -122.35619807,
    },
    "washington_dc": {
        "min_latitude": 38.79151001,
        "max_latitude": 38.99592832,
        "min_longitude": -77.11956952,
        "max_longitude": -76.90903411,
    },
    "fort_worth": {
        "min_latitude": 32.49684288,
        "max_latitude": 33.04686194,
        "min_longitude": -97.61320601,
        "max_longitude": -96.95624817,
    },
}


CITY_TIMEZONES = {
    "dallas": "America/Chicago",
    "fort_worth": "America/Chicago",
    "chicago": "America/Chicago",
    "new_york": "America/New_York",
    "baltimore": "America/New_York",
    "washington_dc": "America/New_York",
    "seattle": "America/Los_Angeles",
    "san_francisco": "America/Los_Angeles",
}


CITY_CANONICAL_MAPPING: dict[
    str,
    dict[str, pl.Expr],
] = {
    "dallas": {
        "source_record_id": text("service_number_id"),

        "occurrence_timestamp": pl.coalesce(
            [
                pl.col("occurrence_timestamp").cast(
                    pl.Datetime("us"),
                    strict=False,
                ),
                parse_date_and_time(
                    "date1_of_occurrence",
                    "time1_of_occurrence",
                ),
            ]
        ),
        "report_timestamp": parse_datetime(
            "date_of_report"
        ),

        "source_offense_code": coalesce_text(
            "nibrs_code",
            "ucr_code",
            "rms_code",
        ),
        "source_offense_category": text("nibrs_crime"),
        "source_offense_description": text(
            "type_of_incident"
        ),

        # These must exist after EPSG:2276 -> EPSG:4326.
        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        "location_label": coalesce_text(
            "incident_address",
            "location1",
        ),
        "location_type": text("type_location"),
        "police_district": text("division"),
        "local_area": text("community"),

        "source_file_uri": text("source_file"),
    },

    "new_york": {
        "source_record_id": text("cmplnt_num"),

        "occurrence_timestamp": parse_nyc_occurrence_timestamp(),
        
        "report_timestamp": parse_datetime("rpt_dt"),

        # PD_CD is more detailed; KY_CD is the fallback.
        "source_offense_code": coalesce_text(
            "pd_cd",
            "ky_cd",
        ),
        "source_offense_category": text("ofns_desc"),
        "source_offense_description": text("pd_desc"),

        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        # NYPD does not provide a general street-address field.
        "location_label": coalesce_text(
            "station_name",
            "parks_nm",
            "hadevelopt",
            "boro_nm",
        ),
        "location_type": text("prem_typ_desc"),
        "police_district": text("addr_pct_cd"),
        "local_area": text("boro_nm"),

        "source_file_uri": text("source_file"),
    },

    "chicago": {
        "source_record_id": text("id"),

        "occurrence_timestamp": parse_datetime("date"),
        "report_timestamp": null_value(),

        "source_offense_code": text("iucr"),
        "source_offense_category": text("primary_type"),
        "source_offense_description": text("description"),

        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        "location_label": text("block"),
        "location_type": text("location_description"),
        "police_district": text("district"),
        "local_area": text("community_area"),

        "source_file_uri": text("source_file"),
    },

    "baltimore": {
        "source_record_id": text("RowID"),

        "occurrence_timestamp": pl.coalesce(
            [
                parse_datetime("CrimeDateTime"),
                parse_datetime("occurred_at_raw"),
            ]
        ),
        "report_timestamp": null_value(),

        "source_offense_code": text("CrimeCode"),

        # Baltimore exposes one primary textual offense level.
        "source_offense_category": text("Description"),
        "source_offense_description": text("Description"),

        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        "location_label": text("Location"),
        "location_type": text("PremiseType"),
        "police_district": text("New_District"),
        "local_area": text("Neighborhood"),

        "source_file_uri": text("source_url"),
    },

    "seattle": {
        "source_record_id": text("offense_id"),

        "occurrence_timestamp": pl.coalesce(
            [
                parse_datetime("offense_date"),
                parse_datetime("occurred_at_raw"),
            ]
        ),
        "report_timestamp": parse_datetime(
            "report_date_time"
        ),

        "source_offense_code": text(
            "nibrs_offense_code"
        ),
        "source_offense_category": text(
            "offense_category"
        ),
        "source_offense_description": text(
            "nibrs_offense_code_description"
        ),

        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        "location_label": text("block_address"),
        "location_type": null_value(),
        "police_district": text("precinct"),
        "local_area": text("neighborhood"),

        "source_file_uri": text("source_file"),
    },

    "san_francisco": {
        "source_record_id": text("row_id"),

        "occurrence_timestamp": pl.coalesce(
            [
                parse_datetime("incident_datetime"),
                parse_datetime("occurred_at_raw"),
            ]
        ),
        "report_timestamp": parse_datetime(
            "report_datetime"
        ),

        "source_offense_code": text("incident_code"),
        "source_offense_category": text(
            "incident_category"
        ),
        "source_offense_description": text(
            "incident_description"
        ),

        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        "location_label": text("intersection"),
        "location_type": null_value(),
        "police_district": text("police_district"),
        "local_area": text("analysis_neighborhood"),

        "source_file_uri": text("source_file"),
    },

    "washington_dc": {
        "source_record_id": text("OBJECTID"),

        "occurrence_timestamp": pl.coalesce(
            [
                parse_datetime("START_DATE"),
                parse_datetime("occurred_at_raw"),
            ]
        ),
        "report_timestamp": parse_datetime("REPORT_DAT"),

        # DC provides an offense label, not a separate code.
        "source_offense_code": null_value(),
        "source_offense_category": text("OFFENSE"),
        "source_offense_description": text("OFFENSE"),

        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        "location_label": text("BLOCK"),
        "location_type": null_value(),
        "police_district": text("DISTRICT"),
        "local_area": text("NEIGHBORHOOD_CLUSTER"),

        "source_file_uri": text("source_url"),
    },

    "fort_worth": {
        "source_record_id": text("case_no_offense"),

        "occurrence_timestamp": pl.coalesce(
            [
                pl.col("occurrence_timestamp").cast(
                    pl.Datetime("us"),
                    strict=False,
                ),
                parse_datetime("from_date"),
            ]
        ),
        "report_timestamp": parse_datetime(
            "reported_date"
        ),

        "source_offense_code": text("offense"),
        "source_offense_category": text("offense"),
        "source_offense_description": text("offense_desc"),

        "latitude": pl.col("latitude"),
        "longitude": pl.col("longitude"),

        "location_label": coalesce_text(
            "block_address",
            "address",
            "location_1",
        ),
        "location_type": text(
            "locationtypedescription"
        ),
        "police_district": text("division"),

        # No actual neighborhood-like column is present.
        "local_area": null_value(),

        "source_file_uri": coalesce_text(
            "_source_file_uri",
            "source_file",
        ),
    },
}