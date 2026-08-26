import polars as pl

CANONICAL_MAPPING_VERSION = "crime_canonical_v1_5"

CANONICAL_CLASSIFICATION_COLUMNS = (
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
)

SOURCE_PROJECTION_SCHEMA = pl.Schema(
    {
        "source_record_id": pl.String,
        "occurrence_timestamp": pl.Datetime("us"),
        "report_timestamp": pl.Datetime("us"),
        "occurrence_year": pl.Int16,
        "source_offense_code": pl.String,
        "source_offense_category": pl.String,
        "source_offense_description": pl.String,
        "source_auxiliary": pl.String,
        "source_severity": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "location_label": pl.String,
        "location_type": pl.String,
        "police_district": pl.String,
        "local_area": pl.String,
        "source_file_uri": pl.String,
        "ingestion_run_id": pl.String,
        "ingested_at_utc": pl.Datetime(time_unit="us", time_zone="UTC"),
    }
)

CANONICAL_CRIME_SCHEMA = pl.Schema(
    {
        # ------------------------------------------------------------
        # Identity / source
        # ------------------------------------------------------------
        "crime_id": pl.String,
        "source_city": pl.String,
        "source_record_id": pl.String,
        # ------------------------------------------------------------
        # Time
        # ------------------------------------------------------------
        "occurrence_timestamp": pl.Datetime("us"),
        "report_timestamp": pl.Datetime("us"),
        "occurrence_year": pl.Int16,
        "source_timezone": pl.String,
        # ------------------------------------------------------------
        # Original source taxonomy
        # ------------------------------------------------------------
        "source_offense_code": pl.String,
        "source_offense_category": pl.String,
        "source_offense_description": pl.String,
        "source_auxiliary": pl.String,
        "source_severity": pl.String,
        # ------------------------------------------------------------
        # Canonical mapping / taxonomy
        # ------------------------------------------------------------
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
        # ------------------------------------------------------------
        # Mapping auditability
        # ------------------------------------------------------------
        "canonical_mapping_found": pl.Boolean,
        "mapping_confidence": pl.String,
        "review_required": pl.Boolean,
        "mapping_action": pl.String,
        "include_in_model": pl.Boolean,
        # ------------------------------------------------------------
        # Geography
        # ------------------------------------------------------------
        "source_coordinate_bounds_valid": pl.Boolean,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "location_label": pl.String,
        "location_type": pl.String,
        "police_district": pl.String,
        "local_area": pl.String,
        # ------------------------------------------------------------
        # Ingestion provenance
        # ------------------------------------------------------------
        "source_file_uri": pl.String,
        "ingestion_run_id": pl.String,
        "ingested_at_utc": pl.Datetime(
            time_unit="us",
            time_zone="UTC",
        ),
    }
)
