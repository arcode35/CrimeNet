"""Version-aware, auditable crime-location to Census-tract mapping."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession


MAPPING_DEFINITION_VERSION = "tract_point_in_polygon_v1"
LOCATION_KEY_COLUMNS = (
    "tiger_line_year",
    "latitude",
    "longitude",
)
MATCHED_CONTAINS = "matched_contains"
MATCHED_COVERS = "matched_covers"
UNMATCHED = "unmatched"
AMBIGUOUS = "ambiguous"
MATCH_STATUSES = frozenset(
    {
        MATCHED_CONTAINS,
        MATCHED_COVERS,
        UNMATCHED,
        AMBIGUOUS,
    }
)
_COORDINATE_DECIMAL_PLACES = 12


@dataclass(frozen=True)
class LocationFrames:
    """Valid unique mapping candidates and rejected source crime rows."""

    candidates: DataFrame
    quarantine: DataFrame
    source_row_count: int
    invalid_row_count: int


def canonical_coordinate(value: float) -> str:
    """Return the mapping-version coordinate representation."""

    if isinstance(value, bool):
        raise ValueError("Coordinate must be numeric, not boolean.")
    try:
        coordinate = float(value)
        decimal_value = Decimal(str(value))
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError(f"Invalid coordinate value: {value!r}.") from exc
    if not math.isfinite(coordinate):
        raise ValueError(f"Coordinate must be finite: {value!r}.")

    quantum = Decimal(1).scaleb(-_COORDINATE_DECIMAL_PLACES)
    normalized = decimal_value.quantize(
        quantum,
        rounding=ROUND_HALF_UP,
    )
    if normalized == 0:
        normalized = abs(normalized)
    return format(normalized, f".{_COORDINATE_DECIMAL_PLACES}f")


def coordinate_reason_code(
    *,
    latitude: float | None,
    longitude: float | None,
) -> str | None:
    """Classify an invalid coordinate without silently discarding it."""

    if latitude is None or longitude is None:
        return "MISSING_COORDINATE"
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return "NON_FINITE_COORDINATE"
    try:
        normalized_latitude = float(latitude)
        normalized_longitude = float(longitude)
    except (TypeError, ValueError):
        return "NON_FINITE_COORDINATE"
    if not (math.isfinite(normalized_latitude) and math.isfinite(normalized_longitude)):
        return "NON_FINITE_COORDINATE"
    if not -90.0 <= normalized_latitude <= 90.0:
        return "LATITUDE_OUT_OF_RANGE"
    if not -180.0 <= normalized_longitude <= 180.0:
        return "LONGITUDE_OUT_OF_RANGE"
    return None


def location_tract_key(
    *,
    tiger_line_year: int,
    latitude: float,
    longitude: float,
    boundary_definition_version: str,
    source_archive_sha256: str,
    mapping_definition_version: str = MAPPING_DEFINITION_VERSION,
) -> str:
    """Return a stable key independent of files, runs, and environments."""

    if not boundary_definition_version.strip():
        raise ValueError("boundary_definition_version cannot be blank.")
    if not mapping_definition_version.strip():
        raise ValueError("mapping_definition_version cannot be blank.")
    if not re.fullmatch(r"[0-9a-f]{64}", source_archive_sha256):
        raise ValueError("source_archive_sha256 must be a SHA-256 digest.")
    reason = coordinate_reason_code(
        latitude=latitude,
        longitude=longitude,
    )
    if reason is not None:
        raise ValueError(f"Cannot key invalid coordinates: {reason}.")

    payload = json.dumps(
        {
            "mapping_definition_version": mapping_definition_version,
            "boundary_definition_version": boundary_definition_version,
            "tiger_line_year": str(tiger_line_year),
            "latitude": canonical_coordinate(latitude),
            "longitude": canonical_coordinate(longitude),
            "source_archive_sha256": source_archive_sha256,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attach_release_aware_boundary_year(
    crime_dataframe: DataFrame,
    calendar_dataframe: DataFrame,
) -> DataFrame:
    """Attach only the ACS/TIGER vintage public at crime occurrence time."""

    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    duplicate_calendar_keys = (
        calendar_dataframe.groupBy("acs_vintage")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    duplicate_release_dates = (
        calendar_dataframe.groupBy("acs_release_date")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    invalid_calendar_rows = (
        calendar_dataframe.filter(
            F.col("acs_vintage").isNull()
            | F.col("acs_release_date").isNull()
            | F.col("tiger_line_year").isNull()
            | (F.col("tiger_line_year") != F.col("acs_vintage"))
        )
        .limit(1)
        .count()
    )
    if duplicate_calendar_keys or duplicate_release_dates:
        raise ValueError(
            "ACS release calendar contains duplicate vintages or public release dates."
        )
    if invalid_calendar_rows:
        raise ValueError(
            "ACS release calendar contains null or incompatible vintage metadata."
        )

    calendar_window = Window.orderBy("acs_release_date")
    ranges = (
        calendar_dataframe.withColumn(
            "_eligible_start_date",
            F.date_add("acs_release_date", 1),
        )
        .withColumn(
            "_next_eligible_start_date",
            F.lead("_eligible_start_date").over(calendar_window),
        )
        .withColumn(
            "_eligible_end_date",
            F.date_sub("_next_eligible_start_date", 1),
        )
    )
    crimes = crime_dataframe.withColumn(
        "_occurred_date",
        F.to_date("occurred_at"),
    )
    return (
        crimes.alias("crime")
        .join(
            F.broadcast(ranges).alias("calendar"),
            (F.col("crime._occurred_date") >= F.col("calendar._eligible_start_date"))
            & (
                F.col("calendar._eligible_end_date").isNull()
                | (
                    F.col("crime._occurred_date")
                    <= F.col("calendar._eligible_end_date")
                )
            ),
            "left",
        )
        .select(
            "crime.*",
            F.col("calendar.acs_vintage").alias("selected_acs_vintage"),
            F.col("calendar.acs_release_date").alias("selected_acs_release_date"),
            F.col("calendar.tiger_line_year"),
            F.col("calendar.tract_definition_vintage"),
        )
        .drop("_occurred_date")
    )


def _source_column(
    dataframe: DataFrame,
    *candidates: str,
    fallback: object,
) -> Any:
    from pyspark.sql import functions as F

    for candidate in candidates:
        if candidate in dataframe.columns:
            return F.col(candidate)
    return F.lit(fallback)


def split_location_candidates(
    crime_with_calendar: DataFrame,
    *,
    pipeline_run_id: str,
) -> LocationFrames:
    """Return unique valid locations and a reasoned invalid-row quarantine."""

    from pyspark.sql import functions as F

    valid_coordinate = (
        F.col("latitude").isNotNull()
        & F.col("longitude").isNotNull()
        & ~F.isnan("latitude")
        & ~F.isnan("longitude")
        & (F.abs(F.col("latitude")) != float("inf"))
        & (F.abs(F.col("longitude")) != float("inf"))
        & F.col("latitude").between(-90.0, 90.0)
        & F.col("longitude").between(-180.0, 180.0)
    )
    valid = (
        valid_coordinate
        & F.col("occurred_at").isNotNull()
        & F.col("tiger_line_year").isNotNull()
    )

    candidates = (
        crime_with_calendar.filter(valid)
        .select(*LOCATION_KEY_COLUMNS)
        .dropDuplicates(list(LOCATION_KEY_COLUMNS))
    )

    reason_code = (
        F.when(
            F.col("occurred_at").isNull(),
            F.lit("MISSING_OCCURRENCE_TIMESTAMP"),
        )
        .when(
            F.col("tiger_line_year").isNull(),
            F.lit("NO_ELIGIBLE_ACS_VINTAGE"),
        )
        .when(
            F.col("latitude").isNull() | F.col("longitude").isNull(),
            F.lit("MISSING_COORDINATE"),
        )
        .when(
            F.isnan("latitude")
            | F.isnan("longitude")
            | (F.abs(F.col("latitude")) == float("inf"))
            | (F.abs(F.col("longitude")) == float("inf")),
            F.lit("NON_FINITE_COORDINATE"),
        )
        .when(
            ~F.col("latitude").between(-90.0, 90.0),
            F.lit("LATITUDE_OUT_OF_RANGE"),
        )
        .when(
            ~F.col("longitude").between(-180.0, 180.0),
            F.lit("LONGITUDE_OUT_OF_RANGE"),
        )
        .otherwise(F.lit("INVALID_SPATIAL_INPUT"))
    )
    reason_text = (
        F.when(
            reason_code == "MISSING_OCCURRENCE_TIMESTAMP",
            F.lit("Crime occurrence timestamp is null."),
        )
        .when(
            reason_code == "NO_ELIGIBLE_ACS_VINTAGE",
            F.lit("No ACS/TIGER vintage was public on the crime date."),
        )
        .when(
            reason_code == "MISSING_COORDINATE",
            F.lit("Latitude or longitude is null."),
        )
        .when(
            reason_code == "NON_FINITE_COORDINATE",
            F.lit("Latitude or longitude is not finite."),
        )
        .when(
            reason_code == "LATITUDE_OUT_OF_RANGE",
            F.lit("Latitude is outside [-90, 90]."),
        )
        .when(
            reason_code == "LONGITUDE_OUT_OF_RANGE",
            F.lit("Longitude is outside [-180, 180]."),
        )
        .otherwise(F.lit("Spatial mapping input is invalid."))
    )

    payload_columns = [
        column
        for column in (
            "source_system",
            "source_city",
            "source_incident_id",
            "source_offense_id",
            "source_record_id",
            "source_row_hash",
            "occurred_at",
            "latitude",
            "longitude",
            "tiger_line_year",
        )
        if column in crime_with_calendar.columns
    ]
    raw_payload = F.to_json(F.struct(*[F.col(column) for column in payload_columns]))
    source_row_hash = F.coalesce(
        _source_column(
            crime_with_calendar,
            "source_row_hash",
            fallback=None,
        ),
        F.sha2(raw_payload, 256),
    )
    source_system = F.coalesce(
        _source_column(
            crime_with_calendar,
            "source_system",
            "source_city",
            fallback="crime",
        ),
        F.lit("crime"),
    )
    source_file = _source_column(
        crime_with_calendar,
        "source_file",
        fallback=None,
    ).cast("string")

    quarantine = (
        crime_with_calendar.filter(~valid)
        .withColumn("quarantine_reason_code", reason_code)
        .withColumn("quarantine_reason", reason_text)
        .withColumn("source_system", source_system.cast("string"))
        .withColumn("source_file", source_file)
        .withColumn("source_row_hash", source_row_hash)
        .withColumn("raw_payload", raw_payload)
        .withColumn(
            "quarantine_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.lit("location_tract_mapping"),
                    F.col("source_system"),
                    F.col("source_row_hash"),
                    F.col("quarantine_reason_code"),
                ),
                256,
            ),
        )
        .withColumn("pipeline_run_id", F.lit(pipeline_run_id))
        .withColumn("quarantined_at", F.current_timestamp())
        .select(
            "quarantine_id",
            "source_system",
            "source_file",
            "source_row_hash",
            "raw_payload",
            "quarantine_reason_code",
            "quarantine_reason",
            "pipeline_run_id",
            "quarantined_at",
            "occurred_at",
            "latitude",
            "longitude",
            "tiger_line_year",
        )
        .dropDuplicates(["quarantine_id"])
    )

    source_row_count = crime_with_calendar.count()
    invalid_row_count = crime_with_calendar.filter(~valid).count()
    return LocationFrames(
        candidates=candidates,
        quarantine=quarantine,
        source_row_count=source_row_count,
        invalid_row_count=invalid_row_count,
    )


def attach_boundary_source_version(
    candidate_locations: DataFrame,
    boundary_dataframe: DataFrame,
    *,
    boundary_definition_version: str,
) -> DataFrame:
    """Attach the one active source checksum for each requested vintage."""

    from pyspark.sql import functions as F

    active_boundaries = boundary_dataframe.filter(
        F.col("boundary_definition_version") == boundary_definition_version
    )
    metadata_counts = active_boundaries.groupBy("boundary_vintage").agg(
        F.countDistinct("boundary_definition_version").alias("_definition_count"),
        F.countDistinct("source_archive_sha256").alias("_archive_count"),
        F.first(
            "boundary_definition_version",
            ignorenulls=True,
        ).alias("boundary_definition_version"),
        F.first(
            "source_archive_sha256",
            ignorenulls=True,
        ).alias("source_archive_sha256"),
    )
    invalid_metadata = (
        metadata_counts.filter(
            (F.col("_definition_count") != 1) | (F.col("_archive_count") != 1)
        )
        .limit(1)
        .count()
    )
    if invalid_metadata:
        raise ValueError(
            "Each boundary vintage must have exactly one definition "
            "version and source archive checksum."
        )

    enriched = (
        candidate_locations.alias("location")
        .join(
            metadata_counts.alias("metadata"),
            F.col("location.tiger_line_year") == F.col("metadata.boundary_vintage"),
            "left",
        )
        .select(
            *[F.col(f"location.{column}") for column in LOCATION_KEY_COLUMNS],
            F.col("metadata.boundary_definition_version"),
            F.col("metadata.source_archive_sha256"),
        )
    )
    missing_versions = (
        enriched.filter(
            F.col("boundary_definition_version").isNull()
            | F.col("source_archive_sha256").isNull()
        )
        .select("tiger_line_year")
        .distinct()
        .collect()
    )
    if missing_versions:
        years = sorted(int(row["tiger_line_year"]) for row in missing_versions)
        raise ValueError(
            "No active boundary source exists for TIGER/Line years "
            f"{years} and definition {boundary_definition_version!r}."
        )
    return enriched


def validate_spatial_boundary_inputs(
    boundary_dataframe: DataFrame,
    *,
    boundary_definition_version: str,
) -> None:
    """Defensively validate the active boundary slice before a spatial join."""

    from pyspark.sql import functions as F

    required_columns = {
        "boundary_vintage",
        "geoid",
        "tract_geometry",
        "boundary_definition_version",
        "source_archive_sha256",
    }
    missing_columns = sorted(required_columns - set(boundary_dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Census boundary table is missing mapping columns: {missing_columns}."
        )

    active = boundary_dataframe.filter(
        F.col("boundary_definition_version") == boundary_definition_version
    )
    if active.isEmpty():
        raise ValueError(
            "Census boundary table has no rows for definition "
            f"{boundary_definition_version!r}."
        )
    duplicate_keys = (
        active.groupBy("boundary_vintage", "geoid")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    tract_srid = F.expr("ST_SRID(tract_geometry)")
    invalid_rows = (
        active.filter(
            F.col("boundary_vintage").isNull()
            | F.col("geoid").isNull()
            | ~F.col("geoid").rlike(r"^[0-9]{11}$")
            | F.col("tract_geometry").isNull()
            | F.col("source_archive_sha256").isNull()
            | (F.length("source_archive_sha256") != 64)
            | tract_srid.isNull()
            | (tract_srid != 4326)
        )
        .limit(1)
        .count()
    )
    if duplicate_keys:
        raise ValueError(
            "Active Census boundaries contain duplicate (boundary_vintage, geoid) keys."
        )
    if invalid_rows:
        raise ValueError(
            "Active Census boundaries contain invalid keys, geometry, "
            "source checksums, or SRIDs."
        )


def _spark_location_key(
    *,
    mapping_definition_version: str,
) -> Column:
    from pyspark.sql import functions as F

    return F.sha2(
        F.to_json(
            F.struct(
                F.lit(mapping_definition_version).alias("mapping_definition_version"),
                F.col("boundary_definition_version"),
                F.col("tiger_line_year").cast("string").alias("tiger_line_year"),
                F.format_string(
                    f"%.{_COORDINATE_DECIMAL_PLACES}f",
                    F.col("latitude"),
                ).alias("latitude"),
                F.format_string(
                    f"%.{_COORDINATE_DECIMAL_PLACES}f",
                    F.col("longitude"),
                ).alias("longitude"),
                F.col("source_archive_sha256"),
            )
        ),
        256,
    )


def spatially_map_locations(
    candidate_locations: DataFrame,
    boundary_dataframe: DataFrame,
    *,
    boundary_definition_version: str,
    mapping_definition_version: str,
    pipeline_run_id: str,
) -> DataFrame:
    """Map each valid location with ST_Contains/ST_Covers and expose ambiguity."""

    from pyspark.sql import functions as F

    if not mapping_definition_version.strip():
        raise ValueError("mapping_definition_version cannot be blank.")

    enriched_locations = attach_boundary_source_version(
        candidate_locations,
        boundary_dataframe,
        boundary_definition_version=boundary_definition_version,
    ).withColumn(
        "_crime_point",
        F.expr("ST_Point(longitude, latitude, 4326)"),
    )
    tracts = (
        boundary_dataframe.filter(
            F.col("boundary_definition_version") == boundary_definition_version
        )
        .filter(F.col("tract_geometry").isNotNull())
        .select(
            F.col("boundary_vintage").alias("_boundary_vintage"),
            F.col("geoid").alias("_candidate_geoid"),
            "tract_geometry",
        )
    )
    candidates = (
        enriched_locations.join(
            tracts,
            F.col("tiger_line_year") == F.col("_boundary_vintage"),
            "inner",
        )
        .filter(F.expr("ST_Covers(tract_geometry, _crime_point)"))
        .select(
            *LOCATION_KEY_COLUMNS,
            "boundary_definition_version",
            "source_archive_sha256",
            "_candidate_geoid",
            F.expr("ST_Contains(tract_geometry, _crime_point)").alias("_contains"),
        )
    )
    grouping_columns = [
        *LOCATION_KEY_COLUMNS,
        "boundary_definition_version",
        "source_archive_sha256",
    ]
    aggregated_matches = candidates.groupBy(*grouping_columns).agg(
        F.countDistinct("_candidate_geoid").alias("candidate_match_count"),
        F.min("_candidate_geoid").alias("_selected_geoid"),
        F.max(F.col("_contains").cast("int")).alias("_contains_match"),
    )

    return (
        enriched_locations.drop("_crime_point")
        .join(
            aggregated_matches,
            on=grouping_columns,
            how="left",
        )
        .withColumn(
            "candidate_match_count",
            F.coalesce(
                F.col("candidate_match_count"),
                F.lit(0),
            ).cast("int"),
        )
        .withColumn(
            "tract_geoid",
            F.when(
                F.col("candidate_match_count") == 1,
                F.col("_selected_geoid"),
            ).cast("string"),
        )
        .withColumn(
            "match_status",
            F.when(
                F.col("candidate_match_count") == 0,
                F.lit(UNMATCHED),
            )
            .when(
                F.col("candidate_match_count") > 1,
                F.lit(AMBIGUOUS),
            )
            .when(
                F.col("_contains_match") == 1,
                F.lit(MATCHED_CONTAINS),
            )
            .otherwise(F.lit(MATCHED_COVERS)),
        )
        .withColumn(
            "mapping_definition_version",
            F.lit(mapping_definition_version),
        )
        .withColumn(
            "location_tract_key",
            _spark_location_key(
                mapping_definition_version=mapping_definition_version,
            ),
        )
        .withColumn("pipeline_run_id", F.lit(pipeline_run_id))
        .withColumn("mapped_at", F.current_timestamp())
        .drop("_selected_geoid", "_contains_match")
    )


def mapping_issues_to_quarantine(
    mapping_dataframe: DataFrame,
) -> DataFrame:
    """Convert unmatched/ambiguous mapping results into stable quarantine rows."""

    from pyspark.sql import functions as F

    issue_rows = mapping_dataframe.filter(
        F.col("match_status").isin(UNMATCHED, AMBIGUOUS)
    )
    reason_code = F.when(
        F.col("match_status") == AMBIGUOUS,
        F.lit("AMBIGUOUS_TRACT_MATCH"),
    ).otherwise(F.lit("NO_TRACT_MATCH"))
    reason = F.when(
        F.col("match_status") == AMBIGUOUS,
        F.concat(
            F.lit("Coordinate was covered by "),
            F.col("candidate_match_count").cast("string"),
            F.lit(" tract geometries."),
        ),
    ).otherwise(F.lit("Coordinate was not covered by a tract geometry."))
    raw_payload = F.to_json(
        F.struct(
            "tiger_line_year",
            "latitude",
            "longitude",
            "boundary_definition_version",
            "source_archive_sha256",
            "mapping_definition_version",
            "candidate_match_count",
        )
    )
    return (
        issue_rows.withColumn(
            "quarantine_reason_code",
            reason_code,
        )
        .withColumn("quarantine_reason", reason)
        .withColumn("source_system", F.lit("census_tiger_line"))
        .withColumn("source_file", F.lit(None).cast("string"))
        .withColumn(
            "source_row_hash",
            F.col("location_tract_key"),
        )
        .withColumn("raw_payload", raw_payload)
        .withColumn(
            "quarantine_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.lit("location_tract_mapping"),
                    F.col("location_tract_key"),
                    F.col("quarantine_reason_code"),
                ),
                256,
            ),
        )
        .withColumnRenamed("mapped_at", "quarantined_at")
        .select(
            "quarantine_id",
            "source_system",
            "source_file",
            "source_row_hash",
            "raw_payload",
            "quarantine_reason_code",
            "quarantine_reason",
            "pipeline_run_id",
            "quarantined_at",
            F.lit(None).cast("timestamp").alias("occurred_at"),
            "latitude",
            "longitude",
            "tiger_line_year",
        )
        .dropDuplicates(["quarantine_id"])
    )


def validate_mapping_dataframe(
    mapping_dataframe: DataFrame,
    *,
    expected_locations: DataFrame,
    maximum_ambiguous_matches: int,
    maximum_unmatched_rate: float,
) -> None:
    """Validate exact key preservation, uniqueness, statuses, and coverage."""

    from pyspark.sql import functions as F

    if maximum_ambiguous_matches < 0:
        raise ValueError("maximum_ambiguous_matches cannot be negative.")
    if not 0.0 <= maximum_unmatched_rate <= 1.0:
        raise ValueError("maximum_unmatched_rate must be between 0 and 1.")

    required_columns = {
        *LOCATION_KEY_COLUMNS,
        "tract_geoid",
        "boundary_definition_version",
        "source_archive_sha256",
        "mapping_definition_version",
        "location_tract_key",
        "match_status",
        "candidate_match_count",
    }
    missing_columns = sorted(required_columns - set(mapping_dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Location mapping is missing required columns: {missing_columns}."
        )

    duplicate_locations = (
        mapping_dataframe.groupBy(*LOCATION_KEY_COLUMNS)
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    duplicate_mapping_ids = (
        mapping_dataframe.groupBy("location_tract_key")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    missing_keys = (
        expected_locations.select(*LOCATION_KEY_COLUMNS)
        .dropDuplicates()
        .join(
            mapping_dataframe.select(*LOCATION_KEY_COLUMNS),
            on=list(LOCATION_KEY_COLUMNS),
            how="left_anti",
        )
        .limit(1)
        .count()
    )
    unexpected_keys = (
        mapping_dataframe.select(*LOCATION_KEY_COLUMNS)
        .join(
            expected_locations.select(*LOCATION_KEY_COLUMNS).dropDuplicates(),
            on=list(LOCATION_KEY_COLUMNS),
            how="left_anti",
        )
        .limit(1)
        .count()
    )
    invalid_rows = (
        mapping_dataframe.filter(
            F.col("match_status").isNull()
            | ~F.col("match_status").isin(*sorted(MATCH_STATUSES))
            | F.col("location_tract_key").isNull()
            | (F.length("location_tract_key") != 64)
            | F.col("boundary_definition_version").isNull()
            | (F.length("boundary_definition_version") == 0)
            | F.col("source_archive_sha256").isNull()
            | (F.length("source_archive_sha256") != 64)
            | F.col("mapping_definition_version").isNull()
            | (F.length("mapping_definition_version") == 0)
            | F.col("tiger_line_year").isNull()
            | F.col("latitude").isNull()
            | F.col("longitude").isNull()
            | F.isnan("latitude")
            | F.isnan("longitude")
            | (F.abs(F.col("latitude")) == float("inf"))
            | (F.abs(F.col("longitude")) == float("inf"))
            | ~F.col("latitude").between(-90.0, 90.0)
            | ~F.col("longitude").between(-180.0, 180.0)
            | F.col("candidate_match_count").isNull()
            | (F.col("candidate_match_count") < 0)
            | (
                F.col("match_status").isin(
                    MATCHED_CONTAINS,
                    MATCHED_COVERS,
                )
                & (F.col("candidate_match_count") != 1)
            )
            | (
                (F.col("match_status") == UNMATCHED)
                & (F.col("candidate_match_count") != 0)
            )
            | (
                (F.col("match_status") == AMBIGUOUS)
                & (F.col("candidate_match_count") <= 1)
            )
            | (
                F.col("tract_geoid").isNull()
                & F.col("match_status").isin(
                    MATCHED_CONTAINS,
                    MATCHED_COVERS,
                )
            )
            | (
                F.col("tract_geoid").isNotNull()
                & F.col("match_status").isin(
                    UNMATCHED,
                    AMBIGUOUS,
                )
            )
        )
        .limit(1)
        .count()
    )
    metrics = mapping_dataframe.agg(
        F.count("*").alias("row_count"),
        F.sum((F.col("match_status") == AMBIGUOUS).cast("long")).alias(
            "ambiguous_count"
        ),
        F.sum((F.col("match_status") == UNMATCHED).cast("long")).alias(
            "unmatched_count"
        ),
    ).first()
    if metrics is None:
        raise ValueError("Could not calculate location mapping metrics.")
    row_count = int(metrics["row_count"] or 0)
    ambiguous_count = int(metrics["ambiguous_count"] or 0)
    unmatched_count = int(metrics["unmatched_count"] or 0)
    unmatched_rate = unmatched_count / row_count if row_count else 0.0

    failures: list[str] = []
    if duplicate_locations:
        failures.append("duplicate physical location keys")
    if duplicate_mapping_ids:
        failures.append("duplicate location_tract_key values")
    if missing_keys:
        failures.append("missing expected location keys")
    if unexpected_keys:
        failures.append("unexpected location keys")
    if invalid_rows:
        failures.append("invalid match status, metadata, or tract GEOID")
    if ambiguous_count > maximum_ambiguous_matches:
        failures.append(
            "ambiguous match threshold exceeded "
            f"({ambiguous_count}>{maximum_ambiguous_matches})"
        )
    if unmatched_rate > maximum_unmatched_rate:
        failures.append(
            "unmatched rate threshold exceeded "
            f"({unmatched_rate:.8f}>{maximum_unmatched_rate:.8f})"
        )
    if failures:
        raise ValueError(
            "Location-to-tract mapping validation failed: " + "; ".join(failures) + "."
        )


def select_stale_or_missing_locations(
    candidate_locations: DataFrame,
    boundary_dataframe: DataFrame,
    existing_mapping: DataFrame,
    *,
    boundary_definition_version: str,
    mapping_definition_version: str,
) -> DataFrame:
    """Select new or version-stale physical keys for recomputation."""

    from pyspark.sql import functions as F

    required_existing_columns = {
        *LOCATION_KEY_COLUMNS,
        "boundary_definition_version",
        "mapping_definition_version",
        "source_archive_sha256",
    }
    missing = sorted(required_existing_columns - set(existing_mapping.columns))
    if missing:
        raise ValueError(
            "Existing location mapping predates version-aware keys. "
            "Run with --full-rebuild to migrate it; missing columns="
            f"{missing}."
        )

    current_candidates = attach_boundary_source_version(
        candidate_locations,
        boundary_dataframe,
        boundary_definition_version=boundary_definition_version,
    )
    joined = current_candidates.alias("candidate").join(
        existing_mapping.alias("existing"),
        on=(
            (F.col("candidate.tiger_line_year") == F.col("existing.tiger_line_year"))
            & (F.col("candidate.latitude") == F.col("existing.latitude"))
            & (F.col("candidate.longitude") == F.col("existing.longitude"))
        ),
        how="left",
    )
    return (
        joined.filter(
            F.col("existing.tiger_line_year").isNull()
            | ~F.col("existing.mapping_definition_version").eqNullSafe(
                F.lit(mapping_definition_version)
            )
            | ~F.col("existing.boundary_definition_version").eqNullSafe(
                F.lit(boundary_definition_version)
            )
            | ~F.col("existing.source_archive_sha256").eqNullSafe(
                F.col("candidate.source_archive_sha256")
            )
        )
        .select(*[F.col(f"candidate.{column}") for column in LOCATION_KEY_COLUMNS])
        .dropDuplicates(list(LOCATION_KEY_COLUMNS))
    )


def merge_spatial_quarantine(
    spark: SparkSession,
    dataframe: DataFrame,
    *,
    target_table: str,
    pipeline_run_id: str,
) -> None:
    """Insert one observation per stable issue and pipeline run."""

    from crimenet.config.validation import validate_qualified_table_name
    from crimenet.utils.promotion import staging_table_name

    validate_qualified_table_name(target_table)
    if dataframe.isEmpty():
        return

    stage = staging_table_name(
        target_table,
        f"{pipeline_run_id}_spatial_quarantine",
    )
    (
        dataframe.dropDuplicates(["quarantine_id", "pipeline_run_id"])
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(stage)
    )
    try:
        if not spark.catalog.tableExists(target_table):
            spark.sql(
                f"CREATE TABLE {target_table} USING DELTA AS SELECT * FROM {stage}"
            )
            return
        spark.sql(
            f"""
            MERGE INTO {target_table} AS target
            USING {stage} AS source
            ON target.quarantine_id = source.quarantine_id
            AND target.pipeline_run_id = source.pipeline_run_id
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {stage}")
