"""Pure transformations for CrimeNet Silver weather and Gold environment data."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from numbers import Real
from typing import Any

import h3.api.basic_int as h3
import polars as pl
from pvlib import spa

WEATHER_CONTRACT_VERSION = "model_weather_v2"
WEATHER_PROVIDER = "open_meteo"
WEATHER_MODEL = "best_match"
WEATHER_MODEL_SELECTION_POLICY = "open_meteo_default_best_match"
WEATHER_H3_RESOLUTION = 6
OSM_H3_RESOLUTION = 9
WEATHER_ARCHIVE_LAG_DAYS = 7

SILVER_WEATHER_SCHEMA_VERSION = "silver_weather_v1"
ENVIRONMENTAL_SCHEMA_VERSION = "environmental_features_v1"

SILVER_WEATHER_SCHEMA = pl.Schema(
    {
        "h3_cell_id": pl.Int64,
        "hour": pl.Datetime("us", time_zone="UTC"),
        "weather_temperature_2m_c": pl.Float32,
        "weather_relative_humidity_2m_pct": pl.Float32,
        "weather_contract_version": pl.String,
        "weather_provider": pl.String,
        "weather_model": pl.String,
        "weather_model_selection_policy": pl.String,
        "weather_request_id": pl.String,
        "source_object_uri": pl.String,
    }
)

REQUIREMENT_KEY_SCHEMA = pl.Schema(
    {
        "h3_cell_id": pl.Int64,
        "hour": pl.Datetime("us", time_zone="UTC"),
        "event_reference_count": pl.Int64,
        "integration_reference_count": pl.Int64,
    }
)

LIGHTING_SCHEMA = pl.Schema(
    {
        "h3_cell_id": pl.Int64,
        "hour": pl.Datetime("us", time_zone="UTC"),
        "solar_elevation_deg": pl.Float32,
        "solar_zenith_deg": pl.Float32,
        "solar_azimuth_deg": pl.Float32,
        "lighting_condition": pl.String,
        "is_daylight": pl.Boolean,
    }
)

ENVIRONMENTAL_FEATURE_SCHEMA = pl.Schema(
    {
        "h3_cell_id": pl.Int64,
        "hour": pl.Datetime("us", time_zone="UTC"),
        "weather_temperature_2m_c": pl.Float32,
        "weather_relative_humidity_2m_pct": pl.Float32,
        "weather_available": pl.Boolean,
        "solar_elevation_deg": pl.Float32,
        "solar_zenith_deg": pl.Float32,
        "solar_azimuth_deg": pl.Float32,
        "lighting_condition": pl.String,
        "is_daylight": pl.Boolean,
        "event_reference_count": pl.Int64,
        "integration_reference_count": pl.Int64,
    }
)


class EnvironmentalContractError(RuntimeError):
    """An input violates the environmental feature contract."""


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentalContractError(f"{field} must be a JSON object")
    return value


def _require_nonempty_string(envelope: Mapping[str, Any], field: str) -> str:
    value = envelope.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EnvironmentalContractError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_numeric_array(values: object, *, field: str) -> list[float]:
    if not isinstance(values, list):
        raise EnvironmentalContractError(f"hourly.{field} must be an array")
    parsed: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EnvironmentalContractError(
                f"hourly.{field}[{index}] must be a finite number"
            )
        number = float(value)
        if not math.isfinite(number):
            raise EnvironmentalContractError(
                f"hourly.{field}[{index}] must be a finite number"
            )
        parsed.append(number)
    return parsed


def _parse_hour_axis(values: object) -> list[datetime]:
    if not isinstance(values, list) or not values:
        raise EnvironmentalContractError("hourly.time must be a non-empty array")
    if any(not isinstance(value, str) for value in values):
        raise EnvironmentalContractError("hourly.time values must be strings")
    if len(values) != len(set(values)):
        raise EnvironmentalContractError(
            "raw weather object contains duplicate hourly timestamps"
        )

    hours: list[datetime] = []
    for value in values:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M").replace(tzinfo=UTC)
        except ValueError as error:
            raise EnvironmentalContractError(
                f"invalid UTC hourly timestamp: {value!r}"
            ) from error
        hours.append(parsed)

    for previous, current in zip(hours, hours[1:], strict=False):
        if current - previous != timedelta(hours=1):
            raise EnvironmentalContractError(
                "raw weather hourly timestamps must be contiguous and increasing"
            )
    return hours


def normalize_weather_envelope(
    envelope: Mapping[str, Any],
    *,
    source_object_uri: str,
) -> pl.DataFrame:
    """Normalize one immutable Open-Meteo cell-year envelope to hourly rows."""

    expected_strings = {
        "weather_contract_version": WEATHER_CONTRACT_VERSION,
        "provider": WEATHER_PROVIDER,
        "model": WEATHER_MODEL,
        "model_selection_policy": WEATHER_MODEL_SELECTION_POLICY,
    }
    for field, expected in expected_strings.items():
        actual = _require_nonempty_string(envelope, field)
        if actual != expected:
            raise EnvironmentalContractError(
                f"unexpected {field}: expected={expected!r}, actual={actual!r}"
            )

    request_id = _require_nonempty_string(envelope, "request_id")
    if envelope.get("timezone") != "GMT" or envelope.get("utc_offset_seconds") != 0:
        raise EnvironmentalContractError(
            "model_weather_v2 timestamps must use GMT with utc_offset_seconds=0"
        )

    resolution = envelope.get("h3_resolution")
    cell = envelope.get("weather_query_cell_id")
    if isinstance(cell, bool) or not isinstance(cell, int):
        raise EnvironmentalContractError("weather_query_cell_id must be an integer")
    if resolution != WEATHER_H3_RESOLUTION:
        raise EnvironmentalContractError(
            f"h3_resolution must equal {WEATHER_H3_RESOLUTION}"
        )
    if not h3.is_valid_cell(cell) or h3.get_resolution(cell) != WEATHER_H3_RESOLUTION:
        raise EnvironmentalContractError(
            f"weather_query_cell_id is not a valid H3-r6 cell: {cell!r}"
        )

    hourly = _require_mapping(envelope.get("hourly"), field="hourly")
    units = _require_mapping(envelope.get("hourly_units"), field="hourly_units")
    variables = envelope.get("hourly_variables")
    expected_variables = {"temperature_2m", "relative_humidity_2m"}
    if (
        not isinstance(variables, list)
        or any(not isinstance(value, str) for value in variables)
        or len(variables) != len(set(variables))
        or set(variables) != expected_variables
    ):
        raise EnvironmentalContractError(
            "hourly_variables must contain exactly temperature_2m and "
            "relative_humidity_2m"
        )
    expected_units = {
        "time": "iso8601",
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
    }
    for field, expected in expected_units.items():
        if units.get(field) != expected:
            raise EnvironmentalContractError(
                f"unexpected unit for {field}: expected={expected!r}, "
                f"actual={units.get(field)!r}"
            )

    hours = _parse_hour_axis(hourly.get("time"))
    temperature = _validate_numeric_array(
        hourly.get("temperature_2m"), field="temperature_2m"
    )
    humidity = _validate_numeric_array(
        hourly.get("relative_humidity_2m"), field="relative_humidity_2m"
    )
    if len({len(hours), len(temperature), len(humidity)}) != 1:
        raise EnvironmentalContractError(
            "inconsistent raw hourly array lengths: "
            f"time={len(hours)}, temperature_2m={len(temperature)}, "
            f"relative_humidity_2m={len(humidity)}"
        )
    if any(value < 0.0 or value > 100.0 for value in humidity):
        raise EnvironmentalContractError(
            "relative_humidity_2m values must be percentages in [0, 100]"
        )

    try:
        start_date = date.fromisoformat(_require_nonempty_string(envelope, "start_date"))
        end_date = date.fromisoformat(_require_nonempty_string(envelope, "end_date"))
    except ValueError as error:
        raise EnvironmentalContractError("invalid start_date/end_date") from error
    expected_count = ((end_date - start_date).days + 1) * 24
    if end_date < start_date or len(hours) != expected_count:
        raise EnvironmentalContractError(
            "weather object does not contain every declared UTC hour: "
            f"expected={expected_count}, actual={len(hours)}"
        )
    expected_first = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    expected_last = datetime.combine(end_date, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=23
    )
    if hours[0] != expected_first or hours[-1] != expected_last:
        raise EnvironmentalContractError(
            "weather hour axis does not match declared start_date/end_date"
        )

    row_count = len(hours)
    return pl.DataFrame(
        {
            "h3_cell_id": pl.Series([cell] * row_count, dtype=pl.Int64),
            "hour": pl.Series(hours, dtype=pl.Datetime("us", time_zone="UTC")),
            "weather_temperature_2m_c": pl.Series(temperature, dtype=pl.Float32),
            "weather_relative_humidity_2m_pct": pl.Series(
                humidity, dtype=pl.Float32
            ),
            "weather_contract_version": [WEATHER_CONTRACT_VERSION] * row_count,
            "weather_provider": [WEATHER_PROVIDER] * row_count,
            "weather_model": [WEATHER_MODEL] * row_count,
            "weather_model_selection_policy": [
                WEATHER_MODEL_SELECTION_POLICY
            ]
            * row_count,
            "weather_request_id": [request_id] * row_count,
            "source_object_uri": [source_object_uri] * row_count,
        },
        schema=SILVER_WEATHER_SCHEMA,
    )


def _validate_r6_cells(cells: Sequence[object], *, label: str) -> None:
    invalid: list[object] = []
    for raw_cell in cells:
        try:
            cell = int(raw_cell)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            invalid.append(raw_cell)
            continue
        if (
            not h3.is_valid_cell(cell)
            or h3.get_resolution(cell) != WEATHER_H3_RESOLUTION
        ):
            invalid.append(raw_cell)
    if invalid:
        raise EnvironmentalContractError(
            f"{label} contains invalid/non-r6 H3 cells: {invalid[:10]}"
        )


def _duplicate_key_count(frame: pl.DataFrame) -> int:
    return frame.height - frame.select("h3_cell_id", "hour").unique().height


def validate_silver_weather_lazy(frame: pl.LazyFrame) -> dict[str, object]:
    """Validate Silver with bounded-memory aggregate and distinct-cell scans."""

    if frame.collect_schema() != SILVER_WEATHER_SCHEMA:
        raise EnvironmentalContractError(
            f"Silver weather schema mismatch: {frame.collect_schema()}"
        )
    summary = frame.select(
        pl.len().alias("row_count"),
        pl.struct("h3_cell_id", "hour").n_unique().alias("unique_key_count"),
        pl.col("h3_cell_id").n_unique().alias("unique_h3_cells"),
        pl.col("hour").min().alias("min_hour_utc"),
        pl.col("hour").max().alias("max_hour_utc"),
        (
            pl.any_horizontal(
                pl.col("h3_cell_id").is_null(),
                pl.col("hour").is_null(),
                pl.col("weather_temperature_2m_c").is_null(),
                pl.col("weather_relative_humidity_2m_pct").is_null(),
            )
            | ~pl.col("weather_temperature_2m_c").is_finite()
            | ~pl.col("weather_relative_humidity_2m_pct").is_finite()
            | ~pl.col("weather_relative_humidity_2m_pct").is_between(0.0, 100.0)
        )
        .sum()
        .alias("invalid_numeric_rows"),
        (
            (pl.col("hour").dt.minute() != 0)
            | (pl.col("hour").dt.second() != 0)
            | (pl.col("hour").dt.microsecond() != 0)
        )
        .sum()
        .alias("invalid_hour_rows"),
        (
            pl.any_horizontal(
                pl.col("weather_contract_version").is_null(),
                pl.col("weather_provider").is_null(),
                pl.col("weather_model").is_null(),
                pl.col("weather_model_selection_policy").is_null(),
                pl.col("weather_request_id").is_null(),
                pl.col("source_object_uri").is_null(),
            )
            | (pl.col("weather_contract_version") != WEATHER_CONTRACT_VERSION)
            | (pl.col("weather_provider") != WEATHER_PROVIDER)
            | (pl.col("weather_model") != WEATHER_MODEL)
            | (
                pl.col("weather_model_selection_policy")
                != WEATHER_MODEL_SELECTION_POLICY
            )
            | (pl.col("weather_request_id").str.len_chars() == 0)
            | (pl.col("source_object_uri").str.len_chars() == 0)
        )
        .sum()
        .alias("invalid_provenance_rows"),
    ).collect(engine="streaming").row(0, named=True)
    row_count = int(summary["row_count"])
    if row_count == 0:
        raise EnvironmentalContractError("Silver weather contains zero rows")
    duplicate_keys = row_count - int(summary["unique_key_count"])
    if duplicate_keys:
        raise EnvironmentalContractError(
            f"Silver weather contains duplicate (h3_cell_id, hour) keys: {duplicate_keys}"
        )
    cells = (
        frame.select("h3_cell_id")
        .unique()
        .collect(engine="streaming")
        .get_column("h3_cell_id")
        .to_list()
    )
    _validate_r6_cells(
        cells, label="Silver weather"
    )
    invalid_hours = int(summary["invalid_hour_rows"] or 0)
    if invalid_hours:
        raise EnvironmentalContractError(
            f"Silver weather contains non-hourly timestamps: {invalid_hours}"
        )
    invalid_numeric = int(summary["invalid_numeric_rows"] or 0)
    if invalid_numeric:
        raise EnvironmentalContractError(
            f"Silver weather contains invalid numeric rows: {invalid_numeric}"
        )
    invalid_provenance = int(summary["invalid_provenance_rows"] or 0)
    if invalid_provenance:
        raise EnvironmentalContractError(
            "Silver weather contains invalid provenance rows: "
            f"{invalid_provenance}"
        )
    return {
        "row_count": row_count,
        "unique_h3_cells": int(summary["unique_h3_cells"]),
        "min_hour_utc": summary["min_hour_utc"],
        "max_hour_utc": summary["max_hour_utc"],
        "duplicate_key_count": duplicate_keys,
    }


def validate_silver_weather(frame: pl.DataFrame) -> dict[str, object]:
    return validate_silver_weather_lazy(frame.lazy())


def build_r9_to_r6_mapping(cells: Sequence[int]) -> pl.DataFrame:
    unique_cells = sorted({int(cell) for cell in cells})
    invalid = [
        cell
        for cell in unique_cells
        if not h3.is_valid_cell(cell) or h3.get_resolution(cell) != OSM_H3_RESOLUTION
    ]
    if invalid:
        raise EnvironmentalContractError(
            f"integration cells contain invalid/non-r9 H3 IDs: {invalid[:10]}"
        )
    parents = [h3.cell_to_parent(cell, WEATHER_H3_RESOLUTION) for cell in unique_cells]
    return pl.DataFrame(
        {
            "osm_h3_cell_id": pl.Series(unique_cells, dtype=pl.Int64),
            "h3_cell_id": pl.Series(parents, dtype=pl.Int64),
        }
    )


def build_required_environmental_keys(
    *,
    events: pl.LazyFrame,
    integration: pl.LazyFrame,
) -> pl.LazyFrame:
    """Collapse model references to unique H3-r6 × UTC-hour keys and weights."""

    event_points = events.select(
        pl.col("weather_query_cell_id").cast(pl.Int64, strict=True),
        pl.col("occurrence_timestamp_utc").cast(
            pl.Datetime("us", time_zone="UTC"), strict=True
        ),
    )
    invalid_event_rows = int(
        event_points.select(
            pl.any_horizontal(
                pl.col("weather_query_cell_id").is_null(),
                pl.col("occurrence_timestamp_utc").is_null(),
            )
            .sum()
            .alias("invalid_rows")
        ).collect(engine="streaming")[0, "invalid_rows"]
        or 0
    )
    if invalid_event_rows:
        raise EnvironmentalContractError(
            "Event Spine contains null environmental requirement fields: "
            f"rows={invalid_event_rows}"
        )

    event_required = (
        event_points
        .with_columns(
            pl.col("weather_query_cell_id").alias("h3_cell_id"),
            pl.col("occurrence_timestamp_utc").dt.truncate("1h").alias("hour"),
        )
        .group_by("h3_cell_id", "hour")
        .agg(pl.len().cast(pl.Int64).alias("event_reference_count"))
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("integration_reference_count"))
        .select(REQUIREMENT_KEY_SCHEMA.names())
    )

    integration_points = integration.select(
        pl.col("osm_h3_cell_id").cast(pl.Int64, strict=True),
        pl.col("integration_timestamp_utc").cast(
            pl.Datetime("us", time_zone="UTC"), strict=True
        ),
    )
    integration_profile = integration_points.select(
        pl.col("osm_h3_cell_id")
        .drop_nulls()
        .unique()
        .sort()
        .implode()
        .alias("cells"),
        pl.any_horizontal(
            pl.col("osm_h3_cell_id").is_null(),
            pl.col("integration_timestamp_utc").is_null(),
        )
        .sum()
        .alias("invalid_rows"),
    ).collect(engine="streaming")
    invalid_integration_rows = int(integration_profile[0, "invalid_rows"] or 0)
    if invalid_integration_rows:
        raise EnvironmentalContractError(
            "Integration samples contain null environmental requirement fields: "
            f"rows={invalid_integration_rows}"
        )
    integration_cells = integration_profile[0, "cells"]
    mapping = build_r9_to_r6_mapping(integration_cells)
    integration_required = (
        integration_points
        .with_columns(
            pl.col("integration_timestamp_utc").dt.truncate("1h").alias("hour")
        )
        .group_by("osm_h3_cell_id", "hour")
        .agg(pl.len().cast(pl.Int64).alias("integration_reference_count"))
        .join(mapping.lazy(), on="osm_h3_cell_id", how="inner", validate="m:1")
        .group_by("h3_cell_id", "hour")
        .agg(pl.col("integration_reference_count").sum())
        .with_columns(pl.lit(0, dtype=pl.Int64).alias("event_reference_count"))
        .select(REQUIREMENT_KEY_SCHEMA.names())
    )

    return (
        pl.concat([event_required, integration_required], how="vertical_relaxed")
        .group_by("h3_cell_id", "hour")
        .agg(
            pl.col("event_reference_count").sum(),
            pl.col("integration_reference_count").sum(),
        )
        .select(REQUIREMENT_KEY_SCHEMA.names())
    )


def lighting_condition_expression(column: str = "solar_elevation_deg") -> pl.Expr:
    elevation = pl.col(column)
    return (
        pl.when(elevation >= 0.0)
        .then(pl.lit("day"))
        .when(elevation >= -6.0)
        .then(pl.lit("civil_twilight"))
        .when(elevation >= -12.0)
        .then(pl.lit("nautical_twilight"))
        .when(elevation >= -18.0)
        .then(pl.lit("astronomical_twilight"))
        .otherwise(pl.lit("night"))
    )


def classify_lighting(elevations: Sequence[float]) -> list[str]:
    frame = pl.DataFrame(
        {"solar_elevation_deg": pl.Series(elevations, dtype=pl.Float64)}
    ).with_columns(lighting_condition_expression().alias("lighting_condition"))
    return frame.get_column("lighting_condition").to_list()


def compute_lighting_features(keys: pl.DataFrame) -> pl.DataFrame:
    """Calculate pvlib NREL-SPA lighting once per unique H3-r6 × hour."""

    missing = {"h3_cell_id", "hour"} - set(keys.columns)
    if missing:
        raise EnvironmentalContractError(f"lighting keys missing columns: {sorted(missing)}")
    normalized = keys.select(
        pl.col("h3_cell_id").cast(pl.Int64, strict=True),
        pl.col("hour").cast(pl.Datetime("us", time_zone="UTC"), strict=True),
    )
    if sum(normalized.null_count().row(0)) != 0:
        raise EnvironmentalContractError("lighting keys contain nulls")
    duplicates = _duplicate_key_count(normalized)
    if duplicates:
        raise EnvironmentalContractError(f"lighting keys contain duplicates: {duplicates}")
    _validate_r6_cells(
        normalized.get_column("h3_cell_id").unique().to_list(), label="lighting keys"
    )

    parts: list[pl.DataFrame] = []
    for (cell,), group in normalized.sort("h3_cell_id", "hour").group_by(
        "h3_cell_id", maintain_order=True
    ):
        latitude, longitude = h3.cell_to_latlng(int(cell))
        unix_seconds = group.get_column("hour").dt.epoch("s").to_numpy()
        position = spa.solar_position(
            unix_seconds,
            float(latitude),
            float(longitude),
            0.0,
            1013.25,
            12.0,
            67.0,
            0.5667,
        )
        parts.append(
            group.with_columns(
                pl.Series("solar_elevation_deg", position[3], dtype=pl.Float32),
                pl.Series("solar_zenith_deg", position[0], dtype=pl.Float32),
                pl.Series("solar_azimuth_deg", position[4], dtype=pl.Float32),
            ).with_columns(
                lighting_condition_expression().alias("lighting_condition"),
                (pl.col("solar_elevation_deg") >= 0.0).alias("is_daylight"),
            )
        )
    if not parts:
        return pl.DataFrame(schema=LIGHTING_SCHEMA)
    return pl.concat(parts, how="vertical").cast(LIGHTING_SCHEMA).sort(
        "h3_cell_id", "hour"
    )


def build_environmental_features(
    *,
    requirements: pl.DataFrame,
    silver_weather: pl.DataFrame,
) -> pl.DataFrame:
    """Attach nullable weather and always-present deterministic lighting."""

    requirements = requirements.cast(REQUIREMENT_KEY_SCHEMA)
    if requirements.is_empty():
        raise EnvironmentalContractError("environmental requirement universe is empty")
    if _duplicate_key_count(requirements):
        raise EnvironmentalContractError("environmental requirement keys are duplicated")
    _validate_r6_cells(
        requirements.get_column("h3_cell_id").unique().to_list(),
        label="environmental requirements",
    )
    weather_columns = [
        "h3_cell_id",
        "hour",
        "weather_temperature_2m_c",
        "weather_relative_humidity_2m_pct",
    ]
    missing_weather_columns = set(weather_columns) - set(silver_weather.columns)
    if missing_weather_columns:
        raise EnvironmentalContractError(
            f"Silver weather join input is missing columns: {sorted(missing_weather_columns)}"
        )
    weather_join = silver_weather.select(weather_columns).cast(
        {
            "h3_cell_id": pl.Int64,
            "hour": pl.Datetime("us", time_zone="UTC"),
            "weather_temperature_2m_c": pl.Float32,
            "weather_relative_humidity_2m_pct": pl.Float32,
        }
    )
    if _duplicate_key_count(weather_join):
        raise EnvironmentalContractError("Silver weather join keys are duplicated")

    lighting = compute_lighting_features(requirements.select("h3_cell_id", "hour"))
    result = (
        requirements.join(lighting, on=["h3_cell_id", "hour"], how="left", validate="1:1")
        .join(
            weather_join,
            on=["h3_cell_id", "hour"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            (
                pl.col("weather_temperature_2m_c").is_not_null()
                & pl.col("weather_relative_humidity_2m_pct").is_not_null()
            ).alias("weather_available")
        )
        .select(ENVIRONMENTAL_FEATURE_SCHEMA.names())
        .cast(ENVIRONMENTAL_FEATURE_SCHEMA)
        .sort("h3_cell_id", "hour")
    )
    if result.height != requirements.height:
        raise EnvironmentalContractError(
            "environmental construction dropped requirement-domain rows"
        )
    return result


def archive_available_through_hour(
    *,
    today: date | None = None,
    lag_days: int = WEATHER_ARCHIVE_LAG_DAYS,
) -> datetime:
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative")
    cutoff_date = (today or date.today()) - timedelta(days=lag_days)
    return datetime.combine(cutoff_date, datetime.min.time(), tzinfo=UTC) + timedelta(
        hours=23
    )


def validate_environmental_features_lazy(
    frame: pl.LazyFrame,
    *,
    archive_cutoff_hour: datetime | None = None,
) -> dict[str, object]:
    if frame.collect_schema() != ENVIRONMENTAL_FEATURE_SCHEMA:
        raise EnvironmentalContractError(
            f"Gold environmental schema mismatch: {frame.collect_schema()}"
        )
    if archive_cutoff_hour is not None and archive_cutoff_hour.tzinfo is None:
        raise ValueError("archive_cutoff_hour must be timezone-aware")
    cutoff = (
        archive_cutoff_hour.astimezone(UTC)
        if archive_cutoff_hour is not None
        else None
    )
    weather_present = (
        pl.col("weather_temperature_2m_c").is_not_null()
        & pl.col("weather_relative_humidity_2m_pct").is_not_null()
    )
    missing_weather = ~pl.col("weather_available")
    partial_weather = (
        pl.col("weather_temperature_2m_c").is_null()
        != pl.col("weather_relative_humidity_2m_pct").is_null()
    )
    inconsistent_lighting = (
        (pl.col("lighting_condition") != lighting_condition_expression())
        | (pl.col("is_daylight") != (pl.col("solar_elevation_deg") >= 0.0))
    )
    unexpected_missing = (
        missing_weather & (pl.col("hour") <= pl.lit(cutoff))
        if cutoff is not None
        else pl.lit(False)
    )
    summary = frame.select(
        pl.len().alias("row_count"),
        pl.struct("h3_cell_id", "hour").n_unique().alias("unique_key_count"),
        pl.col("h3_cell_id").n_unique().alias("unique_h3_cells"),
        pl.col("hour").min().alias("min_hour_utc"),
        pl.col("hour").max().alias("max_hour_utc"),
        pl.any_horizontal(
            pl.col("h3_cell_id").is_null(),
            pl.col("hour").is_null(),
            pl.col("solar_elevation_deg").is_null(),
            pl.col("solar_zenith_deg").is_null(),
            pl.col("solar_azimuth_deg").is_null(),
            pl.col("lighting_condition").is_null(),
            pl.col("is_daylight").is_null(),
            ~pl.col("solar_elevation_deg").is_finite(),
            ~pl.col("solar_zenith_deg").is_finite(),
            ~pl.col("solar_azimuth_deg").is_finite(),
            ~pl.col("solar_elevation_deg").is_between(-90.0, 90.0),
            ~pl.col("solar_zenith_deg").is_between(0.0, 180.0),
            ~pl.col("solar_azimuth_deg").is_between(0.0, 360.0),
        )
        .sum()
        .alias("missing_lighting_rows"),
        inconsistent_lighting.sum().alias("inconsistent_lighting_rows"),
        (pl.col("weather_available") != weather_present)
        .sum()
        .alias("inconsistent_weather_rows"),
        partial_weather.sum().alias("partial_weather_rows"),
        pl.any_horizontal(
            pl.col("event_reference_count").is_null(),
            pl.col("integration_reference_count").is_null(),
            pl.col("event_reference_count") < 0,
            pl.col("integration_reference_count") < 0,
            (
                pl.col("event_reference_count")
                + pl.col("integration_reference_count")
            )
            <= 0,
        )
        .sum()
        .alias("invalid_reference_count_rows"),
        missing_weather.sum().alias("weather_null_rows"),
        unexpected_missing.sum().alias("unexpected_missing_rows"),
        pl.col("event_reference_count").sum().alias("event_reference_count"),
        pl.when(pl.col("weather_available"))
        .then(pl.col("event_reference_count"))
        .otherwise(0)
        .sum()
        .alias("covered_event_references"),
        pl.col("integration_reference_count")
        .sum()
        .alias("integration_reference_count"),
        pl.when(pl.col("weather_available"))
        .then(pl.col("integration_reference_count"))
        .otherwise(0)
        .sum()
        .alias("covered_integration_references"),
    ).collect(engine="streaming").row(0, named=True)
    row_count = int(summary["row_count"])
    if row_count == 0:
        raise EnvironmentalContractError("Gold environmental store contains zero rows")
    duplicate_keys = row_count - int(summary["unique_key_count"])
    if duplicate_keys:
        raise EnvironmentalContractError(
            f"Gold environmental store contains duplicate keys: {duplicate_keys}"
        )
    cells = (
        frame.select("h3_cell_id")
        .unique()
        .collect(engine="streaming")
        .get_column("h3_cell_id")
        .to_list()
    )
    _validate_r6_cells(cells, label="Gold environmental store")
    missing_lighting = int(summary["missing_lighting_rows"] or 0)
    if missing_lighting:
        raise EnvironmentalContractError(
            f"Gold environmental store has missing lighting rows: {missing_lighting}"
        )
    inconsistent_lighting_rows = int(summary["inconsistent_lighting_rows"] or 0)
    if inconsistent_lighting_rows:
        raise EnvironmentalContractError(
            "Gold environmental lighting classification is inconsistent: "
            f"{inconsistent_lighting_rows}"
        )
    inconsistent_weather_rows = int(summary["inconsistent_weather_rows"] or 0)
    if inconsistent_weather_rows:
        raise EnvironmentalContractError(
            "Gold environmental weather_available flag is inconsistent: "
            f"rows={inconsistent_weather_rows}"
        )
    partial_weather_rows = int(summary["partial_weather_rows"] or 0)
    if partial_weather_rows:
        raise EnvironmentalContractError(
            "Gold environmental weather fields are only partially present: "
            f"rows={partial_weather_rows}"
        )
    invalid_reference_count_rows = int(
        summary["invalid_reference_count_rows"] or 0
    )
    if invalid_reference_count_rows:
        raise EnvironmentalContractError(
            "Gold environmental reference counts are invalid: "
            f"rows={invalid_reference_count_rows}"
        )
    unexpected_missing_rows = int(summary["unexpected_missing_rows"] or 0)
    weather_null = int(summary["weather_null_rows"] or 0)
    total_event_refs = int(summary["event_reference_count"] or 0)
    total_integration_refs = int(summary["integration_reference_count"] or 0)
    covered_event_refs = int(summary["covered_event_references"] or 0)
    covered_integration_refs = int(summary["covered_integration_references"] or 0)
    total_refs = total_event_refs + total_integration_refs
    covered_refs = covered_event_refs + covered_integration_refs

    def pct(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 100.0

    return {
        "row_count": row_count,
        "unique_h3_cells": int(summary["unique_h3_cells"]),
        "min_hour_utc": summary["min_hour_utc"],
        "max_hour_utc": summary["max_hour_utc"],
        "duplicate_key_count": duplicate_keys,
        "missing_lighting_rows": 0,
        "inconsistent_lighting_rows": 0,
        "inconsistent_weather_rows": 0,
        "partial_weather_rows": 0,
        "invalid_reference_count_rows": 0,
        "weather_available_rows": row_count - weather_null,
        "weather_null_rows": weather_null,
        "weather_coverage_pct": pct(row_count - weather_null, row_count),
        "event_reference_count": total_event_refs,
        "event_weather_covered_references": covered_event_refs,
        "event_weighted_weather_coverage_pct": pct(
            covered_event_refs, total_event_refs
        ),
        "integration_reference_count": total_integration_refs,
        "integration_weather_covered_references": covered_integration_refs,
        "integration_weighted_weather_coverage_pct": pct(
            covered_integration_refs, total_integration_refs
        ),
        "point_reference_count": total_refs,
        "point_weather_covered_references": covered_refs,
        "point_weighted_weather_coverage_pct": pct(covered_refs, total_refs),
        "unexpected_archive_eligible_missing_rows": unexpected_missing_rows,
    }


def validate_environmental_features(
    frame: pl.DataFrame,
    *,
    archive_cutoff_hour: datetime | None = None,
) -> dict[str, object]:
    return validate_environmental_features_lazy(
        frame.lazy(), archive_cutoff_hour=archive_cutoff_hour
    )


__all__ = [
    "ENVIRONMENTAL_FEATURE_SCHEMA",
    "ENVIRONMENTAL_SCHEMA_VERSION",
    "EnvironmentalContractError",
    "LIGHTING_SCHEMA",
    "REQUIREMENT_KEY_SCHEMA",
    "SILVER_WEATHER_SCHEMA",
    "SILVER_WEATHER_SCHEMA_VERSION",
    "WEATHER_ARCHIVE_LAG_DAYS",
    "WEATHER_CONTRACT_VERSION",
    "archive_available_through_hour",
    "build_environmental_features",
    "build_r9_to_r6_mapping",
    "build_required_environmental_keys",
    "classify_lighting",
    "compute_lighting_features",
    "lighting_condition_expression",
    "normalize_weather_envelope",
    "validate_environmental_features",
    "validate_environmental_features_lazy",
    "validate_silver_weather",
    "validate_silver_weather_lazy",
]
