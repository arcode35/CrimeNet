"""Validation for identifiers and environment-provided thresholds."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUN_ID_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def validate_identifier(value: str, *, label: str = "identifier") -> str:
    """Return a safe unquoted Spark SQL identifier or raise."""
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {_IDENTIFIER_PATTERN.pattern!r}; "
            f"received {value!r}."
        )
    return value


def validate_qualified_table_name(value: str) -> str:
    """Validate a three-part Unity Catalog table name."""
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError(
            "qualified Unity Catalog table names must contain catalog, schema, "
            f"and table; received {value!r}."
        )
    for index, part in enumerate(parts):
        validate_identifier(part, label=f"table name component {index + 1}")
    return value


def normalize_pipeline_run_id(value: str | None) -> str:
    """Normalize an external run identifier for use in a staging name."""
    normalized = _RUN_ID_PATTERN.sub(
        "_",
        (value or uuid4().hex).strip(),
    ).strip("_")
    if not normalized:
        raise ValueError(
            "pipeline_run_id must contain at least one letter or number."
        )
    return normalized[:96]


def validate_rate(value: float, *, label: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1; received {value}.")
    return value


@dataclass(frozen=True)
class QualityThresholds:
    """Blocking quality thresholds supplied by each bundle target."""

    minimum_row_count: int = 1
    maximum_row_count: int = 2_000_000_000
    minimum_silver_to_bronze_ratio: float = 0.50
    maximum_silver_to_bronze_ratio: float = 1.05
    maximum_critical_null_rate: float = 0.0
    maximum_coordinate_null_rate: float = 1.0
    maximum_quarantine_rate: float = 0.05
    minimum_weather_coverage: float = 0.0
    minimum_lighting_coverage: float = 0.0
    minimum_tract_coverage: float = 0.0
    minimum_acs_coverage: float = 0.0

    def validate(self) -> QualityThresholds:
        if self.minimum_row_count < 0:
            raise ValueError("minimum_row_count cannot be negative.")
        if self.maximum_row_count < self.minimum_row_count:
            raise ValueError(
                "maximum_row_count must be at least minimum_row_count."
            )
        for field_name in (
            "minimum_silver_to_bronze_ratio",
            "maximum_critical_null_rate",
            "maximum_coordinate_null_rate",
            "maximum_quarantine_rate",
            "minimum_weather_coverage",
            "minimum_lighting_coverage",
            "minimum_tract_coverage",
            "minimum_acs_coverage",
        ):
            validate_rate(
                float(getattr(self, field_name)),
                label=field_name,
            )
        if self.maximum_silver_to_bronze_ratio < 0.0:
            raise ValueError(
                "maximum_silver_to_bronze_ratio cannot be negative."
            )
        if (
            self.maximum_silver_to_bronze_ratio
            < self.minimum_silver_to_bronze_ratio
        ):
            raise ValueError(
                "maximum_silver_to_bronze_ratio must be at least "
                "minimum_silver_to_bronze_ratio."
            )
        return self
