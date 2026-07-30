"""Reusable runtime validators for CrimeNet Silver and Gold datasets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from crimenet.contracts.lighting import (
    LIGHTING_DEFINITION_VERSION,
    LIGHTING_KEYS,
)
from crimenet.contracts.silver import SILVER_SCHEMA
from crimenet.contracts.weather import WEATHER_KEYS
from crimenet.quality.models import QualityCheckResult, QualityReport

SILVER_CRIME_DATASET = "silver_crime"
WEATHER_DATASET = "silver_weather"
SOCIOECONOMIC_DATASET = "silver_socioeconomic"
LIGHTING_DATASET = "silver_lighting"
GOLD_DATASET = "gold_crime_features"

SILVER_CRIME_REQUIRED_COLUMNS = (
    *(field.name for field in SILVER_SCHEMA.fields),
    "crime_offense_id",
)

WEATHER_REQUIRED_COLUMNS = (
    "provider",
    "model",
    "h3_resolution",
    "weather_query_cell_id",
    "weather_timestamp",
    "temperature_2m_c",
)

SOCIOECONOMIC_KEYS = ("geoid", "acs_vintage")
SOCIOECONOMIC_RATE_COLUMNS = (
    "poverty_rate",
    "unemployment_rate",
    "vacancy_rate",
    "renter_occupied_rate",
    "no_vehicle_rate",
)
SOCIOECONOMIC_REQUIRED_COLUMNS = (
    *SOCIOECONOMIC_KEYS,
    *SOCIOECONOMIC_RATE_COLUMNS,
)

LIGHTING_REQUIRED_COLUMNS = (
    *LIGHTING_KEYS,
    "query_latitude",
    "query_longitude",
    "solar_elevation_deg",
    "apparent_solar_elevation_deg",
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "lighting_condition",
    "is_daylight",
    "pvlib_version",
)

GOLD_WEATHER_LINEAGE_COLUMNS = (
    "weather_provider",
    "weather_model",
    "weather_h3_resolution",
    "weather_request_id",
    "weather_source_row_hash",
)
GOLD_LIGHTING_LINEAGE_COLUMNS = (
    "lighting_definition_version",
    "lighting_pvlib_version",
)
GOLD_REQUIRED_COLUMNS = (
    "crime_offense_id",
    "occurred_date",
    "selected_acs_vintage",
    "selected_acs_release_date",
    "tract_geoid",
    "socioeconomic_match_found",
    "weather_match_found",
    "lighting_match_found",
    *GOLD_WEATHER_LINEAGE_COLUMNS,
    *GOLD_LIGHTING_LINEAGE_COLUMNS,
)

VALID_SOURCE_CITIES = frozenset({"dallas", "fort_worth", "houston"})
VALID_LIGHTING_CONDITIONS = frozenset(
    {
        "daylight",
        "civil_twilight",
        "nautical_twilight",
        "astronomical_twilight",
        "night",
    }
)


@dataclass(frozen=True)
class GoldCoverageThresholds:
    """Minimum acceptable match rates for Gold enrichments."""

    tract: float = 0.0
    socioeconomic: float = 0.0
    weather: float = 0.0
    lighting: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("tract", self.tract),
            ("socioeconomic", self.socioeconomic),
            ("weather", self.weather),
            ("lighting", self.lighting),
        ):
            _validate_rate(value, name=f"{name} coverage threshold")


@dataclass(frozen=True)
class _ConditionSpec:
    check_name: str
    valid_condition: Column
    message: str
    example_columns: tuple[str, ...]
    blocking: bool = True


def _validate_rate(value: float, *, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _required_columns_result(
    dataframe: DataFrame,
    *,
    dataset: str,
    required_columns: Iterable[str],
) -> QualityCheckResult:
    missing = tuple(
        sorted(set(required_columns).difference(dataframe.columns))
    )
    return QualityCheckResult(
        dataset=dataset,
        check_name="required_columns",
        passed=not missing,
        message=(
            "All required columns are present."
            if not missing
            else f"Missing required columns: {list(missing)}"
        ),
        failed_count=len(missing),
        examples=(
            ({"missing_columns": list(missing)},)
            if missing
            else ()
        ),
    )


def _finish_report(
    dataset: str,
    checks: Sequence[QualityCheckResult],
    *,
    raise_on_failure: bool,
) -> QualityReport:
    report = QualityReport(dataset=dataset, checks=tuple(checks))
    if raise_on_failure:
        report.raise_for_failures()
    return report


def _collect_examples(
    dataframe: DataFrame,
    *,
    condition: Column,
    columns: Sequence[str],
    maximum_examples: int,
) -> tuple[dict[str, Any], ...]:
    selected_columns = [
        column for column in columns if column in dataframe.columns
    ]
    invalid = dataframe.filter(
        ~F.coalesce(condition.cast("boolean"), F.lit(False))
    )
    if selected_columns:
        invalid = invalid.select(*selected_columns)
    return tuple(
        row.asDict(recursive=True)
        for row in invalid.limit(maximum_examples).collect()
    )


def _condition_results(
    dataframe: DataFrame,
    *,
    dataset: str,
    specs: Sequence[_ConditionSpec],
    maximum_examples: int,
) -> tuple[list[QualityCheckResult], int]:
    aggregations: list[Column] = [F.count(F.lit(1)).alias("_total")]
    for index, spec in enumerate(specs):
        valid = F.coalesce(
            spec.valid_condition.cast("boolean"),
            F.lit(False),
        )
        aggregations.append(
            F.coalesce(
                F.sum(F.when(valid, F.lit(0)).otherwise(F.lit(1))),
                F.lit(0),
            ).alias(f"_failed_{index}")
        )

    summary = dataframe.agg(*aggregations).first()
    if summary is None:
        raise RuntimeError(f"Unable to aggregate quality checks for {dataset}")

    total = int(summary["_total"])
    results: list[QualityCheckResult] = []
    for index, spec in enumerate(specs):
        failed_count = int(summary[f"_failed_{index}"])
        examples = (
            _collect_examples(
                dataframe,
                condition=spec.valid_condition,
                columns=spec.example_columns,
                maximum_examples=maximum_examples,
            )
            if failed_count
            else ()
        )
        results.append(
            QualityCheckResult(
                dataset=dataset,
                check_name=spec.check_name,
                passed=failed_count == 0,
                blocking=spec.blocking,
                message=spec.message,
                failed_count=failed_count,
                evaluated_count=total,
                examples=examples,
            )
        )
    return results, total


def _unique_keys_result(
    dataframe: DataFrame,
    *,
    dataset: str,
    keys: Sequence[str],
    check_name: str,
    maximum_examples: int,
) -> QualityCheckResult:
    duplicates = (
        dataframe.groupBy(*keys)
        .count()
        .filter(F.col("count") > 1)
    )
    summary = duplicates.agg(
        F.count(F.lit(1)).alias("duplicate_groups"),
        F.coalesce(
            F.sum(F.col("count") - F.lit(1)),
            F.lit(0),
        ).alias("extra_rows"),
    ).first()
    if summary is None:
        raise RuntimeError(f"Unable to check unique keys for {dataset}")

    duplicate_groups = int(summary["duplicate_groups"])
    extra_rows = int(summary["extra_rows"])
    examples = tuple(
        row.asDict(recursive=True)
        for row in duplicates.select(*keys, "count")
        .limit(maximum_examples)
        .collect()
    )
    return QualityCheckResult(
        dataset=dataset,
        check_name=check_name,
        passed=duplicate_groups == 0,
        message=f"Keys {tuple(keys)} must be unique.",
        failed_count=extra_rows,
        metric_value=duplicate_groups,
        examples=examples,
    )


def _nonnull_key_condition(keys: Sequence[str]) -> Column:
    condition = F.lit(True)
    for key in keys:
        condition = condition & F.col(key).isNotNull()
    return condition


def _silver_schema_result(dataframe: DataFrame) -> QualityCheckResult:
    expected_fields = SILVER_SCHEMA.fields
    actual_by_name = {
        field.name: field.dataType.simpleString()
        for field in dataframe.schema.fields
    }
    mismatches = [
        (
            field.name,
            field.dataType.simpleString(),
            actual_by_name.get(field.name),
        )
        for field in expected_fields
        if actual_by_name.get(field.name) != field.dataType.simpleString()
    ]
    crime_id_type = actual_by_name.get("crime_offense_id")
    if crime_id_type != "string":
        mismatches.append(("crime_offense_id", "string", crime_id_type))

    canonical_order = [
        field.name for field in expected_fields
    ]
    actual_prefix = dataframe.columns[: len(canonical_order)]
    order_matches = actual_prefix == canonical_order
    examples: tuple[dict[str, Any], ...] = ()
    if mismatches or not order_matches:
        examples = (
            {
                "type_mismatches": [
                    {
                        "column": name,
                        "expected": expected,
                        "actual": actual,
                    }
                    for name, expected, actual in mismatches
                ],
                "expected_canonical_prefix": canonical_order,
                "actual_prefix": actual_prefix,
            },
        )
    return QualityCheckResult(
        dataset=SILVER_CRIME_DATASET,
        check_name="canonical_schema",
        passed=not mismatches and order_matches,
        message=(
            "Canonical Silver columns must retain their declared order and "
            "data types; crime_offense_id must be a string."
        ),
        failed_count=len(mismatches) + (not order_matches),
        examples=examples,
    )


def validate_silver_crime(
    dataframe: DataFrame,
    *,
    minimum_occurred_at_coverage: float | None = None,
    maximum_examples: int = 5,
    raise_on_failure: bool = True,
) -> QualityReport:
    """Validate canonical Silver crime identities, schema, and domains."""
    if maximum_examples < 1:
        raise ValueError("maximum_examples must be at least 1")
    if minimum_occurred_at_coverage is not None:
        _validate_rate(
            minimum_occurred_at_coverage,
            name="minimum_occurred_at_coverage",
        )

    required = _required_columns_result(
        dataframe,
        dataset=SILVER_CRIME_DATASET,
        required_columns=SILVER_CRIME_REQUIRED_COLUMNS,
    )
    checks = [required]
    if not required.passed:
        return _finish_report(
            SILVER_CRIME_DATASET,
            checks,
            raise_on_failure=raise_on_failure,
        )

    schema = _silver_schema_result(dataframe)
    checks.append(schema)
    if not schema.passed:
        return _finish_report(
            SILVER_CRIME_DATASET,
            checks,
            raise_on_failure=raise_on_failure,
        )

    coordinate_condition = (
        (
            F.col("latitude").isNull()
            & F.col("longitude").isNull()
        )
        | (
            F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
            & ~F.isnan("latitude")
            & ~F.isnan("longitude")
            & F.col("latitude").between(-90.0, 90.0)
            & F.col("longitude").between(-180.0, 180.0)
        )
    )
    specs = [
        _ConditionSpec(
            "nonnull_crime_offense_id",
            F.col("crime_offense_id").isNotNull(),
            "crime_offense_id must be non-null.",
            ("crime_offense_id", "source_city", "source_record_id"),
        ),
        _ConditionSpec(
            "recognized_source_city",
            F.col("source_city").isin(*sorted(VALID_SOURCE_CITIES)),
            f"source_city must be one of {sorted(VALID_SOURCE_CITIES)}.",
            ("crime_offense_id", "source_city"),
        ),
        _ConditionSpec(
            "nonnull_source_row_hash",
            F.col("source_row_hash").isNotNull()
            & (F.length(F.col("source_row_hash")) > 0),
            "source_row_hash must be non-null and non-empty.",
            ("crime_offense_id", "source_row_hash"),
        ),
        _ConditionSpec(
            "valid_coordinates",
            coordinate_condition,
            (
                "Coordinates must be either a null pair or a valid "
                "latitude/longitude pair."
            ),
            ("crime_offense_id", "latitude", "longitude"),
        ),
    ]
    condition_checks, total = _condition_results(
        dataframe,
        dataset=SILVER_CRIME_DATASET,
        specs=specs,
        maximum_examples=maximum_examples,
    )
    checks.extend(condition_checks)
    checks.append(
        _unique_keys_result(
            dataframe,
            dataset=SILVER_CRIME_DATASET,
            keys=("crime_offense_id",),
            check_name="unique_crime_offense_id",
            maximum_examples=maximum_examples,
        )
    )

    if minimum_occurred_at_coverage is not None:
        occurred_count = dataframe.filter(
            F.col("occurred_at").isNotNull()
        ).count()
        coverage = occurred_count / total if total else 0.0
        coverage_passed = coverage >= minimum_occurred_at_coverage
        checks.append(
            QualityCheckResult(
                dataset=SILVER_CRIME_DATASET,
                check_name="occurred_at_coverage",
                passed=coverage_passed,
                message="occurred_at coverage must meet the configured minimum.",
                failed_count=(
                    0
                    if coverage_passed
                    else max(total - occurred_count, 0)
                ),
                evaluated_count=total,
                metric_value=coverage,
                threshold=minimum_occurred_at_coverage,
                examples=(
                    _collect_examples(
                        dataframe,
                        condition=F.col("occurred_at").isNotNull(),
                        columns=("crime_offense_id", "occurred_at"),
                        maximum_examples=maximum_examples,
                    )
                    if not coverage_passed
                    else ()
                ),
            )
        )

    return _finish_report(
        SILVER_CRIME_DATASET,
        checks,
        raise_on_failure=raise_on_failure,
    )


def validate_weather(
    dataframe: DataFrame,
    *,
    allowed_providers: Iterable[str] = ("open_meteo",),
    allowed_models: Iterable[str] = ("era5_land",),
    allowed_h3_resolutions: Iterable[int] = (6,),
    minimum_temperature_c: float = -100.0,
    maximum_temperature_c: float = 70.0,
    maximum_examples: int = 5,
    raise_on_failure: bool = True,
) -> QualityReport:
    """Validate hourly weather keys, grain, source domains, and temperature."""
    providers = frozenset(allowed_providers)
    models = frozenset(allowed_models)
    resolutions = frozenset(allowed_h3_resolutions)
    if not providers or not models or not resolutions:
        raise ValueError("Allowed weather domains cannot be empty")
    if minimum_temperature_c >= maximum_temperature_c:
        raise ValueError(
            "minimum_temperature_c must be less than maximum_temperature_c"
        )

    required = _required_columns_result(
        dataframe,
        dataset=WEATHER_DATASET,
        required_columns=WEATHER_REQUIRED_COLUMNS,
    )
    checks = [required]
    if not required.passed:
        return _finish_report(
            WEATHER_DATASET,
            checks,
            raise_on_failure=raise_on_failure,
        )

    specs = [
        _ConditionSpec(
            "nonnull_weather_keys",
            _nonnull_key_condition(WEATHER_KEYS),
            f"Weather keys {WEATHER_KEYS} must be non-null.",
            WEATHER_KEYS,
        ),
        _ConditionSpec(
            "hour_aligned_weather_timestamp",
            F.col("weather_timestamp")
            == F.date_trunc("hour", F.col("weather_timestamp")),
            "weather_timestamp must be aligned exactly to the hour.",
            WEATHER_KEYS,
        ),
        _ConditionSpec(
            "recognized_weather_provider",
            F.col("provider").isin(*sorted(providers)),
            f"provider must be one of {sorted(providers)}.",
            ("provider", *WEATHER_KEYS),
        ),
        _ConditionSpec(
            "recognized_weather_model",
            F.col("model").isin(*sorted(models)),
            f"model must be one of {sorted(models)}.",
            ("model", *WEATHER_KEYS),
        ),
        _ConditionSpec(
            "recognized_weather_h3_resolution",
            F.col("h3_resolution").isin(*sorted(resolutions)),
            f"h3_resolution must be one of {sorted(resolutions)}.",
            ("h3_resolution", *WEATHER_KEYS),
        ),
        _ConditionSpec(
            "temperature_bounds",
            F.col("temperature_2m_c").isNull()
            | F.col("temperature_2m_c").between(
                minimum_temperature_c,
                maximum_temperature_c,
            ),
            (
                "Non-null temperature_2m_c values must be between "
                f"{minimum_temperature_c} and {maximum_temperature_c}."
            ),
            ("temperature_2m_c", *WEATHER_KEYS),
        ),
    ]
    condition_checks, _ = _condition_results(
        dataframe,
        dataset=WEATHER_DATASET,
        specs=specs,
        maximum_examples=maximum_examples,
    )
    checks.extend(condition_checks)
    checks.append(
        _unique_keys_result(
            dataframe,
            dataset=WEATHER_DATASET,
            keys=WEATHER_KEYS,
            check_name="unique_weather_keys",
            maximum_examples=maximum_examples,
        )
    )
    return _finish_report(
        WEATHER_DATASET,
        checks,
        raise_on_failure=raise_on_failure,
    )


def validate_socioeconomic(
    dataframe: DataFrame,
    *,
    minimum_acs_vintage: int = 2009,
    maximum_acs_vintage: int = 2100,
    maximum_examples: int = 5,
    raise_on_failure: bool = True,
) -> QualityReport:
    """Validate ACS tract keys, vintages, and derived rate domains."""
    if minimum_acs_vintage > maximum_acs_vintage:
        raise ValueError(
            "minimum_acs_vintage cannot exceed maximum_acs_vintage"
        )
    required = _required_columns_result(
        dataframe,
        dataset=SOCIOECONOMIC_DATASET,
        required_columns=SOCIOECONOMIC_REQUIRED_COLUMNS,
    )
    checks = [required]
    if not required.passed:
        return _finish_report(
            SOCIOECONOMIC_DATASET,
            checks,
            raise_on_failure=raise_on_failure,
        )

    specs = [
        _ConditionSpec(
            "nonnull_socioeconomic_keys",
            _nonnull_key_condition(SOCIOECONOMIC_KEYS),
            f"Socioeconomic keys {SOCIOECONOMIC_KEYS} must be non-null.",
            SOCIOECONOMIC_KEYS,
        ),
        _ConditionSpec(
            "valid_tract_geoid",
            F.col("geoid").rlike(r"^[0-9]{11}$"),
            "geoid must contain exactly 11 decimal digits.",
            SOCIOECONOMIC_KEYS,
        ),
        _ConditionSpec(
            "valid_acs_vintage",
            F.col("acs_vintage").between(
                minimum_acs_vintage,
                maximum_acs_vintage,
            ),
            (
                "acs_vintage must be between "
                f"{minimum_acs_vintage} and {maximum_acs_vintage}."
            ),
            SOCIOECONOMIC_KEYS,
        ),
    ]
    specs.extend(
        _ConditionSpec(
            f"{column_name}_domain",
            F.col(column_name).isNull()
            | F.col(column_name).between(0.0, 1.0),
            f"{column_name} must be null or between 0.0 and 1.0.",
            (*SOCIOECONOMIC_KEYS, column_name),
        )
        for column_name in SOCIOECONOMIC_RATE_COLUMNS
    )
    condition_checks, _ = _condition_results(
        dataframe,
        dataset=SOCIOECONOMIC_DATASET,
        specs=specs,
        maximum_examples=maximum_examples,
    )
    checks.extend(condition_checks)
    checks.append(
        _unique_keys_result(
            dataframe,
            dataset=SOCIOECONOMIC_DATASET,
            keys=SOCIOECONOMIC_KEYS,
            check_name="unique_socioeconomic_keys",
            maximum_examples=maximum_examples,
        )
    )
    return _finish_report(
        SOCIOECONOMIC_DATASET,
        checks,
        raise_on_failure=raise_on_failure,
    )


def _expected_lighting_condition() -> Column:
    elevation = F.col("solar_elevation_deg")
    return (
        F.when(elevation >= 0.0, F.lit("daylight"))
        .when(elevation >= -6.0, F.lit("civil_twilight"))
        .when(elevation >= -12.0, F.lit("nautical_twilight"))
        .when(elevation >= -18.0, F.lit("astronomical_twilight"))
        .otherwise(F.lit("night"))
    )


def validate_lighting(
    dataframe: DataFrame,
    *,
    active_definition_version: str = LIGHTING_DEFINITION_VERSION,
    maximum_examples: int = 5,
    raise_on_failure: bool = True,
) -> QualityReport:
    """Validate lighting key grain, version, solar domains, and labels."""
    if not active_definition_version:
        raise ValueError("active_definition_version cannot be empty")
    required = _required_columns_result(
        dataframe,
        dataset=LIGHTING_DATASET,
        required_columns=LIGHTING_REQUIRED_COLUMNS,
    )
    checks = [required]
    if not required.passed:
        return _finish_report(
            LIGHTING_DATASET,
            checks,
            raise_on_failure=raise_on_failure,
        )

    coordinates_valid = (
        F.col("query_latitude").isNotNull()
        & F.col("query_longitude").isNotNull()
        & ~F.isnan("query_latitude")
        & ~F.isnan("query_longitude")
        & F.col("query_latitude").between(-90.0, 90.0)
        & F.col("query_longitude").between(-180.0, 180.0)
    )
    classification_consistent = (
        (
            F.col("lighting_condition")
            == _expected_lighting_condition()
        )
        & (
            F.col("is_daylight")
            == (F.col("solar_elevation_deg") >= 0.0)
        )
    )
    specs = [
        _ConditionSpec(
            "nonnull_lighting_keys",
            _nonnull_key_condition(LIGHTING_KEYS),
            f"Lighting keys {LIGHTING_KEYS} must be non-null.",
            LIGHTING_KEYS,
        ),
        _ConditionSpec(
            "active_lighting_definition",
            F.col("lighting_definition_version")
            == active_definition_version,
            (
                "lighting_definition_version must equal the active version "
                f"{active_definition_version!r}."
            ),
            LIGHTING_KEYS,
        ),
        _ConditionSpec(
            "valid_lighting_coordinates",
            coordinates_valid,
            "Lighting query coordinates must be non-null and in range.",
            (*LIGHTING_KEYS, "query_latitude", "query_longitude"),
        ),
        _ConditionSpec(
            "solar_elevation_domain",
            F.col("solar_elevation_deg").isNotNull()
            & F.col("solar_elevation_deg").between(-90.0, 90.0),
            "solar_elevation_deg must be between -90.0 and 90.0.",
            (*LIGHTING_KEYS, "solar_elevation_deg"),
        ),
        _ConditionSpec(
            "apparent_solar_elevation_domain",
            F.col("apparent_solar_elevation_deg").isNotNull()
            & F.col("apparent_solar_elevation_deg").between(-90.0, 90.0),
            "apparent_solar_elevation_deg must be between -90.0 and 90.0.",
            (*LIGHTING_KEYS, "apparent_solar_elevation_deg"),
        ),
        _ConditionSpec(
            "solar_zenith_domain",
            F.col("solar_zenith_deg").isNotNull()
            & F.col("solar_zenith_deg").between(0.0, 180.0),
            "solar_zenith_deg must be between 0.0 and 180.0.",
            (*LIGHTING_KEYS, "solar_zenith_deg"),
        ),
        _ConditionSpec(
            "solar_azimuth_domain",
            F.col("solar_azimuth_deg").isNotNull()
            & F.col("solar_azimuth_deg").between(0.0, 360.0),
            "solar_azimuth_deg must be between 0.0 and 360.0.",
            (*LIGHTING_KEYS, "solar_azimuth_deg"),
        ),
        _ConditionSpec(
            "recognized_lighting_condition",
            F.col("lighting_condition").isin(
                *sorted(VALID_LIGHTING_CONDITIONS)
            ),
            (
                "lighting_condition must be one of "
                f"{sorted(VALID_LIGHTING_CONDITIONS)}."
            ),
            (*LIGHTING_KEYS, "lighting_condition"),
        ),
        _ConditionSpec(
            "lighting_classification_consistency",
            classification_consistent,
            (
                "lighting_condition and is_daylight must agree with "
                "solar_elevation_deg."
            ),
            (
                *LIGHTING_KEYS,
                "solar_elevation_deg",
                "lighting_condition",
                "is_daylight",
            ),
        ),
        _ConditionSpec(
            "nonnull_pvlib_version",
            F.col("pvlib_version").isNotNull()
            & (F.length(F.col("pvlib_version")) > 0),
            "pvlib_version must be non-null and non-empty.",
            (*LIGHTING_KEYS, "pvlib_version"),
        ),
    ]
    condition_checks, _ = _condition_results(
        dataframe,
        dataset=LIGHTING_DATASET,
        specs=specs,
        maximum_examples=maximum_examples,
    )
    checks.extend(condition_checks)
    checks.append(
        _unique_keys_result(
            dataframe,
            dataset=LIGHTING_DATASET,
            keys=LIGHTING_KEYS,
            check_name="unique_lighting_keys",
            maximum_examples=maximum_examples,
        )
    )
    return _finish_report(
        LIGHTING_DATASET,
        checks,
        raise_on_failure=raise_on_failure,
    )


def _coverage_result(
    dataframe: DataFrame,
    *,
    total: int,
    check_name: str,
    matched_condition: Column,
    threshold: float,
    maximum_examples: int,
) -> QualityCheckResult:
    matched = dataframe.filter(
        F.coalesce(matched_condition.cast("boolean"), F.lit(False))
    ).count()
    rate = matched / total if total else 0.0
    coverage_passed = rate >= threshold
    return QualityCheckResult(
        dataset=GOLD_DATASET,
        check_name=check_name,
        passed=coverage_passed,
        message=f"{check_name} must meet its configured minimum.",
        failed_count=(
            0 if coverage_passed else max(total - matched, 0)
        ),
        evaluated_count=total,
        metric_value=rate,
        threshold=threshold,
        examples=(
            _collect_examples(
                dataframe,
                condition=matched_condition,
                columns=("crime_offense_id",),
                maximum_examples=maximum_examples,
            )
            if not coverage_passed
            else ()
        ),
    )


def validate_gold(
    dataframe: DataFrame,
    *,
    source_crime_count: int,
    coverage_thresholds: GoldCoverageThresholds | None = None,
    maximum_examples: int = 5,
    raise_on_failure: bool = True,
) -> QualityReport:
    """Validate Gold cardinality, identity, leakage, lineage, and coverage."""
    if source_crime_count < 0:
        raise ValueError("source_crime_count must be nonnegative")
    thresholds = coverage_thresholds or GoldCoverageThresholds()
    required = _required_columns_result(
        dataframe,
        dataset=GOLD_DATASET,
        required_columns=GOLD_REQUIRED_COLUMNS,
    )
    checks = [required]
    if not required.passed:
        return _finish_report(
            GOLD_DATASET,
            checks,
            raise_on_failure=raise_on_failure,
        )

    match_flags_nonnull = (
        F.col("socioeconomic_match_found").isNotNull()
        & F.col("weather_match_found").isNotNull()
        & F.col("lighting_match_found").isNotNull()
    )
    acs_leakage_safe = (
        (
            F.col("selected_acs_vintage").isNull()
            & F.col("selected_acs_release_date").isNull()
        )
        | (
            F.col("selected_acs_vintage").isNotNull()
            & F.col("selected_acs_release_date").isNotNull()
            & F.col("occurred_date").isNotNull()
            & (
                F.col("selected_acs_release_date")
                < F.col("occurred_date")
            )
        )
    )
    weather_lineage = ~F.col("weather_match_found") | (
        _nonnull_key_condition(GOLD_WEATHER_LINEAGE_COLUMNS)
    )
    lighting_lineage = ~F.col("lighting_match_found") | (
        _nonnull_key_condition(GOLD_LIGHTING_LINEAGE_COLUMNS)
        & (
            F.col("lighting_definition_version")
            == LIGHTING_DEFINITION_VERSION
        )
    )
    specs = [
        _ConditionSpec(
            "nonnull_gold_crime_offense_id",
            F.col("crime_offense_id").isNotNull(),
            "Gold crime_offense_id must be non-null.",
            ("crime_offense_id",),
        ),
        _ConditionSpec(
            "calculable_match_metrics",
            match_flags_nonnull,
            "All Gold match indicators must be non-null booleans.",
            (
                "crime_offense_id",
                "socioeconomic_match_found",
                "weather_match_found",
                "lighting_match_found",
            ),
        ),
        _ConditionSpec(
            "leakage_safe_acs_dates",
            acs_leakage_safe,
            (
                "Selected ACS releases must precede the crime occurrence "
                "date; unmatched rows must not claim ACS release lineage."
            ),
            (
                "crime_offense_id",
                "occurred_date",
                "selected_acs_vintage",
                "selected_acs_release_date",
            ),
        ),
        _ConditionSpec(
            "weather_lineage_when_matched",
            weather_lineage,
            (
                "Matched weather rows must carry provider, model, H3 "
                "resolution, request ID, and source-row-hash lineage."
            ),
            (
                "crime_offense_id",
                "weather_match_found",
                *GOLD_WEATHER_LINEAGE_COLUMNS,
            ),
        ),
        _ConditionSpec(
            "lighting_lineage_when_matched",
            lighting_lineage,
            (
                "Matched lighting rows must carry the active definition "
                "version and pvlib version."
            ),
            (
                "crime_offense_id",
                "lighting_match_found",
                *GOLD_LIGHTING_LINEAGE_COLUMNS,
            ),
        ),
    ]
    condition_checks, total = _condition_results(
        dataframe,
        dataset=GOLD_DATASET,
        specs=specs,
        maximum_examples=maximum_examples,
    )
    checks.extend(condition_checks)
    checks.append(
        _unique_keys_result(
            dataframe,
            dataset=GOLD_DATASET,
            keys=("crime_offense_id",),
            check_name="unique_gold_crime_offense_id",
            maximum_examples=maximum_examples,
        )
    )
    checks.extend(
        [
            QualityCheckResult(
                dataset=GOLD_DATASET,
                check_name="source_row_cardinality",
                passed=total == source_crime_count,
                message=(
                    "Gold row count must equal the source Silver crime count."
                ),
                failed_count=abs(total - source_crime_count),
                evaluated_count=total,
                metric_value=total,
                threshold=source_crime_count,
            ),
            QualityCheckResult(
                dataset=GOLD_DATASET,
                check_name="no_join_multiplication",
                passed=total <= source_crime_count,
                message="Gold enrichment joins must not multiply crime rows.",
                failed_count=max(total - source_crime_count, 0),
                evaluated_count=total,
                metric_value=total,
                threshold=source_crime_count,
            ),
        ]
    )
    checks.extend(
        [
            _coverage_result(
                dataframe,
                total=total,
                check_name="tract_coverage",
                matched_condition=F.col("tract_geoid").isNotNull(),
                threshold=thresholds.tract,
                maximum_examples=maximum_examples,
            ),
            _coverage_result(
                dataframe,
                total=total,
                check_name="socioeconomic_coverage",
                matched_condition=F.col("socioeconomic_match_found"),
                threshold=thresholds.socioeconomic,
                maximum_examples=maximum_examples,
            ),
            _coverage_result(
                dataframe,
                total=total,
                check_name="weather_coverage",
                matched_condition=F.col("weather_match_found"),
                threshold=thresholds.weather,
                maximum_examples=maximum_examples,
            ),
            _coverage_result(
                dataframe,
                total=total,
                check_name="lighting_coverage",
                matched_condition=F.col("lighting_match_found"),
                threshold=thresholds.lighting,
                maximum_examples=maximum_examples,
            ),
        ]
    )
    return _finish_report(
        GOLD_DATASET,
        checks,
        raise_on_failure=raise_on_failure,
    )
