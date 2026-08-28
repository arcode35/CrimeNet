"""Frozen CrimeNet geographic CV and final all-city training contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import mean, median
from typing import Any

CANONICAL_GEOCV_VERSION = "crimenet_geocv_v1"
CANONICAL_MODELING_CITIES = frozenset(
    {
        "atlanta",
        "baltimore",
        "chandler_az",
        "chicago",
        "dallas",
        "denver",
        "fort_worth",
        "los_angeles_county_sheriff",
        "marin_county_sheriff_ca",
        "montgomery_county_md",
        "new_york",
        "san_francisco",
        "seattle",
        "sonoma_county_sheriff_ca",
        "washington_dc",
    }
)
CANONICAL_GEOGRAPHIC_FOLDS: dict[str, tuple[str, ...]] = {
    "bay_area": (
        "san_francisco",
        "marin_county_sheriff_ca",
        "sonoma_county_sheriff_ca",
    ),
    "mid_atlantic": (
        "washington_dc",
        "montgomery_county_md",
        "baltimore",
    ),
    "dfw_southwest": ("dallas", "fort_worth", "chandler_az"),
    "major_urban": ("new_york", "chicago", "atlanta"),
    "western_mixed": ("los_angeles_county_sheriff", "denver", "seattle"),
}


def validate_geographic_folds(
    folds: Mapping[str, list[str] | tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Fail closed unless configuration exactly matches the frozen v1 contract."""

    configured = {str(name): tuple(map(str, cities)) for name, cities in folds.items()}
    expected_names = tuple(CANONICAL_GEOGRAPHIC_FOLDS)
    if tuple(configured) != expected_names:
        raise ValueError(
            "Geographic folds must use the deterministic canonical order "
            f"{list(expected_names)}"
        )
    if len(configured) != 5 or any(len(cities) != 3 for cities in configured.values()):
        raise ValueError("Geographic CV requires exactly five folds of three cities")
    flattened = [city for cities in configured.values() for city in cities]
    duplicates = sorted({city for city in flattened if flattened.count(city) > 1})
    if duplicates:
        raise ValueError(f"Duplicate cities across geographic folds: {duplicates}")
    unknown = sorted(set(flattened) - CANONICAL_MODELING_CITIES)
    missing = sorted(CANONICAL_MODELING_CITIES - set(flattened))
    if unknown:
        raise ValueError(f"Unknown geographic-CV cities: {unknown}")
    if missing:
        raise ValueError(f"Missing canonical geographic-CV cities: {missing}")
    if configured != CANONICAL_GEOGRAPHIC_FOLDS:
        raise ValueError("Configured fold membership differs from crimenet_geocv_v1")
    return configured


def resolve_geographic_folds(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    geocv = config.get("geographic_cv")
    if not isinstance(geocv, Mapping) or not bool(geocv.get("enabled", False)):
        raise ValueError("Production HPO requires geographic_cv.enabled=true")
    if str(geocv.get("version", "")) != CANONICAL_GEOCV_VERSION:
        raise ValueError(f"geographic_cv.version must be {CANONICAL_GEOCV_VERSION}")
    folds = geocv.get("folds", CANONICAL_GEOGRAPHIC_FOLDS)
    if not isinstance(folds, Mapping):
        raise ValueError("geographic_cv.folds must be a mapping")
    return validate_geographic_folds(folds)


def validate_exact_modeling_cities(cities: list[str], *, label: str) -> None:
    actual = set(map(str, cities))
    missing = sorted(CANONICAL_MODELING_CITIES - actual)
    unknown = sorted(actual - CANONICAL_MODELING_CITIES)
    if missing or unknown:
        raise ValueError(f"{label} city contract mismatch: missing={missing}, unknown={unknown}")


def aggregate_intensity_oof(
    fold_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate original fold sufficient statistics without reweighting exposure."""

    folds = validate_geographic_folds(CANONICAL_GEOGRAPHIC_FOLDS)
    if tuple(fold_reports) != tuple(folds):
        raise ValueError("OOF reports are missing folds or are out of canonical order")
    city_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold_name, held_out in folds.items():
        report = fold_reports[fold_name]
        rows = list(report["per_city"])
        cities = {str(row["source_city"]) for row in rows}
        if cities != set(held_out):
            raise ValueError(
                f"{fold_name}: validation cities differ from held-out contract: {sorted(cities)}"
            )
        overlap = seen & cities
        if overlap:
            raise ValueError(f"Cities evaluated more than once: {sorted(overlap)}")
        seen.update(cities)
        for raw in rows:
            row = dict(raw)
            row["fold_name"] = fold_name
            row["exposure"] = float(row.get("exposure", row["total_exposure"]))
            if float(row["observed_events"]) <= 0:
                raise ValueError(f"{fold_name}/{row['source_city']}: zero observed events")
            if int(row["integration_rows"]) <= 0 or row["exposure"] <= 0:
                raise ValueError(
                    f"{fold_name}/{row['source_city']}: missing positive integration domain"
                )
            city_rows.append(row)
        pooled = report["global"]
        macro = report["macro_city"]
        fold_rows.append(
            {
                "fold_name": fold_name,
                "validation_city_count": len(cities),
                "macro_nll_per_event": float(macro["mean_nll_per_event"]),
                "pooled_nll_per_event": float(pooled["nll_per_event"]),
                "macro_bits_per_event": float(macro["mean_bits_per_event"]),
                "expected_observed_ratio": float(pooled["expected_observed_ratio"]),
                "calibration_error_pct": float(pooled["calibration_error_pct"]),
                "observed_events": float(pooled["observed_events"]),
                "expected_events": float(pooled["expected_events"]),
                "exposure": float(pooled["total_exposure"]),
                "nll": float(pooled["nll"]),
            }
        )
    validate_exact_modeling_cities(sorted(seen), label="OOF validation")
    if len(city_rows) != 15:
        raise ValueError("OOF validation must contain exactly fifteen city rows")
    total_nll = sum(float(row["nll"]) for row in fold_rows)
    total_observed = sum(float(row["observed_events"]) for row in fold_rows)
    total_expected = sum(float(row["expected_events"]) for row in fold_rows)
    total_exposure = sum(float(row["exposure"]) for row in fold_rows)
    if total_observed <= 0 or total_exposure <= 0:
        raise ValueError("OOF validation requires positive observed events and exposure")
    nll_values = [float(row["nll_per_event"]) for row in city_rows]
    bits_values = [float(row["bits_per_event"]) for row in city_rows]
    calibration = [abs(float(row["calibration_error_pct"])) for row in city_rows]
    values = [*nll_values, *bits_values, *calibration]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("OOF city metrics contain non-finite values")
    ratio = total_expected / total_observed
    metrics = {
        "geocv_macro_nll_per_event": mean(nll_values),
        "geocv_pooled_nll_per_event": total_nll / total_observed,
        "geocv_macro_bits_per_event": mean(bits_values),
        "geocv_median_city_nll_per_event": median(nll_values),
        "geocv_median_city_bits_per_event": median(bits_values),
        "geocv_mean_abs_calibration_error_pct": mean(calibration),
        "geocv_worst_city_bits_per_event": min(bits_values),
        "geocv_p10_city_bits_per_event": _percentile(bits_values, 10.0),
        "geocv_expected_observed_ratio": ratio,
        "geocv_calibration_error_pct": (ratio - 1.0) * 100.0,
        "geocv_total_nll": total_nll,
        "geocv_total_observed_events": total_observed,
        "geocv_total_expected_events": total_expected,
        "geocv_total_exposure": total_exposure,
        "total_oof_nll": total_nll,
        "total_oof_observed_events": total_observed,
        "total_oof_expected_events": total_expected,
        "total_oof_exposure": total_exposure,
    }
    return {"metrics": metrics, "folds": fold_rows, "cities": city_rows}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "CANONICAL_GEOCV_VERSION",
    "CANONICAL_GEOGRAPHIC_FOLDS",
    "CANONICAL_MODELING_CITIES",
    "aggregate_intensity_oof",
    "resolve_geographic_folds",
    "validate_exact_modeling_cities",
    "validate_geographic_folds",
]
