"""Stable identifiers and validation thresholds for Gold features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

GOLD_IDENTITY_VERSION = "crime_offense_id_v1"


def crime_offense_id_from_business_identity(
    business_identity: str,
) -> str:
    """Hash a logical Silver identity into the stable Gold identifier."""

    normalized_identity = business_identity.strip()
    if not normalized_identity:
        raise ValueError("Business identity must not be blank.")

    payload = f"{GOLD_IDENTITY_VERSION}||{normalized_identity}"
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GoldCoverageThresholds:
    """Minimum acceptable enrichment coverage rates."""

    weather: float = 0.0
    lighting: float = 0.0
    socioeconomic: float = 0.0
    tract: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("weather", self.weather),
            ("lighting", self.lighting),
            ("socioeconomic", self.socioeconomic),
            ("tract", self.tract),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} coverage threshold must be between 0 and 1; "
                    f"received {value}."
                )

    def as_metric_thresholds(self) -> dict[str, float]:
        return {
            "weather_match_rate": self.weather,
            "lighting_match_rate": self.lighting,
            "socioeconomic_match_rate": self.socioeconomic,
            "tract_match_rate": self.tract,
        }


def coverage_failures(
    metrics: Mapping[str, object],
    thresholds: GoldCoverageThresholds,
) -> tuple[str, ...]:
    """Return deterministic validation messages for failed coverage checks."""

    failures: list[str] = []

    for metric_name, minimum in thresholds.as_metric_thresholds().items():
        observed = metrics.get(metric_name)

        if observed is None:
            failures.append(f"{metric_name} is missing")
            continue

        if not isinstance(observed, (int, float, str)):
            failures.append(f"{metric_name} is not numeric: {observed!r}")
            continue

        try:
            observed_rate = float(observed)
        except ValueError:
            failures.append(f"{metric_name} is not numeric: {observed!r}")
            continue

        if not 0.0 <= observed_rate <= 1.0:
            failures.append(f"{metric_name} is outside [0, 1]: {observed_rate}")
        elif observed_rate < minimum:
            failures.append(
                f"{metric_name}={observed_rate:.8f} is below minimum={minimum:.8f}"
            )

    return tuple(failures)
