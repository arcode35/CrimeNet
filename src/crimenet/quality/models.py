"""Structured results for runtime data-quality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QualityCheckResult:
    """Outcome of one named quality check."""

    dataset: str
    check_name: str
    passed: bool
    message: str
    blocking: bool = True
    failed_count: int = 0
    evaluated_count: int | None = None
    metric_value: float | int | str | None = None
    threshold: float | int | str | None = None
    examples: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.failed_count < 0:
            raise ValueError("failed_count must be nonnegative")
        if self.evaluated_count is not None and self.evaluated_count < 0:
            raise ValueError("evaluated_count must be nonnegative")
        if self.passed and self.failed_count:
            raise ValueError("A passing check cannot have failed rows")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation where values permit it."""
        return {
            "dataset": self.dataset,
            "check_name": self.check_name,
            "passed": self.passed,
            "blocking": self.blocking,
            "message": self.message,
            "failed_count": self.failed_count,
            "evaluated_count": self.evaluated_count,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class QualityReport:
    """Collection of quality-check outcomes for one dataset."""

    dataset: str
    checks: tuple[QualityCheckResult, ...]

    def __post_init__(self) -> None:
        wrong_dataset = [
            check.check_name
            for check in self.checks
            if check.dataset != self.dataset
        ]
        if wrong_dataset:
            raise ValueError(
                "All checks in a report must use its dataset name. "
                f"Mismatches: {wrong_dataset}"
            )

    @property
    def blocking_failures(self) -> tuple[QualityCheckResult, ...]:
        return tuple(
            check
            for check in self.checks
            if check.blocking and not check.passed
        )

    @property
    def warnings(self) -> tuple[QualityCheckResult, ...]:
        return tuple(
            check
            for check in self.checks
            if not check.blocking and not check.passed
        )

    @property
    def passed(self) -> bool:
        return not self.blocking_failures

    def raise_for_failures(self) -> None:
        """Raise when the report contains a blocking failure."""
        if self.blocking_failures:
            raise QualityValidationError(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }


class QualityValidationError(RuntimeError):
    """Raised when one or more blocking checks fail."""

    def __init__(self, report: QualityReport) -> None:
        self.report = report
        details = "; ".join(
            (
                f"{check.check_name}: {check.message} "
                f"(failed_count={check.failed_count}, "
                f"examples={list(check.examples)})"
            )
            for check in report.blocking_failures
        )
        super().__init__(
            f"{report.dataset} failed "
            f"{len(report.blocking_failures)} blocking quality check(s): "
            f"{details}"
        )
