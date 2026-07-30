from __future__ import annotations

import pytest

from crimenet.quality import (
    QualityCheckResult,
    QualityReport,
    QualityValidationError,
)

pytestmark = pytest.mark.unit


def test_quality_report_exposes_blocking_failures_and_warnings() -> None:
    blocking = QualityCheckResult(
        dataset="example",
        check_name="blocking",
        passed=False,
        message="blocking failure",
        failed_count=1,
        examples=({"id": "bad"},),
    )
    warning = QualityCheckResult(
        dataset="example",
        check_name="warning",
        passed=False,
        blocking=False,
        message="non-blocking warning",
        failed_count=2,
    )
    report = QualityReport(
        dataset="example",
        checks=(blocking, warning),
    )

    assert not report.passed
    assert report.blocking_failures == (blocking,)
    assert report.warnings == (warning,)
    assert report.as_dict()["passed"] is False

    with pytest.raises(QualityValidationError) as error:
        report.raise_for_failures()

    assert error.value.report is report
    assert "blocking" in str(error.value)
    assert "examples=[{'id': 'bad'}]" in str(error.value)


def test_quality_result_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="passing check"):
        QualityCheckResult(
            dataset="example",
            check_name="impossible",
            passed=True,
            message="bad result",
            failed_count=1,
        )


def test_quality_report_rejects_mixed_dataset_results() -> None:
    result = QualityCheckResult(
        dataset="other",
        check_name="check",
        passed=True,
        message="passed",
    )
    with pytest.raises(ValueError, match="dataset name"):
        QualityReport(dataset="example", checks=(result,))
