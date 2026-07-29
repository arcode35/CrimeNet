from __future__ import annotations

import pytest

from crimenet.utils.promotion import (
    drop_staging_table,
    normalize_pipeline_run_id,
    promote_staged_delta_table,
    staging_table_name,
)


class RecordingSpark:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def sql(self, statement: str) -> None:
        self.statements.append(statement)


def test_staging_name_is_scoped_to_target_and_run() -> None:
    assert staging_table_name(
        "catalog.gold.crime_features",
        "job/123 attempt:2",
    ) == ("catalog.gold.crime_features__staging__job_123_attempt_2")


def test_invalid_staging_identifiers_fail_early() -> None:
    with pytest.raises(ValueError, match="qualified"):
        staging_table_name("crime_features", "run-1")

    with pytest.raises(
        ValueError,
        match="alphanumeric|letter or number",
    ):
        normalize_pipeline_run_id("!!!")


def test_promotion_and_cleanup_quote_identifiers() -> None:
    spark = RecordingSpark()

    promote_staged_delta_table(
        spark,
        staging_table=("catalog.gold.crime_features__staging__run_1"),
        target_table="catalog.gold.crime_features",
    )
    drop_staging_table(
        spark,
        "catalog.gold.crime_features__staging__run_1",
    )

    assert spark.statements == [
        (
            "CREATE OR REPLACE TABLE "
            "`catalog`.`gold`.`crime_features` DEEP CLONE "
            "`catalog`.`gold`."
            "`crime_features__staging__run_1`"
        ),
        ("DROP TABLE IF EXISTS `catalog`.`gold`.`crime_features__staging__run_1`"),
    ]
