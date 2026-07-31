"""Canonical Silver data-quality gates."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from crimenet.canonical.schema import (
    CANONICAL_NON_NULL_COLUMNS,
    validate_canonical_schema,
)


class CanonicalBuildError(
    RuntimeError
):
    """A hard canonical invariant failed."""


@dataclass(frozen=True)
class CityBuildMetrics:
    city: str
    rows: int
    distinct_record_ids: int
    null_record_ids: int
    duplicate_record_ids: int
    source_city_mismatches: int
    coordinate_pair_mismatches: int
    contract_violations: int


def _count_when(
    condition: object,
) -> object:
    return F.sum(
        F.when(
            condition,
            F.lit(1),
        )
        .otherwise(F.lit(0))
    ).cast("bigint")


def audit_city(
    dataframe: DataFrame,
    *,
    city: str,
) -> CityBuildMetrics:
    validate_canonical_schema(
        dataframe
    )

    record_id_valid = (
        F.col(
            "canonical_record_id"
        ).isNotNull()
        & (
            F.trim(
                F.col(
                    "canonical_record_id"
                )
            )
            != ""
        )
    )

    coordinate_mismatch = (
        (
            F.col("latitude").isNull()
            & F.col(
                "longitude"
            ).isNotNull()
        )
        | (
            F.col(
                "latitude"
            ).isNotNull()
            & F.col("longitude").isNull()
        )
    )

    contract_violation = F.lit(False)

    for column_name in sorted(
        CANONICAL_NON_NULL_COLUMNS
    ):
        condition = F.col(
            column_name
        ).isNull()

        if column_name in {
            "canonical_record_id",
            "source_city",
            "source_record_id",
            "canonical_schema_version",
            "key_definition_version",
            "offense_taxonomy_version",
        }:
            condition = (
                condition
                | (
                    F.trim(
                        F.col(
                            column_name
                        ).cast("string")
                    )
                    == ""
                )
            )

        contract_violation = (
            contract_violation
            | condition
        )

    result = (
        dataframe
        .agg(
            F.count(F.lit(1))
            .cast("bigint")
            .alias("rows"),

            F.countDistinct(
                "canonical_record_id"
            )
            .cast("bigint")
            .alias("distinct_ids"),

            _count_when(
                ~record_id_valid
            ).alias("null_ids"),

            _count_when(
                record_id_valid
            ).alias("nonblank_ids"),

            _count_when(
                ~F.col(
                    "source_city"
                ).eqNullSafe(
                    F.lit(city)
                )
            ).alias(
                "city_mismatches"
            ),

            _count_when(
                coordinate_mismatch
            ).alias(
                "coordinate_mismatches"
            ),

            _count_when(
                contract_violation
            ).alias(
                "contract_violations"
            ),
        )
        .first()
    )

    if result is None:
        raise CanonicalBuildError(
            f"No DQ result for {city}"
        )

    distinct_ids = int(
        result["distinct_ids"] or 0
    )

    nonblank_ids = int(
        result["nonblank_ids"] or 0
    )

    return CityBuildMetrics(
        city=city,
        rows=int(
            result["rows"] or 0
        ),
        distinct_record_ids=(
            distinct_ids
        ),
        null_record_ids=int(
            result["null_ids"] or 0
        ),
        duplicate_record_ids=(
            nonblank_ids
            - distinct_ids
        ),
        source_city_mismatches=int(
            result[
                "city_mismatches"
            ]
            or 0
        ),
        coordinate_pair_mismatches=int(
            result[
                "coordinate_mismatches"
            ]
            or 0
        ),
        contract_violations=int(
            result[
                "contract_violations"
            ]
            or 0
        ),
    )


def enforce_hard_gates(
    metrics: CityBuildMetrics,
) -> None:
    failures: list[str] = []

    if metrics.rows == 0:
        failures.append(
            "output contains zero rows"
        )

    if metrics.null_record_ids:
        failures.append(
            "null/blank canonical IDs: "
            f"{metrics.null_record_ids:,}"
        )

    if metrics.duplicate_record_ids:
        failures.append(
            "duplicate canonical IDs: "
            f"{metrics.duplicate_record_ids:,}"
        )

    if metrics.source_city_mismatches:
        failures.append(
            "source-city mismatches: "
            f"{metrics.source_city_mismatches:,}"
        )

    if metrics.coordinate_pair_mismatches:
        failures.append(
            "coordinate-pair mismatches: "
            f"{metrics.coordinate_pair_mismatches:,}"
        )

    if metrics.contract_violations:
        failures.append(
            "non-null contract violations: "
            f"{metrics.contract_violations:,}"
        )

    if failures:
        details = "\n  - ".join(
            failures
        )

        raise CanonicalBuildError(
            f"{metrics.city} failed DQ:\n"
            f"  - {details}"
        )