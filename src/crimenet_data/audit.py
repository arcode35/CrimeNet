#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq


# =============================================================================
# CONFIG
# =============================================================================

BUCKET = "crimenet-data"

STATIC_PREFIX = (
    "gold/national_feature_store/latest/h3_r9/"
)

ANNUAL_PREFIX = (
    "gold/national_feature_store/"
    "temporal/h3_r9/annual/"
)

OUTPUT_DIR = Path(
    "artifacts/national_feature_audit"
)

H3_CANDIDATES = (
    "osm_h3_cell_id",
    "h3_cell_id",
    "h3",
    "h3_r9",
    "h3_index",
)

STATE_CANDIDATES = (
    "state_fips",
    "STATEFP",
    "statefp",
)

SOCIO_FEATURES = (
    "socio_population",
    "socio_median_age",
    "socio_median_household_income",
    "socio_poverty_rate",
    "socio_unemployment_rate",
    "socio_vacancy_rate",
    "socio_renter_occupied_rate",
    "socio_no_vehicle_rate",
)


STATE_NAMES = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
}


# The three requested investigations.
TARGETS = {
    2022: None,               # all states
    2023: {"09"},             # Connecticut
    2026: {"32", "11", "04"}, # Nevada, DC, Arizona
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FeatureStats:
    denominator: int = 0
    rows_present: int = 0
    valid: int = 0
    null: int = 0
    nonfinite: int = 0

    @property
    def coverage_pct(self) -> float:
        return pct(
            self.valid,
            self.denominator,
        )

    @property
    def present_row_valid_pct(self) -> float:
        return pct(
            self.valid,
            self.rows_present,
        )


# =============================================================================
# HELPERS
# =============================================================================

def pct(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def normalize_h3(
    value,
) -> str | None:
    if value is None:
        return None

    return str(value)


def normalize_state(
    value,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    try:
        return f"{int(text):02d}"
    except ValueError:
        return text


def make_clients():
    endpoint = os.environ[
        "B2_ENDPOINT_URL"
    ]

    key_id = os.environ[
        "B2_KEY_ID"
    ]

    application_key = os.environ[
        "B2_APPLICATION_KEY"
    ]

    region = os.environ.get(
        "B2_REGION",
        "us-east-005",
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=(
            application_key
        ),
        region_name=region,
    )

    fs = pafs.S3FileSystem(
        access_key=key_id,
        secret_key=application_key,
        region=region,
        endpoint_override=endpoint,
        scheme="https",
    )

    return s3, fs


def list_parquet_keys(
    s3,
    prefix: str,
) -> list[str]:
    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    keys = []

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=prefix,
    ):
        for obj in page.get(
            "Contents",
            [],
        ):
            key = obj["Key"]

            if key.lower().endswith(
                ".parquet"
            ):
                keys.append(key)

    return sorted(keys)


def parquet_schema(
    fs,
    key: str,
) -> pa.Schema:
    path = f"{BUCKET}/{key}"

    with fs.open_input_file(
        path
    ) as f:
        return pq.ParquetFile(
            f
        ).schema_arrow


def find_column(
    schema: pa.Schema,
    candidates: tuple[str, ...],
) -> str:
    names = set(schema.names)

    for candidate in candidates:
        if candidate in names:
            return candidate

    raise KeyError(
        f"Could not find any of "
        f"{candidates} in schema:\n"
        f"{schema}"
    )


def validity(
    arr: pa.Array,
) -> tuple[
    pa.Array,
    pa.Array,
    pa.Array,
]:
    """
    Returns:

        valid_mask
        null_mask
        nonfinite_mask
    """

    null_mask = pc.is_null(
        arr
    )

    nonnull_mask = pc.invert(
        null_mask
    )

    if pa.types.is_floating(
        arr.type
    ):
        finite_mask = pc.is_finite(
            arr
        )

        nonfinite_mask = pc.and_(
            nonnull_mask,
            pc.invert(
                finite_mask
            ),
        )

        valid_mask = pc.and_(
            nonnull_mask,
            finite_mask,
        )

    else:
        nonfinite_mask = (
            pa.array(
                [False] * len(arr),
                type=pa.bool_(),
            )
        )

        valid_mask = (
            nonnull_mask
        )

    return (
        valid_mask,
        null_mask,
        nonfinite_mask,
    )


def count_true(
    mask: pa.Array,
) -> int:
    value = pc.sum(
        mask.cast(
            pa.int64()
        )
    )

    if not value.is_valid:
        return 0

    return int(
        value.as_py()
    )


# =============================================================================
# BUILD AUTHORITATIVE H3 -> STATE MAP
# =============================================================================

def build_h3_state_map(
    s3,
    fs,
):
    keys = list_parquet_keys(
        s3,
        STATIC_PREFIX,
    )

    if not keys:
        raise RuntimeError(
            f"No static feature shards "
            f"found under "
            f"s3://{BUCKET}/"
            f"{STATIC_PREFIX}"
        )

    schema = parquet_schema(
        fs,
        keys[0],
    )

    h3_col = find_column(
        schema,
        H3_CANDIDATES,
    )

    state_col = find_column(
        schema,
        STATE_CANDIDATES,
    )

    print("=" * 110)
    print(
        "BUILDING AUTHORITATIVE "
        "H3 -> STATE MAP"
    )
    print("=" * 110)

    print(
        f"H3 column:    {h3_col}"
    )
    print(
        f"State column: {state_col}"
    )
    print(
        f"Shards:       {len(keys):,}"
    )

    h3_to_state: dict[
        str,
        str,
    ] = {}

    ambiguous_h3 = set()
    null_state = 0

    for i, key in enumerate(
        keys,
        start=1,
    ):
        path = f"{BUCKET}/{key}"

        with fs.open_input_file(
            path
        ) as f:
            pf = pq.ParquetFile(
                f
            )

            for batch in pf.iter_batches(
                columns=[
                    h3_col,
                    state_col,
                ],
                batch_size=250_000,
            ):
                h3_values = (
                    batch.column(0)
                    .to_pylist()
                )

                state_values = (
                    batch.column(1)
                    .to_pylist()
                )

                for (
                    raw_h3,
                    raw_state,
                ) in zip(
                    h3_values,
                    state_values,
                    strict=True,
                ):
                    h3 = normalize_h3(
                        raw_h3
                    )

                    state = (
                        normalize_state(
                            raw_state
                        )
                    )

                    if h3 is None:
                        continue

                    if state is None:
                        null_state += 1
                        continue

                    previous = (
                        h3_to_state.get(
                            h3
                        )
                    )

                    if (
                        previous is not None
                        and previous != state
                    ):
                        ambiguous_h3.add(
                            h3
                        )
                        continue

                    h3_to_state[
                        h3
                    ] = state

        if (
            i == 1
            or i % 10 == 0
            or i == len(keys)
        ):
            print(
                f"processed "
                f"{i:,}/"
                f"{len(keys):,}"
            )

    for h3 in ambiguous_h3:
        h3_to_state.pop(
            h3,
            None,
        )

    state_denominators = Counter(
        h3_to_state.values()
    )

    print()
    print(
        f"H3/state mappings: "
        f"{len(h3_to_state):,}"
    )

    print(
        f"Null-state rows:    "
        f"{null_state:,}"
    )

    print(
        f"Ambiguous H3s:      "
        f"{len(ambiguous_h3):,}"
    )

    return (
        h3_to_state,
        state_denominators,
    )


# =============================================================================
# AUDIT YEAR
# =============================================================================

def audit_year(
    s3,
    fs,
    year: int,
    h3_to_state: dict[str, str],
    state_denominators: Counter,
    target_states: set[str] | None,
):
    prefix = (
        f"{ANNUAL_PREFIX}"
        f"as_of_year={year}/"
    )

    keys = list_parquet_keys(
        s3,
        prefix,
    )

    if not keys:
        raise RuntimeError(
            f"No annual shards found "
            f"for {year}"
        )

    schema = parquet_schema(
        fs,
        keys[0],
    )

    h3_col = find_column(
        schema,
        H3_CANDIDATES,
    )

    missing_features = (
        set(SOCIO_FEATURES)
        - set(schema.names)
    )

    if missing_features:
        raise RuntimeError(
            f"{year} missing fields: "
            f"{sorted(missing_features)}"
        )

    if target_states is None:
        target_states = set(
            state_denominators
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    feature_stats = {
        state: {
            feature: FeatureStats(
                denominator=(
                    state_denominators[
                        state
                    ]
                )
            )
            for feature
            in SOCIO_FEATURES
        }
        for state
        in target_states
    }

    state_present = Counter()
    state_complete = Counter()

    # Useful for proving which individual
    # H3 cells are broken.
    incomplete_h3 = defaultdict(
        list
    )

    seen_h3 = set()

    duplicate_rows = 0
    outside_universe = 0

    print()
    print("=" * 110)
    print(
        f"AUDITING YEAR {year}"
    )
    print("=" * 110)

    print(
        "States: "
        + ", ".join(
            STATE_NAMES.get(
                state,
                state,
            )
            for state
            in sorted(
                target_states
            )
        )
    )

    columns = [
        h3_col,
        *SOCIO_FEATURES,
    ]

    for i, key in enumerate(
        keys,
        start=1,
    ):
        path = f"{BUCKET}/{key}"

        with fs.open_input_file(
            path
        ) as f:
            pf = pq.ParquetFile(
                f
            )

            for batch in pf.iter_batches(
                columns=columns,
                batch_size=100_000,
            ):
                table = (
                    pa.Table
                    .from_batches(
                        [batch]
                    )
                )

                h3_values = (
                    table[h3_col]
                    .combine_chunks()
                    .to_pylist()
                )

                masks = {}

                for feature in (
                    SOCIO_FEATURES
                ):
                    arr = (
                        table[feature]
                        .combine_chunks()
                    )

                    (
                        valid_mask,
                        null_mask,
                        nonfinite_mask,
                    ) = validity(
                        arr
                    )

                    masks[
                        feature
                    ] = {
                        "valid": (
                            valid_mask
                            .to_pylist()
                        ),
                        "null": (
                            null_mask
                            .to_pylist()
                        ),
                        "nonfinite": (
                            nonfinite_mask
                            .to_pylist()
                        ),
                    }

                for row_idx, raw_h3 in (
                    enumerate(
                        h3_values
                    )
                ):
                    h3 = normalize_h3(
                        raw_h3
                    )

                    if h3 is None:
                        continue

                    if h3 in seen_h3:
                        duplicate_rows += 1
                        continue

                    seen_h3.add(h3)

                    state = (
                        h3_to_state.get(
                            h3
                        )
                    )

                    if state is None:
                        outside_universe += 1
                        continue

                    if (
                        state
                        not in target_states
                    ):
                        continue

                    state_present[
                        state
                    ] += 1

                    all_valid = True

                    for feature in (
                        SOCIO_FEATURES
                    ):
                        stat = (
                            feature_stats[
                                state
                            ][feature]
                        )

                        stat.rows_present += 1

                        if (
                            masks[
                                feature
                            ][
                                "valid"
                            ][
                                row_idx
                            ]
                        ):
                            stat.valid += 1

                        else:
                            all_valid = False

                            if (
                                masks[
                                    feature
                                ][
                                    "null"
                                ][
                                    row_idx
                                ]
                            ):
                                stat.null += 1

                            if (
                                masks[
                                    feature
                                ][
                                    "nonfinite"
                                ][
                                    row_idx
                                ]
                            ):
                                (
                                    stat
                                    .nonfinite
                                ) += 1

                    if all_valid:
                        state_complete[
                            state
                        ] += 1

                    else:
                        incomplete_h3[
                            state
                        ].append(
                            h3
                        )

        if (
            i == 1
            or i % 25 == 0
            or i == len(keys)
        ):
            print(
                f"processed "
                f"{i:,}/"
                f"{len(keys):,}"
            )

    return {
        "year": year,
        "feature_stats": (
            feature_stats
        ),
        "state_present": (
            state_present
        ),
        "state_complete": (
            state_complete
        ),
        "state_denominators": (
            state_denominators
        ),
        "incomplete_h3": (
            incomplete_h3
        ),
        "duplicates": (
            duplicate_rows
        ),
        "outside_universe": (
            outside_universe
        ),
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_state_feature_report(
    result,
    state: str,
):
    year = result[
        "year"
    ]

    denominator = (
        result[
            "state_denominators"
        ][state]
    )

    present = (
        result[
            "state_present"
        ][state]
    )

    complete = (
        result[
            "state_complete"
        ][state]
    )

    print()
    print("=" * 120)
    print(
        f"{STATE_NAMES.get(state, state).upper()} "
        f"{year}"
    )
    print("=" * 120)

    print(
        f"Target H3 cells:       "
        f"{denominator:,}"
    )

    print(
        f"Rows present:          "
        f"{present:,} "
        f"({pct(present, denominator):.4f}%)"
    )

    print(
        f"All features complete: "
        f"{complete:,} "
        f"({pct(complete, denominator):.4f}%)"
    )

    print(
        f"Missing entirely:      "
        f"{denominator - present:,}"
    )

    print(
        f"Present but incomplete:"
        f" {present - complete:,}"
    )

    print()
    print(
        f"{'feature':<44}"
        f"{'valid':>13}"
        f"{'target %':>12}"
        f"{'present %':>13}"
        f"{'null':>13}"
        f"{'nonfinite':>13}"
    )

    print("-" * 110)

    ranked = sorted(
        SOCIO_FEATURES,
        key=lambda feature:
        result[
            "feature_stats"
        ][state][feature]
        .coverage_pct,
    )

    for feature in ranked:
        stat = (
            result[
                "feature_stats"
            ][state][feature]
        )

        print(
            f"{feature:<44}"
            f"{stat.valid:>13,}"
            f"{stat.coverage_pct:>11.4f}%"
            f"{stat.present_row_valid_pct:>12.4f}%"
            f"{stat.null:>13,}"
            f"{stat.nonfinite:>13,}"
        )

    worst = ranked[0]

    print()
    print(
        f"Worst feature: "
        f"{worst}"
    )

    worst_stats = (
        result[
            "feature_stats"
        ][state][worst]
    )

    print(
        f"Coverage:      "
        f"{worst_stats.coverage_pct:.4f}%"
    )

    print(
        f"Null rows:     "
        f"{worst_stats.null:,}"
    )


def print_2022_summary(
    result,
):
    print()
    print("=" * 140)
    print(
        "2022 STATE × FEATURE "
        "MISSINGNESS DIAGNOSIS"
    )
    print("=" * 140)

    rows = []

    for state in (
        result[
            "feature_stats"
        ]
    ):
        denominator = (
            result[
                "state_denominators"
            ][state]
        )

        present = (
            result[
                "state_present"
            ][state]
        )

        complete = (
            result[
                "state_complete"
            ][state]
        )

        worst_feature = min(
            SOCIO_FEATURES,
            key=lambda feature:
            result[
                "feature_stats"
            ][state][feature]
            .coverage_pct,
        )

        worst_stats = (
            result[
                "feature_stats"
            ][state][
                worst_feature
            ]
        )

        rows.append(
            (
                state,
                denominator,
                present,
                complete,
                worst_feature,
                worst_stats.coverage_pct,
                worst_stats.null,
            )
        )

    rows.sort(
        key=lambda row:
        pct(
            row[3],
            row[1],
        )
    )

    print()
    print(
        f"{'rank':>5} "
        f"{'state':<24}"
        f"{'target':>12}"
        f"{'spatial':>11}"
        f"{'complete':>12}"
        f"{'complete %':>12}"
        f"{'worst feature':<38}"
        f"{'feature %':>12}"
        f"{'nulls':>12}"
    )

    print("-" * 145)

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        (
            state,
            denominator,
            present,
            complete,
            worst_feature,
            feature_pct,
            nulls,
        ) = row

        print(
            f"{rank:>5} "
            f"{STATE_NAMES.get(state, state):<24}"
            f"{denominator:>12,}"
            f"{pct(present, denominator):>10.3f}%"
            f"{complete:>12,}"
            f"{pct(complete, denominator):>11.3f}%"
            f"{worst_feature:<38}"
            f"{feature_pct:>11.3f}%"
            f"{nulls:>12,}"
        )


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_feature_stats(
    results: dict[int, dict],
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / "state_feature_missingness.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "year",
                "state_fips",
                "state_name",
                "feature",
                "target_h3",
                "rows_present",
                "valid",
                "null",
                "nonfinite",
                "target_coverage_pct",
                "present_row_valid_pct",
            ]
        )

        for year, result in (
            sorted(
                results.items()
            )
        ):
            for state, features in (
                result[
                    "feature_stats"
                ].items()
            ):
                for (
                    feature,
                    stat,
                ) in features.items():
                    writer.writerow(
                        [
                            year,
                            state,
                            STATE_NAMES.get(
                                state,
                                state,
                            ),
                            feature,
                            stat.denominator,
                            stat.rows_present,
                            stat.valid,
                            stat.null,
                            stat.nonfinite,
                            (
                                f"{stat.coverage_pct:.6f}"
                            ),
                            (
                                f"{stat.present_row_valid_pct:.6f}"
                            ),
                        ]
                    )

    print()
    print(
        f"Wrote: {path}"
    )


def export_incomplete_h3(
    results: dict[int, dict],
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / "incomplete_h3_cells.csv"
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "year",
                "state_fips",
                "state_name",
                "h3",
            ]
        )

        for year, result in (
            sorted(
                results.items()
            )
        ):
            for (
                state,
                h3_values,
            ) in result[
                "incomplete_h3"
            ].items():
                for h3 in h3_values:
                    writer.writerow(
                        [
                            year,
                            state,
                            STATE_NAMES.get(
                                state,
                                state,
                            ),
                            h3,
                        ]
                    )

    print(
        f"Wrote: {path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    s3, fs = make_clients()

    (
        h3_to_state,
        state_denominators,
    ) = build_h3_state_map(
        s3,
        fs,
    )

    results = {}

    # -----------------------------------------------------------------
    # 1. 2022 — ALL STATES
    # -----------------------------------------------------------------

    results[2022] = audit_year(
        s3,
        fs,
        2022,
        h3_to_state,
        state_denominators,
        target_states=None,
    )

    print_2022_summary(
        results[2022]
    )

    # -----------------------------------------------------------------
    # 2. CONNECTICUT 2023
    # -----------------------------------------------------------------

    results[2023] = audit_year(
        s3,
        fs,
        2023,
        h3_to_state,
        state_denominators,
        target_states={
            "09",
        },
    )

    print_state_feature_report(
        results[2023],
        "09",
    )

    # -----------------------------------------------------------------
    # 3. CURRENT 2026 PROBLEM STATES
    # -----------------------------------------------------------------

    results[2026] = audit_year(
        s3,
        fs,
        2026,
        h3_to_state,
        state_denominators,
        target_states={
            "32",  # Nevada
            "11",  # DC
            "04",  # Arizona
        },
    )

    for state in (
        "32",
        "11",
        "04",
    ):
        print_state_feature_report(
            results[2026],
            state,
        )

    # -----------------------------------------------------------------
    # OUTPUT ARTIFACTS
    # -----------------------------------------------------------------

    export_feature_stats(
        results
    )

    export_incomplete_h3(
        results
    )

    print()
    print("=" * 120)
    print("AUDIT COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()