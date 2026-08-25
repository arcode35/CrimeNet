#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import polars as pl
import polars_h3 as plh3

from crimenet_data.resources.crime_lake import CrimeLakeResources


HISTORY_ROOT = (
    "s3://crimenet-data/gold/national_feature_store/"
    "temporal/h3_r9/history"
)
ANNUAL_ROOT = (
    "s3://crimenet-data/gold/national_feature_store/"
    "temporal/h3_r9/annual"
)

DEFAULT_OUT = Path("artifacts/audits/exact_temporal_asof")

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(40)
pl.Config.set_fmt_str_lengths(100)


def pct_expr(numer: str, denom: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denom) > 0)
        .then(
            100.0
            * pl.col(numer).cast(pl.Float64)
            / pl.col(denom).cast(pl.Float64)
        )
        .otherwise(None)
        .alias(alias)
    )


def weighted(condition: pl.Expr, weight: str, alias: str) -> pl.Expr:
    return (
        pl.when(condition)
        .then(pl.col(weight))
        .otherwise(0)
        .sum()
        .alias(alias)
    )


def print_section(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def save_df(df: pl.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.parquet"
    df.write_parquet(path, compression="zstd")
    print(f"[saved] {path}")


def history_scan(lake: CrimeLakeResources) -> pl.LazyFrame:
    return pl.scan_parquet(
        f"{HISTORY_ROOT}/feature_available_date=*/version_id=*/part-*.parquet",
        storage_options=lake.storage_options,
        credential_provider=None,
        hive_partitioning=False,
    )


def annual_scan(lake: CrimeLakeResources) -> pl.LazyFrame:
    return pl.scan_parquet(
        f"{ANNUAL_ROOT}/as_of_year=*/part-*.parquet",
        storage_options=lake.storage_options,
        credential_provider=None,
        hive_partitioning=False,
    )


def localize_event_times(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Silver occurrence_timestamp is source-local wall-clock time.
    Convert it to UTC before comparing with feature_available_at.

    Ambiguous DST fall-back times use the EARLIEST possible UTC instant,
    which is conservative for leakage safety: it cannot make later features
    appear available earlier than they really were.

    Non-existent spring-forward local times become null and are audited.
    """
    timezones = (
        events
        .select("source_timezone")
        .drop_nulls()
        .unique()
        .sort("source_timezone")
        .get_column("source_timezone")
        .to_list()
    )

    parts: list[pl.DataFrame] = []

    for tz in timezones:
        part = (
            events
            .filter(pl.col("source_timezone") == tz)
            .with_columns(
                pl.col("occurrence_timestamp")
                .dt.replace_time_zone(
                    tz,
                    ambiguous="earliest",
                    non_existent="null",
                )
                .dt.convert_time_zone("UTC")
                .alias("event_at_utc")
            )
        )
        parts.append(part)

    null_tz = events.filter(pl.col("source_timezone").is_null())
    if null_tz.height:
        parts.append(
            null_tz.with_columns(
                pl.lit(None, dtype=pl.Datetime("us", "UTC"))
                .alias("event_at_utc")
            )
        )

    localized = pl.concat(parts, how="vertical_relaxed")

    tz_audit = (
        localized.lazy()
        .group_by("source_timezone")
        .agg(
            pl.len().alias("events"),
            pl.col("event_at_utc").null_count().alias("utc_conversion_nulls"),
        )
        .with_columns(
            pct_expr("utc_conversion_nulls", "events", "utc_conversion_null_pct")
        )
        .sort("utc_conversion_nulls", descending=True)
        .collect()
    )

    return localized, tz_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Local directory for audit artifacts.",
    )
    parser.add_argument(
        "--all-silver",
        action="store_true",
        help="Audit all Silver rows instead of include_in_model rows only.",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=500,
        help="Number of worst/example rows to persist per diagnostic.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    lake = CrimeLakeResources()

    # ==================================================================================
    # A. SILVER EVENTS
    # ==================================================================================
    print_section("A. LOAD MODELED SILVER EVENTS")

    silver = lake.scan_silver_snapshot()
    if not args.all_silver:
        silver = silver.filter(
            pl.col("include_in_model").fill_null(False)
        )

    event_cols = [
        "crime_id",
        "source_city",
        "occurrence_timestamp",
        "occurrence_year",
        "source_timezone",
        "latitude",
        "longitude",
    ]

    events = (
        silver
        .select(event_cols)
        .with_columns(
            plh3.latlng_to_cell(
                "latitude",
                "longitude",
                resolution=9,
                return_dtype=pl.UInt64,
            )
            .cast(pl.Int64, strict=False)
            .alias("osm_h3_cell_id")
        )
        .collect(engine="streaming")
    )

    print(f"Silver events loaded: {events.height:,}")

    events, tz_audit = localize_event_times(events)
    print("\nTIMEZONE CONVERSION AUDIT")
    print(tz_audit)
    save_df(tz_audit, out_dir, "timezone_conversion_audit")

    event_quality = events.select(
        pl.len().alias("events"),
        pl.col("crime_id").n_unique().alias("unique_crime_ids"),
        pl.col("osm_h3_cell_id").null_count().alias("null_h3"),
        pl.col("occurrence_timestamp").null_count().alias("null_occurrence_timestamp"),
        pl.col("source_timezone").null_count().alias("null_source_timezone"),
        pl.col("event_at_utc").null_count().alias("null_event_at_utc"),
        (
            pl.col("occurrence_timestamp").dt.year()
            != pl.col("occurrence_year")
        )
        .fill_null(False)
        .sum()
        .alias("occurrence_year_mismatch"),
    )
    print("\nEVENT QUALITY")
    print(event_quality)

    # ==================================================================================
    # B. HISTORY STORE CONTRACT
    # ==================================================================================
    print_section("B. LOAD AND ATTACK TEMPORAL HISTORY STORE")

    hscan = history_scan(lake)
    hschema = hscan.collect_schema()
    hnames = set(hschema.names())

    required = {
        "osm_h3_cell_id",
        "feature_available_at",
        "feature_version_id",
    }
    missing_required = sorted(required - hnames)
    if missing_required:
        raise RuntimeError(
            f"History store missing required columns: {missing_required}"
        )

    optional_candidates = [
        "osm_available_at",
        "osm_snapshot_date",
        "osm_snapshot_year",
        "acs_release_date",
        "acs_vintage",
        "tiger_release_date",
        "tiger_line_year",
        "tract_geoid",
        "state_fips",
        "_socioeconomic_matched",
    ]
    optional = [c for c in optional_candidates if c in hnames]

    history_cols = [
        "osm_h3_cell_id",
        "feature_available_at",
        "feature_version_id",
        *optional,
    ]

    print("History metadata columns:")
    for c in history_cols:
        print(f"  - {c}: {hschema[c]}")

    history = (
        hscan
        .select(history_cols)
        .with_columns(
            pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False),
        )
        .collect(engine="streaming")
    )

    print(f"\nHistory rows loaded: {history.height:,}")
    print(
        f"Unique H3 cells: "
        f"{history.get_column('osm_h3_cell_id').n_unique():,}"
    )
    print(
        f"Unique feature versions: "
        f"{history.get_column('feature_version_id').n_unique():,}"
    )
    print(
        f"Feature availability range: "
        f"{history.get_column('feature_available_at').min()} -> "
        f"{history.get_column('feature_available_at').max()}"
    )

    # Hard key uniqueness check.
    dup_history_keys = (
        history.lazy()
        .group_by(["osm_h3_cell_id", "feature_available_at"])
        .agg(
            pl.len().alias("rows"),
            pl.col("feature_version_id").n_unique().alias("version_ids"),
        )
        .filter(pl.col("rows") > 1)
        .sort("rows", descending=True)
        .collect()
    )

    print("\nDUPLICATE (H3, feature_available_at) KEYS")
    print(dup_history_keys.head(args.worst))
    save_df(
        dup_history_keys.head(args.worst),
        out_dir,
        "duplicate_history_keys",
    )

    if dup_history_keys.height:
        raise RuntimeError(
            "Temporal history violates unique (H3, feature_available_at) key. "
            "Do not trust an as-of join until this is fixed."
        )

    # Component availability must never exceed the composite availability time.
    component_availability_cols = [
        c
        for c in [
            "osm_available_at",
            "acs_release_date",
            "tiger_release_date",
        ]
        if c in history.columns
    ]

    component_contract_exprs: list[pl.Expr] = []
    for c in component_availability_cols:
        component_contract_exprs.append(
            (
                pl.col(c).is_not_null()
                & (
                    pl.col(c)
                    > pl.col("feature_available_at")
                )
            )
            .sum()
            .alias(f"{c}_after_feature_available_at")
        )

    if component_contract_exprs:
        component_contract = history.select(component_contract_exprs)
        print("\nCOMPONENT -> FEATURE AVAILABILITY CONTRACT")
        print(component_contract)
        save_df(
            component_contract,
            out_dir,
            "component_availability_contract",
        )

    # Sort per H3 to derive next legal feature transition.
    history = (
        history
        .sort(["osm_h3_cell_id", "feature_available_at"])
        .with_columns(
            pl.col("feature_available_at")
            .shift(-1)
            .over("osm_h3_cell_id")
            .alias("next_feature_available_at")
        )
    )

    # H3 temporal support bounds used to classify missing events.
    support = (
        history.lazy()
        .group_by("osm_h3_cell_id")
        .agg(
            pl.col("feature_available_at")
            .min()
            .alias("first_feature_available_at"),
            pl.col("feature_available_at")
            .max()
            .alias("last_feature_available_at"),
            pl.len().alias("history_versions"),
            pl.col("feature_version_id")
            .n_unique()
            .alias("unique_feature_versions"),
        )
        .collect()
    )

    # ==================================================================================
    # C. EXACT BACKWARD AS-OF JOIN
    # ==================================================================================
    print_section("C. FULL EXACT BACKWARD AS-OF JOIN")

    joinable = events.filter(
        pl.col("osm_h3_cell_id").is_not_null()
        & pl.col("event_at_utc").is_not_null()
    )

    unjoinable = events.filter(
        pl.col("osm_h3_cell_id").is_null()
        | pl.col("event_at_utc").is_null()
    )

    print(f"Joinable events:   {joinable.height:,}")
    print(f"Unjoinable events: {unjoinable.height:,}")

    # Polars join_asof requires the as-of key to be sorted.
    # The `by` predicate constrains matches to the same H3 cell.
    left = joinable.sort("event_at_utc")
    right = history.sort("feature_available_at")

    t0 = perf_counter()
    joined = left.join_asof(
        right,
        left_on="event_at_utc",
        right_on="feature_available_at",
        by="osm_h3_cell_id",
        strategy="backward",
        allow_exact_matches=True,
    )
    join_seconds = perf_counter() - t0
    print(f"As-of join completed in {join_seconds:,.1f}s")

    joined = (
        joined
        .join(
            support,
            on="osm_h3_cell_id",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("feature_available_at")
            .is_not_null()
            .alias("_history_matched")
        )
        .with_columns(
            pl.when(pl.col("_history_matched"))
            .then(
                pl.col("event_at_utc")
                - pl.col("feature_available_at")
            )
            .otherwise(None)
            .alias("feature_age"),

            (
                pl.col("_history_matched")
                & (
                    pl.col("feature_available_at")
                    > pl.col("event_at_utc")
                )
            )
            .alias("_future_feature_leak"),

            (
                pl.col("_history_matched")
                & pl.col("next_feature_available_at").is_not_null()
                & (
                    pl.col("event_at_utc")
                    >= pl.col("next_feature_available_at")
                )
            )
            .alias("_not_latest_legal_version"),
        )
    )

    # ==================================================================================
    # D. LEAKAGE / LATEST-LEGAL-VERSION PROOF
    # ==================================================================================
    print_section("D. HARD TEMPORAL CORRECTNESS INVARIANTS")

    invariant_exprs: list[pl.Expr] = [
        pl.col("_history_matched").sum().alias("matched_events"),
        (~pl.col("_history_matched")).sum().alias("joinable_missing_events"),
        pl.col("_future_feature_leak").sum().alias("future_feature_leaks"),
        pl.col("_not_latest_legal_version")
        .sum()
        .alias("not_latest_legal_version"),
    ]

    for c in component_availability_cols:
        invariant_exprs.append(
            (
                pl.col("_history_matched")
                & pl.col(c).is_not_null()
                & (
                    pl.col(c)
                    > pl.col("event_at_utc")
                )
            )
            .sum()
            .alias(f"future_{c}_events")
        )

    invariants = joined.select(invariant_exprs)
    print(invariants)
    save_df(invariants, out_dir, "temporal_invariants")

    inv = invariants.row(0, named=True)
    hard_failures = {
        k: int(v)
        for k, v in inv.items()
        if (
            k == "future_feature_leaks"
            or k == "not_latest_legal_version"
            or k.startswith("future_")
        )
        and int(v) != 0
    }

    # Persist actual violating rows, if any.
    violations = joined.filter(
        pl.col("_future_feature_leak")
        | pl.col("_not_latest_legal_version")
    )
    if violations.height:
        cols = [
            "crime_id",
            "source_city",
            "event_at_utc",
            "osm_h3_cell_id",
            "feature_available_at",
            "next_feature_available_at",
            "feature_version_id",
            "_future_feature_leak",
            "_not_latest_legal_version",
        ]
        cols += [
            c for c in component_availability_cols
            if c in violations.columns
        ]
        save_df(
            violations.select(cols).head(args.worst),
            out_dir,
            "hard_temporal_violations",
        )

    # ==================================================================================
    # E. MISSINGNESS FORENSICS
    # ==================================================================================
    print_section("E. EXACT MISSINGNESS FORENSICS")

    missing_joinable = (
        joined
        .filter(~pl.col("_history_matched"))
        .with_columns(
            pl.when(pl.col("first_feature_available_at").is_null())
            .then(pl.lit("h3_absent_from_history"))
            .when(
                pl.col("event_at_utc")
                < pl.col("first_feature_available_at")
            )
            .then(pl.lit("event_before_first_feature"))
            .otherwise(pl.lit("unexplained_asof_miss"))
            .alias("missing_reason")
        )
    )

    missing_reason = (
        missing_joinable.lazy()
        .group_by("missing_reason")
        .agg(pl.len().alias("events"))
        .sort("events", descending=True)
        .collect()
    )

    if unjoinable.height:
        unjoinable_reason = (
            unjoinable.lazy()
            .with_columns(
                pl.when(pl.col("osm_h3_cell_id").is_null())
                .then(pl.lit("null_h3"))
                .when(pl.col("event_at_utc").is_null())
                .then(pl.lit("invalid_event_utc"))
                .otherwise(pl.lit("unknown_unjoinable"))
                .alias("missing_reason")
            )
            .group_by("missing_reason")
            .agg(pl.len().alias("events"))
            .collect()
        )
        missing_reason = pl.concat(
            [missing_reason, unjoinable_reason],
            how="vertical_relaxed",
        ).sort("events", descending=True)

    print(missing_reason)
    save_df(missing_reason, out_dir, "missing_reasons")

    if missing_joinable.height:
        missing_examples = (
            missing_joinable
            .select(
                "crime_id",
                "source_city",
                "occurrence_timestamp",
                "event_at_utc",
                "occurrence_year",
                "osm_h3_cell_id",
                "missing_reason",
                "first_feature_available_at",
                "last_feature_available_at",
                "history_versions",
            )
            .sort(
                ["missing_reason", "source_city", "event_at_utc"]
            )
            .head(args.worst)
        )
        save_df(
            missing_examples,
            out_dir,
            "missing_event_examples",
        )

    # ==================================================================================
    # F. COVERAGE
    # ==================================================================================
    print_section("F. EVENT-WEIGHTED HISTORY COVERAGE")

    total_events = events.height
    matched_events = int(
        joined.get_column("_history_matched").sum()
    )
    missing_events = total_events - matched_events

    global_coverage = pl.DataFrame(
        {
            "events": [total_events],
            "matched_events": [matched_events],
            "missing_events": [missing_events],
            "coverage_pct": [
                100.0 * matched_events / total_events
                if total_events
                else None
            ],
            "joinable_events": [joinable.height],
            "unjoinable_events": [unjoinable.height],
        }
    )
    print(global_coverage)
    save_df(global_coverage, out_dir, "global_history_coverage")

    by_city = (
        joined.lazy()
        .group_by("source_city")
        .agg(
            pl.len().alias("joinable_events"),
            pl.col("_history_matched")
            .sum()
            .alias("matched_events"),
            (~pl.col("_history_matched"))
            .sum()
            .alias("missing_events"),
            pl.col("osm_h3_cell_id")
            .n_unique()
            .alias("unique_h3_cells"),
        )
        .with_columns(
            pct_expr(
                "matched_events",
                "joinable_events",
                "coverage_pct",
            )
        )
        .sort("coverage_pct")
        .collect()
    )
    print("\nBY CITY")
    print(by_city)
    save_df(by_city, out_dir, "history_coverage_by_city")

    by_year = (
        joined.lazy()
        .group_by("occurrence_year")
        .agg(
            pl.len().alias("joinable_events"),
            pl.col("_history_matched")
            .sum()
            .alias("matched_events"),
            (~pl.col("_history_matched"))
            .sum()
            .alias("missing_events"),
            pl.col("osm_h3_cell_id")
            .n_unique()
            .alias("unique_h3_cells"),
        )
        .with_columns(
            pct_expr(
                "matched_events",
                "joinable_events",
                "coverage_pct",
            )
        )
        .sort("occurrence_year")
        .collect()
    )
    print("\nBY YEAR")
    print(by_year)
    save_df(by_year, out_dir, "history_coverage_by_year")

    by_city_year = (
        joined.lazy()
        .group_by(["source_city", "occurrence_year"])
        .agg(
            pl.len().alias("joinable_events"),
            pl.col("_history_matched")
            .sum()
            .alias("matched_events"),
            (~pl.col("_history_matched"))
            .sum()
            .alias("missing_events"),
            pl.col("osm_h3_cell_id")
            .n_unique()
            .alias("unique_h3_cells"),
        )
        .with_columns(
            pct_expr(
                "matched_events",
                "joinable_events",
                "coverage_pct",
            )
        )
        .sort(["source_city", "occurrence_year"])
        .collect()
    )
    save_df(
        by_city_year,
        out_dir,
        "history_coverage_by_city_year",
    )

    # ==================================================================================
    # G. FEATURE STALENESS
    # ==================================================================================
    print_section("G. FEATURE STALENESS / AGE AT EVENT TIME")

    matched = joined.filter(pl.col("_history_matched"))

    staleness_global = matched.select(
        pl.len().alias("events"),
        pl.col("feature_age").min().alias("min_age"),
        pl.col("feature_age").median().alias("median_age"),
        pl.col("feature_age").quantile(0.90).alias("p90_age"),
        pl.col("feature_age").quantile(0.95).alias("p95_age"),
        pl.col("feature_age").quantile(0.99).alias("p99_age"),
        pl.col("feature_age").max().alias("max_age"),
        (pl.col("feature_age") > pl.duration(days=30))
        .sum()
        .alias("gt_30d"),
        (pl.col("feature_age") > pl.duration(days=90))
        .sum()
        .alias("gt_90d"),
        (pl.col("feature_age") > pl.duration(days=365))
        .sum()
        .alias("gt_1y"),
        (pl.col("feature_age") > pl.duration(days=730))
        .sum()
        .alias("gt_2y"),
    )
    print(staleness_global)
    save_df(staleness_global, out_dir, "staleness_global")

    staleness_city = (
        matched.lazy()
        .group_by("source_city")
        .agg(
            pl.len().alias("events"),
            pl.col("feature_age").median().alias("median_age"),
            pl.col("feature_age").quantile(0.95).alias("p95_age"),
            pl.col("feature_age").quantile(0.99).alias("p99_age"),
            pl.col("feature_age").max().alias("max_age"),
            (pl.col("feature_age") > pl.duration(days=365))
            .sum()
            .alias("gt_1y"),
        )
        .sort("p99_age", descending=True)
        .collect()
    )
    print("\nSTALENESS BY CITY")
    print(staleness_city)
    save_df(staleness_city, out_dir, "staleness_by_city")

    staleness_year = (
        matched.lazy()
        .group_by("occurrence_year")
        .agg(
            pl.len().alias("events"),
            pl.col("feature_age").median().alias("median_age"),
            pl.col("feature_age").quantile(0.95).alias("p95_age"),
            pl.col("feature_age").quantile(0.99).alias("p99_age"),
            pl.col("feature_age").max().alias("max_age"),
        )
        .sort("occurrence_year")
        .collect()
    )
    save_df(staleness_year, out_dir, "staleness_by_year")

    worst_stale_cols = [
        "crime_id",
        "source_city",
        "occurrence_timestamp",
        "event_at_utc",
        "osm_h3_cell_id",
        "feature_available_at",
        "next_feature_available_at",
        "feature_version_id",
        "feature_age",
    ]
    worst_stale_cols += [
        c
        for c in [
            "osm_available_at",
            "osm_snapshot_date",
            "acs_release_date",
            "acs_vintage",
            "tiger_release_date",
            "tiger_line_year",
            "tract_geoid",
        ]
        if c in matched.columns
    ]

    worst_stale = (
        matched
        .select(worst_stale_cols)
        .sort("feature_age", descending=True)
        .head(args.worst)
    )
    save_df(worst_stale, out_dir, "worst_stale_events")

    # ==================================================================================
    # H. COMPONENT-LEVEL LEAKAGE AND AGE
    # ==================================================================================
    print_section("H. OSM / ACS / TIGER COMPONENT AUDIT")

    component_summary_exprs: list[pl.Expr] = []
    for c in component_availability_cols:
        component_summary_exprs.extend(
            [
                pl.col(c).null_count().alias(f"{c}_nulls"),
                (
                    pl.col(c).is_not_null()
                    & (
                        pl.col(c)
                        > pl.col("event_at_utc")
                    )
                )
                .sum()
                .alias(f"{c}_future"),
                (
                    pl.col("event_at_utc") - pl.col(c)
                )
                .median()
                .alias(f"{c}_median_age"),
                (
                    pl.col("event_at_utc") - pl.col(c)
                )
                .quantile(0.99)
                .alias(f"{c}_p99_age"),
            ]
        )

    if component_summary_exprs:
        component_summary = matched.select(
            component_summary_exprs
        )
        print(component_summary)
        save_df(
            component_summary,
            out_dir,
            "component_event_time_audit",
        )
    else:
        print("No component availability columns found.")

    if "_socioeconomic_matched" in matched.columns:
        socio_global = matched.select(
            pl.len().alias("matched_events"),
            pl.col("_socioeconomic_matched")
            .fill_null(False)
            .sum()
            .alias("socio_matched_events"),
        ).with_columns(
            pct_expr(
                "socio_matched_events",
                "matched_events",
                "socio_coverage_pct",
            )
        )
        print("\nSOCIOECONOMIC COVERAGE AMONG HISTORY-MATCHED EVENTS")
        print(socio_global)
        save_df(
            socio_global,
            out_dir,
            "history_socio_global",
        )

        socio_city = (
            matched.lazy()
            .group_by("source_city")
            .agg(
                pl.len().alias("matched_events"),
                pl.col("_socioeconomic_matched")
                .fill_null(False)
                .sum()
                .alias("socio_matched_events"),
            )
            .with_columns(
                pct_expr(
                    "socio_matched_events",
                    "matched_events",
                    "socio_coverage_pct",
                )
            )
            .sort("socio_coverage_pct")
            .collect()
        )
        print(socio_city)
        save_df(
            socio_city,
            out_dir,
            "history_socio_by_city",
        )

    # ==================================================================================
    # I. VERSION SELECTION DISTRIBUTION
    # ==================================================================================
    print_section("I. FEATURE VERSION SELECTION")

    version_usage = (
        matched.lazy()
        .group_by("feature_version_id")
        .agg(
            pl.len().alias("events"),
            pl.col("osm_h3_cell_id").n_unique().alias("h3_cells"),
            pl.col("feature_available_at").min().alias("available_min"),
            pl.col("feature_available_at").max().alias("available_max"),
        )
        .sort("events", descending=True)
        .collect()
    )
    print(version_usage.head(100))
    save_df(version_usage, out_dir, "feature_version_usage")

    # ==================================================================================
    # J. COMPARE EXACT HISTORY AGAINST ANNUAL CACHE
    # ==================================================================================
    print_section("J. EXACT HISTORY VS ANNUAL CACHE")

    ascan = annual_scan(lake)
    aschema = ascan.collect_schema()
    anames = set(aschema.names())

    annual_required = {
        "osm_h3_cell_id",
        "as_of_year",
    }
    missing_annual = sorted(annual_required - anames)
    if missing_annual:
        raise RuntimeError(
            f"Annual store missing columns: {missing_annual}"
        )

    annual_cols = [
        "osm_h3_cell_id",
        "as_of_year",
    ]
    for c in [
        "feature_available_at",
        "feature_version_id",
        "_socioeconomic_matched",
    ]:
        if c in anames:
            annual_cols.append(c)

    annual = (
        ascan
        .select(annual_cols)
        .with_columns(
            pl.col("osm_h3_cell_id").cast(pl.Int64, strict=False),
            pl.col("as_of_year").cast(pl.Int16, strict=False),
        )
        .collect(engine="streaming")
    )

    annual_dups = (
        annual.lazy()
        .group_by(["osm_h3_cell_id", "as_of_year"])
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
        .collect()
    )
    print(f"Annual duplicate keys: {annual_dups.height:,}")

    if annual_dups.height:
        save_df(
            annual_dups.head(args.worst),
            out_dir,
            "annual_duplicate_keys",
        )
        raise RuntimeError(
            "Annual serving table violates unique (H3, as_of_year) key."
        )

    # Collapse exact-history result to city/year/H3 so the annual comparison
    # remains cheap but preserves exact event weights.
    exact_cell_year = (
        joined.lazy()
        .group_by(
            [
                "source_city",
                "occurrence_year",
                "osm_h3_cell_id",
            ]
        )
        .agg(
            pl.len().alias("events"),
            pl.col("_history_matched")
            .sum()
            .alias("history_matched_events"),
        )
        .collect()
    )

    annual_key = (
        annual
        .select(
            pl.col("osm_h3_cell_id"),
            pl.col("as_of_year")
            .alias("occurrence_year"),
        )
        .with_columns(
            pl.lit(True).alias("_annual_matched")
        )
    )

    compare = (
        exact_cell_year
        .join(
            annual_key,
            on=["osm_h3_cell_id", "occurrence_year"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("_annual_matched")
            .fill_null(False)
        )
        .with_columns(
            (
                pl.col("_annual_matched")
                & (
                    pl.col("history_matched_events")
                    < pl.col("events")
                )
            )
            .alias("_annual_but_history_regression"),

            (
                ~pl.col("_annual_matched")
                & (
                    pl.col("history_matched_events") > 0
                )
            )
            .alias("_history_gain_over_annual"),
        )
    )

    annual_vs_history = compare.select(
        pl.col("events").sum().alias("events"),
        weighted(
            pl.col("_annual_matched"),
            "events",
            "annual_matched_events",
        ),
        pl.col("history_matched_events")
        .sum()
        .alias("history_matched_events"),
        weighted(
            pl.col("_annual_but_history_regression"),
            "events",
            "annual_but_history_regression_events",
        ),
        weighted(
            pl.col("_history_gain_over_annual"),
            "events",
            "history_gain_over_annual_cell_year_events",
        ),
    ).with_columns(
        pct_expr(
            "annual_matched_events",
            "events",
            "annual_coverage_pct",
        ),
        pct_expr(
            "history_matched_events",
            "events",
            "history_coverage_pct",
        ),
    )

    print(annual_vs_history)
    save_df(
        annual_vs_history,
        out_dir,
        "annual_vs_history",
    )

    regressions = compare.filter(
        pl.col("_annual_but_history_regression")
    )
    if regressions.height:
        save_df(
            regressions
            .sort("events", descending=True)
            .head(args.worst),
            out_dir,
            "annual_history_regressions",
        )

    gains = compare.filter(
        pl.col("_history_gain_over_annual")
    )
    if gains.height:
        save_df(
            gains
            .sort("events", descending=True)
            .head(args.worst),
            out_dir,
            "history_gains_over_annual",
        )

    # ==================================================================================
    # K. CELL-YEAR SUPPORT / CONCENTRATION
    # ==================================================================================
    print_section("K. CELL-YEAR SUPPORT AND FAILURE CONCENTRATION")

    cell_year = (
        joined.lazy()
        .group_by(
            [
                "source_city",
                "occurrence_year",
                "osm_h3_cell_id",
            ]
        )
        .agg(
            pl.len().alias("events"),
            pl.col("_history_matched")
            .sum()
            .alias("matched_events"),
        )
        .with_columns(
            (
                pl.col("matched_events")
                == pl.col("events")
            ).alias("fully_matched"),
            (
                pl.col("matched_events") == 0
            ).alias("fully_missing"),
        )
        .collect()
    )

    cell_year_city = (
        cell_year.lazy()
        .group_by("source_city")
        .agg(
            pl.len().alias("crime_cell_years"),
            pl.col("fully_matched").sum().alias("fully_matched_cell_years"),
            pl.col("fully_missing").sum().alias("fully_missing_cell_years"),
        )
        .with_columns(
            pct_expr(
                "fully_matched_cell_years",
                "crime_cell_years",
                "cell_year_coverage_pct",
            )
        )
        .sort("cell_year_coverage_pct")
        .collect()
    )
    print(cell_year_city)
    save_df(
        cell_year_city,
        out_dir,
        "cell_year_coverage_by_city",
    )

    # ==================================================================================
    # L. FINAL VERDICT
    # ==================================================================================
    print_section("L. FINAL VERDICT")

    unexplained = 0
    if missing_joinable.height:
        unexplained = missing_joinable.filter(
            pl.col("missing_reason") == "unexplained_asof_miss"
        ).height

    annual_regression_events = int(
        annual_vs_history.row(0, named=True)[
            "annual_but_history_regression_events"
        ]
    )

    elapsed = perf_counter() - started

    verdict = {
        "modeled_only": not args.all_silver,
        "events": total_events,
        "joinable_events": joinable.height,
        "matched_events": matched_events,
        "missing_events": missing_events,
        "history_coverage_pct": (
            100.0 * matched_events / total_events
            if total_events
            else None
        ),
        "future_feature_leaks": int(
            inv["future_feature_leaks"]
        ),
        "not_latest_legal_version": int(
            inv["not_latest_legal_version"]
        ),
        "unexplained_asof_misses": unexplained,
        "annual_but_history_regression_events": annual_regression_events,
        "duplicate_history_keys": dup_history_keys.height,
        "runtime_seconds": elapsed,
    }

    for k, v in hard_failures.items():
        verdict[k] = v

    verdict_path = out_dir / "verdict.json"
    verdict_path.write_text(
        json.dumps(verdict, indent=2, default=str)
    )

    print(json.dumps(verdict, indent=2, default=str))
    print(f"\nAudit artifacts: {out_dir.resolve()}")

    fatal = {
        "duplicate_history_keys": dup_history_keys.height,
        "future_feature_leaks": int(inv["future_feature_leaks"]),
        "not_latest_legal_version": int(
            inv["not_latest_legal_version"]
        ),
        "unexplained_asof_misses": unexplained,
        "annual_but_history_regression_events": annual_regression_events,
    }
    fatal.update(hard_failures)

    bad = {k: v for k, v in fatal.items() if v != 0}

    if bad:
        print("\nHARD AUDIT FAILURES:")
        for k, v in bad.items():
            print(f"  {k}: {v:,}")
        raise SystemExit(2)

    print("\nHARD TEMPORAL QUALITY GATE: PASS")


if __name__ == "__main__":
    main()
