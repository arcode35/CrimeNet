"""Read-only jurisdiction coordinate-bounds audit for current Silver."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from crimenet_data.assets.crime.common.source_bounds import (
    get_source_coordinate_bounds,
    globally_valid_coordinate_expr,
    source_bounds_expr,
)
from crimenet_data.assets.crime.sources import SILVER_SOURCE_KEYS
from crimenet_data.resources.crime_lake import CrimeLakeResources

SAFETY_THRESHOLD_PCT = 5.0
AUDIT_COLUMNS = [
    "crime_id",
    "source_city",
    "occurrence_timestamp",
    "latitude",
    "longitude",
    "source_offense_code",
    "source_offense_category",
    "source_offense_description",
    "include_in_model",
]


def scan_silver_source_for_audit(
    crime_lake: CrimeLakeResources,
    *,
    snapshot_uri: str,
    source_city: str,
) -> pl.LazyFrame:
    """Scan current or legacy Silver without requiring the future canonical schema."""

    source_uri = f"{snapshot_uri}/source_city={source_city}"
    scanned = pl.scan_parquet(
        f"{source_uri}/**/*.parquet",
        storage_options=crime_lake.storage_options,
        credential_provider=None,
        hive_partitioning=True,
        hive_schema={
            "snapshot_id": pl.String,
            "source_city": pl.String,
            "occurrence_year": pl.Int16,
        },
    )
    missing = sorted(set(AUDIT_COLUMNS) - set(scanned.collect_schema().names()))
    if missing:
        raise RuntimeError(
            f"Silver source {source_city!r} is missing audit columns: {missing}"
        )
    return scanned.select(AUDIT_COLUMNS)


def bound_excess_expressions(source_city: str) -> list[pl.Expr]:
    bounds = get_source_coordinate_bounds(source_city)
    return [
        (bounds.min_lat - pl.col("latitude"))
        .clip(lower_bound=0.0)
        .alias("lat_below_by"),
        (pl.col("latitude") - bounds.max_lat)
        .clip(lower_bound=0.0)
        .alias("lat_above_by"),
        (bounds.min_lon - pl.col("longitude"))
        .clip(lower_bound=0.0)
        .alias("lon_below_by"),
        (pl.col("longitude") - bounds.max_lon)
        .clip(lower_bound=0.0)
        .alias("lon_above_by"),
    ]


def audit_source_frame(
    frame: pl.DataFrame,
    source_city: str,
) -> tuple[dict[str, object], pl.DataFrame]:
    """Return one source summary and its 100 most extreme valid outliers."""

    globally_valid = globally_valid_coordinate_expr()
    inside = source_bounds_expr(source_city)
    outside = globally_valid & ~inside
    included = pl.col("include_in_model").fill_null(False)
    summary = (
        frame.lazy()
        .select(
            pl.lit(source_city).alias("source_city"),
            pl.len().alias("rows"),
            (~included).sum().alias("already_excluded_rows"),
            (~globally_valid).sum().alias("globally_invalid_coordinate_rows"),
            inside.sum().alias("inside_source_bounds_rows"),
            outside.sum().alias("outside_source_bounds_rows"),
            pl.when(pl.len() > 0)
            .then(100.0 * outside.sum() / pl.len())
            .otherwise(0.0)
            .alias("outside_source_bounds_pct"),
            included.sum().alias("modeled_rows_before"),
            (included & inside).sum().alias("modeled_rows_after"),
            (included & ~inside).sum().alias("newly_excluded_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    modeled_before = int(summary["modeled_rows_before"])
    newly_excluded = int(summary["newly_excluded_rows"])
    summary["newly_excluded_modeled_pct"] = (
        100.0 * newly_excluded / modeled_before if modeled_before else 0.0
    )
    summary["safety_threshold_exceeded"] = (
        float(summary["newly_excluded_modeled_pct"]) > SAFETY_THRESHOLD_PCT
    )

    outliers = (
        frame.lazy()
        .filter(outside)
        .with_columns(bound_excess_expressions(source_city))
        .with_columns(
            pl.max_horizontal(
                "lat_below_by",
                "lat_above_by",
                "lon_below_by",
                "lon_above_by",
            ).alias("max_bound_excess_degrees")
        )
        .select(
            "crime_id",
            "source_city",
            "occurrence_timestamp",
            "latitude",
            "longitude",
            "source_offense_code",
            "source_offense_category",
            "source_offense_description",
            "lat_below_by",
            "lat_above_by",
            "lon_below_by",
            "lon_above_by",
            "max_bound_excess_degrees",
        )
        .sort("max_bound_excess_degrees", descending=True)
        .head(100)
        .collect()
    )
    return summary, outliers


def run_audit(
    *,
    crime_lake: CrimeLakeResources,
    output_dir: Path,
    snapshot_uri: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Audit every modeled source without writing to the lake."""

    resolved_snapshot = snapshot_uri or crime_lake.resolve_current_silver_snapshot()
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_dir = output_dir / "outliers"
    outlier_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    outlier_frames: list[pl.DataFrame] = []
    for source_city in SILVER_SOURCE_KEYS:
        frame = scan_silver_source_for_audit(
            crime_lake,
            snapshot_uri=resolved_snapshot,
            source_city=source_city,
        ).collect(engine="streaming")
        summary, outliers = audit_source_frame(frame, source_city)
        summaries.append(summary)
        outlier_frames.append(outliers)
        outliers.write_csv(outlier_dir / f"{source_city}.csv")
        print(json.dumps(summary, sort_keys=True, default=str), flush=True)

    summary_frame = pl.DataFrame(summaries).sort("source_city")
    all_outliers = pl.concat(outlier_frames, how="vertical_relaxed")
    top_outliers = all_outliers.sort("max_bound_excess_degrees", descending=True).head(
        20
    )
    summary_frame.write_csv(output_dir / "source_summary.csv")
    all_outliers.write_parquet(
        output_dir / "top_100_outliers_per_source.parquet",
        compression="zstd",
    )
    top_outliers.write_csv(output_dir / "top_20_outliers_overall.csv")

    modeled_before = int(summary_frame["modeled_rows_before"].sum())
    newly_excluded = int(summary_frame["newly_excluded_rows"].sum())
    report: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "silver_snapshot_uri": resolved_snapshot,
        "source_count": summary_frame.height,
        "total_rows": int(summary_frame["rows"].sum()),
        "currently_model_eligible_rows": modeled_before,
        "newly_excluded_rows": newly_excluded,
        "estimated_model_eligible_rows_after": modeled_before - newly_excluded,
        "safety_threshold_pct": SAFETY_THRESHOLD_PCT,
        "sources_exceeding_safety_threshold": summary_frame.filter(
            pl.col("safety_threshold_exceeded")
        )["source_city"].to_list(),
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print("\nPER_SOURCE_AUDIT\n" + summary_frame.write_csv(), flush=True)
    print("\nTOP_20_OUTLIERS\n" + top_outliers.write_csv(), flush=True)
    print("\nAUDIT_REPORT\n" + json.dumps(report, indent=2), flush=True)
    return summary_frame, top_outliers, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/audits/source_coordinate_bounds"),
    )
    parser.add_argument("--snapshot-uri")
    args = parser.parse_args()
    _, _, report = run_audit(
        crime_lake=CrimeLakeResources(),
        output_dir=args.output_dir,
        snapshot_uri=args.snapshot_uri,
    )
    unsafe = report["sources_exceeding_safety_threshold"]
    if unsafe:
        raise RuntimeError(
            f"Source-coordinate-bounds safety threshold exceeded: {unsafe}"
        )


if __name__ == "__main__":
    main()
