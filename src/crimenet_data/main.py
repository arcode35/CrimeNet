#!/usr/bin/env python3

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass

import boto3
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq


# =============================================================================
# CONFIG
# =============================================================================

BUCKET = "crimenet-data"

# Static national OSM/static feature store.
#
# Adjust ONLY this prefix if your static store has a different path.
STATIC_PREFIX = "gold/national_feature_store/latest/h3_r9/"

# Confirmed from your B2 listing.
SOCIO_ANNUAL_PREFIX = (
    "gold/national_feature_store/temporal/h3_r9/annual/"
)

YEARS = range(2014, 2027)

H3_CANDIDATES = (
    "h3_cell_id",
    "h3",
    "h3_r9",
    "osm_h3_cell_id",
    "h3_index",
)

OSM_FEATURES = (
    "osm_poi_density_per_km2",
    "osm_nightlife_poi_density_per_km2",
    "osm_food_poi_density_per_km2",
    "osm_retail_poi_density_per_km2",
    "osm_transit_poi_density_per_km2",
    "osm_road_length_density_m_per_km2",
    "osm_major_road_density_m_per_km2",
    "osm_intersection_density_per_km2",
    "osm_dead_end_density_per_km2",
    "osm_building_density_per_km2",
    "osm_major_road_length_ratio",
    "osm_residential_road_length_ratio",
    "osm_service_road_length_ratio",
    "osm_one_way_road_length_ratio",
    "osm_tracked_poi_category_entropy",
    "osm_land_use_category_entropy",
    "osm_commercial_residential_mix_ratio",
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


# =============================================================================
# HELPERS
# =============================================================================

@dataclass
class StoreSummary:
    rows: int = 0
    unique_h3: int = 0
    duplicate_rows: int = 0
    complete_rows: int = 0


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def make_clients():
    endpoint = os.environ["B2_ENDPOINT_URL"]
    key_id = os.environ["B2_KEY_ID"]
    application_key = os.environ["B2_APPLICATION_KEY"]
    region = os.environ.get(
        "B2_REGION",
        "us-east-005",
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=application_key,
        region_name=region,
    )

    arrow_fs = pafs.S3FileSystem(
        access_key=key_id,
        secret_key=application_key,
        region=region,
        endpoint_override=endpoint,
        scheme="https",
    )

    return s3, arrow_fs


def list_parquet_keys(
    s3,
    prefix: str,
) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")

    keys = []

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.lower().endswith(".parquet"):
                keys.append(key)

    return sorted(keys)


def discover_schema(
    arrow_fs,
    key: str,
) -> pa.Schema:
    path = f"{BUCKET}/{key}"

    with arrow_fs.open_input_file(path) as f:
        return pq.ParquetFile(f).schema_arrow


def find_h3_column(
    schema: pa.Schema,
) -> str:
    names = set(schema.names)

    for candidate in H3_CANDIDATES:
        if candidate in names:
            return candidate

    # Last-resort discovery.
    for name in schema.names:
        lower = name.lower()

        if "h3" in lower and (
            "cell" in lower
            or "index" in lower
            or lower == "h3"
        ):
            return name

    raise KeyError(
        "Could not identify H3 column. "
        f"Available columns: {schema.names}"
    )


def valid_numeric_mask(
    arr: pa.Array,
) -> pa.Array:
    valid = pc.invert(
        pc.is_null(arr)
    )

    if pa.types.is_floating(
        arr.type
    ):
        valid = pc.and_(
            valid,
            pc.is_finite(arr),
        )

    return valid


def count_true(
    arr: pa.Array,
) -> int:
    value = pc.sum(
        arr.cast(pa.int64())
    )

    if not value.is_valid:
        return 0

    return int(value.as_py())


def normalize_h3(value) -> str | None:
    if value is None:
        return None

    return str(value)


def combine_masks(
    masks: list[pa.Array],
) -> pa.Array:
    if not masks:
        raise ValueError(
            "Cannot combine empty mask list"
        )

    result = masks[0]

    for mask in masks[1:]:
        result = pc.and_(
            result,
            mask,
        )

    return result


# =============================================================================
# OSM AUDIT
# =============================================================================

def audit_osm(
    s3,
    arrow_fs,
):
    keys = list_parquet_keys(
        s3,
        STATIC_PREFIX,
    )

    if not keys:
        raise RuntimeError(
            f"No Parquet files under "
            f"s3://{BUCKET}/{STATIC_PREFIX}"
        )

    schema = discover_schema(
        arrow_fs,
        keys[0],
    )

    h3_col = find_h3_column(
        schema
    )

    available = set(
        schema.names
    )

    osm_features = [
        col
        for col in OSM_FEATURES
        if col in available
    ]

    if not osm_features:
        raise RuntimeError(
            "No expected OSM feature columns "
            "found in static store.\n"
            f"Schema:\n{schema}"
        )

    missing_expected = sorted(
        set(OSM_FEATURES)
        - set(osm_features)
    )

    print()
    print("=" * 110)
    print("OSM STATIC NATIONAL COVERAGE")
    print("=" * 110)

    print(
        f"Prefix: s3://{BUCKET}/"
        f"{STATIC_PREFIX}"
    )
    print(f"Shards: {len(keys):,}")
    print(f"H3 key: {h3_col}")
    print(
        f"OSM features found: "
        f"{len(osm_features):,}"
    )

    if missing_expected:
        print(
            "Expected OSM columns not found:"
        )

        for col in missing_expected:
            print(f"  {col}")

    rows = 0
    complete_rows = 0

    feature_valid = Counter()
    feature_null = Counter()
    feature_nonfinite = Counter()

    h3_cells: set[str] = set()
    duplicate_rows = 0

    columns = [
        h3_col,
        *osm_features,
    ]

    for i, key in enumerate(
        keys,
        start=1,
    ):
        path = f"{BUCKET}/{key}"

        with arrow_fs.open_input_file(
            path
        ) as f:
            pf = pq.ParquetFile(f)

            for batch in pf.iter_batches(
                columns=columns,
                batch_size=100_000,
            ):
                table = pa.Table.from_batches(
                    [batch]
                )

                rows += table.num_rows

                h3_values = (
                    table[h3_col]
                    .combine_chunks()
                    .to_pylist()
                )

                for raw_h3 in h3_values:
                    h3 = normalize_h3(
                        raw_h3
                    )

                    if h3 is None:
                        continue

                    if h3 in h3_cells:
                        duplicate_rows += 1
                    else:
                        h3_cells.add(h3)

                masks = []

                for feature in osm_features:
                    arr = (
                        table[feature]
                        .combine_chunks()
                    )

                    mask = (
                        valid_numeric_mask(
                            arr
                        )
                    )

                    masks.append(mask)

                    feature_valid[
                        feature
                    ] += count_true(
                        mask
                    )

                    feature_null[
                        feature
                    ] += arr.null_count

                    if pa.types.is_floating(
                        arr.type
                    ):
                        nonnull = pc.invert(
                            pc.is_null(arr)
                        )

                        nonfinite = pc.and_(
                            nonnull,
                            pc.invert(
                                pc.is_finite(
                                    arr
                                )
                            ),
                        )

                        feature_nonfinite[
                            feature
                        ] += count_true(
                            nonfinite
                        )

                complete_rows += count_true(
                    combine_masks(masks)
                )

        if (
            i == 1
            or i % 25 == 0
            or i == len(keys)
        ):
            print(
                f"  processed "
                f"{i:,}/{len(keys):,} shards"
            )

    print()
    print(
        f"{'Rows':<48}"
        f"{rows:>16,}"
    )
    print(
        f"{'Unique H3 cells':<48}"
        f"{len(h3_cells):>16,}"
    )
    print(
        f"{'Duplicate H3 rows':<48}"
        f"{duplicate_rows:>16,}"
    )
    print(
        f"{'Rows with every OSM feature valid':<48}"
        f"{complete_rows:>16,}  "
        f"{pct(complete_rows, rows):8.4f}%"
    )

    print()
    print(
        f"{'OSM feature':<52}"
        f"{'valid':>14}"
        f"{'coverage':>14}"
        f"{'null':>14}"
        f"{'nonfinite':>14}"
    )
    print("-" * 110)

    for feature in osm_features:
        print(
            f"{feature:<52}"
            f"{feature_valid[feature]:>14,}"
            f"{pct(feature_valid[feature], rows):>13.5f}%"
            f"{feature_null[feature]:>14,}"
            f"{feature_nonfinite[feature]:>14,}"
        )

    return h3_cells


# =============================================================================
# SOCIOECONOMIC AUDIT
# =============================================================================

def audit_socio_year(
    s3,
    arrow_fs,
    year: int,
    osm_universe: set[str],
):
    prefix = (
        f"{SOCIO_ANNUAL_PREFIX}"
        f"as_of_year={year}/"
    )

    keys = list_parquet_keys(
        s3,
        prefix,
    )

    if not keys:
        return {
            "year": year,
            "shards": 0,
            "rows": 0,
            "unique_h3": 0,
            "duplicates": 0,
            "osm_overlap": 0,
            "spatial_coverage": 0.0,
            "complete_socio": 0,
            "complete_coverage": 0.0,
            "missing_from_osm_universe": len(
                osm_universe
            ),
            "extra_cells": 0,
            "feature_valid": {},
            "feature_null": {},
            "feature_nonfinite": {},
        }

    schema = discover_schema(
        arrow_fs,
        keys[0],
    )

    h3_col = find_h3_column(
        schema
    )

    available = set(
        schema.names
    )

    socio_features = [
        feature
        for feature in SOCIO_FEATURES
        if feature in available
    ]

    if not socio_features:
        raise RuntimeError(
            f"{year}: no expected socioeconomic "
            f"columns found.\n"
            f"Schema:\n{schema}"
        )

    rows = 0
    complete_socio = 0

    feature_valid = Counter()
    feature_null = Counter()
    feature_nonfinite = Counter()

    h3_cells: set[str] = set()
    duplicate_rows = 0

    columns = [
        h3_col,
        *socio_features,
    ]

    for key in keys:
        path = f"{BUCKET}/{key}"

        with arrow_fs.open_input_file(
            path
        ) as f:
            pf = pq.ParquetFile(f)

            for batch in pf.iter_batches(
                columns=columns,
                batch_size=100_000,
            ):
                table = (
                    pa.Table
                    .from_batches([batch])
                )

                rows += table.num_rows

                h3_values = (
                    table[h3_col]
                    .combine_chunks()
                    .to_pylist()
                )

                for raw_h3 in h3_values:
                    h3 = normalize_h3(
                        raw_h3
                    )

                    if h3 is None:
                        continue

                    if h3 in h3_cells:
                        duplicate_rows += 1
                    else:
                        h3_cells.add(h3)

                masks = []

                for feature in (
                    socio_features
                ):
                    arr = (
                        table[feature]
                        .combine_chunks()
                    )

                    mask = (
                        valid_numeric_mask(
                            arr
                        )
                    )

                    masks.append(mask)

                    feature_valid[
                        feature
                    ] += count_true(
                        mask
                    )

                    feature_null[
                        feature
                    ] += arr.null_count

                    if pa.types.is_floating(
                        arr.type
                    ):
                        nonnull = pc.invert(
                            pc.is_null(arr)
                        )

                        nonfinite = pc.and_(
                            nonnull,
                            pc.invert(
                                pc.is_finite(
                                    arr
                                )
                            ),
                        )

                        feature_nonfinite[
                            feature
                        ] += (
                            count_true(
                                nonfinite
                            )
                        )

                complete_socio += (
                    count_true(
                        combine_masks(
                            masks
                        )
                    )
                )

    overlap = (
        h3_cells
        & osm_universe
    )

    missing = (
        osm_universe
        - h3_cells
    )

    extra = (
        h3_cells
        - osm_universe
    )

    return {
        "year": year,
        "shards": len(keys),
        "rows": rows,
        "unique_h3": len(h3_cells),
        "duplicates": duplicate_rows,
        "osm_overlap": len(overlap),
        "spatial_coverage": pct(
            len(overlap),
            len(osm_universe),
        ),
        "complete_socio": complete_socio,
        "complete_coverage": pct(
            complete_socio,
            len(osm_universe),
        ),
        "missing_from_osm_universe": len(
            missing
        ),
        "extra_cells": len(extra),
        "feature_valid": dict(
            feature_valid
        ),
        "feature_null": dict(
            feature_null
        ),
        "feature_nonfinite": dict(
            feature_nonfinite
        ),
        "socio_features": socio_features,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    s3, arrow_fs = make_clients()

    print("=" * 110)
    print(
        "CRIMENET NATIONAL OSM + "
        "SOCIOECONOMIC COVERAGE AUDIT"
    )
    print("=" * 110)

    # -----------------------------------------------------------------
    # OSM establishes the static national H3 universe.
    # -----------------------------------------------------------------

    osm_universe = audit_osm(
        s3,
        arrow_fs,
    )

    print()
    print("=" * 110)
    print(
        "ANNUAL SOCIOECONOMIC COVERAGE "
        "AGAINST OSM H3 UNIVERSE"
    )
    print("=" * 110)

    print(
        f"\nOSM national H3 universe: "
        f"{len(osm_universe):,} cells\n"
    )

    results = []

    for year in YEARS:
        print(
            f"Auditing socioeconomic "
            f"year {year}..."
        )

        result = audit_socio_year(
            s3,
            arrow_fs,
            year,
            osm_universe,
        )

        results.append(
            result
        )

    # -----------------------------------------------------------------
    # Annual summary
    # -----------------------------------------------------------------

    print()
    print("=" * 125)
    print("ANNUAL SUMMARY")
    print("=" * 125)

    print(
        f"{'year':<7}"
        f"{'rows':>15}"
        f"{'unique H3':>15}"
        f"{'duplicates':>13}"
        f"{'H3 overlap':>15}"
        f"{'spatial %':>12}"
        f"{'complete socio':>18}"
        f"{'complete %':>12}"
        f"{'missing':>14}"
        f"{'extra':>12}"
    )

    print("-" * 125)

    for r in results:
        print(
            f"{r['year']:<7}"
            f"{r['rows']:>15,}"
            f"{r['unique_h3']:>15,}"
            f"{r['duplicates']:>13,}"
            f"{r['osm_overlap']:>15,}"
            f"{r['spatial_coverage']:>11.4f}%"
            f"{r['complete_socio']:>18,}"
            f"{r['complete_coverage']:>11.4f}%"
            f"{r['missing_from_osm_universe']:>14,}"
            f"{r['extra_cells']:>12,}"
        )

    # -----------------------------------------------------------------
    # Per-feature / per-year detail
    # -----------------------------------------------------------------

    print()
    print("=" * 125)
    print(
        "SOCIOECONOMIC FEATURE COMPLETENESS "
        "BY YEAR"
    )
    print("=" * 125)

    for r in results:
        if not r.get(
            "socio_features"
        ):
            continue

        print()
        print(
            f"YEAR {r['year']}"
        )
        print("-" * 90)

        print(
            f"{'feature':<45}"
            f"{'valid':>14}"
            f"{'row coverage':>16}"
            f"{'null':>14}"
            f"{'nonfinite':>14}"
        )

        for feature in (
            r["socio_features"]
        ):
            valid = (
                r["feature_valid"]
                .get(feature, 0)
            )

            nulls = (
                r["feature_null"]
                .get(feature, 0)
            )

            nonfinite = (
                r["feature_nonfinite"]
                .get(feature, 0)
            )

            print(
                f"{feature:<45}"
                f"{valid:>14,}"
                f"{pct(valid, r['rows']):>15.5f}%"
                f"{nulls:>14,}"
                f"{nonfinite:>14,}"
            )

    # -----------------------------------------------------------------
    # Overall temporal stats
    # -----------------------------------------------------------------

    valid_results = [
        r
        for r in results
        if r["rows"] > 0
    ]

    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    print(
        f"OSM national universe: "
        f"{len(osm_universe):,} H3-r9 cells"
    )

    print(
        f"Socio years present:   "
        f"{len(valid_results):,}/"
        f"{len(list(YEARS))}"
    )

    if valid_results:
        min_spatial = min(
            r["spatial_coverage"]
            for r in valid_results
        )

        max_spatial = max(
            r["spatial_coverage"]
            for r in valid_results
        )

        mean_spatial = sum(
            r["spatial_coverage"]
            for r in valid_results
        ) / len(valid_results)

        min_complete = min(
            r["complete_coverage"]
            for r in valid_results
        )

        max_complete = max(
            r["complete_coverage"]
            for r in valid_results
        )

        mean_complete = sum(
            r["complete_coverage"]
            for r in valid_results
        ) / len(valid_results)

        print()
        print(
            "Socio spatial coverage:"
        )
        print(
            f"  minimum: {min_spatial:.5f}%"
        )
        print(
            f"  mean:    {mean_spatial:.5f}%"
        )
        print(
            f"  maximum: {max_spatial:.5f}%"
        )

        print()
        print(
            "Complete socioeconomic coverage:"
        )
        print(
            f"  minimum: {min_complete:.5f}%"
        )
        print(
            f"  mean:    {mean_complete:.5f}%"
        )
        print(
            f"  maximum: {max_complete:.5f}%"
        )

    print()
    print("=" * 110)
    print("DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()