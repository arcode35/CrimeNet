#!/usr/bin/env python3
"""
Seed national CrimeNet source data into Backblaze B2.

Uploads to B2 bucket `crimenet-data` by default:

  bronze/osm/us/snapshot_year=YYYY/us-YY0101.osm.pbf
  bronze/osm/us/latest/us-latest.osm.pbf
  bronze/acs/acs5/tract/vintage=YYYY/tracts.parquet
  bronze/tiger/tract/year=YYYY/state_fips=SS/tl_YYYY_SS_tract.zip

Defaults are chosen for CrimeNet's historical window:
  * OSM yearly snapshots: 2014..current year, plus latest
  * ACS 5-year tract vintages: 2014..2024 (2024 is the latest ACS5 vintage
    available when this script was written)
  * TIGER tract geometry: matching ACS vintages, 2014..2024

The H3-r9 -> tract GEOID mapping is intentionally NOT generated here. It is a
Silver/derived spatial index, not a source dataset. The TIGER tract polygons
uploaded by this script are the geometry source needed to build that mapping.

Required environment variables:
  B2_KEY_ID
  B2_APPLICATION_KEY
  B2_ENDPOINT_URL      e.g. https://s3.us-east-005.backblazeb2.com
  CENSUS_API_KEY

Optional:
  B2_BUCKET            default: crimenet-data

Dependencies:
  python >= 3.11
  boto3
  botocore
  requests
  polars
  pyarrow

Example:
  uv pip install boto3 requests polars pyarrow

  export B2_KEY_ID='...'
  export B2_APPLICATION_KEY='...'
  export B2_ENDPOINT_URL='https://s3.us-east-005.backblazeb2.com'
  export CENSUS_API_KEY='...'

  python seed_crimenet_national_b2.py

Safe to rerun: immutable snapshot/TIGER/ACS keys are skipped if already in B2.
Use --refresh-latest-osm to overwrite the mutable `latest` OSM object.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import boto3
import polars as pl
import requests
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError


# 50 states + District of Columbia. Territories can be enabled explicitly.
STATE_FIPS_50_DC = (
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56",
)
TERRITORY_FIPS = ("60", "66", "69", "72", "78")

# ACS detailed-table variables used by the existing CrimeNet socioeconomic schema.
# E = estimate, M = margin of error.
ACS_COLUMNS: dict[str, tuple[str, str, str]] = {
    "population": ("B01003_001E", "B01003_001M", "int"),
    "median_age": ("B01002_001E", "B01002_001M", "float"),
    "median_household_income": ("B19013_001E", "B19013_001M", "float"),
    "poverty_universe": ("B17001_001E", "B17001_001M", "int"),
    "population_below_poverty": ("B17001_002E", "B17001_002M", "int"),
    "civilian_labor_force": ("B23025_003E", "B23025_003M", "int"),
    "unemployed_population": ("B23025_005E", "B23025_005M", "int"),
    "housing_units": ("B25002_001E", "B25002_001M", "int"),
    "occupied_housing_units": ("B25002_002E", "B25002_002M", "int"),
    "vacant_housing_units": ("B25002_003E", "B25002_003M", "int"),
    "occupied_units_tenure_universe": ("B25003_001E", "B25003_001M", "int"),
    "renter_occupied_units": ("B25003_003E", "B25003_003M", "int"),
    "households_vehicle_universe": ("B08201_001E", "B08201_001M", "int"),
    "households_no_vehicle": ("B08201_002E", "B08201_002M", "int"),
}

OUTPUT_SCHEMA_ORDER = [
    "geoid",
    "geography_name",
    "county_fips",
    "tract_code",
    "period_start_year",
    "period_end_year",
    "geography_type",
    "population",
    "population_moe",
    "median_age",
    "median_age_moe",
    "median_household_income",
    "median_household_income_moe",
    "poverty_universe",
    "poverty_universe_moe",
    "population_below_poverty",
    "population_below_poverty_moe",
    "civilian_labor_force",
    "civilian_labor_force_moe",
    "unemployed_population",
    "unemployed_population_moe",
    "housing_units",
    "housing_units_moe",
    "occupied_housing_units",
    "occupied_housing_units_moe",
    "vacant_housing_units",
    "vacant_housing_units_moe",
    "occupied_units_tenure_universe",
    "occupied_units_tenure_universe_moe",
    "renter_occupied_units",
    "renter_occupied_units_moe",
    "households_vehicle_universe",
    "households_vehicle_universe_moe",
    "households_no_vehicle",
    "households_no_vehicle_moe",
    "poverty_rate",
    "unemployment_rate",
    "vacancy_rate",
    "renter_occupied_rate",
    "no_vehicle_rate",
    "source_file",
    "bronze_ingested_at",
    "metro",
    "acs_vintage",
    "state_fips",
]

PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{stamp}] {msg}", flush=True)


def parse_year_range(spec: str) -> list[int]:
    spec = spec.strip()
    if ":" in spec:
        a, b = spec.split(":", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError(f"bad year range: {spec}")
        return list(range(start, end + 1))
    return sorted({int(x) for x in spec.split(",") if x.strip()})


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def b2_region_from_endpoint(endpoint: str) -> str | None:
    host = urlparse(endpoint).hostname or ""
    # s3.us-east-005.backblazeb2.com -> us-east-005
    parts = host.split(".")
    if len(parts) >= 4 and parts[0] == "s3":
        return parts[1]
    return os.environ.get("B2_REGION")


def make_s3_client():
    endpoint = required_env("B2_ENDPOINT_URL").rstrip("/")
    region = b2_region_from_endpoint(endpoint)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=required_env("B2_KEY_ID"),
        aws_secret_access_key=required_env("B2_APPLICATION_KEY"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 12, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=180,
            max_pool_connections=64,
            s3={"addressing_style": "path"},
        ),
    )


TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=64 * 1024**2,
    max_concurrency=8,
    use_threads=True,
)


def b2_object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_file(s3, bucket: str, path: Path, key: str, *, metadata: dict[str, str] | None = None) -> None:
    log(f"B2 upload -> b2://{bucket}/{key} ({path.stat().st_size / 1024**3:.2f} GiB)")
    extra = {"Metadata": metadata or {}}
    s3.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs=extra,
        Config=TRANSFER_CONFIG,
    )


def put_json(s3, bucket: str, key: str, obj: object) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    attempts: int = 8,
    **kwargs,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = session.request(method, url, **kwargs)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt == attempts:
                break
            delay = min(60.0, 2.0 ** (attempt - 1))
            log(f"retry {attempt}/{attempts} for {url}: {exc}; sleeping {delay:.0f}s")
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def download_resumable(url: str, dest: Path, *, session: requests.Session | None = None) -> Path:
    """Download URL to dest with HTTP Range resume and retries."""
    own_session = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "CrimeNet-national-seed/1.1", "Accept-Encoding": "identity"})
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        for outer_attempt in range(1, 9):
            existing = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
            try:
                resp = session.get(url, stream=True, timeout=(30, 180), headers=headers)
                if resp.status_code == 416 and existing > 0:
                    log(f"download already complete? {dest.name} ({existing / 1024**3:.2f} GiB)")
                    return dest
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()

                # If server ignored Range, restart the local file.
                append = existing > 0 and resp.status_code == 206
                mode = "ab" if append else "wb"
                if not append:
                    existing = 0

                total_header = resp.headers.get("Content-Range") or resp.headers.get("Content-Length")
                total: int | None = None
                if resp.headers.get("Content-Range") and "/" in resp.headers["Content-Range"]:
                    try:
                        total = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
                    except ValueError:
                        total = None
                elif resp.headers.get("Content-Length"):
                    try:
                        total = existing + int(resp.headers["Content-Length"])
                    except ValueError:
                        total = None

                log(
                    f"download <- {url} | resume={existing / 1024**3:.2f} GiB"
                    + (f" total={total / 1024**3:.2f} GiB" if total else "")
                )
                written = existing
                last_report = time.monotonic()
                with dest.open(mode) as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024**2):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 20:
                            if total:
                                pct = 100.0 * written / total
                                log(f"  {dest.name}: {written/1024**3:.2f}/{total/1024**3:.2f} GiB ({pct:.1f}%)")
                            else:
                                log(f"  {dest.name}: {written/1024**3:.2f} GiB")
                            last_report = now

                if total is not None and dest.stat().st_size != total:
                    raise IOError(f"short download: got {dest.stat().st_size}, expected {total}")
                return dest

            except (requests.RequestException, OSError, IOError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                if outer_attempt == 8:
                    raise
                delay = min(60.0, 2.0 ** (outer_attempt - 1))
                log(f"download retry {outer_attempt}/8: {url}: {exc}; sleeping {delay:.0f}s")
                time.sleep(delay)
        raise RuntimeError("unreachable")
    finally:
        if own_session:
            session.close()


def source_to_b2(
    s3,
    bucket: str,
    url: str,
    key: str,
    work_dir: Path,
    *,
    force: bool = False,
) -> str:
    if not force and b2_object_exists(s3, bucket, key):
        log(f"SKIP exists b2://{bucket}/{key}")
        return "exists"

    filename = key.rsplit("/", 1)[-1]
    # Avoid collisions when many TIGER files are processed concurrently.
    token = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    local = work_dir / f"{token}-{filename}"
    try:
        download_resumable(url, local)
        upload_file(
            s3,
            bucket,
            local,
            key,
            metadata={"source-url": url[:1900]},
        )
        log(f"DONE b2://{bucket}/{key}")
        return "uploaded"
    finally:
        local.unlink(missing_ok=True)


def sanitize_numeric_expr(raw_col: str, out_col: str, dtype: pl.DataType) -> pl.Expr:
    x = pl.col(raw_col).cast(pl.Float64, strict=False)
    # Census uses large negative sentinels such as -666666666 / -999999999.
    cleaned = pl.when(x <= -100_000_000).then(None).otherwise(x)
    return cleaned.cast(dtype, strict=False).alias(out_col)


def rate_expr(numerator: str, denominator: str, out: str) -> pl.Expr:
    return (
        pl.when(
            pl.col(denominator).is_not_null()
            & (pl.col(denominator) > 0)
            & pl.col(numerator).is_not_null()
            & (pl.col(numerator) >= 0)
        )
        .then(pl.col(numerator).cast(pl.Float64) / pl.col(denominator).cast(pl.Float64))
        .otherwise(None)
        .alias(out)
    )


def fetch_acs_state(year: int, state_fips: str, census_key: str, session: requests.Session) -> pl.DataFrame:
    estimate_and_moe: list[str] = []
    for est, moe, _kind in ACS_COLUMNS.values():
        estimate_and_moe.extend([est, moe])

    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(estimate_and_moe),
        "for": "tract:*",
        "in": f"state:{state_fips}",
        "key": census_key,
    }
    resp = request_with_retry(session, "GET", base_url, params=params, timeout=(30, 180))
    payload = resp.json()
    if not payload or len(payload) < 2:
        raise RuntimeError(f"empty ACS response for year={year} state={state_fips}")
    header, rows = payload[0], payload[1:]
    return pl.DataFrame(rows, schema=header, orient="row")


def build_acs_year(year: int, state_fipses: Iterable[str], census_key: str, workers: int) -> pl.DataFrame:
    log(f"ACS5 {year}: fetching tract data")
    session = requests.Session()
    session.headers.update({"User-Agent": "CrimeNet-national-seed/1.1"})

    # Census API is not a bulk-download CDN. Keep concurrency conservative.
    acs_workers = max(1, min(workers, 6))

    def one(state: str) -> tuple[str, pl.DataFrame]:
        # Per-thread Session avoids sharing connection state.
        s = requests.Session()
        s.headers.update({"User-Agent": "CrimeNet-national-seed/1.1"})
        try:
            return state, fetch_acs_state(year, state, census_key, s)
        finally:
            s.close()

    frames: list[tuple[str, pl.DataFrame]] = []
    with cf.ThreadPoolExecutor(max_workers=acs_workers) as pool:
        futs = {pool.submit(one, state): state for state in state_fipses}
        for fut in cf.as_completed(futs):
            state, frame = fut.result()
            frames.append((state, frame))
            log(f"ACS5 {year}: state {state}: {frame.height:,} tracts")
    session.close()

    # State is also returned by API; stable ordering makes output deterministic.
    raw = pl.concat([f for _s, f in sorted(frames, key=lambda x: x[0])], how="vertical")

    exprs: list[pl.Expr] = []
    for out_name, (est, moe, kind) in ACS_COLUMNS.items():
        dtype = pl.Int64 if kind == "int" else pl.Float64
        exprs.append(sanitize_numeric_expr(est, out_name, dtype))
        # Median variables need floating MOE; count MOEs remain int to match existing schema.
        moe_dtype = pl.Float64 if kind == "float" else pl.Int64
        exprs.append(sanitize_numeric_expr(moe, f"{out_name}_moe", moe_dtype))

    ingested_at = datetime.now(timezone.utc).replace(tzinfo=None)
    transformed = (
        raw.with_columns(exprs)
        .with_columns(
            (pl.col("state") + pl.col("county") + pl.col("tract")).alias("geoid"),
            pl.col("NAME").alias("geography_name"),
            (pl.col("state") + pl.col("county")).alias("county_fips"),
            pl.col("tract").alias("tract_code"),
            pl.lit(year - 4, dtype=pl.Int64).alias("period_start_year"),
            pl.lit(year, dtype=pl.Int64).alias("period_end_year"),
            pl.lit("tract").alias("geography_type"),
            pl.lit(f"census_api_acs5_{year}_tract").alias("source_file"),
            pl.lit(ingested_at).cast(pl.Datetime("ns")).alias("bronze_ingested_at"),
            pl.lit(None, dtype=pl.String).alias("metro"),
            pl.lit(year, dtype=pl.Int64).alias("acs_vintage"),
            pl.col("state").alias("state_fips"),
        )
        .with_columns(
            rate_expr("population_below_poverty", "poverty_universe", "poverty_rate"),
            rate_expr("unemployed_population", "civilian_labor_force", "unemployment_rate"),
            rate_expr("vacant_housing_units", "housing_units", "vacancy_rate"),
            rate_expr("renter_occupied_units", "occupied_units_tenure_universe", "renter_occupied_rate"),
            rate_expr("households_no_vehicle", "households_vehicle_universe", "no_vehicle_rate"),
        )
        .select(OUTPUT_SCHEMA_ORDER)
        .sort("geoid")
    )

    if transformed.select(pl.col("geoid").n_unique()).item() != transformed.height:
        raise RuntimeError(f"ACS5 {year}: duplicate tract GEOIDs detected")

    return transformed


def upload_acs_year(
    s3,
    bucket: str,
    year: int,
    state_fipses: Iterable[str],
    census_key: str,
    work_dir: Path,
    workers: int,
    *,
    force: bool,
) -> str:
    key = f"bronze/acs/acs5/tract/vintage={year}/tracts.parquet"
    if not force and b2_object_exists(s3, bucket, key):
        log(f"SKIP exists b2://{bucket}/{key}")
        return "exists"

    df = build_acs_year(year, state_fipses, census_key, workers)
    local = work_dir / f"acs5_{year}_us_tracts.parquet"
    try:
        df.write_parquet(local, compression="zstd", statistics=True)
        log(f"ACS5 {year}: {df.height:,} rows, {local.stat().st_size/1024**2:.1f} MiB")
        upload_file(
            s3,
            bucket,
            local,
            key,
            metadata={
                "source": "US Census ACS 5-year Detailed Tables API",
                "geography": "tract",
                "vintage": str(year),
                "rate-scale": "fraction-0-to-1",
            },
        )
        return "uploaded"
    finally:
        local.unlink(missing_ok=True)


def osm_source_for_year(year: int) -> tuple[str, str]:
    """Return (source URL, B2 key) for the annual OSM snapshot.

    Geofabrik only exposes whole-US annual aggregate snapshots from 2021 onward.
    For 2014-2020, use the corresponding North America aggregate snapshot; downstream
    national-US processing should clip/filter to the US H3 domain.
    """
    yy = year % 100
    if year <= 2020:
        filename = f"north-america-{yy:02d}0101.osm.pbf"
        return (
            f"https://download.geofabrik.de/{filename}",
            f"bronze/osm/north-america/snapshot_year={year}/{filename}",
        )

    filename = f"us-{yy:02d}0101.osm.pbf"
    return (
        f"https://download.geofabrik.de/north-america/{filename}",
        f"bronze/osm/us/snapshot_year={year}/{filename}",
    )


def tiger_url(year: int, state_fips: str) -> str:
    return f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"


def tiger_key(year: int, state_fips: str) -> str:
    return f"bronze/tiger/tract/year={year}/state_fips={state_fips}/tl_{year}_{state_fips}_tract.zip"


def main() -> int:
    current_year = datetime.now(timezone.utc).year
    parser = argparse.ArgumentParser(description="Seed national CrimeNet datasets into Backblaze B2")
    parser.add_argument("--bucket", default=os.environ.get("B2_BUCKET", "crimenet-data"))
    parser.add_argument("--work-dir", default=None, help="Temporary working directory; default uses system temp")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent TIGER / ACS state workers")
    parser.add_argument("--acs-years", default="2014:2024")
    parser.add_argument("--tiger-years", default="2014:2024")
    parser.add_argument("--osm-years", default=f"2014:{current_year}")
    parser.add_argument("--include-territories", action="store_true")
    parser.add_argument("--skip-osm", action="store_true")
    parser.add_argument("--skip-acs", action="store_true")
    parser.add_argument("--skip-tiger", action="store_true")
    parser.add_argument("--skip-latest-osm", action="store_true")
    parser.add_argument("--refresh-latest-osm", action="store_true")
    parser.add_argument("--overwrite-acs", action="store_true")
    parser.add_argument("--overwrite-tiger", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    s3 = make_s3_client()
    bucket = args.bucket
    census_key = None if args.skip_acs else required_env("CENSUS_API_KEY")

    states = list(STATE_FIPS_50_DC)
    if args.include_territories:
        states.extend(TERRITORY_FIPS)

    # Validate bucket access immediately.
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception as exc:
        raise SystemExit(f"Cannot access B2 bucket {bucket!r}: {exc}") from exc

    root_context = tempfile.TemporaryDirectory(prefix="crimenet-national-") if args.work_dir is None else None
    work_dir = Path(root_context.name if root_context else args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "states": states,
        "acs_years": parse_year_range(args.acs_years),
        "tiger_years": parse_year_range(args.tiger_years),
        "osm_years": parse_year_range(args.osm_years),
        "include_territories": args.include_territories,
        "outputs": {},
    }

    try:
        if not args.skip_acs:
            assert census_key is not None
            acs_results = {}
            for year in parse_year_range(args.acs_years):
                acs_results[str(year)] = upload_acs_year(
                    s3,
                    bucket,
                    year,
                    states,
                    census_key,
                    work_dir,
                    args.workers,
                    force=args.overwrite_acs,
                )
            manifest["outputs"]["acs"] = acs_results  # type: ignore[index]

        if not args.skip_tiger:
            tiger_jobs = [
                (year, state)
                for year in parse_year_range(args.tiger_years)
                for state in states
            ]
            log(f"TIGER tract files: {len(tiger_jobs):,} objects")
            tiger_counts = {"uploaded": 0, "exists": 0}

            def tiger_one(job: tuple[int, str]) -> str:
                year, state = job
                return source_to_b2(
                    s3,
                    bucket,
                    tiger_url(year, state),
                    tiger_key(year, state),
                    work_dir,
                    force=args.overwrite_tiger,
                )

            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(tiger_one, tiger_jobs):
                    tiger_counts[result] = tiger_counts.get(result, 0) + 1
            manifest["outputs"]["tiger"] = tiger_counts  # type: ignore[index]

        if not args.skip_osm:
            # OSM is deliberately processed serially: each snapshot is multi-GB and
            # we only need enough local disk for one national PBF at a time.
            osm_results = {}
            for year in parse_year_range(args.osm_years):
                osm_url, osm_key = osm_source_for_year(year)
                result = source_to_b2(
                    s3,
                    bucket,
                    osm_url,
                    osm_key,
                    work_dir,
                    force=False,
                )
                osm_results[str(year)] = {
                    "result": result,
                    "source_url": osm_url,
                    "b2_key": osm_key,
                    "source_region": "north-america" if year <= 2020 else "us",
                }

            if not args.skip_latest_osm:
                latest_url = "https://download.geofabrik.de/north-america/us-latest.osm.pbf"
                latest_key = "bronze/osm/us/latest/us-latest.osm.pbf"
                osm_results["latest"] = source_to_b2(
                    s3,
                    bucket,
                    latest_url,
                    latest_key,
                    work_dir,
                    force=args.refresh_latest_osm,
                )
            manifest["outputs"]["osm"] = osm_results  # type: ignore[index]

        put_json(s3, bucket, "bronze/_manifests/national_seed_manifest.json", manifest)
        log("ALL DONE")
        log(f"manifest -> b2://{bucket}/bronze/_manifests/national_seed_manifest.json")
        return 0
    finally:
        if root_context is not None:
            root_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
