from __future__ import annotations

import csv
import fnmatch
import glob
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from crimenet_data.assets.crime.sources.base import SourcePattern

LOG = logging.getLogger(__name__)
S3ClientFactory = Callable[[], object]


def _split_reader_options(
    read_options: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    options = dict(read_options)
    strategy = str(options.pop("strategy", "native"))
    return strategy, options

def _fnmatch_globstar(path: str, pattern: str) -> bool:
    """Match S3 object keys while allowing **/ to represent zero directories."""

    if fnmatch.fnmatchcase(path, pattern):
        return True

    # fnmatch doesn't give **/ glob.glob(recursive=True) semantics.
    # Allow each "/**/" segment to match zero directory components.
    reduced = pattern
    while "/**/" in reduced:
        reduced = reduced.replace("/**/", "/", 1)
        if fnmatch.fnmatchcase(path, reduced):
            return True

    # Also support patterns beginning with **/.
    if pattern.startswith("**/"):
        reduced = pattern
        while reduced.startswith("**/"):
            reduced = reduced[3:]
            if fnmatch.fnmatchcase(path, reduced):
                return True

    return False

def _iter_object_uris(
    uri_pattern: str, s3_client_factory: S3ClientFactory
) -> list[str]:
    if not uri_pattern.startswith("s3://"):
        return sorted(glob.glob(uri_pattern, recursive=True))

    parsed = urlparse(uri_pattern)
    bucket = parsed.netloc
    key_pattern = parsed.path.lstrip("/")
    wildcard_positions = [
        position
        for token in ("*", "?", "[")
        if (position := key_pattern.find(token)) >= 0
    ]
    prefix = (
        key_pattern[: min(wildcard_positions)] if wildcard_positions else key_pattern
    )
    prefix = prefix.rsplit("/", 1)[0] + "/" if "/" in prefix else ""
    client = s3_client_factory()
    paginator = client.get_paginator("list_objects_v2")
    matches: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if _fnmatch_globstar(key, key_pattern):
                matches.append(f"s3://{bucket}/{key}")
    return sorted(matches)


def _read_object_bytes(uri: str, s3_client_factory: S3ClientFactory) -> bytes:
    if not uri.startswith("s3://"):
        return Path(uri).read_bytes()
    parsed = urlparse(uri)
    response = s3_client_factory().get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )
    return response["Body"].read()


def _normalize_ragged_row(
    row: list[str],
    width: int,
    overflow_index: int | None,
) -> tuple[list[str], bool]:
    if len(row) == width:
        return row, False
    if len(row) < width:
        return [*row, *([""] * (width - len(row)))], True
    if overflow_index is None:
        return row[:width], True
    overflow = len(row) - width
    merged = ",".join(row[overflow_index : overflow_index + overflow + 1])
    return [*row[:overflow_index], merged, *row[overflow_index + overflow + 1 :]], True


def _read_python_csv(
    uri_pattern: str,
    options: dict[str, object],
    s3_client_factory: S3ClientFactory,
) -> pl.LazyFrame:
    encoding = str(options.pop("encoding", "utf-8"))
    expected_columns = tuple(options.pop("expected_columns", ()))
    overflow_column = options.pop("overflow_column", None)
    if options:
        raise ValueError(f"Unsupported tolerant CSV options: {sorted(options)}")

    frames: list[pl.DataFrame] = []
    for uri in _iter_object_uris(uri_pattern, s3_client_factory):
        text = _read_object_bytes(uri, s3_client_factory).decode(encoding)
        lines = text.splitlines()
        if not lines:
            continue
        header = next(csv.reader([lines[0]], strict=False))
        if expected_columns and tuple(header) != expected_columns:
            raise ValueError(
                f"CSV header mismatch for {uri}: expected {expected_columns}, got {tuple(header)}"
            )
        overflow_index = (
            header.index(str(overflow_column)) if overflow_column in header else None
        )
        columns: dict[str, list[str | None]] = {name: [] for name in header}
        repaired = 0
        for line_number, line in enumerate(lines[1:], start=2):
            try:
                row = next(csv.reader([line], strict=False))
            except csv.Error as error:
                LOG.warning(
                    "csv_record_parse_failed",
                    extra={"source_file_uri": uri, "line_number": line_number},
                    exc_info=error,
                )
                row = [line]
            row, changed = _normalize_ragged_row(row, len(header), overflow_index)
            repaired += int(changed)
            for name, value in zip(header, row, strict=True):
                columns[name].append(value or None)
        if repaired:
            LOG.warning(
                "csv_records_repaired",
                extra={"source_file_uri": uri, "repaired_records": repaired},
            )
        frame = pl.DataFrame(columns, schema={name: pl.String for name in header})
        frames.append(frame.with_columns(pl.lit(uri).alias("__landing_object_uri")))
    if not frames:
        raise FileNotFoundError(f"No CSV objects matched {uri_pattern!r}")
    return pl.concat(frames, how="diagonal_relaxed").lazy()


def _geojson_rows(payload: bytes, source_file_uri: str) -> Iterable[dict[str, object]]:
    document = json.loads(payload)
    if document.get("type") != "FeatureCollection":
        raise ValueError(
            f"GeoJSON object is not a FeatureCollection: {source_file_uri}"
        )
    for feature in document.get("features", []):
        properties = dict(feature.get("properties") or {})
        geometry = feature.get("geometry")
        properties["geometry_type"] = geometry.get("type") if geometry else None
        properties["geometry_json"] = (
            json.dumps(geometry, separators=(",", ":"), sort_keys=True)
            if geometry is not None
            else None
        )
        properties["__landing_object_uri"] = source_file_uri
        yield properties


def _read_geojson(
    uri_pattern: str,
    options: dict[str, object],
    s3_client_factory: S3ClientFactory,
) -> pl.LazyFrame:
    if options:
        raise ValueError(f"Unsupported GeoJSON read options: {sorted(options)}")
    frames: list[pl.DataFrame] = []
    for uri in _iter_object_uris(uri_pattern, s3_client_factory):
        rows = list(_geojson_rows(_read_object_bytes(uri, s3_client_factory), uri))
        if rows:
            frames.append(pl.DataFrame(rows, strict=False))
    if not frames:
        raise FileNotFoundError(f"No GeoJSON features matched {uri_pattern!r}")
    return pl.concat(frames, how="diagonal_relaxed").lazy()


def read_source_pattern(
    uri: str,
    pattern: SourcePattern,
    *,
    storage_options: Mapping[str, str] | None = None,
    s3_client_factory: S3ClientFactory | None = None,
) -> pl.LazyFrame:
    """Read one configured transport pattern without source-name branching."""

    strategy, options = _split_reader_options(pattern.read_options)
    client_factory = s3_client_factory or (lambda: None)
    if pattern.format == "parquet":
        if strategy != "native":
            raise ValueError(f"Unsupported Parquet reader strategy: {strategy!r}")
        return pl.scan_parquet(
            uri,
            hive_partitioning=True,
            storage_options=dict(storage_options or {}),
            credential_provider=None,
            include_file_paths="__landing_object_uri",
            **options,
        )
    if pattern.format == "csv":
        if strategy == "python_tolerant":
            return _read_python_csv(uri, options, client_factory)
        if strategy != "native":
            raise ValueError(f"Unsupported CSV reader strategy: {strategy!r}")
        return pl.scan_csv(
            uri,
            storage_options=dict(storage_options or {}),
            credential_provider=None,
            include_file_paths="__landing_object_uri",
            **options,
        )
    if pattern.format == "geojson":
        if strategy != "native":
            raise ValueError(f"Unsupported GeoJSON reader strategy: {strategy!r}")
        return _read_geojson(uri, options, client_factory)
    raise ValueError(f"Unsupported source format: {pattern.format!r}")
