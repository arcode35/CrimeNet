"""Replay-safe Census TIGER/Line tract archive landing and normalization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


TIGER_BASE_URL = "https://www2.census.gov/geo/tiger"
BOUNDARY_DEFINITION_VERSION = "tiger_line_tract_wgs84_v1"
TIGER_SOURCE_SYSTEM = "census_tiger_line"

_STATE_FIPS_PATTERN = re.compile(r"^[0-9]{2}$")
_GEOID_PATTERN = re.compile(r"^[0-9]{11}$")
_REQUIRED_ARCHIVE_EXTENSIONS = frozenset({".shp", ".shx", ".dbf", ".prj"})


class TigerBoundaryError(RuntimeError):
    """Base class for boundary-source failures."""


class TigerDownloadError(TigerBoundaryError):
    """Raised when a TIGER/Line archive cannot be downloaded safely."""


class TigerArchiveError(TigerBoundaryError):
    """Raised when a downloaded archive is incomplete or corrupt."""


class TigerNormalizationError(TigerBoundaryError):
    """Raised when a TIGER/Line shapefile cannot be normalized."""


class MissingBoundaryDependency(TigerNormalizationError):
    """Raised when the optional GeoPandas runtime dependency is unavailable."""


@dataclass(frozen=True)
class TigerArchive:
    """A validated immutable source archive."""

    year: int
    state_fips: str
    source_url: str
    path: Path
    sha256: str
    size_bytes: int
    archive_id: str


@dataclass(frozen=True)
class BoundaryIssue:
    """An auditable rejected source archive or shapefile record."""

    quarantine_id: str
    reason_code: str
    reason: str
    source_file: str
    source_row_hash: str
    raw_payload: str
    boundary_vintage: int
    state_fips: str
    geoid: str | None
    source_archive_sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "quarantine_id": self.quarantine_id,
            "source_system": TIGER_SOURCE_SYSTEM,
            "source_file": self.source_file,
            "source_row_hash": self.source_row_hash,
            "raw_payload": self.raw_payload,
            "quarantine_reason_code": self.reason_code,
            "quarantine_reason": self.reason,
            "boundary_vintage": self.boundary_vintage,
            "state_fips": self.state_fips,
            "geoid": self.geoid,
            "source_archive_sha256": self.source_archive_sha256,
        }


@dataclass(frozen=True)
class LandingResult:
    """All successful archives and failures from one requested landing run."""

    archives: tuple[TigerArchive, ...]
    issues: tuple[BoundaryIssue, ...]


@dataclass(frozen=True)
class NormalizationResult:
    """All valid normalized rows and explicitly rejected records."""

    records: tuple[dict[str, object], ...]
    issues: tuple[BoundaryIssue, ...]


def _stable_digest(parts: Iterable[object]) -> str:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_state_fips(state_fips: str) -> str:
    if not _STATE_FIPS_PATTERN.fullmatch(state_fips):
        raise ValueError(
            f"state_fips must contain exactly two digits; received {state_fips!r}."
        )
    return state_fips


def validate_tiger_year(year: int) -> int:
    if year < 2007 or year > 2100:
        raise ValueError(
            f"TIGER/Line year must be between 2007 and 2100; received {year}."
        )
    return year


def tiger_tract_archive_name(*, year: int, state_fips: str) -> str:
    validate_tiger_year(year)
    validate_state_fips(state_fips)
    return f"tl_{year}_{state_fips}_tract.zip"


def build_tiger_tract_url(
    *,
    year: int,
    state_fips: str,
    base_url: str = TIGER_BASE_URL,
) -> str:
    """Build the canonical Census URL for one tract archive."""

    archive_name = tiger_tract_archive_name(
        year=year,
        state_fips=state_fips,
    )
    return f"{base_url.rstrip('/')}/TIGER{year}/TRACT/{archive_name}"


def tiger_tract_landing_path(
    landing_directory: str | Path,
    *,
    year: int,
    state_fips: str,
) -> Path:
    """Return a deterministic, version-partitioned landing path."""

    archive_name = tiger_tract_archive_name(
        year=year,
        state_fips=state_fips,
    )
    return (
        Path(landing_directory)
        / "tiger_line"
        / "tract"
        / f"year={year}"
        / f"state={state_fips}"
        / archive_name
    )


def boundary_record_id(
    *,
    year: int,
    state_fips: str,
    geoid: str,
    source_archive_sha256: str,
    definition_version: str = BOUNDARY_DEFINITION_VERSION,
) -> str:
    """Return a stable identity that changes with source/algorithm version."""

    validate_tiger_year(year)
    validate_state_fips(state_fips)
    if not _GEOID_PATTERN.fullmatch(geoid):
        raise ValueError(f"Invalid tract GEOID: {geoid!r}.")
    if not definition_version.strip():
        raise ValueError("definition_version cannot be blank.")
    if not re.fullmatch(r"[0-9a-f]{64}", source_archive_sha256):
        raise ValueError("source_archive_sha256 must be a lowercase SHA-256.")

    return _stable_digest(
        (
            definition_version,
            year,
            state_fips,
            geoid,
            source_archive_sha256,
        )
    )


def build_tiger_session() -> requests.Session:
    """Create a bounded retrying session for immutable public archives."""

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def _archive_member_names(archive_path: Path) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise TigerArchiveError(
                    "TIGER/Line archive failed its CRC check at member "
                    f"{corrupt_member!r}: {archive_path}."
                )
            names = tuple(archive.namelist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise TigerArchiveError(
            f"Invalid TIGER/Line ZIP archive: {archive_path}."
        ) from exc

    for name in names:
        member_path = PurePosixPath(name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise TigerArchiveError(
                f"TIGER/Line archive contains an unsafe member path: {name!r}."
            )
    return names


def validate_tiger_archive(
    archive_path: str | Path,
    *,
    year: int,
    state_fips: str,
) -> tuple[str, ...]:
    """Verify ZIP integrity and the required shapefile sidecars."""

    path = Path(archive_path)
    expected_stem = tiger_tract_archive_name(
        year=year,
        state_fips=state_fips,
    ).removesuffix(".zip")
    names = _archive_member_names(path)
    member_extensions = {
        PurePosixPath(name).suffix.lower()
        for name in names
        if PurePosixPath(name).stem.lower() == expected_stem.lower()
    }
    missing_extensions = sorted(_REQUIRED_ARCHIVE_EXTENSIONS - member_extensions)
    if missing_extensions:
        raise TigerArchiveError(
            "TIGER/Line tract archive is missing required shapefile "
            f"members for {expected_stem}: {missing_extensions}."
        )
    return names


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_from_path(
    path: Path,
    *,
    year: int,
    state_fips: str,
    source_url: str,
) -> TigerArchive:
    validate_tiger_archive(
        path,
        year=year,
        state_fips=state_fips,
    )
    checksum = file_sha256(path)
    size_bytes = path.stat().st_size
    archive_id = _stable_digest((year, state_fips, source_url, checksum))
    return TigerArchive(
        year=year,
        state_fips=state_fips,
        source_url=source_url,
        path=path,
        sha256=checksum,
        size_bytes=size_bytes,
        archive_id=archive_id,
    )


def _write_manifest(archive: TigerArchive) -> None:
    manifest_path = archive.path.with_suffix(archive.path.suffix + ".manifest.json")
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            if (
                existing.get("archive_id") == archive.archive_id
                and existing.get("sha256") == archive.sha256
                and existing.get("size_bytes") == archive.size_bytes
                and existing.get("source_url") == archive.source_url
            ):
                return

    temporary_path = manifest_path.with_name(f"{manifest_path.name}.tmp-{uuid4().hex}")
    payload = {
        "archive_id": archive.archive_id,
        "boundary_vintage": archive.year,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "sha256": archive.sha256,
        "size_bytes": archive.size_bytes,
        "source_system": TIGER_SOURCE_SYSTEM,
        "source_url": archive.source_url,
        "state_fips": archive.state_fips,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_tiger_archive(
    *,
    year: int,
    state_fips: str,
    landing_directory: str | Path,
    session: requests.Session | None = None,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    maximum_archive_bytes: int = 1_000_000_000,
    base_url: str = TIGER_BASE_URL,
) -> TigerArchive:
    """Atomically land one archive, reusing only a fully validated cache hit."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if maximum_archive_bytes <= 0:
        raise ValueError("maximum_archive_bytes must be positive.")

    destination = tiger_tract_landing_path(
        landing_directory,
        year=year,
        state_fips=state_fips,
    )
    source_url = build_tiger_tract_url(
        year=year,
        state_fips=state_fips,
        base_url=base_url,
    )

    if destination.exists() and not overwrite:
        try:
            cached = _archive_from_path(
                destination,
                year=year,
                state_fips=state_fips,
                source_url=source_url,
            )
        except TigerArchiveError:
            # A partial or corrupt cache file is never accepted.  The valid
            # replacement is still written atomically below.
            pass
        else:
            _write_manifest(cached)
            return cached

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.part-{uuid4().hex}")
    owns_session = session is None
    active_session = session or build_tiger_session()
    response: requests.Response | None = None

    try:
        try:
            response = active_session.get(
                source_url,
                headers={"Accept": "application/zip"},
                stream=True,
                timeout=(30.0, timeout_seconds),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status_code = getattr(
                getattr(exc, "response", None),
                "status_code",
                None,
            )
            raise TigerDownloadError(
                "Census TIGER/Line request failed for "
                f"year={year}, state={state_fips}, "
                f"status={status_code!r}."
            ) from exc

        try:
            byte_count = 0
            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    byte_count += len(chunk)
                    if byte_count > maximum_archive_bytes:
                        raise TigerDownloadError(
                            "Census TIGER/Line archive exceeded the configured "
                            f"maximum of {maximum_archive_bytes} bytes."
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except requests.RequestException as exc:
            raise TigerDownloadError(
                "Census TIGER/Line response stream failed for "
                f"year={year}, state={state_fips}."
            ) from exc

        if byte_count == 0:
            raise TigerDownloadError(
                "Census TIGER/Line returned an empty archive for "
                f"year={year}, state={state_fips}."
            )

        validate_tiger_archive(
            temporary_path,
            year=year,
            state_fips=state_fips,
        )
        os.replace(temporary_path, destination)
        landed = _archive_from_path(
            destination,
            year=year,
            state_fips=state_fips,
            source_url=source_url,
        )
        _write_manifest(landed)
        return landed
    finally:
        temporary_path.unlink(missing_ok=True)
        if response is not None:
            response.close()
        if owns_session:
            active_session.close()


def _landing_issue(
    *,
    year: int,
    state_fips: str,
    destination: Path,
    source_url: str,
    error: Exception,
) -> BoundaryIssue:
    reason_code = (
        "TIGER_ARCHIVE_NOT_FOUND"
        if isinstance(error, TigerDownloadError) and "status=404" in str(error)
        else "TIGER_ARCHIVE_LANDING_FAILED"
    )
    raw_payload = json.dumps(
        {
            "boundary_vintage": year,
            "source_url": source_url,
            "state_fips": state_fips,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    stable_request_identity = json.dumps(
        {
            "boundary_vintage": year,
            "state_fips": state_fips,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    source_row_hash = hashlib.sha256(
        stable_request_identity.encode("utf-8")
    ).hexdigest()
    return BoundaryIssue(
        quarantine_id=_stable_digest(
            (
                TIGER_SOURCE_SYSTEM,
                year,
                state_fips,
                source_row_hash,
                reason_code,
            )
        ),
        reason_code=reason_code,
        reason=str(error),
        source_file=str(destination),
        source_row_hash=source_row_hash,
        raw_payload=raw_payload,
        boundary_vintage=year,
        state_fips=state_fips,
        geoid=None,
        source_archive_sha256=None,
    )


def land_tiger_archives(
    *,
    years: Iterable[int],
    state_fips: str,
    landing_directory: str | Path,
    overwrite: bool = False,
    timeout_seconds: float = 180.0,
    maximum_archive_bytes: int = 1_000_000_000,
    base_url: str = TIGER_BASE_URL,
    session: requests.Session | None = None,
) -> LandingResult:
    """Attempt every requested vintage and return failures for quarantine."""

    validated_years = tuple(sorted({validate_tiger_year(year) for year in years}))
    if not validated_years:
        raise ValueError("At least one TIGER/Line year is required.")
    validate_state_fips(state_fips)

    owns_session = session is None
    active_session = session or build_tiger_session()
    archives: list[TigerArchive] = []
    issues: list[BoundaryIssue] = []
    try:
        for year in validated_years:
            destination = tiger_tract_landing_path(
                landing_directory,
                year=year,
                state_fips=state_fips,
            )
            source_url = build_tiger_tract_url(
                year=year,
                state_fips=state_fips,
                base_url=base_url,
            )
            try:
                archive = download_tiger_archive(
                    year=year,
                    state_fips=state_fips,
                    landing_directory=landing_directory,
                    session=active_session,
                    overwrite=overwrite,
                    timeout_seconds=timeout_seconds,
                    maximum_archive_bytes=maximum_archive_bytes,
                    base_url=base_url,
                )
            except Exception as exc:
                issues.append(
                    _landing_issue(
                        year=year,
                        state_fips=state_fips,
                        destination=destination,
                        source_url=source_url,
                        error=exc,
                    )
                )
            else:
                archives.append(archive)
    finally:
        if owns_session:
            active_session.close()

    return LandingResult(tuple(archives), tuple(issues))


def _import_geopandas() -> Any:
    try:
        import geopandas
    except ImportError as exc:
        raise MissingBoundaryDependency(
            "TIGER/Line normalization requires GeoPandas with a "
            "shapefile engine. Install `geopandas` in the wheel runtime."
        ) from exc
    return geopandas


def _resolve_geoid_column(columns: Iterable[object]) -> str:
    by_upper_name = {str(column).upper(): str(column) for column in columns}
    for candidate in ("GEOID", "GEOID20", "GEOID10"):
        if candidate in by_upper_name:
            return by_upper_name[candidate]
    raise TigerNormalizationError(
        "TIGER/Line tract shapefile does not contain a GEOID column."
    )


def _json_safe_raw_payload(
    row: Any,
    *,
    excluded_columns: set[str],
) -> str:
    payload = {
        str(column): None if value is None else str(value)
        for column, value in row.items()
        if str(column) not in excluded_columns
    }
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _record_issue(
    *,
    archive: TigerArchive,
    geoid: str | None,
    reason_code: str,
    reason: str,
    raw_payload: str,
) -> BoundaryIssue:
    source_row_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
    return BoundaryIssue(
        quarantine_id=_stable_digest(
            (
                TIGER_SOURCE_SYSTEM,
                archive.year,
                archive.state_fips,
                archive.sha256,
                source_row_hash,
                reason_code,
            )
        ),
        reason_code=reason_code,
        reason=reason,
        source_file=str(archive.path),
        source_row_hash=source_row_hash,
        raw_payload=raw_payload,
        boundary_vintage=archive.year,
        state_fips=archive.state_fips,
        geoid=geoid,
        source_archive_sha256=archive.sha256,
    )


def normalization_failure_issue(
    archive: TigerArchive,
    error: Exception,
) -> BoundaryIssue:
    """Convert an archive-level normalization failure into quarantine data."""

    raw_payload = json.dumps(
        {
            "boundary_vintage": archive.year,
            "source_archive_sha256": archive.sha256,
            "state_fips": archive.state_fips,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _record_issue(
        archive=archive,
        geoid=None,
        reason_code="TIGER_ARCHIVE_NORMALIZATION_FAILED",
        reason=str(error),
        raw_payload=raw_payload,
    )


def normalize_tiger_archive(
    archive: TigerArchive,
    *,
    definition_version: str = BOUNDARY_DEFINITION_VERSION,
) -> NormalizationResult:
    """Normalize one tract shapefile to WGS84 WKT with explicit rejects."""

    if not definition_version.strip():
        raise ValueError("definition_version cannot be blank.")

    geopandas = _import_geopandas()
    try:
        source = geopandas.read_file(str(archive.path))
    except Exception as exc:
        raise TigerNormalizationError(
            f"Could not read TIGER/Line archive {archive.path}."
        ) from exc

    if source.empty:
        raise TigerNormalizationError(
            f"TIGER/Line archive contains no tract records: {archive.path}."
        )
    if source.crs is None:
        raise TigerNormalizationError(
            f"TIGER/Line archive has no declared CRS: {archive.path}."
        )

    geoid_column = _resolve_geoid_column(source.columns)
    geometry_column = source.geometry.name
    try:
        normalized = source.to_crs(epsg=4326)
    except Exception as exc:
        raise TigerNormalizationError(
            f"Could not transform {archive.path} to EPSG:4326."
        ) from exc

    preliminary_records: list[dict[str, object]] = []
    issues: list[BoundaryIssue] = []

    for _, row in normalized.iterrows():
        raw_payload = _json_safe_raw_payload(
            row,
            excluded_columns={geometry_column},
        )
        raw_geoid = row.get(geoid_column)
        geoid = None if raw_geoid is None else str(raw_geoid).strip()
        geometry = row.get(geometry_column)

        if geoid is None or not _GEOID_PATTERN.fullmatch(geoid):
            issues.append(
                _record_issue(
                    archive=archive,
                    geoid=geoid,
                    reason_code="INVALID_TRACT_GEOID",
                    reason=("Tract GEOID must contain exactly 11 digits."),
                    raw_payload=raw_payload,
                )
            )
            continue
        if not geoid.startswith(archive.state_fips):
            issues.append(
                _record_issue(
                    archive=archive,
                    geoid=geoid,
                    reason_code="TRACT_STATE_MISMATCH",
                    reason=(
                        f"Tract GEOID {geoid} does not belong to "
                        f"state {archive.state_fips}."
                    ),
                    raw_payload=raw_payload,
                )
            )
            continue
        if geometry is None or geometry.is_empty:
            issues.append(
                _record_issue(
                    archive=archive,
                    geoid=geoid,
                    reason_code="EMPTY_TRACT_GEOMETRY",
                    reason="Tract geometry is null or empty.",
                    raw_payload=raw_payload,
                )
            )
            continue
        if not geometry.is_valid:
            issues.append(
                _record_issue(
                    archive=archive,
                    geoid=geoid,
                    reason_code="INVALID_TRACT_GEOMETRY",
                    reason="Tract geometry is topologically invalid.",
                    raw_payload=raw_payload,
                )
            )
            continue

        preliminary_records.append(
            {
                "geoid": geoid,
                "boundary_vintage": archive.year,
                "state_fips": archive.state_fips,
                "wkt_geometry": geometry.wkt,
                "boundary_definition_version": definition_version,
                "source_archive_sha256": archive.sha256,
                "source_archive_id": archive.archive_id,
                "boundary_record_id": boundary_record_id(
                    year=archive.year,
                    state_fips=archive.state_fips,
                    geoid=geoid,
                    source_archive_sha256=archive.sha256,
                    definition_version=definition_version,
                ),
            }
        )

    geoid_counts = Counter(str(record["geoid"]) for record in preliminary_records)
    records: list[dict[str, object]] = []
    quarantined_duplicate_geoids: set[str] = set()
    for record in preliminary_records:
        geoid = str(record["geoid"])
        if geoid_counts[geoid] == 1:
            records.append(record)
            continue
        if geoid in quarantined_duplicate_geoids:
            continue
        quarantined_duplicate_geoids.add(geoid)
        raw_payload = json.dumps(
            {
                "boundary_vintage": archive.year,
                "geoid": geoid,
                "source_archive_sha256": archive.sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        issues.append(
            _record_issue(
                archive=archive,
                geoid=geoid,
                reason_code="DUPLICATE_TRACT_GEOID",
                reason=(
                    "The source archive contains more than one row for "
                    f"tract GEOID {geoid}."
                ),
                raw_payload=raw_payload,
            )
        )

    return NormalizationResult(tuple(records), tuple(issues))


def normalize_tiger_archives(
    archives: Iterable[TigerArchive],
    *,
    definition_version: str = BOUNDARY_DEFINITION_VERSION,
) -> NormalizationResult:
    """Normalize every archive and retain all record-level rejects."""

    records: list[dict[str, object]] = []
    issues: list[BoundaryIssue] = []
    for archive in sorted(
        archives,
        key=lambda item: (item.year, item.state_fips),
    ):
        result = normalize_tiger_archive(
            archive,
            definition_version=definition_version,
        )
        records.extend(result.records)
        issues.extend(result.issues)
    return NormalizationResult(tuple(records), tuple(issues))


def create_boundary_dataframe(
    spark: SparkSession,
    records: Iterable[dict[str, object]],
) -> DataFrame:
    """Create a typed boundary DataFrame and materialize native geometry."""

    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    materialized = list(records)
    schema = StructType(
        [
            StructField("geoid", StringType(), False),
            StructField("boundary_vintage", IntegerType(), False),
            StructField("state_fips", StringType(), False),
            StructField("wkt_geometry", StringType(), False),
            StructField(
                "boundary_definition_version",
                StringType(),
                False,
            ),
            StructField("source_archive_sha256", StringType(), False),
            StructField("source_archive_id", StringType(), False),
            StructField("boundary_record_id", StringType(), False),
        ]
    )
    rows = [
        tuple(record[field.name] for field in schema.fields) for record in materialized
    ]
    return (
        spark.createDataFrame(rows, schema=schema)
        .withColumn(
            "tract_geometry",
            F.expr("ST_GeomFromWKT(wkt_geometry, 4326)"),
        )
        .drop("wkt_geometry")
    )


def validate_boundary_dataframe(
    dataframe: DataFrame,
    *,
    expected_years: Iterable[int],
    state_fips: str,
    minimum_tracts_per_vintage: int = 1,
) -> None:
    """Run blocking completeness, key, identifier, and SRID checks."""

    from pyspark.sql import functions as F

    if minimum_tracts_per_vintage < 1:
        raise ValueError("minimum_tracts_per_vintage must be positive.")
    expected = {validate_tiger_year(year) for year in expected_years}
    if not expected:
        raise ValueError("expected_years cannot be empty.")
    validate_state_fips(state_fips)

    required_columns = {
        "geoid",
        "boundary_vintage",
        "state_fips",
        "tract_geometry",
        "boundary_definition_version",
        "source_archive_sha256",
        "source_archive_id",
        "boundary_record_id",
    }
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Boundary candidate is missing required columns: {missing_columns}."
        )

    counts = {
        int(row["boundary_vintage"]): int(row["count"])
        for row in dataframe.groupBy("boundary_vintage").count().collect()
    }
    actual_years = set(counts)
    if actual_years != expected:
        raise ValueError(
            "Boundary candidate does not contain the exact requested "
            f"vintages: expected={sorted(expected)}, "
            f"actual={sorted(actual_years)}."
        )
    sparse_years = {
        year: count
        for year, count in counts.items()
        if count < minimum_tracts_per_vintage
    }
    if sparse_years:
        raise ValueError(
            "Boundary candidate contains too few tracts for one or more "
            f"vintages: {sparse_years}."
        )

    duplicate_business_keys = (
        dataframe.groupBy("boundary_vintage", "geoid")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    duplicate_record_ids = (
        dataframe.groupBy("boundary_record_id")
        .count()
        .filter(F.col("count") != 1)
        .limit(1)
        .count()
    )
    tract_srid = F.expr("ST_SRID(tract_geometry)")
    invalid_records = (
        dataframe.filter(
            F.col("geoid").isNull()
            | ~F.col("geoid").rlike(r"^[0-9]{11}$")
            | ~F.col("geoid").startswith(state_fips)
            | F.col("state_fips").isNull()
            | (F.col("state_fips") != state_fips)
            | F.col("tract_geometry").isNull()
            | F.col("boundary_record_id").isNull()
            | (F.length("boundary_record_id") != 64)
            | F.col("source_archive_sha256").isNull()
            | (F.length("source_archive_sha256") != 64)
            | tract_srid.isNull()
            | (tract_srid != 4326)
        )
        .limit(1)
        .count()
    )

    if duplicate_business_keys:
        raise ValueError(
            "Boundary candidate contains duplicate (boundary_vintage, geoid) keys."
        )
    if duplicate_record_ids:
        raise ValueError("Boundary candidate has duplicate record IDs.")
    if invalid_records:
        raise ValueError(
            "Boundary candidate contains an invalid GEOID, geometry, "
            "state, checksum, record ID, or SRID."
        )


def boundary_issues_to_dataframe(
    spark: SparkSession,
    issues: Iterable[BoundaryIssue],
    *,
    pipeline_run_id: str,
    definition_version: str = BOUNDARY_DEFINITION_VERSION,
) -> DataFrame:
    """Create the domain quarantine DataFrame with stable identities."""

    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    issue_dicts = [issue.as_dict() for issue in issues]
    schema = StructType(
        [
            StructField("quarantine_id", StringType(), False),
            StructField("source_system", StringType(), False),
            StructField("source_file", StringType(), True),
            StructField("source_row_hash", StringType(), False),
            StructField("raw_payload", StringType(), False),
            StructField(
                "quarantine_reason_code",
                StringType(),
                False,
            ),
            StructField("quarantine_reason", StringType(), False),
            StructField("boundary_vintage", IntegerType(), False),
            StructField("state_fips", StringType(), False),
            StructField("geoid", StringType(), True),
            StructField(
                "source_archive_sha256",
                StringType(),
                True,
            ),
        ]
    )
    rows = [
        tuple(issue[field.name] for field in schema.fields) for issue in issue_dicts
    ]
    return (
        spark.createDataFrame(rows, schema=schema)
        .dropDuplicates(["quarantine_id"])
        .withColumn("pipeline_run_id", F.lit(pipeline_run_id))
        .withColumn(
            "boundary_definition_version",
            F.lit(definition_version),
        )
        .withColumn("quarantined_at", F.current_timestamp())
    )


def merge_boundary_quarantine(
    spark: SparkSession,
    dataframe: DataFrame,
    *,
    target_table: str,
    pipeline_run_id: str,
) -> None:
    """Insert one observation per stable issue and pipeline run."""

    from crimenet.config.validation import validate_qualified_table_name
    from crimenet.utils.promotion import staging_table_name

    validate_qualified_table_name(target_table)
    if dataframe.isEmpty():
        return

    stage = staging_table_name(
        target_table,
        f"{pipeline_run_id}_boundary_quarantine",
    )
    (
        dataframe.dropDuplicates(["quarantine_id", "pipeline_run_id"])
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(stage)
    )
    try:
        if not spark.catalog.tableExists(target_table):
            spark.sql(
                f"CREATE TABLE {target_table} USING DELTA AS SELECT * FROM {stage}"
            )
            return

        spark.sql(
            f"""
            MERGE INTO {target_table} AS target
            USING {stage} AS source
            ON target.quarantine_id = source.quarantine_id
            AND target.pipeline_run_id = source.pipeline_run_id
            WHEN NOT MATCHED THEN INSERT *
            """
        )
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {stage}")
