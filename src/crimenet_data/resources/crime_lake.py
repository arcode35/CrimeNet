from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import dagster as dg
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError

from crimenet_data.assets.crime.canonical.schema import CANONICAL_CRIME_SCHEMA
from crimenet_data.assets.crime.ingestion.readers import read_source_pattern
from crimenet_data.assets.crime.sources import SOURCE_KEYS, get_source

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUCCESS_MARKER = "_SUCCESS"
LATEST_POINTER = "_latest.json"
SNAPSHOT_MANIFEST = "manifest.json"

# Compatibility names for the existing Bronze/Silver implementation. Canonical
# object names are defined once because every immutable CrimeNet dataset uses
# the same publication contract.
BRONZE_SUCCESS_MARKER = SUCCESS_MARKER
BRONZE_LATEST_POINTER = LATEST_POINTER
SILVER_SUCCESS_MARKER = SUCCESS_MARKER
SILVER_LATEST_POINTER = LATEST_POINTER
SILVER_MANIFEST = SNAPSHOT_MANIFEST


@dataclass(frozen=True)
class BronzeSnapshotPointer:
    """The durable pointer to a source's current completed Bronze snapshot."""

    source_key: str
    snapshot_id: str
    created_at: datetime

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "source_key": self.source_key,
                "snapshot_id": self.snapshot_id,
                "created_at": self.created_at.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_json(
        cls,
        payload: bytes,
        *,
        expected_source_key: str,
    ) -> BronzeSnapshotPointer:
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Malformed Bronze snapshot pointer for {expected_source_key!r}"
            ) from error

        if not isinstance(document, dict):
            raise TypeError(
                f"Malformed Bronze snapshot pointer for {expected_source_key!r}: "
                "expected a JSON object"
            )

        try:
            source_key = document["source_key"]
            snapshot_id = document["snapshot_id"]
            created_at_raw = document["created_at"]
        except KeyError as error:
            raise ValueError(
                f"Malformed Bronze snapshot pointer for {expected_source_key!r}: "
                f"missing {error.args[0]!r}"
            ) from error

        if source_key != expected_source_key:
            raise ValueError(
                f"Bronze snapshot pointer source mismatch: expected "
                f"{expected_source_key!r}, found {source_key!r}"
            )
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("Bronze snapshot pointer has an invalid snapshot_id")
        if not isinstance(created_at_raw, str):
            raise TypeError("Bronze snapshot pointer has an invalid created_at")
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except ValueError as error:
            raise ValueError(
                "Bronze snapshot pointer has an invalid created_at"
            ) from error
        if created_at.tzinfo is None:
            raise ValueError(
                "Bronze snapshot pointer created_at must include a timezone"
            )

        return cls(
            source_key=source_key,
            snapshot_id=snapshot_id,
            created_at=created_at.astimezone(UTC),
        )


@dataclass(frozen=True)
class SilverSnapshotPointer:
    """The published pointer to one immutable unified Silver snapshot."""

    snapshot_id: str
    snapshot_uri: str
    created_at_utc: datetime
    mapping_version: str

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "snapshot_id": self.snapshot_id,
                "snapshot_uri": self.snapshot_uri,
                "created_at_utc": self.created_at_utc.isoformat(),
                "mapping_version": self.mapping_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_json(cls, payload: bytes) -> SilverSnapshotPointer:
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Malformed Silver snapshot pointer") from error
        if not isinstance(document, dict):
            raise TypeError("Malformed Silver snapshot pointer: expected an object")
        required = {
            "snapshot_id",
            "snapshot_uri",
            "created_at_utc",
            "mapping_version",
        }
        missing = required - set(document)
        if missing:
            raise ValueError(
                f"Malformed Silver snapshot pointer: missing {sorted(missing)}"
            )
        try:
            created_at = datetime.fromisoformat(str(document["created_at_utc"]))
        except ValueError as error:
            raise ValueError(
                "Silver snapshot pointer has an invalid created_at_utc"
            ) from error
        if created_at.tzinfo is None:
            raise ValueError(
                "Silver snapshot pointer created_at_utc must include a timezone"
            )
        values = {
            name: document[name]
            for name in ("snapshot_id", "snapshot_uri", "mapping_version")
        }
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError("Silver snapshot pointer contains an invalid string")
        return cls(
            snapshot_id=values["snapshot_id"],
            snapshot_uri=values["snapshot_uri"],
            created_at_utc=created_at.astimezone(UTC),
            mapping_version=values["mapping_version"],
        )


class CrimeLakeResources(dg.ConfigurableResource):
    """Crime lake paths and Backblaze B2 storage configuration"""

    bucket: str = "s3://crimenet-data"
    crosswalk_path: str | None = None
    local_fixture_root: str | None = None

    @property
    def storage_options(self) -> dict[str, object]:
        """Options accepted by Polars for B2's S3-compatible API."""

        if not self.bucket.startswith("s3://"):
            return {}

        endpoint = os.environ.get("B2_ENDPOINT_URL")
        key_id = os.environ.get("B2_KEY_ID")
        app_key = os.environ.get("B2_APPLICATION_KEY")
        region = os.environ.get("B2_REGION", "us-east-005")

        missing = [
            name
            for name, value in (
                ("B2_ENDPOINT_URL", endpoint),
                ("B2_KEY_ID", key_id),
                ("B2_APPLICATION_KEY", app_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing Backblaze B2 configuration: " + ", ".join(missing)
            )

        return {
            "aws_access_key_id": key_id,
            "aws_secret_access_key": app_key,
            "aws_region": region,
            "aws_endpoint_url": endpoint,
            # Aggressive retries for B2 intermittent 500 "internal incident"
            "max_retries": 15,
            "retry_timeout_ms": 300_000,          # 5 min total budget
            "retry_init_backoff_ms": 400,
            "retry_max_backoff_ms": 20_000,
            "retry_base_multiplier": 2.0,
        }

    def s3_client(self):
        """Build a boto3 client from the same centralized B2 configuration."""

        options = self.storage_options
        if not options:
            raise ValueError("An S3 client requires an s3:// CrimeNet bucket")
        return boto3.client(
            "s3",
            endpoint_url=options["aws_endpoint_url"],
            aws_access_key_id=options["aws_access_key_id"],
            aws_secret_access_key=options["aws_secret_access_key"],
            region_name=options["aws_region"],
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                connect_timeout=30,
                read_timeout=120,
                max_pool_connections=32,
            ),
        )

    @property
    def raw_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/raw_files"

    @property
    def landing_root(self) -> str:
        return f"{self.raw_root}/landing"

    @property
    def bronze_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/bronze"

    @property
    def silver_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/silver"

    @property
    def gold_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/gold"

    @property
    def quality_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/quality"

    @property
    def reference_root(self) -> str:
        """Canonical CrimeNet-owned reference inputs in the landing layer."""

        return f"{self.landing_root}/reference"

    @property
    def integration_reference_root(self) -> str:
        return f"{self.reference_root}/integration_sampling"

    @property
    def base_domain_uri(self) -> str:
        """Audited authoritative H3 reporting support used for integration."""

        return f"{self.integration_reference_root}/base_domain_h3.csv"

    @property
    def temporal_coverage_uri(self) -> str:
        """Audited, outcome-independent source temporal coverage catalog."""

        return f"{self.integration_reference_root}/source_temporal_coverage.csv"

    @property
    def model_weather_v2_root(self) -> str:
        return f"{self.landing_root}/weather/model_weather_v2/open_meteo"

    @property
    def model_weather_v2_best_match_root(self) -> str:
        return f"{self.model_weather_v2_root}/best_match"

    @property
    def silver_environmental_root(self) -> str:
        return f"{self.silver_root}/environmental"

    @property
    def silver_weather_root(self) -> str:
        return f"{self.silver_environmental_root}/weather"

    @property
    def silver_weather_latest_pointer_uri(self) -> str:
        return self.latest_pointer_uri(self.silver_weather_root)

    def silver_weather_snapshot_uri(self, snapshot_id: str) -> str:
        return self.snapshot_uri(self.silver_weather_root, snapshot_id)

    def silver_weather_year_uri(self, snapshot_uri: str, year: int) -> str:
        return f"{snapshot_uri.rstrip('/')}/year={int(year)}/part-00000.parquet"

    def silver_weather_year_glob(self, snapshot_uri: str, year: int) -> str:
        return f"{snapshot_uri.rstrip('/')}/year={int(year)}/*.parquet"

    @property
    def final_model_table_root(self) -> str:
        """Canonical immutable Gold final model-table dataset root."""

        return f"{self.gold_root}/final_model_table"

    @property
    def final_model_table_latest_pointer_uri(self) -> str:
        """Pointer to the currently published final model-table snapshot."""

        return self.latest_pointer_uri(self.final_model_table_root)

    def final_model_table_snapshot_uri(self, snapshot_id: str) -> str:
        """Return the immutable URI for one final model-table snapshot."""

        return self.snapshot_uri(self.final_model_table_root, snapshot_id)

    @staticmethod
    def final_model_table_parquet_glob(snapshot_uri: str) -> str:
        """Parquet glob for one immutable final model-table snapshot."""

        return f"{snapshot_uri.rstrip('/')}/**/*.parquet"

    @property
    def environmental_features_root(self) -> str:
        return f"{self.gold_root}/environmental_features"

    @property
    def environmental_features_latest_pointer_uri(self) -> str:
        return self.latest_pointer_uri(self.environmental_features_root)

    def environmental_features_snapshot_uri(self, snapshot_id: str) -> str:
        return self.snapshot_uri(self.environmental_features_root, snapshot_id)

    def environmental_features_year_uri(self, snapshot_uri: str, year: int) -> str:
        return f"{snapshot_uri.rstrip('/')}/year={int(year)}/part-00000.parquet"

    def environmental_features_year_glob(self, snapshot_uri: str, year: int) -> str:
        return f"{snapshot_uri.rstrip('/')}/year={int(year)}/*.parquet"

    @staticmethod
    def environmental_features_parquet_glob(snapshot_uri: str) -> str:
        """Parquet glob for one immutable Gold environmental-features snapshot."""

        return f"{snapshot_uri.rstrip('/')}/**/*.parquet"

    @property
    def national_feature_store_root(self) -> str:
        return f"{self.gold_root}/national_feature_store"

    @property
    def national_feature_latest_h3_r9_root(self) -> str:
        return f"{self.national_feature_store_root}/latest/h3_r9"

    @property
    def national_temporal_annual_h3_r9_root(self) -> str:
        return f"{self.national_feature_store_root}/temporal/h3_r9/annual"

    @property
    def national_temporal_history_root(self) -> str:
        return f"{self.national_feature_store_root}/temporal/h3_r9/history"

    @property
    def national_temporal_history_glob(self) -> str:
        return (
            f"{self.national_temporal_history_root}/"
            "feature_available_date=*/version_id=*/part-*.parquet"
        )

    @property
    def event_spine_root(self) -> str:
        return f"{self.gold_root}/event_spine"

    @property
    def event_spine_latest_pointer_uri(self) -> str:
        return self.latest_pointer_uri(self.event_spine_root)

    def event_spine_snapshot_uri(self, snapshot_id: str) -> str:
        return self.snapshot_uri(self.event_spine_root, snapshot_id)

    def event_spine_manifest_uri(self, snapshot_uri: str) -> str:
        return self.snapshot_manifest_uri(snapshot_uri)

    def event_spine_success_uri(self, snapshot_uri: str) -> str:
        return self.snapshot_success_uri(snapshot_uri)

    @staticmethod
    def event_spine_parquet_glob(snapshot_uri: str) -> str:
        return f"{snapshot_uri.rstrip('/')}/**/*.parquet"

    def event_spine_source_year_glob(
        self,
        snapshot_uri: str,
        *,
        source_city: str,
        occurrence_year: int,
    ) -> str:
        return (
            f"{snapshot_uri.rstrip('/')}/source_city={source_city}/"
            f"occurrence_year={occurrence_year}/**/*.parquet"
        )

    @property
    def integration_root(self) -> str:
        return f"{self.gold_root}/integration_sampling"

    @property
    def integration_latest_pointer_uri(self) -> str:
        return self.latest_pointer_uri(self.integration_root)

    def integration_snapshot_uri(self, snapshot_id: str) -> str:
        return self.snapshot_uri(self.integration_root, snapshot_id)

    def integration_manifest_uri(self, snapshot_uri: str) -> str:
        return self.snapshot_manifest_uri(snapshot_uri)

    def integration_success_uri(self, snapshot_uri: str) -> str:
        return self.snapshot_success_uri(snapshot_uri)

    def integration_domain_uri(self, snapshot_uri: str, source_city: str) -> str:
        return (
            f"{snapshot_uri.rstrip('/')}/domain/source_city={source_city}/"
            "part-00000.parquet"
        )

    def integration_samples_prefix(self, snapshot_uri: str, source_city: str) -> str:
        return f"{snapshot_uri.rstrip('/')}/samples/source_city={source_city}"

    def integration_sample_part_uri(
        self,
        snapshot_uri: str,
        source_city: str,
        part_index: int,
    ) -> str:
        if part_index < 0:
            raise ValueError("Integration sample part index must be non-negative")
        return (
            f"{self.integration_samples_prefix(snapshot_uri, source_city)}/"
            f"part-{part_index:05d}.parquet"
        )

    def integration_sample_uris_from_manifest(
        self,
        snapshot_uri: str,
        manifest: Mapping[str, object],
    ) -> list[str]:
        """Resolve every immutable integration-sample Parquet part from its manifest.

        The integration manifest is authoritative for the number of sample parts
        written per source.  This avoids broad prefix scans and guarantees the
        final model table reads exactly the published integration snapshot.
        """

        snapshot_uri = snapshot_uri.rstrip("/")
        expected_root = str(
            manifest.get("snapshot_root")
            or manifest.get("snapshot_uri")
            or ""
        ).rstrip("/")
        if expected_root and expected_root != snapshot_uri:
            raise RuntimeError(
                "Integration manifest snapshot URI mismatch: "
                f"selected={snapshot_uri!r}, manifest={expected_root!r}"
            )

        sources = manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise RuntimeError(
                "Integration manifest contains no per-source sample metadata"
            )

        uris: list[str] = []
        seen_sources: set[str] = set()

        for source_info in sources:
            if not isinstance(source_info, Mapping):
                raise RuntimeError(
                    "Integration manifest contains a malformed source record"
                )

            source_city = str(source_info.get("source_city", "")).strip()
            if not source_city:
                raise RuntimeError(
                    "Integration manifest source record is missing source_city"
                )
            if source_city in seen_sources:
                raise RuntimeError(
                    f"Integration manifest contains duplicate source_city={source_city!r}"
                )
            seen_sources.add(source_city)

            raw_part_count = source_info.get("sample_part_count")
            try:
                part_count = int(raw_part_count or 0)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{source_city}: invalid sample_part_count={raw_part_count!r}"
                ) from error

            if part_count <= 0:
                raise RuntimeError(
                    f"{source_city}: integration manifest has invalid "
                    f"sample_part_count={part_count}"
                )

            uris.extend(
                self.integration_sample_part_uri(
                    snapshot_uri,
                    source_city,
                    part_index,
                )
                for part_index in range(part_count)
            )

        if not uris:
            raise RuntimeError(
                "Integration manifest resolved no sample Parquet objects"
            )

        return uris

    @staticmethod
    def _validate_snapshot_id(snapshot_id: str) -> None:
        if not snapshot_id or "/" in snapshot_id or "\\" in snapshot_id:
            raise ValueError(f"Invalid snapshot ID: {snapshot_id!r}")

    @classmethod
    def snapshot_uri(cls, root: str, snapshot_id: str) -> str:
        cls._validate_snapshot_id(snapshot_id)
        return f"{root.rstrip('/')}/snapshot_id={snapshot_id}"

    @staticmethod
    def latest_pointer_uri(root: str) -> str:
        return f"{root.rstrip('/')}/{LATEST_POINTER}"

    @staticmethod
    def snapshot_manifest_uri(snapshot_uri: str) -> str:
        return f"{snapshot_uri.rstrip('/')}/{SNAPSHOT_MANIFEST}"

    @staticmethod
    def snapshot_success_uri(snapshot_uri: str) -> str:
        return f"{snapshot_uri.rstrip('/')}/{SUCCESS_MARKER}"

    def storage_options_for(self, uri: str) -> dict[str, object] | None:
        return self.storage_options if uri.startswith("s3://") else None

    def list_object_uris(self, root_uri: str, *, suffix: str = "") -> list[str]:
        """List objects below one CrimeLake root for local or S3-backed tests/jobs."""

        root_uri = root_uri.rstrip("/")
        if not root_uri.startswith("s3://"):
            root = Path(root_uri)
            if not root.exists():
                return []
            return sorted(
                str(path)
                for path in root.rglob("*")
                if path.is_file() and (not suffix or path.name.endswith(suffix))
            )
        bucket, prefix = self._s3_location(f"{root_uri}/placeholder")
        paginator = self.s3_client().get_paginator("list_objects_v2")
        base_prefix = prefix.removesuffix("placeholder")
        return sorted(
            f"s3://{bucket}/{key}"
            for page in paginator.paginate(Bucket=bucket, Prefix=base_prefix)
            for item in page.get("Contents", [])
            if isinstance((key := item.get("Key")), str)
            and (not suffix or key.endswith(suffix))
        )

    def upload_local_file(self, local_path: Path, destination_uri: str) -> None:
        """Publish a completed local staging file to one canonical object URI."""

        if not destination_uri.startswith("s3://"):
            destination = Path(destination_uri)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_path, destination)
            return
        bucket, key = self._s3_location(destination_uri)
        self.s3_client().upload_file(str(local_path), bucket, key)

    def source_root(self, source_key: str) -> str:
        get_source(source_key)
        return f"{self.landing_root}/{source_key}"

    def source_uris(self, source_key: str) -> tuple[str, ...]:
        root = self.source_root(source_key).rstrip("/")
        return tuple(
            f"{root}/{pattern.glob}"
            for pattern in get_source(source_key).config.patterns
        )

    def source_uri(self, source_key: str) -> str:
        """Return a single configured URI, rejecting multi-era source layouts."""

        uris = self.source_uris(source_key)
        if len(uris) != 1:
            raise ValueError(
                f"Source {source_key!r} has multiple source URIs; use source_uris()"
            )
        return uris[0]

    def resolve_source_path(self, source_key: str, schema: str) -> str:
        get_source(source_key)
        roots = {
            "bronze": f"{self.bronze_root}/crime/{source_key}",
            "silver": self.silver_crime_offenses_uri,
            "gold": f"{self.gold_root}/crime/{source_key}",
        }
        try:
            return roots[schema]
        except KeyError as error:
            raise KeyError(
                f"{schema!r} is not a valid crime lake layer. "
                f"Valid layers: {sorted(roots)}"
            ) from error

    def bronze_snapshot_uri(self, source_key: str, snapshot_id: str) -> str:
        """Return the immutable path assigned to one Bronze materialization."""

        root = self.resolve_source_path(source_key, "bronze")
        self._validate_snapshot_id(snapshot_id)
        return self.snapshot_uri(root, snapshot_id)

    def _bronze_pointer_uri(self, source_key: str) -> str:
        root = self.resolve_source_path(source_key, "bronze")
        return f"{root}/{BRONZE_LATEST_POINTER}"

    @staticmethod
    def _s3_location(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError(f"Invalid S3 object URI: {uri!r}")
        return parsed.netloc, parsed.path.lstrip("/")

    def _object_exists(self, uri: str) -> bool:
        if not uri.startswith("s3://"):
            return Path(uri).is_file()
        bucket, key = self._s3_location(uri)
        try:
            self.s3_client().head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def _read_object(self, uri: str) -> bytes:
        if not uri.startswith("s3://"):
            try:
                return Path(uri).read_bytes()
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Bronze metadata object not found: {uri}"
                ) from error
        bucket, key = self._s3_location(uri)
        try:
            response = self.s3_client().get_object(Bucket=bucket, Key=key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"Bronze metadata object not found: {uri}"
                ) from error
            raise
        return response["Body"].read()

    def _write_object(
        self,
        uri: str,
        payload: bytes,
        *,
        content_type: str,
    ) -> None:
        if not uri.startswith("s3://"):
            path = Path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == BRONZE_LATEST_POINTER:
                temporary = path.with_name(f".{path.name}.tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
            else:
                path.write_bytes(payload)
            return
        bucket, key = self._s3_location(uri)
        self.s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
        )

    def resolve_event_spine_snapshot(
        self,
        *,
        snapshot_override_uri: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Resolve and validate one complete immutable Gold event-spine snapshot."""

        if snapshot_override_uri:
            snapshot_uri = snapshot_override_uri.rstrip("/")
        else:
            pointer_uri = self.event_spine_latest_pointer_uri
            try:
                pointer = json.loads(self._read_object(pointer_uri))
                snapshot_uri = str(pointer["snapshot_uri"]).rstrip("/")
            except (
                KeyError,
                TypeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as error:
                raise RuntimeError(
                    f"Malformed event-spine latest pointer: {pointer_uri}"
                ) from error

        expected_prefix = f"{self.event_spine_root}/snapshot_id="
        if not snapshot_uri.startswith(expected_prefix):
            raise ValueError(
                "Event-spine snapshot is outside the canonical root: "
                f"{snapshot_uri!r}"
            )
        if not self._object_exists(self.event_spine_success_uri(snapshot_uri)):
            raise RuntimeError(
                f"Event-spine snapshot is not complete: {snapshot_uri}"
            )

        manifest_uri = self.event_spine_manifest_uri(snapshot_uri)
        try:
            manifest = json.loads(self._read_object(manifest_uri))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Malformed event-spine manifest: {manifest_uri}"
            ) from error
        if not isinstance(manifest, dict):
            raise RuntimeError(
                f"Malformed event-spine manifest: {manifest_uri}"
            )
        manifest_snapshot_uri = str(manifest.get("snapshot_uri", "")).rstrip("/")
        if manifest_snapshot_uri and manifest_snapshot_uri != snapshot_uri:
            raise RuntimeError(
                "Event-spine manifest snapshot URI mismatch: "
                f"selected={snapshot_uri!r}, manifest={manifest_snapshot_uri!r}"
            )
        return snapshot_uri, manifest

    def _resolve_current_snapshot(
        self,
        *,
        root_uri: str,
        pointer_uri: str,
        manifest_snapshot_field: str,
        label: str,
    ) -> tuple[str, dict[str, object]]:
        try:
            pointer = json.loads(self._read_object(pointer_uri))
            snapshot_id = str(pointer["snapshot_id"])
            snapshot_uri = str(pointer["snapshot_uri"]).rstrip("/")
        except (
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise RuntimeError(f"Malformed {label} latest pointer: {pointer_uri}") from error
        if not snapshot_id or snapshot_uri != self.snapshot_uri(root_uri, snapshot_id):
            raise ValueError(
                f"{label} pointer snapshot identity is invalid: "
                f"snapshot_id={snapshot_id!r}, snapshot_uri={snapshot_uri!r}"
            )
        if not self._object_exists(self.snapshot_success_uri(snapshot_uri)):
            raise RuntimeError(f"{label} snapshot is not complete: {snapshot_uri}")
        manifest_uri = self.snapshot_manifest_uri(snapshot_uri)
        try:
            manifest = json.loads(self._read_object(manifest_uri))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Malformed {label} manifest: {manifest_uri}") from error
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Malformed {label} manifest: {manifest_uri}")
        recorded_uri = str(manifest.get(manifest_snapshot_field, "")).rstrip("/")
        if recorded_uri != snapshot_uri:
            raise RuntimeError(
                f"{label} manifest snapshot URI mismatch: "
                f"selected={snapshot_uri!r}, manifest={recorded_uri!r}"
            )
        if str(manifest.get("snapshot_id", "")) != snapshot_id:
            raise RuntimeError(
                f"{label} manifest snapshot ID mismatch: "
                f"selected={snapshot_id!r}, manifest={manifest.get('snapshot_id')!r}"
            )
        if not self._snapshot_has_parquet(snapshot_uri):
            raise RuntimeError(f"{label} snapshot contains no Parquet: {snapshot_uri}")
        return snapshot_uri, manifest

    def resolve_current_integration_snapshot(self) -> tuple[str, dict[str, object]]:
        return self._resolve_current_snapshot(
            root_uri=self.integration_root,
            pointer_uri=self.integration_latest_pointer_uri,
            manifest_snapshot_field="snapshot_root",
            label="integration-sampling",
        )

    def resolve_current_silver_weather_snapshot(self) -> tuple[str, dict[str, object]]:
        return self._resolve_current_snapshot(
            root_uri=self.silver_weather_root,
            pointer_uri=self.silver_weather_latest_pointer_uri,
            manifest_snapshot_field="snapshot_uri",
            label="Silver weather",
        )
    def resolve_national_temporal_history_files(
        self,
    ) -> tuple[list[str], dict[str, object]]:
        """Resolve and freeze the current national temporal-history object set.

        The national temporal history is not published as a snapshot directory with
        ``_latest.json``/``manifest.json``/``_SUCCESS``. It is an immutable
        partitioned history rooted at ``national_temporal_history_root``.

        For downstream lineage, this resolver lists the exact Parquet objects that
        exist now and derives a deterministic object-set identity from their URIs.
        The returned manifest is therefore a frozen lineage descriptor for the
        precise history object set consumed by the caller.
        """
        root_uri = self.national_temporal_history_root.rstrip("/")
        parquet_uris = self.list_object_uris(root_uri, suffix=".parquet")

        if not parquet_uris:
            raise RuntimeError(
                "National temporal history contains no Parquet objects: "
                f"{root_uri}"
            )

        # list_object_uris() is already sorted, but sort again here so the lineage
        # identity remains deterministic even if that helper changes later.
        parquet_uris = sorted(parquet_uris)
        object_set_payload = "\n".join(parquet_uris).encode("utf-8")
        object_set_sha256 = hashlib.sha256(object_set_payload).hexdigest()

        manifest: dict[str, object] = {
            "snapshot_id": f"objectset-{object_set_sha256[:24]}",
            "root_uri": root_uri,
            "object_count": len(parquet_uris),
            "object_set_sha256": object_set_sha256,
            "objects": parquet_uris,
        }
        return parquet_uris, manifest

    def resolve_current_environmental_features_snapshot(
        self,
    ) -> tuple[str, dict[str, object]]:
        return self._resolve_current_snapshot(
            root_uri=self.environmental_features_root,
            pointer_uri=self.environmental_features_latest_pointer_uri,
            manifest_snapshot_field="snapshot_uri",
            label="Gold environmental features",
        )

    def resolve_current_final_model_table_snapshot(
        self,
    ) -> tuple[str, dict[str, object]]:
        """Resolve and validate the currently published final model-table snapshot."""

        return self._resolve_current_snapshot(
            root_uri=self.final_model_table_root,
            pointer_uri=self.final_model_table_latest_pointer_uri,
            manifest_snapshot_field="snapshot_uri",
            label="Gold final model table",
        )

    def _snapshot_has_parquet(self, snapshot_uri: str) -> bool:
        if not snapshot_uri.startswith("s3://"):
            return any(Path(snapshot_uri).rglob("*.parquet"))
        bucket, prefix = self._s3_location(f"{snapshot_uri}/placeholder")
        response = self.s3_client().list_objects_v2(
            Bucket=bucket,
            Prefix=prefix.removesuffix("placeholder"),
        )
        return any(
            str(item.get("Key", "")).endswith(".parquet")
            for item in response.get("Contents", [])
        )

    def _prefix_has_objects(self, uri: str) -> bool:
        if not uri.startswith("s3://"):
            path = Path(uri)
            return path.exists() and any(path.rglob("*"))
        bucket, prefix = self._s3_location(f"{uri.rstrip('/')}/placeholder")
        response = self.s3_client().list_objects_v2(
            Bucket=bucket,
            Prefix=prefix.removesuffix("placeholder"),
            MaxKeys=1,
        )
        return bool(response.get("KeyCount", 0))

    def _parquet_file_count(self, snapshot_uri: str) -> int:
        if not snapshot_uri.startswith("s3://"):
            return sum(1 for _ in Path(snapshot_uri).rglob("*.parquet"))
        bucket, prefix = self._s3_location(f"{snapshot_uri.rstrip('/')}/placeholder")
        paginator = self.s3_client().get_paginator("list_objects_v2")
        return sum(
            1
            for page in paginator.paginate(
                Bucket=bucket,
                Prefix=prefix.removesuffix("placeholder"),
            )
            for item in page.get("Contents", [])
            if str(item.get("Key", "")).endswith(".parquet")
        )

    def write_bronze_snapshot(
        self,
        lf: pl.LazyFrame,
        *,
        source_key: str,
        snapshot_id: str,
        partitioning_columns: Sequence[str],
    ) -> str:
        """Stream a partitioned Parquet snapshot without mutating completed data.

        An incomplete prefix may be reused by a Dagster retry with the same run ID.
        It remains invisible to readers until ``complete_bronze_snapshot`` writes
        the completion marker and advances the pointer. Failed prefixes are left
        in place deliberately for later, separately managed garbage collection.
        """

        if not partitioning_columns:
            raise ValueError("Bronze snapshots require at least one partition column")
        snapshot_uri = self.bronze_snapshot_uri(source_key, snapshot_id)
        success_uri = f"{snapshot_uri}/{BRONZE_SUCCESS_MARKER}"
        if self._object_exists(success_uri):
            return snapshot_uri

        lf.sink_parquet(
            pl.PartitionBy(
                snapshot_uri,
                key=list(partitioning_columns),
                include_key=False,
            ),
            compression="zstd",
            compression_level=3,
            storage_options=self.storage_options,
            credential_provider=None,
            mkdir=True,
            engine="streaming",
        )
        return snapshot_uri

    def complete_bronze_snapshot(
        self,
        *,
        source_key: str,
        snapshot_id: str,
        created_at: datetime,
    ) -> str:
        """Mark a written snapshot complete, then atomically publish its pointer."""

        if created_at.tzinfo is None:
            raise ValueError("Bronze snapshot created_at must include a timezone")
        snapshot_uri = self.bronze_snapshot_uri(source_key, snapshot_id)
        if not self._snapshot_has_parquet(snapshot_uri):
            raise RuntimeError(
                f"Cannot complete Bronze snapshot without Parquet files: {snapshot_uri}"
            )

        success_uri = f"{snapshot_uri}/{BRONZE_SUCCESS_MARKER}"
        if not self._object_exists(success_uri):
            self._write_object(
                success_uri,
                b"",
                content_type="application/octet-stream",
            )

        pointer = BronzeSnapshotPointer(
            source_key=source_key,
            snapshot_id=snapshot_id,
            created_at=created_at.astimezone(UTC),
        )
        pointer_uri = self._bronze_pointer_uri(source_key)
        if self._object_exists(pointer_uri):
            current = BronzeSnapshotPointer.from_json(
                self._read_object(pointer_uri),
                expected_source_key=source_key,
            )
            current_order = (current.created_at, current.snapshot_id)
            candidate_order = (pointer.created_at, pointer.snapshot_id)
            if current_order >= candidate_order:
                return snapshot_uri
        self._write_object(
            pointer_uri,
            pointer.to_json(),
            content_type="application/json",
        )
        return snapshot_uri

    def resolve_current_bronze_snapshot(self, source_key: str) -> str:
        """Resolve and validate the published completed snapshot for a source."""

        pointer_uri = self._bronze_pointer_uri(source_key)
        pointer = BronzeSnapshotPointer.from_json(
            self._read_object(pointer_uri),
            expected_source_key=source_key,
        )
        snapshot_uri = self.bronze_snapshot_uri(source_key, pointer.snapshot_id)
        if not self._object_exists(f"{snapshot_uri}/{BRONZE_SUCCESS_MARKER}"):
            raise RuntimeError(
                f"Bronze snapshot pointer references an incomplete snapshot: {snapshot_uri}"
            )
        return snapshot_uri

    def scan_bronze_snapshot(
        self,
        source_key: str,
        *,
        snapshot_uri: str | None = None,
    ) -> pl.LazyFrame:
        """Scan the current complete Bronze Parquet snapshot through S3."""

        current_uri = snapshot_uri or self.resolve_current_bronze_snapshot(source_key)
        source_root = self.resolve_source_path(source_key, "bronze")
        if not current_uri.startswith(f"{source_root}/snapshot_id="):
            raise ValueError(
                f"Bronze snapshot URI is outside source root {source_root!r}: "
                f"{current_uri!r}"
            )
        if not self._object_exists(f"{current_uri}/{BRONZE_SUCCESS_MARKER}"):
            raise RuntimeError(f"Cannot scan incomplete Bronze snapshot: {current_uri}")
        return pl.scan_parquet(
            f"{current_uri}/**/*.parquet",
            hive_partitioning=True,
            storage_options=self.storage_options,
            credential_provider=None,
        )

    @property
    def canonical_crosswalk_uri(self) -> str:
        return self.crosswalk_path or (
            f"{self.reference_root}/canonical_crime_crosswalk_v1_5.csv"
        )

    @property
    def silver_crime_offenses_root(self) -> str:
        return f"{self.silver_root.rstrip('/')}/crime_offenses"

    @property
    def silver_crime_offenses_uri(self) -> str:
        """Compatibility alias for the one logical unified Silver root."""

        return self.silver_crime_offenses_root

    def silver_snapshot_uri(self, snapshot_id: str) -> str:
        self._validate_snapshot_id(snapshot_id)
        return self.snapshot_uri(self.silver_crime_offenses_root, snapshot_id)

    @staticmethod
    def silver_source_parquet_glob(snapshot_uri: str, source_city: str) -> str:
        return (
            f"{snapshot_uri.rstrip('/')}/source_city={source_city}/**/*.parquet"
        )

    def _silver_pointer_uri(self) -> str:
        return f"{self.silver_crime_offenses_root}/{SILVER_LATEST_POINTER}"

    def resolve_crosswalk(self) -> pl.LazyFrame:
        storage_options = (
            self.storage_options
            if self.canonical_crosswalk_uri.startswith("s3://")
            else {}
        )
        return pl.scan_csv(
            self.canonical_crosswalk_uri,
            storage_options=storage_options,
            credential_provider=None,
        )

    def canonical_crosswalk_sha256(self) -> str:
        return hashlib.sha256(
            self._read_object(self.canonical_crosswalk_uri)
        ).hexdigest()

    def scan_source(self, source_key: str) -> pl.LazyFrame:
        config = get_source(source_key).config
        root = self.source_root(source_key).rstrip("/")
        frames: list[pl.LazyFrame] = []

        for pattern in config.patterns:
            uri = f"{root}/{pattern.glob}"
            frames.append(
                read_source_pattern(
                    uri,
                    pattern,
                    storage_options=self.storage_options,
                    s3_client_factory=self.s3_client,
                )
            )

        if len(frames) == 1:
            return frames[0]
        return pl.concat(frames, how="diagonal_relaxed")

    def resolve_current_silver_snapshot(self) -> str:
        pointer = SilverSnapshotPointer.from_json(
            self._read_object(self._silver_pointer_uri())
        )
        expected_uri = self.silver_snapshot_uri(pointer.snapshot_id)
        if pointer.snapshot_uri != expected_uri:
            raise ValueError(
                "Silver snapshot pointer URI mismatch: "
                f"expected {expected_uri!r}, found {pointer.snapshot_uri!r}"
            )
        if not self._object_exists(f"{pointer.snapshot_uri}/{SILVER_SUCCESS_MARKER}"):
            raise RuntimeError(
                "Silver snapshot pointer references an incomplete snapshot: "
                f"{pointer.snapshot_uri}"
            )
        return pointer.snapshot_uri

    def _scan_silver_snapshot(
        self,
        snapshot_uri: str,
        *,
        require_success: bool,
    ) -> pl.LazyFrame:
        root_prefix = f"{self.silver_crime_offenses_root}/snapshot_id="
        if not snapshot_uri.startswith(root_prefix):
            raise ValueError(
                "Silver snapshot URI is outside the unified Silver root: "
                f"{snapshot_uri!r}"
            )
        if require_success and not self._object_exists(
            f"{snapshot_uri}/{SILVER_SUCCESS_MARKER}"
        ):
            raise RuntimeError(
                f"Cannot scan incomplete Silver snapshot: {snapshot_uri}"
            )
        scanned = pl.scan_parquet(
            f"{snapshot_uri}/**/*.parquet",
            storage_options=self.storage_options,
            credential_provider=None,
            hive_partitioning=True,
            hive_schema={
                "snapshot_id": pl.String,
                "source_city": pl.String,
                "occurrence_year": pl.Int16,
            },
        )
        available = set(scanned.collect_schema().names())
        missing = set(CANONICAL_CRIME_SCHEMA) - available
        if missing:
            raise ValueError(
                f"Silver snapshot is missing canonical columns: {sorted(missing)}"
            )
        return scanned.select(
            pl.col(name).cast(dtype, strict=False).alias(name)
            for name, dtype in CANONICAL_CRIME_SCHEMA.items()
        )

    def scan_silver_snapshot(
        self,
        snapshot_uri: str | None = None,
    ) -> pl.LazyFrame:
        current_uri = snapshot_uri or self.resolve_current_silver_snapshot()
        return self._scan_silver_snapshot(current_uri, require_success=True)

    def read_silver_manifest(
        self,
        snapshot_uri: str | None = None,
    ) -> dict[str, object]:
        current_uri = snapshot_uri or self.resolve_current_silver_snapshot()
        if not self._object_exists(f"{current_uri}/{SILVER_SUCCESS_MARKER}"):
            raise RuntimeError(f"Cannot read incomplete Silver snapshot: {current_uri}")
        try:
            document = json.loads(self._read_object(f"{current_uri}/{SILVER_MANIFEST}"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Malformed Silver manifest: {current_uri}") from error
        if not isinstance(document, dict):
            raise TypeError("Malformed Silver manifest: expected an object")
        return document

    def get_source_fixture(self, source_key: str) -> pl.LazyFrame:
        get_source(source_key)
        root = (
            Path(self.local_fixture_root)
            if self.local_fixture_root
            else PROJECT_ROOT / "tests" / "fixtures"
        )
        path = root / "data" / "cities" / f"{source_key}.parquet"
        return pl.scan_parquet(path, use_statistics=True)

    def get_crosswalk_fixture(self) -> pl.LazyFrame:
        path = (
            Path(self.crosswalk_path)
            if self.crosswalk_path
            else PROJECT_ROOT
            / "src"
            / "crimenet_data"
            / "artifacts"
            / "canonical_crime_crosswalk_v1_5.csv"
        )
        return pl.scan_csv(path)

    @staticmethod
    def canonical_schema_document() -> dict[str, str]:
        return {name: str(dtype) for name, dtype in CANONICAL_CRIME_SCHEMA.items()}

    @classmethod
    def canonical_schema_sha256(cls) -> str:
        payload = json.dumps(
            cls.canonical_schema_document(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _silver_quality_summary(
        lf: pl.LazyFrame,
        *,
        mapping_version: str,
    ) -> dict[str, object]:
        actual_schema = lf.collect_schema()
        if actual_schema != CANONICAL_CRIME_SCHEMA:
            raise ValueError(
                "Silver snapshot schema mismatch: "
                f"expected={CANONICAL_CRIME_SCHEMA}, actual={actual_schema}"
            )
        mapped = pl.col("canonical_mapping_found").fill_null(False)

        # A source row is considered to have populated taxonomy only when
        # one of that source's configured canonical-crosswalk keys is
        # populated. Do not treat unrelated source taxonomy fields as
        # evidence that a mapping should exist.
        #
        # Example: Atlanta current APD rows can legitimately have
        # source_offense_code="NOT_APPL" while
        # source_offense_description is null. Atlanta keys its crosswalk on
        # source_offense_description, so those rows are intentionally
        # unmatched rather than populated-key crosswalk failures.
        source_taxonomy_expressions: list[pl.Expr] = []

        for source_key in SOURCE_KEYS:
            crosswalk_keys = tuple(get_source(source_key).config.crosswalk_keys)
            if not crosswalk_keys:
                continue

            key_populated = pl.any_horizontal(
                *(
                    pl.col(column)
                    .cast(pl.String, strict=False)
                    .fill_null("")
                    .str.strip_chars()
                    .ne("")
                    for column in crosswalk_keys
                )
            )

            source_taxonomy_expressions.append(
                (pl.col("source_city") == source_key) & key_populated
            )

        taxonomy_populated = (
            pl.any_horizontal(*source_taxonomy_expressions).fill_null(False)
            if source_taxonomy_expressions
            else pl.lit(False)
        )
        summary = (
            lf.select(
                pl.len().alias("row_count"),
                pl.col("source_city").n_unique().alias("source_count"),
                pl.col("crime_id").n_unique().alias("unique_crime_ids"),
                pl.col("crime_id").null_count().alias("null_crime_ids"),
                pl.col("source_city").null_count().alias("null_source_cities"),
                pl.col("source_record_id").null_count().alias("null_source_record_ids"),
                pl.col("occurrence_timestamp")
                .null_count()
                .alias("null_occurrence_timestamps"),
                pl.col("occurrence_year").null_count().alias("null_occurrence_years"),
                pl.col("latitude").null_count().alias("null_latitudes"),
                pl.col("longitude").null_count().alias("null_longitudes"),
                pl.col("source_coordinate_bounds_valid")
                .null_count()
                .alias("null_source_coordinate_bounds_valid"),
                (~pl.col("latitude").is_finite())
                .fill_null(False)
                .sum()
                .alias("nonfinite_latitudes"),
                (~pl.col("longitude").is_finite())
                .fill_null(False)
                .sum()
                .alias("nonfinite_longitudes"),
                (
                    (~pl.col("latitude").is_between(-90.0, 90.0))
                    | (~pl.col("longitude").is_between(-180.0, 180.0))
                )
                .fill_null(False)
                .sum()
                .alias("world_bounds_violations"),
                ((pl.col("latitude") == 0.0) & (pl.col("longitude") == 0.0))
                .fill_null(False)
                .sum()
                .alias("zero_zero_coordinates"),
                (~pl.col("occurrence_year").is_between(2014, 2026))
                .fill_null(False)
                .sum()
                .alias("occurrence_year_out_of_range"),
                (
                    pl.col("occurrence_timestamp").dt.year()
                    != pl.col("occurrence_year").cast(pl.Int32)
                )
                .fill_null(False)
                .sum()
                .alias("occurrence_year_mismatch"),
                ((~mapped) & taxonomy_populated)
                .sum()
                .alias("unexpected_populated_unmapped_rows"),
                pl.col("review_required")
                .fill_null(False)
                .sum()
                .alias("review_required_rows"),
                pl.col("include_in_model")
                .fill_null(False)
                .sum()
                .alias("include_in_model_rows"),
                (mapped & (pl.col("mapping_version").fill_null("") != mapping_version))
                .sum()
                .alias("wrong_mapping_version_rows"),
                (
                    pl.col("include_in_model").fill_null(False)
                    & (pl.col("mapping_action").fill_null("") != "map")
                )
                .sum()
                .alias("included_but_not_map_rows"),
                (
                    pl.col("include_in_model").fill_null(False)
                    & ~pl.col("source_coordinate_bounds_valid").fill_null(False)
                )
                .sum()
                .alias("included_outside_source_bounds_rows"),
                (~pl.col("source_coordinate_bounds_valid").fill_null(False))
                .sum()
                .alias("outside_source_bounds_rows"),
                (
                    (pl.col("mapping_action").fill_null("") == "map")
                    & ~pl.col("include_in_model").fill_null(False)
                    & pl.col("source_coordinate_bounds_valid").fill_null(False)
                    & pl.col("occurrence_timestamp_valid").fill_null(False)
                )
                .sum()
                .alias("map_but_not_included_rows"),
                (
                    pl.col("include_in_model").fill_null(False)
                    & pl.any_horizontal(
                        pl.col("canonical_family_code").is_null(),
                        pl.col("canonical_offense_family").is_null(),
                        pl.col("canonical_subtype_code").is_null(),
                        pl.col("canonical_offense_subtype").is_null(),
                    )
                )
                .sum()
                .alias("modeled_rows_missing_taxonomy"),
                (
                    pl.col("crime_id")
                    != pl.concat_str(
                        [pl.col("source_city"), pl.col("source_record_id")],
                        separator=":",
                    )
                )
                .fill_null(False)
                .sum()
                .alias("crime_id_contract_violations"),
                (
                    pl.col("mapping_action").is_not_null()
                    & ~pl.col("mapping_action").is_in(
                        ["map", "drop", "exclude_non_criminal"]
                    )
                )
                .sum()
                .alias("unknown_mapping_action_rows"),
                pl.col("occurrence_timestamp").min().alias("min_occurrence_timestamp"),
                pl.col("occurrence_timestamp").max().alias("max_occurrence_timestamp"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
        summary["duplicate_crime_ids"] = int(summary["row_count"]) - int(
            summary["unique_crime_ids"]
        )
        failures = [
            name
            for name in (
                "null_crime_ids",
                "null_source_cities",
                "null_source_record_ids",
                "null_occurrence_timestamps",
                "null_occurrence_years",
                "null_latitudes",
                "null_longitudes",
                "null_source_coordinate_bounds_valid",
                "nonfinite_latitudes",
                "nonfinite_longitudes",
                "world_bounds_violations",
                "zero_zero_coordinates",
                "occurrence_year_out_of_range",
                "occurrence_year_mismatch",
                "unexpected_populated_unmapped_rows",
                "review_required_rows",
                "wrong_mapping_version_rows",
                "included_but_not_map_rows",
                "included_outside_source_bounds_rows",
                "map_but_not_included_rows",
                "modeled_rows_missing_taxonomy",
                "crime_id_contract_violations",
                "unknown_mapping_action_rows",
                "duplicate_crime_ids",
            )
            if int(summary[name]) != 0
        ]
        if int(summary["row_count"]) == 0:
            failures.append("row_count")
        if failures:
            raise RuntimeError(
                "Silver snapshot quality gate failed: "
                f"checks={failures}, summary={summary}"
            )
        return summary

    def _encoded_partition_paths(self, snapshot_uri: str) -> list[str]:
        if not snapshot_uri.startswith("s3://"):
            return [
                str(path.relative_to(Path(snapshot_uri)))
                for path in Path(snapshot_uri).rglob("*")
                if "%3D" in path.name
            ]
        bucket, prefix = self._s3_location(f"{snapshot_uri.rstrip('/')}/placeholder")
        paginator = self.s3_client().get_paginator("list_objects_v2")
        return [
            str(item.get("Key", ""))
            for page in paginator.paginate(
                Bucket=bucket,
                Prefix=prefix.removesuffix("placeholder"),
            )
            for item in page.get("Contents", [])
            if "%3D" in str(item.get("Key", ""))
        ][:25]

    def publish_silver_snapshot(
        self,
        lf: pl.LazyFrame,
        *,
        snapshot_id: str,
        created_at_utc: datetime,
        mapping_version: str,
        schema_version: str,
        crosswalk_sha256: str,
        source_snapshots: Mapping[str, str],
        per_source: Sequence[Mapping[str, object]],
        git_commit_sha: str | None = None,
    ) -> dict[str, object]:
        """Write, validate, and atomically publish one immutable Silver snapshot."""

        if created_at_utc.tzinfo is None:
            raise ValueError("Silver snapshot created_at_utc must include a timezone")
        snapshot_uri = self.silver_snapshot_uri(snapshot_id)
        if self._prefix_has_objects(snapshot_uri):
            raise RuntimeError(f"Silver snapshot prefix already exists: {snapshot_uri}")

        partition_columns = ["source_city", "occurrence_year"]
        prewrite = self._silver_quality_summary(lf, mapping_version=mapping_version)
        if int(prewrite["source_count"]) != len(source_snapshots):
            raise RuntimeError(
                "Silver source count does not match recorded Bronze snapshots: "
                f"rows={prewrite['source_count']}, inputs={len(source_snapshots)}"
            )

        lf.sink_parquet(
            pl.PartitionBy(
                snapshot_uri,
                key=partition_columns,
                include_key=False,
            ),
            compression="zstd",
            compression_level=3,
            storage_options=self.storage_options,
            credential_provider=None,
            mkdir=True,
            engine="streaming",
        )
        if not self._snapshot_has_parquet(snapshot_uri):
            raise RuntimeError(
                f"Silver snapshot write produced no Parquet files: {snapshot_uri}"
            )
        encoded_paths = self._encoded_partition_paths(snapshot_uri)
        if encoded_paths:
            raise RuntimeError(
                "Silver snapshot contains URL-encoded Hive partition paths: "
                f"{encoded_paths}"
            )

        readback = self._scan_silver_snapshot(snapshot_uri, require_success=False)
        postwrite = self._silver_quality_summary(
            readback,
            mapping_version=mapping_version,
        )
        if int(postwrite["row_count"]) != int(prewrite["row_count"]):
            raise RuntimeError(
                "Silver snapshot read-back row count mismatch: "
                f"expected={prewrite['row_count']}, actual={postwrite['row_count']}"
            )

        manifest: dict[str, object] = {
            "snapshot_id": snapshot_id,
            "snapshot_uri": snapshot_uri,
            "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
            "schema_version": schema_version,
            "mapping_version": mapping_version,
            "crosswalk_sha256": crosswalk_sha256,
            "row_count": int(postwrite["row_count"]),
            "include_in_model_rows": int(postwrite["include_in_model_rows"]),
            "source_count": int(postwrite["source_count"]),
            "partition_columns": partition_columns,
            "parquet_file_count": self._parquet_file_count(snapshot_uri),
            "min_occurrence_timestamp": (
                postwrite["min_occurrence_timestamp"].isoformat()
                if postwrite["min_occurrence_timestamp"] is not None
                else None
            ),
            "max_occurrence_timestamp": (
                postwrite["max_occurrence_timestamp"].isoformat()
                if postwrite["max_occurrence_timestamp"] is not None
                else None
            ),
            "unexpected_populated_unmapped_rows": int(
                postwrite["unexpected_populated_unmapped_rows"]
            ),
            "review_required_rows": int(postwrite["review_required_rows"]),
            "outside_source_bounds_rows": int(postwrite["outside_source_bounds_rows"]),
            "source_snapshots": dict(source_snapshots),
            "per_source": [dict(row) for row in per_source],
            "canonical_schema": self.canonical_schema_document(),
            "canonical_schema_sha256": self.canonical_schema_sha256(),
        }
        if git_commit_sha:
            manifest["git_commit_sha"] = git_commit_sha

        self._write_object(
            f"{snapshot_uri}/{SILVER_MANIFEST}",
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            content_type="application/json",
        )
        self._write_object(
            f"{snapshot_uri}/{SILVER_SUCCESS_MARKER}",
            b"",
            content_type="application/octet-stream",
        )
        pointer = SilverSnapshotPointer(
            snapshot_id=snapshot_id,
            snapshot_uri=snapshot_uri,
            created_at_utc=created_at_utc.astimezone(UTC),
            mapping_version=mapping_version,
        )
        self._write_object(
            self._silver_pointer_uri(),
            pointer.to_json(),
            content_type="application/json",
        )
        return manifest


__all__ = [
    "SOURCE_KEYS",
    "BronzeSnapshotPointer",
    "CrimeLakeResources",
    "SilverSnapshotPointer",
]
