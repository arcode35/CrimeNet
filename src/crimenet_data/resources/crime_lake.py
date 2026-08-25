from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import dagster as dg
import deltalake
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError

from crimenet_data.assets.crime.ingestion.readers import read_source_pattern
from crimenet_data.assets.crime.sources import SOURCE_KEYS, get_source

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BRONZE_SUCCESS_MARKER = "_SUCCESS"
BRONZE_LATEST_POINTER = "_latest.json"


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


class CrimeLakeResources(dg.ConfigurableResource):
    """Crime lake paths and Backblaze B2 storage configuration"""

    bucket: str = "s3://crimenet-data"
    delta_bucket: str = "b2://crimenet-data"
    crosswalk_path: str | None = None
    local_fixture_root: str | None = None

    @property
    def storage_options(self) -> dict[str, object]:
        """Options accepted by Polars for B2's S3-compatible API."""

        if not self.bucket.startswith("s3://"):
            return {}

        environment = {
            "B2_ENDPOINT_URL": os.environ.get("B2_ENDPOINT_URL"),
            "B2_KEY_ID": os.environ.get("B2_KEY_ID"),
            "B2_APPLICATION_KEY": os.environ.get("B2_APPLICATION_KEY"),
            "B2_REGION": os.environ.get("B2_REGION", "us-east-005"),
        }
        missing = sorted(name for name, value in environment.items() if not value)
        if missing:
            raise RuntimeError(
                "Missing Backblaze B2 configuration: " + ", ".join(missing)
            )

        return {
            "aws_access_key_id": environment["B2_KEY_ID"],
            "aws_secret_access_key": environment["B2_APPLICATION_KEY"],
            "aws_region": environment["B2_REGION"],
            "aws_endpoint_url": environment["B2_ENDPOINT_URL"],
            "max_retries": 5,
        }

    @property
    def delta_storage_options(self) -> dict[str, str]:
        """Native B2/OpenDAL options retained for Delta-backed layers."""

        environment = {
            "B2_KEY_ID": os.environ.get("B2_KEY_ID"),
            "B2_APPLICATION_KEY": os.environ.get("B2_APPLICATION_KEY"),
            "B2_BUCKET_ID": os.environ.get("B2_BUCKET_ID"),
        }

        missing = sorted(name for name, value in environment.items() if not value)
        if missing:
            raise RuntimeError(
                "Missing native Backblaze B2 configuration: " + ", ".join(missing)
            )

        return {
            "opendal.application_key_id": environment["B2_KEY_ID"],
            "opendal.application_key": environment["B2_APPLICATION_KEY"],
            "opendal.bucket_id": environment["B2_BUCKET_ID"],
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
    def landing_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/raw_files/landing"

    @property
    def delta_root(self) -> str:
        return self.delta_bucket.rstrip("/")

    @property
    def bronze_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/bronze"

    @property
    def silver_root(self) -> str:
        return f"{self.delta_root}/silver"

    @property
    def gold_root(self) -> str:
        return f"{self.delta_root}/gold"

    @property
    def quality_root(self) -> str:
        return f"{self.bucket.rstrip('/')}/quality"

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
            "silver": f"{self.silver_root}/crime/{source_key}",
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
        if not snapshot_id or "/" in snapshot_id or "\\" in snapshot_id:
            raise ValueError(f"Invalid Bronze snapshot ID: {snapshot_id!r}")
        return f"{root}/snapshot_id={snapshot_id}"

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
            f"{self.landing_root}/reference/canonical_crime_crosswalk_v1_3.csv"
        )

    def resolve_crosswalk(self) -> pl.LazyFrame:
        return pl.scan_csv(
            self.canonical_crosswalk_uri,
            storage_options=self.storage_options,
            credential_provider=None,
        )

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

    def scan_source_delta(self, source_key: str, schema: str) -> pl.LazyFrame:
        if schema == "bronze":
            raise ValueError("Bronze is Parquet-backed; use scan_bronze_snapshot()")
        delta_uri = self.resolve_source_path(source_key, schema)

        # Resolve the current Delta snapshot through native B2/OpenDAL.
        table = deltalake.DeltaTable(
            delta_uri,
            storage_options=self.delta_storage_options,
        )

        b2_files = table.file_uris()

        if not b2_files:
            raise RuntimeError(f"Delta table contains no active files: {delta_uri}")

        delta_prefix = self.delta_bucket.rstrip("/") + "/"
        s3_prefix = self.bucket.rstrip("/") + "/"

        s3_files = []

        for uri in b2_files:
            if not uri.startswith(delta_prefix):
                raise ValueError(
                    f"Unexpected Delta file URI {uri!r}; "
                    f"expected prefix {delta_prefix!r}"
                )

            s3_files.append(s3_prefix + uri.removeprefix(delta_prefix))

        # Polars reads the physical Parquet files through B2's
        # S3-compatible endpoint.
        return pl.scan_parquet(
            s3_files,
            storage_options=self.storage_options,
            hive_partitioning=True,
        )

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
        root = (
            Path(self.local_fixture_root)
            if self.local_fixture_root
            else PROJECT_ROOT / "tests" / "fixtures"
        )
        path = root / "data" / "references" / "canonical_crime_crosswalk_v1_3.csv"
        return pl.scan_csv(path)

    def write_delta_table(
        self,
        lf: pl.LazyFrame,
        target_uri: str,
        partitioning_columns: Sequence[str],
    ) -> None:
        lf.sink_delta(
            target_uri,
            mode="overwrite",
            storage_options=self.delta_storage_options,
            credential_provider=None,
            delta_write_options={
                "schema_mode": "overwrite",
                "partition_by": list(partitioning_columns),
                "writer_properties": deltalake.WriterProperties(
                    compression="zstd",
                    compression_level=3,
                ),
            },
        )


__all__ = ["SOURCE_KEYS", "BronzeSnapshotPointer", "CrimeLakeResources"]
