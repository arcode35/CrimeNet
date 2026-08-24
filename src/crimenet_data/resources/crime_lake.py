from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import boto3
import dagster as dg
import deltalake
import polars as pl
from botocore.config import Config

from crimenet_data.assets.crime.ingestion.readers import read_source_pattern
from crimenet_data.assets.crime.sources import SOURCE_KEYS, get_source

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CrimeLakeResources(dg.ConfigurableResource):
    """Crime lake paths and Backblaze B2 storage configuration"""

    bucket: str = "s3://crimenet-data"
    delta_bucket: str = "b2://crimenet-data"
    crosswalk_path: str | None = None
    local_fixture_root: str | None = None

    @property
    def storage_options(self) -> dict[str, str]:
        """Options accepted by Polars and delta-rs for S3-compatible B2"""

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
        }
    @property
    def delta_storage_options(self) -> dict[str, str]:
        environment = {
            "B2_KEY_ID": os.environ.get("B2_KEY_ID"),
            "B2_APPLICATION_KEY": os.environ.get("B2_APPLICATION_KEY"),
            "B2_BUCKET_ID": os.environ.get("B2_BUCKET_ID"),
        }

        missing = sorted(
            name for name, value in environment.items() if not value
        )
        if missing:
            raise RuntimeError(
                "Missing native Backblaze B2 configuration: "
                + ", ".join(missing)
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
        return f"{self.delta_root}/bronze"

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
        delta_uri = self.resolve_source_path(source_key, schema)

        # Resolve the current Delta snapshot through native B2/OpenDAL.
        table = deltalake.DeltaTable(
            delta_uri,
            storage_options=self.delta_storage_options,
        )

        b2_files = table.file_uris()

        if not b2_files:
            raise RuntimeError(
                f"Delta table contains no active files: {delta_uri}"
            )

        delta_prefix = self.delta_bucket.rstrip("/") + "/"
        s3_prefix = self.bucket.rstrip("/") + "/"

        s3_files = []

        for uri in b2_files:
            if not uri.startswith(delta_prefix):
                raise ValueError(
                    f"Unexpected Delta file URI {uri!r}; "
                    f"expected prefix {delta_prefix!r}"
                )

            s3_files.append(
                s3_prefix + uri.removeprefix(delta_prefix)
            )

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

    def write_crimenet_table(
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


__all__ = ["SOURCE_KEYS", "CrimeLakeResources"]
