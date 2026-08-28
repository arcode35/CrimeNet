"""Single access path for immutable final-model-table Parquet snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from crimenet_data.resources.crime_lake import CrimeLakeResources

ALLOWED_ML_SPLITS = frozenset({"train", "validation"})
REQUIRED_COLUMNS = frozenset(
    {
        "model_row_id",
        "row_type",
        "event_indicator",
        "is_observed_event",
        "event_count",
        "integration_weight_cell_seconds",
        "source_city",
        "split",
    }
)


@dataclass(frozen=True)
class ResolvedModelTable:
    snapshot_id: str
    snapshot_uri: str
    schema_version: str
    manifest: dict[str, Any]
    lake: CrimeLakeResources
    local_root: str | None = None

    @property
    def scan_root(self) -> str:
        return self.local_root or self.snapshot_uri

    @property
    def lineage(self) -> dict[str, object]:
        keys = (
            "event_spine_snapshot_id",
            "integration_snapshot_id",
            "environmental_snapshot_id",
            "temporal_history_snapshot_id",
        )
        return {
            "final_model_snapshot_id": self.snapshot_id,
            "final_model_snapshot_uri": self.snapshot_uri,
            "final_model_schema_version": self.schema_version,
            **{key: self.manifest.get(key) for key in keys},
        }

    def scan_split(self, split: str) -> pl.LazyFrame:
        if split not in ALLOWED_ML_SPLITS:
            raise ValueError(
                f"ML access is sealed to {sorted(ALLOWED_ML_SPLITS)}; got {split!r}"
            )
        root = self.scan_root.rstrip("/")
        glob = f"{root}/split={split}/source_city=*/*.parquet"
        frame = pl.scan_parquet(
            glob,
            storage_options=self.lake.storage_options_for(root),
            credential_provider=None,
            hive_partitioning=True,
        )
        if "snapshot_id" not in frame.collect_schema().names():
            frame = frame.with_columns(
                pl.lit(self.snapshot_id, dtype=pl.String).alias("snapshot_id")
            )
        missing = sorted(REQUIRED_COLUMNS - set(frame.collect_schema().names()))
        if missing:
            raise RuntimeError(f"Final model table is missing columns: {missing}")
        return frame


def _read_local_manifest(root: str) -> dict[str, Any]:
    path = Path(root)
    if not path.is_dir() or not any(path.glob("split=*/source_city=*/*.parquet")):
        raise ValueError(f"Local model-table root is not partitioned Parquet: {root}")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Local model-table override requires manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Local model-table manifest must be a JSON object")
    return manifest


def resolve_model_table(
    *,
    lake: CrimeLakeResources | None = None,
    snapshot_override_uri: str | None = None,
    local_root: str | None = None,
) -> ResolvedModelTable:
    """Resolve once and pin one canonical identity for the complete ML run."""

    lake = lake or CrimeLakeResources()
    if local_root:
        manifest = _read_local_manifest(local_root)
        snapshot_id = str(manifest.get("snapshot_id", ""))
        snapshot_uri = str(manifest.get("snapshot_uri", ""))
        canonical_id = snapshot_uri.rsplit("/snapshot_id=", 1)[-1]
        if (
            not snapshot_id
            or "/snapshot_id=" not in snapshot_uri
            or canonical_id != snapshot_id
            or "/" in canonical_id
        ):
            raise ValueError(
                "Local override manifest must retain its canonical immutable identity"
            )
    else:
        snapshot_uri, manifest = lake.resolve_final_model_table_snapshot(
            snapshot_override_uri=snapshot_override_uri
        )
        snapshot_id = str(manifest["snapshot_id"])
    schema_version = str(manifest.get("schema_version", ""))
    if not schema_version:
        raise RuntimeError("Final model-table manifest lacks schema_version")
    manifest_columns = manifest.get("columns")
    if not isinstance(manifest_columns, list):
        raise RuntimeError("Final model-table manifest lacks its columns contract")
    missing_manifest_columns = sorted(REQUIRED_COLUMNS - set(manifest_columns))
    if missing_manifest_columns:
        raise RuntimeError(
            "Final model-table manifest is missing columns: "
            f"{missing_manifest_columns}"
        )
    return ResolvedModelTable(
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri,
        schema_version=schema_version,
        manifest=dict(manifest),
        lake=lake,
        local_root=local_root,
    )


def enrich_config_with_lineage(
    config: dict[str, Any], table: ResolvedModelTable
) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    result.setdefault("data", {}).update(table.lineage)
    result["data"]["test_split_used"] = False
    return result


__all__ = [
    "ALLOWED_ML_SPLITS",
    "REQUIRED_COLUMNS",
    "ResolvedModelTable",
    "enrich_config_with_lineage",
    "resolve_model_table",
]
