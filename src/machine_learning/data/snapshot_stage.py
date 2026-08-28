"""Resumable, test-sealed local staging for immutable HPO snapshots."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from botocore.exceptions import ClientError

from crimenet_data.resources.crime_lake import CrimeLakeResources
from machine_learning.data.model_table import resolve_model_table

HPO_STAGE_MANIFEST = "_HPO_STAGE_MANIFEST.json"
HPO_ALLOWED_SPLITS = ("train", "validation")
_REQUIRED_METADATA = ("manifest.json",)
_OPTIONAL_METADATA = ("_SUCCESS",)


@dataclass(frozen=True)
class RemoteSnapshotObject:
    key: str
    relative_path: str
    size: int


@dataclass(frozen=True)
class SnapshotStagePlan:
    snapshot_id: str
    snapshot_uri: str
    bucket: str
    prefix: str
    objects: tuple[RemoteSnapshotObject, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.objects)

    def split_bytes(self, split: str) -> int:
        prefix = f"split={split}/"
        return sum(
            item.size for item in self.objects if item.relative_path.startswith(prefix)
        )


def _parse_s3_snapshot_uri(snapshot_uri: str) -> tuple[str, str]:
    parsed = urlparse(snapshot_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected an immutable s3:// snapshot URI, got {snapshot_uri!r}")
    return parsed.netloc, parsed.path.strip("/")


def _is_not_found(error: ClientError) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def _head_object_size(client: Any, *, bucket: str, key: str) -> int | None:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if _is_not_found(error):
            return None
        raise
    return int(response["ContentLength"])


def _list_prefix(client: Any, *, bucket: str, prefix: str) -> Iterable[dict[str, Any]]:
    token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        yield from response.get("Contents", [])
        if not response.get("IsTruncated"):
            break
        token = str(response["NextContinuationToken"])


def plan_hpo_snapshot_stage(
    *,
    snapshot_uri: str,
    snapshot_id: str,
    lake: CrimeLakeResources | None = None,
) -> SnapshotStagePlan:
    """List only train/validation objects plus known root metadata."""

    lake = lake or CrimeLakeResources()
    client = lake.s3_client()
    bucket, prefix = _parse_s3_snapshot_uri(snapshot_uri)
    objects: list[RemoteSnapshotObject] = []

    for split in HPO_ALLOWED_SPLITS:
        relative_prefix = f"split={split}/"
        remote_prefix = f"{prefix}/{relative_prefix}"
        found = 0
        for raw in _list_prefix(client, bucket=bucket, prefix=remote_prefix):
            key = str(raw["Key"])
            relative = key.removeprefix(f"{prefix}/")
            if not relative.startswith(relative_prefix):
                raise RuntimeError(f"Unexpected object returned for {remote_prefix}: {key}")
            objects.append(
                RemoteSnapshotObject(
                    key=key,
                    relative_path=relative,
                    size=int(raw["Size"]),
                )
            )
            found += 1
        if not found:
            raise RuntimeError(f"Canonical snapshot contains no split={split} objects")

    for name in (*_REQUIRED_METADATA, *_OPTIONAL_METADATA):
        key = f"{prefix}/{name}"
        size = _head_object_size(client, bucket=bucket, key=key)
        if size is None:
            if name in _REQUIRED_METADATA:
                raise RuntimeError(f"Canonical snapshot is missing required metadata: {name}")
            continue
        objects.append(RemoteSnapshotObject(key=key, relative_path=name, size=size))

    relative_paths = [item.relative_path for item in objects]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("Remote snapshot staging plan contains duplicate object paths")
    if any(path.startswith("split=test/") for path in relative_paths):
        raise RuntimeError("HPO staging plan must never contain split=test")
    return SnapshotStagePlan(
        snapshot_id=snapshot_id,
        snapshot_uri=snapshot_uri.rstrip("/"),
        bucket=bucket,
        prefix=prefix,
        objects=tuple(sorted(objects, key=lambda item: item.relative_path)),
    )


def assert_test_split_not_staged(local_root: Path) -> None:
    if (local_root / "split=test").exists() or any(
        path.name == "split=test" for path in local_root.rglob("split=test")
    ):
        raise RuntimeError(f"Local HPO snapshot contains forbidden split=test: {local_root}")


def _download_one(
    *,
    client: Any,
    bucket: str,
    item: RemoteSnapshotObject,
    local_root: Path,
) -> tuple[str, bool]:
    destination = local_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == item.size:
        return item.relative_path, False

    partial = destination.with_name(destination.name + ".part")
    if partial.is_file() and partial.stat().st_size == item.size:
        partial.replace(destination)
        return item.relative_path, False
    partial.unlink(missing_ok=True)
    client.download_file(bucket, item.key, str(partial))
    actual_size = partial.stat().st_size
    if actual_size != item.size:
        raise RuntimeError(
            f"Incomplete staged object {item.relative_path}: "
            f"expected={item.size}, actual={actual_size}"
        )
    partial.replace(destination)
    return item.relative_path, True


def stage_hpo_snapshot(
    *,
    plan: SnapshotStagePlan,
    stage_dir: Path,
    lake: CrimeLakeResources | None = None,
    workers: int = 12,
) -> Path:
    """Resume-safe stage of a prelisted immutable snapshot to local storage."""

    started = time.perf_counter()
    lake = lake or CrimeLakeResources()
    local_root = stage_dir.expanduser().resolve() / f"snapshot_id={plan.snapshot_id}"
    local_root.mkdir(parents=True, exist_ok=True)
    assert_test_split_not_staged(local_root)
    client = lake.s3_client()
    downloaded = 0

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [
            executor.submit(
                _download_one,
                client=client,
                bucket=plan.bucket,
                item=item,
                local_root=local_root,
            )
            for item in plan.objects
        ]
        for future in as_completed(futures):
            _relative, changed = future.result()
            downloaded += int(changed)

    assert_test_split_not_staged(local_root)
    for item in plan.objects:
        path = local_root / item.relative_path
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"Staged snapshot validation failed for {item.relative_path}")

    manifest = json.loads((local_root / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("snapshot_id")) != plan.snapshot_id:
        raise RuntimeError("Local staged manifest changed canonical snapshot_id")
    if str(manifest.get("snapshot_uri", "")).rstrip("/") != plan.snapshot_uri:
        raise RuntimeError("Local staged manifest changed canonical snapshot_uri")
    resolved = resolve_model_table(local_root=str(local_root), lake=lake)
    if resolved.snapshot_id != plan.snapshot_id or resolved.snapshot_uri != plan.snapshot_uri:
        raise RuntimeError("Local staged snapshot failed canonical lineage validation")
    for split in HPO_ALLOWED_SPLITS:
        resolved.scan_split(split).collect_schema()

    stage_record = {
        "version": 1,
        "complete": True,
        "source_snapshot_id": plan.snapshot_id,
        "source_snapshot_uri": plan.snapshot_uri,
        "local_root": str(local_root),
        "object_count": len(plan.objects),
        "total_bytes": plan.total_bytes,
        "downloaded_objects_this_run": downloaded,
        "elapsed_seconds": time.perf_counter() - started,
        "objects": [asdict(item) | {"complete": True} for item in plan.objects],
        "test_split_staged": False,
    }
    stage_manifest = local_root / HPO_STAGE_MANIFEST
    temporary = stage_manifest.with_suffix(stage_manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(stage_record, indent=2, sort_keys=True))
    temporary.replace(stage_manifest)
    return local_root


def projected_hpo_cache_bytes(plan: SnapshotStagePlan) -> dict[str, int]:
    """Conservative Arrow-IPC projection from compressed Parquet byte counts."""

    expansion = 2.5
    train = plan.split_bytes("train")
    validation = plan.split_bytes("validation")
    return {
        "explore": int((train * 0.25 + validation * 0.25) * expansion),
        "full_train": int(train * expansion),
        "full_validation": int(validation * expansion),
    }


def preflight_hpo_disk(
    *,
    plan: SnapshotStagePlan,
    stage_dir: Path,
    cache_dir: Path,
    stage_enabled: bool,
) -> dict[str, int]:
    projections = projected_hpo_cache_bytes(plan)
    local_root = stage_dir.expanduser().resolve() / f"snapshot_id={plan.snapshot_id}"
    staged_actual = sum(
        item.size
        for item in plan.objects
        if (local_root / item.relative_path).is_file()
        and (local_root / item.relative_path).stat().st_size == item.size
    )
    stage_remaining = max(plan.total_bytes - staged_actual, 0) if stage_enabled else 0
    cache_actual = 0
    if cache_dir.exists():
        for metadata_path in cache_dir.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                arrow_path = Path(str(metadata["ipc_path"]))
            except (KeyError, OSError, TypeError, json.JSONDecodeError):
                continue
            if (
                metadata.get("version") == 5
                and metadata.get("final_model_snapshot_id") == plan.snapshot_id
                and arrow_path.is_file()
                and arrow_path.stat().st_size == int(metadata.get("bytes", -1))
            ):
                cache_actual += arrow_path.stat().st_size
    cache_remaining = max(sum(projections.values()) - cache_actual, 0)
    required = stage_remaining + cache_remaining
    safety_margin = max(10 * 1024**3, int(required * 0.15))
    probe = stage_dir if stage_dir.exists() else stage_dir.parent
    probe.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(probe).free
    report = {
        "remote_parquet_bytes": sum(
            item.size for item in plan.objects if item.relative_path.endswith(".parquet")
        ),
        "local_staged_snapshot_bytes": staged_actual,
        "explore_cache_projected_bytes": projections["explore"],
        "full_train_cache_projected_bytes": projections["full_train"],
        "full_validation_cache_projected_bytes": projections["full_validation"],
        "free_disk_bytes": free,
        "safety_margin_bytes": safety_margin,
        "remaining_required_bytes": required,
    }
    if required + safety_margin > free:
        raise RuntimeError(
            "Insufficient local disk for staged HPO snapshot and global Arrow caches: "
            f"required={required:,}, safety_margin={safety_margin:,}, free={free:,}, "
            f"cache_dir={cache_dir}"
        )
    return report


__all__ = [
    "HPO_STAGE_MANIFEST",
    "RemoteSnapshotObject",
    "SnapshotStagePlan",
    "assert_test_split_not_staged",
    "plan_hpo_snapshot_stage",
    "preflight_hpo_disk",
    "projected_hpo_cache_bytes",
    "stage_hpo_snapshot",
]
