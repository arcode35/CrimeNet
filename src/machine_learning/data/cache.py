"""Immutable HPO stage-cache identities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_identity(
    *,
    cache_version: str,
    snapshot_id: str,
    schema_version: str,
    feature_contract_hash: str,
    model_family: str,
    model_module: str,
    split: str,
    fraction: float,
    seed: int,
    target_column: str | None = None,
) -> str:
    if split == "test":
        raise ValueError("Test split may not be cached")
    payload: dict[str, Any] = {
        "cache_version": cache_version,
        "snapshot_id": snapshot_id,
        "schema_version": schema_version,
        "feature_contract_hash": feature_contract_hash,
        "model_family": model_family,
        "model_module": model_module,
        "split": split,
        "fraction": fraction,
        "seed": seed,
        "target_column": target_column,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = ["cache_identity"]
