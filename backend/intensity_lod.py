from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = (
    Path.home()
    / "crimenet-serving"
    / "data"
    / "national_feature_store"
)

STATIC_ROOT = ROOT / "mmap"
LOD_ROOT = STATIC_ROOT / "lod"

LOD_RESOLUTIONS = (
    4,
    5,
    6,
    7,
    8,
)


def atomic_save_npy(
    path: Path,
    array: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open(
        "wb"
    ) as f:

        np.save(
            f,
            array,
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def write_lod_intensities(
    snapshot_dir: Path,
    intensity_r9: np.ndarray,
) -> None:

    intensity_r9 = np.asarray(
        intensity_r9
    )

    if intensity_r9.ndim != 1:
        raise RuntimeError(
            "Expected 1D r9 intensity array"
        )

    if not np.isfinite(
        intensity_r9
    ).all():
        raise RuntimeError(
            "Non-finite r9 intensity"
        )

    if (
        intensity_r9
        < 0
    ).any():
        raise RuntimeError(
            "Negative r9 intensity"
        )


    # bincount returns float64 with weights.
    # Reuse this one conversion for all five levels.
    weights = np.asarray(
        intensity_r9,
        dtype=np.float64,
    )


    output_metadata = {

        "schema":
            "crimenet_intensity_lod_v1",

        "source_resolution":
            9,

        "aggregation":
            "sum_r9_child_intensity",

        "units":
            "events_per_second",

        "resolutions":
            {},
    }


    for resolution in (
        LOD_RESOLUTIONS
    ):

        static_dir = (
            LOD_ROOT
            / f"r{resolution}"
        )

        keys = np.load(
            static_dir
            / "h3_keys.npy",
            mmap_mode="r",
        )

        mapping = np.load(
            static_dir
            / "r9_to_parent_idx.npy",
            mmap_mode="r",
        )

        if len(
            mapping
        ) != len(
            intensity_r9
        ):

            raise RuntimeError(
                f"r{resolution}: mapping length mismatch"
            )


        summed = np.bincount(
            mapping,
            weights=weights,
            minlength=len(
                keys
            ),
        )


        if len(
            summed
        ) != len(
            keys
        ):

            raise RuntimeError(
                f"r{resolution}: aggregation length mismatch"
            )


        if not np.isfinite(
            summed
        ).all():

            raise RuntimeError(
                f"r{resolution}: non-finite aggregated intensity"
            )


        summed_f32 = summed.astype(
            np.float32,
            copy=False,
        )


        out = (
            snapshot_dir
            / "lod"
            / f"r{resolution}"
            / "intensity.npy"
        )


        atomic_save_npy(
            out,
            summed_f32,
        )


        # Conservation check.
        original_total = float(
            np.sum(
                weights,
                dtype=np.float64,
            )
        )

        aggregate_total = float(
            np.sum(
                summed,
                dtype=np.float64,
            )
        )


        if not np.isclose(
            original_total,
            aggregate_total,
            rtol=1e-10,
            atol=1e-12,
        ):

            raise RuntimeError(
                f"r{resolution}: intensity conservation failed: "
                f"{original_total} != {aggregate_total}"
            )


        output_metadata[
            "resolutions"
        ][
            str(
                resolution
            )
        ] = {

            "cells":
                len(
                    keys
                ),

            "total_events_per_second":
                aggregate_total,
        }


        print(
            f"LOD r{resolution}: "
            f"{len(keys):,} cells",
            flush=True,
        )


    metadata_path = (
        snapshot_dir
        / "lod"
        / "metadata.json"
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = metadata_path.with_name(
        metadata_path.name + ".tmp"
    )

    with tmp.open(
        "w"
    ) as f:

        json.dump(
            output_metadata,
            f,
            indent=2,
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        metadata_path,
    )
