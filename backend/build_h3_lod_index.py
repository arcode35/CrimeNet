from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from h3.api import basic_int as h3i


ROOT = (
    Path.home()
    / "crimenet-serving"
    / "data"
    / "national_feature_store"
)

STATIC_ROOT = ROOT / "mmap"
ENV_ROOT = ROOT / "environmental"
LOD_ROOT = STATIC_ROOT / "lod"

BASE_RESOLUTION = 9
LOD_RESOLUTIONS = (4, 5, 6, 7, 8)

PARENT_CHUNK_SIZE = 1_000_000


def compute_parent_keys(
    child_keys: np.ndarray,
    parent_resolution: int,
) -> np.ndarray:

    n = len(child_keys)

    result = np.empty(
        n,
        dtype=np.uint64,
    )

    for start in range(
        0,
        n,
        PARENT_CHUNK_SIZE,
    ):

        end = min(
            start + PARENT_CHUNK_SIZE,
            n,
        )

        result[start:end] = np.fromiter(
            (
                h3i.cell_to_parent(
                    int(cell),
                    parent_resolution,
                )
                for cell in child_keys[
                    start:end
                ]
            ),
            dtype=np.uint64,
            count=end - start,
        )

        print(
            f"r{parent_resolution}: "
            f"{end:,}/{n:,}",
            flush=True,
        )

    return result


def unique_parent_index(
    child_keys: np.ndarray,
    parent_resolution: int,
) -> tuple[np.ndarray, np.ndarray]:

    parents = compute_parent_keys(
        child_keys,
        parent_resolution,
    )

    unique_keys, inverse = np.unique(
        parents,
        return_inverse=True,
    )

    del parents

    if len(unique_keys) >= 2**31:
        raise RuntimeError(
            "Parent index exceeds int32 range"
        )

    inverse = inverse.astype(
        np.int32,
        copy=False,
    )

    return unique_keys, inverse


def save_level(
    resolution: int,
    keys: np.ndarray,
    r9_to_parent: np.ndarray,
) -> None:

    out = (
        LOD_ROOT
        / f"r{resolution}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = np.bincount(
        r9_to_parent,
        minlength=len(keys),
    ).astype(
        np.uint32,
        copy=False,
    )

    if int(
        counts.sum(
            dtype=np.uint64
        )
    ) != len(
        r9_to_parent
    ):

        raise RuntimeError(
            f"r{resolution}: child-count mismatch"
        )

    np.save(
        out / "h3_keys.npy",
        np.asarray(
            keys,
            dtype=np.uint64,
        ),
    )

    np.save(
        out / "r9_to_parent_idx.npy",
        np.asarray(
            r9_to_parent,
            dtype=np.int32,
        ),
    )

    np.save(
        out / "r9_child_count.npy",
        counts,
    )

    print(
        f"saved r{resolution}: "
        f"{len(keys):,} parents",
        flush=True,
    )


def validate_level(
    resolution: int,
    r9_keys: np.ndarray,
    parent_keys: np.ndarray,
    mapping: np.ndarray,
) -> None:

    rng = np.random.default_rng(
        42 + resolution
    )

    sample_size = min(
        1000,
        len(r9_keys),
    )

    sample = rng.choice(
        len(r9_keys),
        size=sample_size,
        replace=False,
    )

    for row in sample:

        expected = h3i.cell_to_parent(
            int(
                r9_keys[
                    row
                ]
            ),
            resolution,
        )

        actual = int(
            parent_keys[
                mapping[
                    row
                ]
            ]
        )

        if expected != actual:
            raise RuntimeError(
                f"r{resolution}: mapping validation failed "
                f"at row {row}"
            )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        LOD_ROOT.exists()
        and any(
            LOD_ROOT.iterdir()
        )
        and not args.force
    ):
        raise RuntimeError(
            f"{LOD_ROOT} already contains files. "
            "Use --force to rebuild."
        )

    LOD_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Canonical r9 domain
    # --------------------------------------------------------

    r9_keys = np.load(
        STATIC_ROOT / "h3_keys.npy",
        mmap_mode="r",
    )

    n_r9 = len(
        r9_keys
    )

    print(
        f"r9 domain: {n_r9:,}",
        flush=True,
    )


    # --------------------------------------------------------
    # r9 -> r8
    # --------------------------------------------------------

    r8_keys, r9_to_r8 = (
        unique_parent_index(
            r9_keys,
            8,
        )
    )

    save_level(
        8,
        r8_keys,
        r9_to_r8,
    )

    validate_level(
        8,
        r9_keys,
        r8_keys,
        r9_to_r8,
    )


    # --------------------------------------------------------
    # r8 -> r7, then compose r9 -> r7
    # --------------------------------------------------------

    r7_keys, r8_to_r7 = (
        unique_parent_index(
            r8_keys,
            7,
        )
    )

    r9_to_r7 = (
        r8_to_r7[
            r9_to_r8
        ]
    )

    save_level(
        7,
        r7_keys,
        r9_to_r7,
    )

    validate_level(
        7,
        r9_keys,
        r7_keys,
        r9_to_r7,
    )

    del r8_to_r7
    del r9_to_r7


    # --------------------------------------------------------
    # Existing exact r9 -> r6 environmental mapping
    # --------------------------------------------------------

    r6_keys = np.load(
        ENV_ROOT / "r6_keys.npy",
        mmap_mode="r",
    )

    r9_to_r6 = np.load(
        ENV_ROOT / "r9_to_r6_idx.npy",
        mmap_mode="r",
    )

    if len(
        r9_to_r6
    ) != n_r9:

        raise RuntimeError(
            "Existing r9->r6 mapping length mismatch"
        )

    r9_to_r6_i32 = np.asarray(
        r9_to_r6,
        dtype=np.int32,
    )

    save_level(
        6,
        r6_keys,
        r9_to_r6_i32,
    )

    validate_level(
        6,
        r9_keys,
        r6_keys,
        r9_to_r6_i32,
    )


    # --------------------------------------------------------
    # r6 -> r5, compose r9 -> r5
    # --------------------------------------------------------

    r5_keys, r6_to_r5 = (
        unique_parent_index(
            r6_keys,
            5,
        )
    )

    r9_to_r5 = (
        r6_to_r5[
            r9_to_r6_i32
        ]
    )

    save_level(
        5,
        r5_keys,
        r9_to_r5,
    )

    validate_level(
        5,
        r9_keys,
        r5_keys,
        r9_to_r5,
    )


    # --------------------------------------------------------
    # r5 -> r4, compose r9 -> r4
    # --------------------------------------------------------

    r4_keys, r5_to_r4 = (
        unique_parent_index(
            r5_keys,
            4,
        )
    )

    r9_to_r4 = (
        r5_to_r4[
            r9_to_r5
        ]
    )

    save_level(
        4,
        r4_keys,
        r9_to_r4,
    )

    validate_level(
        4,
        r9_keys,
        r4_keys,
        r9_to_r4,
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    counts = {}

    for resolution in (
        4,
        5,
        6,
        7,
        8,
    ):

        keys = np.load(
            LOD_ROOT
            / f"r{resolution}"
            / "h3_keys.npy",
            mmap_mode="r",
        )

        counts[
            str(
                resolution
            )
        ] = len(
            keys
        )


    metadata = {

        "schema":
            "crimenet_h3_lod_index_v1",

        "base_resolution":
            BASE_RESOLUTION,

        "base_cell_count":
            n_r9,

        "lod_resolutions":
            list(
                LOD_RESOLUTIONS
            ),

        "parent_cell_counts":
            counts,

        "aggregation":
            "sum_r9_child_intensity",
    }


    with (
        LOD_ROOT
        / "metadata.json"
    ).open(
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
