from __future__ import annotations

import threading
from pathlib import Path

import h3
import numpy as np


ROOT = (
    Path.home()
    / "crimenet-serving"
    / "data"
    / "national_feature_store"
)

STATIC_ROOT = ROOT / "mmap"
LOD_ROOT = STATIC_ROOT / "lod"

MIN_RESOLUTION = 4
MAX_RESOLUTION = 9

RESOLUTIONS = tuple(
    range(
        MIN_RESOLUTION,
        MAX_RESOLUTION + 1,
    )
)


class ViewportTooLarge(
    RuntimeError
):

    def __init__(
        self,
        cell_count: int,
    ) -> None:

        super().__init__(
            "Viewport is too large "
            "even at minimum H3 resolution"
        )

        self.cell_count = (
            cell_count
        )


class ViewportLOD:

    def __init__(
        self,
    ) -> None:

        self.lock = (
            threading.RLock()
        )

        self.snapshot_id = None
        self.snapshot_path = None

        self.keys = {}
        self.child_counts = {}


        # ----------------------------------------------------
        # r9
        # ----------------------------------------------------

        self.keys[
            9
        ] = np.load(
            STATIC_ROOT
            / "h3_keys.npy",
            mmap_mode="r",
        )


        # ----------------------------------------------------
        # r4-r8
        # ----------------------------------------------------

        for resolution in range(
            4,
            9,
        ):

            static_dir = (
                LOD_ROOT
                / f"r{resolution}"
            )

            self.keys[
                resolution
            ] = np.load(
                static_dir
                / "h3_keys.npy",
                mmap_mode="r",
            )

            self.child_counts[
                resolution
            ] = np.load(
                static_dir
                / "r9_child_count.npy",
                mmap_mode="r",
            )


        self.intensity = {}


    # ========================================================
    # Synchronize to active intensity snapshot
    # ========================================================

    def sync(
        self,
        snapshot_id: str,
        snapshot_path: Path,
    ) -> None:

        with self.lock:

            if (
                self.snapshot_id
                == snapshot_id
            ):
                return


        new_intensity = {}


        # r9
        r9 = np.load(
            snapshot_path
            / "intensity.npy",
            mmap_mode="r",
        )

        if len(
            r9
        ) != len(
            self.keys[
                9
            ]
        ):

            raise RuntimeError(
                "r9 intensity length mismatch"
            )

        new_intensity[
            9
        ] = r9


        # r4-r8
        for resolution in range(
            4,
            9,
        ):

            path = (
                snapshot_path
                / "lod"
                / f"r{resolution}"
                / "intensity.npy"
            )

            values = np.load(
                path,
                mmap_mode="r",
            )

            if len(
                values
            ) != len(
                self.keys[
                    resolution
                ]
            ):

                raise RuntimeError(
                    f"r{resolution}: intensity length mismatch"
                )

            new_intensity[
                resolution
            ] = values


        # Switch only after everything validates.
        with self.lock:

            self.intensity = (
                new_intensity
            )

            self.snapshot_id = (
                snapshot_id
            )

            self.snapshot_path = (
                snapshot_path
            )


        print(
            "Viewport LOD synchronized | "
            f"snapshot={snapshot_id}",
            flush=True,
        )


    # ========================================================
    # Resolution selection
    #
    # IMPORTANT:
    # Start coarse and move finer.
    #
    # Never try to fill the entire USA at r9 just to discover
    # that the request is too large.
    # ========================================================

    def choose_cells(
        self,
        *,
        west: float,
        south: float,
        east: float,
        north: float,
        max_cells: int,
    ) -> tuple[
        int,
        list[str],
    ]:

        polygon = h3.LatLngPoly(
            [
                (
                    south,
                    west,
                ),
                (
                    south,
                    east,
                ),
                (
                    north,
                    east,
                ),
                (
                    north,
                    west,
                ),
            ]
        )


        best_resolution = None
        best_cells = None

        first_count = 0


        for resolution in (
            RESOLUTIONS
        ):

            cells = list(
                h3.h3shape_to_cells(
                    polygon,
                    resolution,
                )
            )

            count = len(
                cells
            )


            if (
                resolution
                == MIN_RESOLUTION
            ):

                first_count = count


            if count > max_cells:

                break


            best_resolution = (
                resolution
            )

            best_cells = (
                cells
            )


        if best_resolution is None:

            raise ViewportTooLarge(
                first_count
            )


        return (
            best_resolution,
            best_cells
            or [],
        )


    # ========================================================
    # Resolution-aware lookup
    # ========================================================

    def lookup_many(
        self,
        cells: list[str],
        resolution: int,
    ) -> list[dict]:

        if not cells:
            return []


        with self.lock:

            if (
                resolution
                not in self.intensity
            ):

                raise RuntimeError(
                    "Viewport LOD has no "
                    "active intensity snapshot"
                )


            index = (
                self.keys[
                    resolution
                ]
            )

            intensity = (
                self.intensity[
                    resolution
                ]
            )

            child_count_index = (
                None
                if resolution == 9
                else self.child_counts[
                    resolution
                ]
            )

        # Captured mmap references remain valid if a new snapshot becomes
        # active; all CPU-heavy lookup work can safely run without the lock.
        keys = np.asarray(
                [
                    h3.str_to_int(
                        cell
                    )
                    for cell in cells
                ],
                dtype=np.uint64,
            )


        rows = np.searchsorted(
                index,
                keys,
            )


        valid = (
                rows
                < len(
                    index
                )
            )


        matched = np.zeros(
                len(
                    cells
                ),
                dtype=bool,
            )


        positions = np.flatnonzero(
                valid
            )


        if len(
                positions
            ):

            matched[
                    positions
                ] = (
                    index[
                        rows[
                            positions
                        ]
                    ]
                    ==
                    keys[
                        positions
                    ]
                )


        positions = np.flatnonzero(
                matched
            )


        if not len(
                positions
            ):

            return []


        matched_rows = (
                rows[
                    positions
                ]
            )


        summed_per_second = np.asarray(
                intensity[
                    matched_rows
                ],
                dtype=np.float32,
            )


        if resolution == 9:

            child_counts = np.ones(
                    len(
                        matched_rows
                    ),
                    dtype=np.uint32,
                )

        else:

            child_counts = np.asarray(
                    child_count_index[
                        matched_rows
                    ],
                    dtype=np.uint32,
                )


        results = []


        for i, position in enumerate(
            positions
        ):

            total_hourly = (
                float(
                    summed_per_second[
                        i
                    ]
                )
                * 3600.0
            )


            children = int(
                child_counts[
                    i
                ]
            )


            # Stable spatial-risk value for map color.
            #
            # This is the mean intensity of modeled r9
            # children rather than the parent sum.
            mean_r9_hourly = (
                total_hourly
                / children
                if children > 0
                else 0.0
            )


            results.append(
                {

                    "h3":
                        cells[
                            int(
                                position
                            )
                        ],

                    # Correct expected-event intensity over
                    # the entire displayed H3 parent.
                    "events_per_hour":
                        total_hourly,

                    # Use this for map color / density.
                    "mean_r9_events_per_hour":
                        mean_r9_hourly,

                    "modeled_r9_cells":
                        children,
                }
            )


        return results
