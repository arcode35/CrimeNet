from __future__ import annotations

import json
import threading
from pathlib import Path

import h3
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from mark_runtime import MarkRuntime


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path.home()
    / "crimenet-serving"
    / "data"
    / "national_feature_store"
)

STATIC_ROOT = (
    ROOT
    / "mmap"
)

LOD_ROOT = (
    STATIC_ROOT
    / "lod"
)

INTENSITY_POINTER = (
    ROOT
    / "intensity_current.json"
)


# ============================================================
# Configuration
# ============================================================

H3_RESOLUTION = 9

MIN_VIEWPORT_RESOLUTION = 4
MAX_VIEWPORT_RESOLUTION = H3_RESOLUTION

# Render-budget target. The viewport endpoint automatically lowers
# H3 resolution until the visible surface fits under this budget.
MAX_VIEWPORT_CELLS = 25_000


# ============================================================
# Helpers
# ============================================================

def load_json(
    path: Path,
) -> dict:

    with path.open("r") as f:
        return json.load(f)


def viewport_polygon(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> h3.LatLngPoly:

    return h3.LatLngPoly(
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


def choose_viewport_cells(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    max_cells: int,
) -> tuple[
    int,
    list[str],
    int,
]:
    """
    Choose the finest H3 resolution whose rectangular viewport
    polyfill stays within the render budget.

    Resolution search intentionally proceeds coarse -> fine. This
    avoids constructing a potentially enormous r9 polyfill merely
    to discover that the browser cannot render it.
    """

    polygon = viewport_polygon(
        west=west,
        south=south,
        east=east,
        north=north,
    )

    best_resolution = None
    best_cells = None
    best_candidate_count = None

    coarsest_count = 0

    for resolution in range(
        MIN_VIEWPORT_RESOLUTION,
        MAX_VIEWPORT_RESOLUTION + 1,
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
            == MIN_VIEWPORT_RESOLUTION
        ):
            coarsest_count = count

        if count > max_cells:
            break

        best_resolution = resolution
        best_cells = cells
        best_candidate_count = count

    if best_resolution is None:
        raise HTTPException(
            status_code=413,
            detail={
                "message":
                    (
                        "Viewport exceeds rendering "
                        "budget even at H3-r"
                        f"{MIN_VIEWPORT_RESOLUTION}"
                    ),
                "resolution":
                    MIN_VIEWPORT_RESOLUTION,
                "cells":
                    coarsest_count,
                "maximum":
                    max_cells,
            },
        )

    return (
        best_resolution,
        best_cells
        or [],
        int(
            best_candidate_count
            or 0
        ),
    )


# ============================================================
# Intensity snapshot store
# ============================================================

class IntensityStore:

    def __init__(self) -> None:

        self.lock = (
            threading.RLock()
        )

        self.h3_keys = np.load(
            STATIC_ROOT
            / "h3_keys.npy",
            mmap_mode="r",
        )

        # ----------------------------------------------------
        # Static H3 LOD indexes.
        #
        # r9 is the canonical model domain. r4-r8 keys and
        # child counts are generated once by build_h3_lod_index.py.
        # ----------------------------------------------------

        self.lod_h3_keys = {
            H3_RESOLUTION:
                self.h3_keys,
        }

        self.lod_child_counts = {}

        for resolution in range(
            MIN_VIEWPORT_RESOLUTION,
            H3_RESOLUTION,
        ):

            level_root = (
                LOD_ROOT
                / f"r{resolution}"
            )

            level_keys_path = (
                level_root
                / "h3_keys.npy"
            )

            child_counts_path = (
                level_root
                / "r9_child_count.npy"
            )

            if not level_keys_path.exists():
                raise RuntimeError(
                    "Missing static LOD H3 index: "
                    f"{level_keys_path}"
                )

            if not child_counts_path.exists():
                raise RuntimeError(
                    "Missing static LOD child counts: "
                    f"{child_counts_path}"
                )

            level_keys = np.load(
                level_keys_path,
                mmap_mode="r",
            )

            child_counts = np.load(
                child_counts_path,
                mmap_mode="r",
            )

            if (
                len(level_keys)
                != len(child_counts)
            ):
                raise RuntimeError(
                    f"H3-r{resolution} LOD keys and "
                    "child counts disagree"
                )

            self.lod_h3_keys[
                resolution
            ] = level_keys

            self.lod_child_counts[
                resolution
            ] = child_counts

        self.pointer_mtime_ns = None

        self.snapshot_id = None
        self.snapshot_path = None
        self.valid_utc_hour = None

        self.log_intensity = None
        self.intensity = None
        self.lod_intensity = {}

        self.reload_if_needed(
            force=True
        )


    # ========================================================
    # Snapshot reload
    # ========================================================

    def reload_if_needed(
        self,
        force: bool = False,
    ) -> bool:

        stat = (
            INTENSITY_POINTER
            .stat()
        )

        mtime_ns = (
            stat.st_mtime_ns
        )

        if (
            not force
            and
            self.pointer_mtime_ns
            == mtime_ns
        ):

            return False


        pointer = load_json(
            INTENSITY_POINTER
        )


        snapshot_path = Path(
            pointer[
                "snapshot_path"
            ]
        )


        if not snapshot_path.exists():

            raise RuntimeError(
                "Intensity snapshot does "
                f"not exist: {snapshot_path}"
            )


        metadata = load_json(
            snapshot_path
            / "metadata.json"
        )


        if (
            pointer[
                "snapshot_id"
            ]
            != metadata[
                "snapshot_id"
            ]
        ):

            raise RuntimeError(
                "Intensity pointer snapshot ID "
                "does not match metadata"
            )


        if (
            pointer[
                "valid_utc_hour"
            ]
            != metadata[
                "valid_utc_hour"
            ]
        ):

            raise RuntimeError(
                "Intensity pointer UTC hour "
                "does not match metadata"
            )


        new_log_intensity = np.load(
            snapshot_path
            / "log_intensity.npy",
            mmap_mode="r",
        )


        new_intensity = np.load(
            snapshot_path
            / "intensity.npy",
            mmap_mode="r",
        )

        new_lod_intensity = {
            H3_RESOLUTION:
                new_intensity,
        }

        for resolution in range(
            MIN_VIEWPORT_RESOLUTION,
            H3_RESOLUTION,
        ):

            lod_path = (
                snapshot_path
                / "lod"
                / f"r{resolution}"
                / "intensity.npy"
            )

            if not lod_path.exists():
                raise RuntimeError(
                    "Active intensity snapshot is "
                    "missing LOD artifact: "
                    f"{lod_path}"
                )

            lod_values = np.load(
                lod_path,
                mmap_mode="r",
            )

            expected_rows = len(
                self.lod_h3_keys[
                    resolution
                ]
            )

            if (
                len(lod_values)
                != expected_rows
            ):
                raise RuntimeError(
                    f"H3-r{resolution} intensity row "
                    "count does not match static LOD "
                    f"index: {len(lod_values):,} != "
                    f"{expected_rows:,}"
                )

            new_lod_intensity[
                resolution
            ] = lod_values


        if (
            len(
                new_log_intensity
            )
            != len(
                self.h3_keys
            )
        ):

            raise RuntimeError(
                "log_intensity.npy row count "
                "does not match H3 index"
            )


        if (
            len(
                new_intensity
            )
            != len(
                self.h3_keys
            )
        ):

            raise RuntimeError(
                "intensity.npy row count "
                "does not match H3 index"
            )


        # Only switch the active snapshot after
        # every new artifact has validated.
        with self.lock:

            self.snapshot_id = (
                pointer[
                    "snapshot_id"
                ]
            )

            self.snapshot_path = (
                snapshot_path
            )

            self.valid_utc_hour = (
                pointer[
                    "valid_utc_hour"
                ]
            )

            self.log_intensity = (
                new_log_intensity
            )

            self.intensity = (
                new_intensity
            )

            self.lod_intensity = (
                new_lod_intensity
            )

            self.pointer_mtime_ns = (
                mtime_ns
            )


        print(
            "Loaded intensity snapshot | "
            f"id={self.snapshot_id} "
            f"utc={self.valid_utc_hour}",
            flush=True,
        )

        return True


    # ========================================================
    # State
    # ========================================================

    def current_state(
        self,
    ) -> dict:

        self.reload_if_needed()

        with self.lock:

            return {

                "snapshot_id":
                    self.snapshot_id,

                "valid_utc_hour":
                    self.valid_utc_hour,

                "cells":
                    len(
                        self.h3_keys
                    ),
            }


    # ========================================================
    # H3 -> row
    # ========================================================

    def row_for_h3(
        self,
        cell: str,
    ) -> int | None:

        key = np.uint64(
            h3.str_to_int(
                cell
            )
        )


        row = int(
            np.searchsorted(
                self.h3_keys,
                key,
            )
        )


        if (
            row
            >= len(
                self.h3_keys
            )
        ):

            return None


        if (
            self.h3_keys[
                row
            ]
            != key
        ):

            return None


        return row


    # ========================================================
    # Point lookup
    # ========================================================

    def lookup_h3(
        self,
        cell: str,
    ) -> dict | None:

        self.reload_if_needed()


        row = self.row_for_h3(
            cell
        )


        if row is None:

            return None


        with self.lock:

            log_lambda = float(
                self.log_intensity[
                    row
                ]
            )

            lambda_per_second = float(
                self.intensity[
                    row
                ]
            )

            snapshot_id = (
                self.snapshot_id
            )

            valid_utc_hour = (
                self.valid_utc_hour
            )


        return {

            "h3":
                cell,

            "row":
                row,

            "snapshot_id":
                snapshot_id,

            "valid_utc_hour":
                valid_utc_hour,

            "log_intensity":
                log_lambda,

            "events_per_second":
                lambda_per_second,

            "events_per_hour":
                lambda_per_second
                * 3600.0,
        }


    # ========================================================
    # Atomic point lookup + snapshot provenance
    #
    # Used by the combined mark endpoint so intensity and
    # mark inference are guaranteed to use one snapshot ID.
    # ========================================================

    def lookup_for_prediction(
        self,
        cell: str,
    ) -> tuple[
        dict,
        str,
        Path,
    ] | None:

        self.reload_if_needed()


        row = self.row_for_h3(
            cell
        )


        if row is None:

            return None


        with self.lock:

            snapshot_id = (
                str(
                    self.snapshot_id
                )
            )

            snapshot_path = Path(
                self.snapshot_path
            )

            valid_utc_hour = (
                self.valid_utc_hour
            )

            log_lambda = float(
                self.log_intensity[
                    row
                ]
            )

            lambda_per_second = float(
                self.intensity[
                    row
                ]
            )


        intensity_result = {

            "h3":
                cell,

            "row":
                row,

            "snapshot_id":
                snapshot_id,

            "valid_utc_hour":
                valid_utc_hour,

            "log_intensity":
                log_lambda,

            "events_per_second":
                lambda_per_second,

            "events_per_hour":
                lambda_per_second
                * 3600.0,
        }


        return (
            intensity_result,
            snapshot_id,
            snapshot_path,
        )


    # ========================================================
    # Batch lookup
    # ========================================================

    def lookup_many(
        self,
        cells: list[str],
    ) -> list[dict]:

        self.reload_if_needed()


        if not cells:

            return []


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
            self.h3_keys,
            keys,
        )


        valid = (
            rows
            < len(
                self.h3_keys
            )
        )


        matched = np.zeros(
            len(
                cells
            ),
            dtype=bool,
        )


        valid_positions = (
            np.flatnonzero(
                valid
            )
        )


        if len(
            valid_positions
        ):

            matched[
                valid_positions
            ] = (

                self.h3_keys[
                    rows[
                        valid_positions
                    ]
                ]

                ==

                keys[
                    valid_positions
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


        with self.lock:

            lambda_values = np.asarray(
                self.intensity[
                    matched_rows
                ],
                dtype=np.float32,
            )

            snapshot_id = (
                self.snapshot_id
            )

            valid_utc_hour = (
                self.valid_utc_hour
            )


        return [

            {

                "h3":
                    cells[
                        int(
                            position
                        )
                    ],

                "events_per_hour":
                    float(
                        lambda_values[
                            i
                        ]
                    )
                    * 3600.0,
            }

            for (
                i,
                position,
            )
            in enumerate(
                positions
            )
        ]


    # ========================================================
    # Resolution-aware viewport lookup
    # ========================================================

    def lookup_viewport(
        self,
        cells: list[str],
        resolution: int,
    ) -> tuple[
        list[dict],
        dict,
    ]:

        self.reload_if_needed()

        if (
            resolution
            < MIN_VIEWPORT_RESOLUTION
            or
            resolution
            > H3_RESOLUTION
        ):
            raise RuntimeError(
                "Unsupported viewport H3 "
                f"resolution: {resolution}"
            )

        index = (
            self.lod_h3_keys[
                resolution
            ]
        )

        if not cells:

            with self.lock:

                state = {
                    "snapshot_id":
                        self.snapshot_id,
                    "valid_utc_hour":
                        self.valid_utc_hour,
                }

            return (
                [],
                state,
            )

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

        valid_positions = (
            np.flatnonzero(
                valid
            )
        )

        if len(
            valid_positions
        ):

            matched[
                valid_positions
            ] = (
                index[
                    rows[
                        valid_positions
                    ]
                ]
                ==
                keys[
                    valid_positions
                ]
            )

        positions = np.flatnonzero(
            matched
        )

        if not len(
            positions
        ):

            with self.lock:

                state = {
                    "snapshot_id":
                        self.snapshot_id,
                    "valid_utc_hour":
                        self.valid_utc_hour,
                }

            return (
                [],
                state,
            )

        matched_rows = (
            rows[
                positions
            ]
        )

        # Snapshot metadata and values are captured under the
        # same lock. reload_if_needed() switches r9 + r4-r8
        # mmaps atomically, so a rollover cannot mix hours.
        with self.lock:

            lambda_values = np.asarray(
                self.lod_intensity[
                    resolution
                ][
                    matched_rows
                ],
                dtype=np.float32,
            )

            if (
                resolution
                == H3_RESOLUTION
            ):

                child_counts = np.ones(
                    len(
                        matched_rows
                    ),
                    dtype=np.uint32,
                )

            else:

                child_counts = np.asarray(
                    self.lod_child_counts[
                        resolution
                    ][
                        matched_rows
                    ],
                    dtype=np.uint32,
                )

            state = {
                "snapshot_id":
                    self.snapshot_id,
                "valid_utc_hour":
                    self.valid_utc_hour,
            }

        results = []

        for (
            i,
            position,
        ) in enumerate(
            positions
        ):

            events_per_hour = (
                float(
                    lambda_values[
                        i
                    ]
                )
                * 3600.0
            )

            modeled_r9_cells = int(
                child_counts[
                    i
                ]
            )

            mean_r9_events_per_hour = (
                events_per_hour
                / modeled_r9_cells
                if modeled_r9_cells
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

                    # Correct expected event rate over the
                    # entire returned H3 cell.
                    "events_per_hour":
                        events_per_hour,

                    # Stable spatial-density visualization
                    # value across H3 resolutions.
                    "mean_r9_events_per_hour":
                        mean_r9_events_per_hour,

                    # Number of canonical modeled r9 cells
                    # represented by this cell.
                    "modeled_r9_cells":
                        modeled_r9_cells,
                }
            )

        return (
            results,
            state,
        )


# ============================================================
# Runtime initialization
# ============================================================

print(
    "Loading CrimeNet intensity store...",
    flush=True,
)

store = (
    IntensityStore()
)


print(
    "Loading CrimeNet mark runtime...",
    flush=True,
)

mark_runtime = (
    MarkRuntime()
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="CrimeNet API",
    version="0.3.0",
)


# ============================================================
# Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],

    allow_credentials=True,

    allow_methods=[
        "GET",
    ],

    allow_headers=[
        "*",
    ],
)


app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)


# ============================================================
# Health
# ============================================================

@app.get(
    "/health"
)
def health():

    state = (
        store.current_state()
    )


    return {

        "status":
            "ok",

        **state,

        "mark_model": {

            "status":
                "ready",

            "run_id":
                mark_runtime
                .MARK_MODEL_RUN_ID
                if hasattr(
                    mark_runtime,
                    "MARK_MODEL_RUN_ID",
                )
                else
                "7efda77cdaec4a66a30321ea50b12ec8",

            "classes":
                mark_runtime
                .num_classes,

            "labels_available":
                mark_runtime
                .labels_available,
        },
    }


# ============================================================
# Intensity — point
# ============================================================

@app.get(
    "/api/v1/intensity/point"
)
def intensity_point(

    lat: float = Query(
        ge=-90,
        le=90,
    ),

    lon: float = Query(
        ge=-180,
        le=180,
    ),
):


    cell = (
        h3.latlng_to_cell(
            lat,
            lon,
            H3_RESOLUTION,
        )
    )


    result = (
        store.lookup_h3(
            cell
        )
    )


    if result is None:

        raise HTTPException(
            status_code=404,

            detail=(
                "Coordinate is outside "
                "the national serving domain"
            ),
        )


    center_lat, center_lon = (
        h3.cell_to_latlng(
            cell
        )
    )


    return {

        **result,

        "center": {

            "lat":
                center_lat,

            "lon":
                center_lon,
        },
    }


# ============================================================
# Intensity — direct H3
# ============================================================

@app.get(
    "/api/v1/intensity/cell/{cell}"
)
def intensity_cell(
    cell: str,
):


    if not h3.is_valid_cell(
        cell
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid H3 cell",
        )


    if (
        h3.get_resolution(
            cell
        )
        != H3_RESOLUTION
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                f"Expected H3 resolution "
                f"{H3_RESOLUTION}"
            ),
        )


    result = (
        store.lookup_h3(
            cell
        )
    )


    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Cell not in serving domain"
            ),
        )


    return result


# ============================================================
# Intensity — viewport
# ============================================================

@app.get(
    "/api/v1/intensity/viewport"
)
def intensity_viewport(

    west: float = Query(
        ge=-180,
        le=180,
    ),

    south: float = Query(
        ge=-90,
        le=90,
    ),

    east: float = Query(
        ge=-180,
        le=180,
    ),

    north: float = Query(
        ge=-90,
        le=90,
    ),
):


    if (
        south
        >= north
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "south must be less "
                "than north"
            ),
        )


    if (
        west
        >= east
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "Dateline-crossing "
                "viewports are not supported"
            ),
        )


    (
        resolution,
        cells,
        candidate_count,
    ) = choose_viewport_cells(

        west=west,
        south=south,
        east=east,
        north=north,

        max_cells=(
            MAX_VIEWPORT_CELLS
        ),
    )


    (
        values,
        state,
    ) = store.lookup_viewport(
        cells,
        resolution,
    )


    return {

        "snapshot_id":
            state[
                "snapshot_id"
            ],

        "valid_utc_hour":
            state[
                "valid_utc_hour"
            ],

        "resolution":
            resolution,

        "aggregation":
            (
                "native_r9"
                if (
                    resolution
                    == H3_RESOLUTION
                )
                else
                "sum_r9_child_intensity"
            ),

        "visualization_metric":
            "mean_r9_events_per_hour",

        # Number of cells in the rectangular H3 polyfill
        # before filtering to the serving domain.
        "candidate_count":
            candidate_count,

        # Number of cells actually returned from the
        # national serving domain.
        "count":
            len(
                values
            ),

        "cells":
            values,
    }


# ============================================================
# Combined intensity + mark prediction
# ============================================================

def combined_prediction(
    cell: str,
    top_k: int,
) -> dict:


    lookup = (
        store.lookup_for_prediction(
            cell
        )
    )


    if lookup is None:

        raise HTTPException(
            status_code=404,

            detail=(
                "Cell not in serving domain"
            ),
        )


    (
        intensity,
        snapshot_id,
        snapshot_path,
    ) = lookup


    mark = (
        mark_runtime.predict(

            cell=cell,

            snapshot_id=(
                snapshot_id
            ),

            intensity_snapshot_path=(
                snapshot_path
            ),

            top_k=(
                top_k
            ),
        )
    )


    overall_events_per_hour = float(
        intensity[
            "events_per_hour"
        ]
    )


    distribution = []


    for item in (
        mark[
            "distribution"
        ]
    ):

        probability = float(
            item[
                "probability"
            ]
        )


        distribution.append(
            {

                **item,

                # ------------------------------------------------
                # Mark-specific point-process intensity:
                #
                # lambda_k(t,x)
                # =
                # lambda(t,x)
                # *
                # P(M=k | event,t,x)
                # ------------------------------------------------

                "events_per_hour":
                    (
                        overall_events_per_hour
                        * probability
                    ),
            }
        )


    return {

        "h3":
            cell,

        "snapshot_id":
            snapshot_id,

        "valid_utc_hour":
            intensity[
                "valid_utc_hour"
            ],

        "intensity": {

            "log_intensity":
                intensity[
                    "log_intensity"
                ],

            "events_per_second":
                intensity[
                    "events_per_second"
                ],

            "events_per_hour":
                overall_events_per_hour,
        },

        "mark": {

            "model_run_id":
                mark[
                    "model_run_id"
                ],

            "num_classes":
                mark[
                    "num_classes"
                ],

            "labels_available":
                mark[
                    "labels_available"
                ],

            "distribution":
                distribution,
        },
    }


# ============================================================
# Combined prediction — coordinate
# ============================================================

@app.get(
    "/api/v1/predict/point"
)
def predict_point(

    lat: float = Query(
        ge=-90,
        le=90,
    ),

    lon: float = Query(
        ge=-180,
        le=180,
    ),

    top_k: int = Query(
        default=5,
        ge=1,
        le=87,
    ),
):


    cell = (
        h3.latlng_to_cell(
            lat,
            lon,
            H3_RESOLUTION,
        )
    )


    result = (
        combined_prediction(
            cell,
            top_k,
        )
    )


    center_lat, center_lon = (
        h3.cell_to_latlng(
            cell
        )
    )


    result[
        "center"
    ] = {

        "lat":
            center_lat,

        "lon":
            center_lon,
    }


    return result


# ============================================================
# Combined prediction — H3
# ============================================================

@app.get(
    "/api/v1/predict/cell/{cell}"
)
def predict_cell(

    cell: str,

    top_k: int = Query(
        default=5,
        ge=1,
        le=87,
    ),
):


    if not h3.is_valid_cell(
        cell
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid H3 cell",
        )


    if (
        h3.get_resolution(
            cell
        )
        != H3_RESOLUTION
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                f"Expected H3 resolution "
                f"{H3_RESOLUTION}"
            ),
        )


    return combined_prediction(
        cell,
        top_k,
    )
