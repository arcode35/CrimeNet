from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


SERVING_ROOT = Path.home() / "crimenet-serving"
ROOT = SERVING_ROOT / "data" / "national_feature_store"
ENV_ROOT = ROOT / "environmental"
ENV_SNAPSHOT_ROOT = ENV_ROOT / "environmental_snapshots"
INTENSITY_SNAPSHOT_ROOT = ROOT / "intensity_snapshots"
INTENSITY_POINTER = ROOT / "intensity_current.json"
TIMELINE_PATH = ROOT / "intensity_timeline.json"

ENV_BUILDER = SERVING_ROOT / "build_environmental_snapshot.py"
INTENSITY_BUILDER = SERVING_ROOT / "build_national_intensity.py"


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def atomic_json_write(path: Path, payload: dict) -> None:
    tmp = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_hour(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def hour_text(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def snapshot_id(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")


def run_checked(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=SERVING_ROOT, check=True)


def validate_snapshot(path: Path, expected_hour: datetime) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing snapshot metadata: {metadata_path}")

    metadata = load_json(metadata_path)
    actual = parse_hour(metadata["valid_utc_hour"])
    if actual != expected_hour:
        raise RuntimeError(
            "Snapshot hour mismatch: "
            f"expected={hour_text(expected_hour)}, actual={hour_text(actual)}, path={path}"
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Number of future hourly snapshots to materialize (default: 24).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse already-published future snapshots. By default future hours are "
            "rebuilt so they use the newest weather forecast run."
        ),
    )
    args = parser.parse_args()

    if args.hours < 1 or args.hours > 168:
        raise ValueError("--hours must be between 1 and 168")

    if not INTENSITY_POINTER.exists():
        raise FileNotFoundError(INTENSITY_POINTER)

    current_pointer = load_json(INTENSITY_POINTER)
    current_hour = parse_hour(current_pointer["valid_utc_hour"])
    current_snapshot_path = Path(current_pointer["snapshot_path"])
    validate_snapshot(current_snapshot_path, current_hour)

    generated_at = datetime.now(timezone.utc)

    entries: list[dict] = [
        {
            "snapshot_id": current_pointer["snapshot_id"],
            "valid_utc_hour": hour_text(current_hour),
            "horizon_hours": 0,
            "kind": "live",
        }
    ]

    for horizon in range(1, args.hours + 1):
        target = current_hour + timedelta(hours=horizon)
        sid = snapshot_id(target)
        env_dir = ENV_SNAPSHOT_ROOT / sid
        intensity_dir = INTENSITY_SNAPSHOT_ROOT / sid

        print(
            f"\n=== forecast +{horizon:03d}h | {hour_text(target)} | {sid} ===",
            flush=True,
        )

        reuse_env = False
        reuse_intensity = False

        if args.reuse_existing and env_dir.exists():
            validate_snapshot(env_dir, target)
            reuse_env = True

        if args.reuse_existing and intensity_dir.exists():
            validate_snapshot(intensity_dir, target)
            reuse_intensity = True

        if not reuse_env:
            run_checked(
                [
                    sys.executable,
                    str(ENV_BUILDER),
                    "--expected-valid-utc-hour",
                    hour_text(target),
                    "--no-publish-current",
                ]
            )
            validate_snapshot(env_dir, target)

        if not reuse_intensity:
            run_checked(
                [
                    sys.executable,
                    str(INTENSITY_BUILDER),
                    "--environmental-snapshot-path",
                    str(env_dir),
                    "--expected-valid-utc-hour",
                    hour_text(target),
                    "--no-publish-current",
                ]
            )
            validate_snapshot(intensity_dir, target)

        entries.append(
            {
                "snapshot_id": sid,
                "valid_utc_hour": hour_text(target),
                "horizon_hours": horizon,
                "kind": "forecast",
            }
        )

        # Publish incrementally so the UI can use completed hours even if a
        # later hour fails. The live pointer is intentionally untouched.
        atomic_json_write(
            TIMELINE_PATH,
            {
                "schema": "crimenet_intensity_timeline_v1",
                "generated_at_utc": generated_at.isoformat(),
                "as_of_utc_hour": hour_text(current_hour),
                "hours_requested": args.hours,
                "hours_available": len(entries) - 1,
                "snapshots": entries,
            },
        )

    print(f"\nTimeline published: {TIMELINE_PATH}")
    print(f"Available snapshots: {len(entries)} (live + {len(entries)-1} forecast)")
    print("intensity_current.json was not advanced by forecast generation.")


if __name__ == "__main__":
    main()
