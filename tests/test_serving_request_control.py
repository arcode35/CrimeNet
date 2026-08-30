from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from request_control import BoundedAdmission, CapacityExceeded  # noqa: E402


def test_expensive_request_concurrency_and_waiting_are_bounded() -> None:
    admission = BoundedAdmission(limit=2, max_waiters=1, wait_seconds=0.05)
    release = threading.Event()
    both_started = threading.Barrier(3)

    def occupy() -> None:
        with admission.slot():
            both_started.wait(timeout=1)
            release.wait(timeout=1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(occupy) for _ in range(2)]
        both_started.wait(timeout=1)
        assert admission.state().in_flight == 2

        with pytest.raises(CapacityExceeded):
            admission.acquire()

        release.set()
        for future in futures:
            future.result(timeout=1)

    assert admission.state().in_flight == 0
    assert admission.state().waiting == 0

    # Capacity is immediately reusable after the burst.
    with admission.slot():
        assert admission.state().in_flight == 1


def test_burst_is_rejected_without_an_unbounded_wait_queue() -> None:
    admission = BoundedAdmission(limit=2, max_waiters=2, wait_seconds=0.02)
    release = threading.Event()
    occupied = threading.Barrier(3)

    def occupy() -> None:
        with admission.slot():
            occupied.wait(timeout=1)
            release.wait(timeout=1)

    def burst_request() -> str:
        try:
            with admission.slot():
                return "admitted"
        except CapacityExceeded:
            return "busy"

    with ThreadPoolExecutor(max_workers=14) as pool:
        holders = [pool.submit(occupy) for _ in range(2)]
        occupied.wait(timeout=1)
        burst = [pool.submit(burst_request) for _ in range(12)]
        assert [future.result(timeout=1) for future in burst] == ["busy"] * 12
        assert admission.state().in_flight == 2
        assert admission.state().waiting == 0
        release.set()
        for future in holders:
            future.result(timeout=1)

    assert admission.state().in_flight == 0
