from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


class CapacityExceeded(RuntimeError):
    """Raised before expensive work when an admission queue is full."""


@dataclass(frozen=True)
class AdmissionState:
    in_flight: int
    waiting: int


class BoundedAdmission:
    """Small concurrency limit with a separately bounded, short-wait queue."""

    def __init__(self, limit: int, max_waiters: int, wait_seconds: float) -> None:
        if limit < 1 or max_waiters < 0 or wait_seconds < 0:
            raise ValueError("Invalid bounded-admission configuration")
        self._semaphore = threading.BoundedSemaphore(limit)
        self._max_waiters = max_waiters
        self._wait_seconds = wait_seconds
        self._state_lock = threading.Lock()
        self._in_flight = 0
        self._waiting = 0

    def _mark_acquired(self) -> None:
        with self._state_lock:
            self._in_flight += 1

    def acquire(self) -> None:
        if self._semaphore.acquire(blocking=False):
            self._mark_acquired()
            return

        with self._state_lock:
            if self._waiting >= self._max_waiters:
                raise CapacityExceeded("expensive request queue is full")
            self._waiting += 1

        try:
            acquired = self._semaphore.acquire(timeout=self._wait_seconds)
        finally:
            with self._state_lock:
                self._waiting -= 1

        if not acquired:
            raise CapacityExceeded("expensive request capacity is busy")
        self._mark_acquired()

    def release(self) -> None:
        with self._state_lock:
            self._in_flight -= 1
        self._semaphore.release()

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def state(self) -> AdmissionState:
        with self._state_lock:
            return AdmissionState(in_flight=self._in_flight, waiting=self._waiting)
