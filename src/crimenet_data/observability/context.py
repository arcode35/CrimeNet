# crimenet_data/logging/context.py

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

from structlog.contextvars import bound_contextvars


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    values = {
        key: value
        for key, value in values.items()
        if value is not None
    }

    with bound_contextvars(**values):
        yield