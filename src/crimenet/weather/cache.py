from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from crimenet.weather.open_meteo_client import (
    WeatherRequest,
)


def get_weather_cache_path(
    cache_directory: str | Path,
    request: WeatherRequest,
) -> Path:
    return (
        Path(cache_directory)
        / request.model
        / f"year={request.start_date.year}"
        / f"{request.request_id}.json"
    )


def write_weather_cache(
    *,
    cache_path: Path,
    response: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                response,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            cache_path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True,
        )