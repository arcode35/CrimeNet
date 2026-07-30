from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from crimenet.weather.open_meteo_client import (
    WeatherRequest,
)


def weather_request_from_manifest_row(
    row: Mapping[str, Any],
) -> WeatherRequest:
    start_date = row["start_date"]
    end_date = row["end_date"]

    if not isinstance(start_date, date):
        raise TypeError(
            "start_date must be a datetime.date"
        )

    if not isinstance(end_date, date):
        raise TypeError(
            "end_date must be a datetime.date"
        )

    return WeatherRequest(
        request_id=str(row["request_id"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        weather_query_cell_id=int(
            row["weather_query_cell_id"]
        ),
        h3_resolution=int(
            row["h3_resolution"]
        ),
        latitude=float(
            row["query_latitude"]
        ),
        longitude=float(
            row["query_longitude"]
        ),
        start_date=start_date,
        end_date=end_date,
        timezone=str(row["timezone"]),
        cell_selection=str(
            row["cell_selection"]
        ),
        hourly_variables=tuple(
            str(variable)
            for variable in row[
                "hourly_variables"
            ]
        ),
    )