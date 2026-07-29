from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from crimenet.weather.open_meteo_client import (
    WeatherRequest,
    build_weather_request_id,
    normalize_hourly_variables,
    validate_weather_request,
)


def weather_request_from_manifest_row(
    row: Mapping[str, Any],
) -> WeatherRequest:
    start_date = row["start_date"]
    end_date = row["end_date"]

    if (
        not isinstance(start_date, date)
        or isinstance(start_date, datetime)
    ):
        raise TypeError(
            "start_date must be a datetime.date"
        )

    if (
        not isinstance(end_date, date)
        or isinstance(end_date, datetime)
    ):
        raise TypeError(
            "end_date must be a datetime.date"
        )

    raw_variables = row["hourly_variables"]
    if isinstance(raw_variables, str) or not isinstance(
        raw_variables,
        (list, tuple),
    ):
        raise TypeError(
            "hourly_variables must be an array of strings"
        )
    if any(
        not isinstance(variable, str)
        for variable in raw_variables
    ):
        raise TypeError(
            "hourly_variables must contain only strings"
        )

    provider = str(row["provider"])
    model = str(row["model"])
    weather_query_cell_id = int(
        row["weather_query_cell_id"]
    )
    h3_resolution = int(row["h3_resolution"])
    timezone = str(row["timezone"])
    cell_selection = str(row["cell_selection"])
    hourly_variables = normalize_hourly_variables(
        raw_variables
    )
    request_id = str(row["request_id"])

    expected_request_id = build_weather_request_id(
        provider=provider,
        model=model,
        weather_query_cell_id=weather_query_cell_id,
        start_date=start_date,
        end_date=end_date,
        hourly_variables=hourly_variables,
        timezone=timezone,
        cell_selection=cell_selection,
        h3_resolution=h3_resolution,
    )
    if request_id != expected_request_id:
        raise ValueError(
            "Manifest request_id does not match its deterministic "
            f"identity: expected={expected_request_id}, "
            f"actual={request_id}"
        )

    request = WeatherRequest(
        request_id=request_id,
        provider=provider,
        model=model,
        weather_query_cell_id=weather_query_cell_id,
        h3_resolution=h3_resolution,
        latitude=float(row["query_latitude"]),
        longitude=float(row["query_longitude"]),
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
        cell_selection=cell_selection,
        hourly_variables=hourly_variables,
    )
    validate_weather_request(request)
    return request
