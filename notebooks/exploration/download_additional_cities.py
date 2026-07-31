#!/usr/bin/env python3
"""Download San Francisco, Seattle, Baltimore, and Washington, DC crime data.

The downloader preserves every source column as a string in a raw Parquet
layer and adds a small set of typed lineage columns.

Default coverage:
    San Francisco: 2019-2025
    Seattle:       2019-2025
    Baltimore:     2022-2025 (source begins in 2022)
    Washington DC: 2019-2025

Output layout:
    data/raw/
        san_francisco_crime/
            source_dataset=wg3w-h783/
                occurrence_year=2019/
                    part-00000.parquet
                    ...
                    _manifest.json
        seattle_crime/
            source_dataset=tazs-3rd5/
                occurrence_year=2019/
                    ...
        baltimore_crime/
            source_dataset=baltimore_nibrs_group_a/
                occurrence_year=2022/
                    ...
        washington_dc_crime/
            source_dataset=dc_crime_incidents/
                occurrence_year=2019/
                    ...

The script is resumable at the city-year partition level. A completed
partition is skipped after its manifest and Parquet row count are validated.
Interrupted hidden staging directories are discarded and rebuilt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw"

SF_DOMAIN = "https://data.sfgov.org"
SF_DATASET_ID = "wg3w-h783"

SEATTLE_DOMAIN = "https://data.seattle.gov"
SEATTLE_DATASET_ID = "tazs-3rd5"

BALTIMORE_LAYER_URL = (
    "https://services1.arcgis.com/UWYHeuuJISiGmgXx/"
    "arcgis/rest/services/NIBRS_GroupA_Crime_Data/"
    "FeatureServer/0"
)

ARCGIS_SHARING_URL = "https://www.arcgis.com/sharing/rest"

CITY_CHOICES = (
    "san_francisco",
    "seattle",
    "baltimore",
    "washington_dc",
)

DEFAULT_START_YEAR = 2019
DEFAULT_END_YEAR = 2025

SOURCE_METADATA_COLUMNS: tuple[tuple[str, pl.DataType], ...] = (
    ("source_city", pl.String),
    ("source_dataset_id", pl.String),
    ("source_dataset_kind", pl.String),
    ("source_dataset", pl.String),
    ("source_record_id", pl.String),
    ("occurred_at_raw", pl.String),
    ("occurred_at_end_raw", pl.String),
    ("latitude_raw", pl.String),
    ("longitude_raw", pl.String),
    ("occurrence_year", pl.Int16),
    ("downloaded_at_utc", pl.String),
    ("source_url", pl.String),
)


@dataclass(frozen=True)
class SocrataSource:
    city: str
    output_directory: str
    domain: str
    dataset_id: str
    date_field: str
    record_id_field: str
    latitude_field: str
    longitude_field: str
    order_fields: tuple[str, ...]


SOCRATA_SOURCES = {
    "san_francisco": SocrataSource(
        city="san_francisco",
        output_directory="san_francisco_crime",
        domain=SF_DOMAIN,
        dataset_id=SF_DATASET_ID,
        date_field="incident_datetime",
        record_id_field="row_id",
        latitude_field="latitude",
        longitude_field="longitude",
        order_fields=(
            "incident_datetime",
            "row_id",
        ),
    ),
    "seattle": SocrataSource(
        city="seattle",
        output_directory="seattle_crime",
        domain=SEATTLE_DOMAIN,
        dataset_id=SEATTLE_DATASET_ID,
        date_field="offense_date",
        record_id_field="offense_id",
        latitude_field="latitude",
        longitude_field="longitude",
        order_fields=(
            "offense_date",
            "offense_id",
        ),
    ),
}


class DownloadError(RuntimeError):
    """Raised when a source returns invalid or incomplete data."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_session() -> Session:
    retry = Retry(
        total=7,
        connect=7,
        read=7,
        status=7,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        backoff_factor=1.0,
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=2,
        pool_maxsize=2,
    )

    session = Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "CrimeNet local crime ingestion/1.0",
        }
    )

    app_token = os.environ.get(
        "SOCRATA_APP_TOKEN",
        "",
    ).strip()

    if app_token:
        session.headers["X-App-Token"] = app_token

    return session


def request_json(
    session: Session,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout: tuple[int, int] = (30, 180),
) -> Any:
    try:
        response = session.get(
            url,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise DownloadError(
            f"Request timed out: {url}"
        ) from exc
    except requests.RequestException as exc:
        response_text = ""
        status_code: int | None = None

        if getattr(exc, "response", None) is not None:
            status_code = exc.response.status_code
            response_text = exc.response.text[:1000]

        raise DownloadError(
            "Request failed: "
            f"url={url}, "
            f"status={status_code}, "
            f"response={response_text!r}"
        ) from exc

    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise DownloadError(
            "Source returned invalid JSON: "
            f"url={response.url}, "
            f"status={response.status_code}, "
            f"response={response.text[:1000]!r}"
        ) from exc

    if (
        isinstance(payload, Mapping)
        and "error" in payload
        and payload["error"]
    ):
        raise DownloadError(
            f"Source returned an API error: {payload['error']!r}"
        )

    return payload


def value_to_raw_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def case_insensitive_value(
    record: Mapping[str, Any],
    field_name: str | None,
) -> Any:
    if field_name is None:
        return None

    if field_name in record:
        return record[field_name]

    target = field_name.casefold()

    for key, value in record.items():
        if str(key).casefold() == target:
            return value

    return None


def create_page_frame(
    *,
    records: Sequence[Mapping[str, Any]],
    source_fields: Sequence[str],
    source_city: str,
    source_dataset_id: str,
    source_dataset_kind: str,
    source_dataset: str,
    source_url: str,
    occurrence_year: int,
    record_id_field: str,
    occurred_at_field: str,
    occurred_at_end_field: str | None,
    latitude_field: str,
    longitude_field: str,
    downloaded_at_utc: str,
) -> pl.DataFrame:
    columns = [
        *source_fields,
        *[
            name
            for name, _dtype
            in SOURCE_METADATA_COLUMNS
        ],
    ]

    schema: list[tuple[str, pl.DataType]] = [
        *[
            (field, pl.String)
            for field in source_fields
        ],
        *SOURCE_METADATA_COLUMNS,
    ]

    rows: list[list[Any]] = []

    for record in records:
        source_values = [
            value_to_raw_string(
                record.get(field)
            )
            for field in source_fields
        ]

        source_record_id = value_to_raw_string(
            case_insensitive_value(
                record,
                record_id_field,
            )
        )

        occurred_at_raw = value_to_raw_string(
            case_insensitive_value(
                record,
                occurred_at_field,
            )
        )

        occurred_at_end_raw = value_to_raw_string(
            case_insensitive_value(
                record,
                occurred_at_end_field,
            )
        )

        latitude_raw = value_to_raw_string(
            case_insensitive_value(
                record,
                latitude_field,
            )
        )

        longitude_raw = value_to_raw_string(
            case_insensitive_value(
                record,
                longitude_field,
            )
        )

        if latitude_raw is None:
            latitude_raw = value_to_raw_string(
                record.get("geometry_y")
            )

        if longitude_raw is None:
            longitude_raw = value_to_raw_string(
                record.get("geometry_x")
            )

        metadata_values = [
            source_city,
            source_dataset_id,
            source_dataset_kind,
            source_dataset,
            source_record_id,
            occurred_at_raw,
            occurred_at_end_raw,
            latitude_raw,
            longitude_raw,
            occurrence_year,
            downloaded_at_utc,
            source_url,
        ]

        rows.append(
            [
                *source_values,
                *metadata_values,
            ]
        )

    return pl.DataFrame(
        rows,
        schema=schema,
        orient="row",
    ).select(columns)


def write_page(
    frame: pl.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.write_parquet(
        destination,
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=131_072,
    )


def partition_directory(
    output_root: Path,
    *,
    output_directory: str,
    source_dataset: str,
    year: int,
) -> Path:
    return (
        output_root
        / output_directory
        / f"source_dataset={source_dataset}"
        / f"occurrence_year={year}"
    )


def validate_existing_partition(
    destination: Path,
) -> bool:
    manifest_path = destination / "_manifest.json"

    if not destination.exists():
        return False

    if not manifest_path.exists():
        raise DownloadError(
            "Partition exists without _manifest.json: "
            f"{destination}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    parquet_paths = sorted(
        destination.glob("part-*.parquet")
    )

    if not parquet_paths:
        raise DownloadError(
            f"Partition contains no Parquet files: {destination}"
        )

    actual_rows = (
        pl.scan_parquet(
            [
                str(path)
                for path in parquet_paths
            ]
        )
        .select(
            pl.len().alias("rows")
        )
        .collect()
        .item()
    )

    expected_rows = int(
        manifest["record_count"]
    )

    if actual_rows != expected_rows:
        raise DownloadError(
            "Existing partition failed row-count validation: "
            f"path={destination}, "
            f"expected={expected_rows}, "
            f"actual={actual_rows}"
        )

    return True


def clean_staging_directories(
    destination: Path,
) -> None:
    pattern = (
        f".{destination.name}.*.tmp"
    )

    for path in destination.parent.glob(pattern):
        if path.is_dir():
            shutil.rmtree(path)


def prepare_staging_directory(
    destination: Path,
    *,
    overwrite: bool,
) -> Path | None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_staging_directories(
        destination
    )

    if destination.exists():
        if not overwrite:
            if validate_existing_partition(
                destination
            ):
                return None

        shutil.rmtree(destination)

    staging = destination.parent / (
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    staging.mkdir(
        parents=True,
        exist_ok=False,
    )

    return staging


def finalize_partition(
    *,
    staging: Path,
    destination: Path,
    manifest: Mapping[str, Any],
) -> None:
    manifest_path = staging / "_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    actual_rows = (
        pl.scan_parquet(
            [
                str(path)
                for path in sorted(
                    staging.glob(
                        "part-*.parquet"
                    )
                )
            ]
        )
        .select(
            pl.len().alias("rows")
        )
        .collect()
        .item()
    )

    expected_rows = int(
        manifest["record_count"]
    )

    if actual_rows != expected_rows:
        raise DownloadError(
            "Staged partition failed row-count validation: "
            f"expected={expected_rows}, "
            f"actual={actual_rows}, "
            f"staging={staging}"
        )

    os.replace(
        staging,
        destination,
    )


def socrata_metadata(
    session: Session,
    source: SocrataSource,
) -> tuple[list[str], str]:
    url = (
        f"{source.domain}/api/views/"
        f"{source.dataset_id}.json"
    )

    payload = request_json(
        session,
        url,
    )

    if not isinstance(payload, Mapping):
        raise DownloadError(
            f"Invalid Socrata metadata: {url}"
        )

    columns_payload = payload.get(
        "columns"
    )

    if not isinstance(
        columns_payload,
        list,
    ):
        raise DownloadError(
            f"Socrata metadata has no columns: {url}"
        )

    fields = [
        str(column["fieldName"])
        for column in columns_payload
        if (
            isinstance(column, Mapping)
            and column.get("fieldName")
        )
    ]

    if not fields:
        raise DownloadError(
            f"Socrata metadata returned zero fields: {url}"
        )

    return fields, url


def socrata_year_predicate(
    date_field: str,
    year: int,
) -> str:
    return (
        f"{date_field} >= "
        f"'{year:04d}-01-01T00:00:00.000' "
        f"AND {date_field} < "
        f"'{year + 1:04d}-01-01T00:00:00.000'"
    )


def socrata_count(
    session: Session,
    source: SocrataSource,
    *,
    year: int,
) -> int:
    endpoint = (
        f"{source.domain}/resource/"
        f"{source.dataset_id}.json"
    )

    payload = request_json(
        session,
        endpoint,
        params={
            "$select": "count(*) AS record_count",
            "$where": socrata_year_predicate(
                source.date_field,
                year,
            ),
        },
    )

    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], Mapping)
    ):
        raise DownloadError(
            "Unexpected Socrata count response: "
            f"city={source.city}, year={year}, "
            f"payload={payload!r}"
        )

    return int(
        payload[0]["record_count"]
    )


def download_socrata_year(
    session: Session,
    source: SocrataSource,
    *,
    year: int,
    output_root: Path,
    page_size: int,
    pause_seconds: float,
    overwrite: bool,
) -> None:
    destination = partition_directory(
        output_root,
        output_directory=(
            source.output_directory
        ),
        source_dataset=source.dataset_id,
        year=year,
    )

    staging = prepare_staging_directory(
        destination,
        overwrite=overwrite,
    )

    if staging is None:
        LOGGER.info(
            "Skipping verified partition: %s",
            destination,
        )
        return

    source_fields, metadata_url = (
        socrata_metadata(
            session,
            source,
        )
    )

    required_fields = {
        source.date_field,
        source.record_id_field,
        source.latitude_field,
        source.longitude_field,
    }

    missing_fields = (
        required_fields
        - set(source_fields)
    )

    if missing_fields:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise DownloadError(
            "Socrata source is missing required fields: "
            f"city={source.city}, "
            f"fields={sorted(missing_fields)}"
        )

    expected_count = socrata_count(
        session,
        source,
        year=year,
    )

    endpoint = (
        f"{source.domain}/resource/"
        f"{source.dataset_id}.json"
    )

    downloaded_at = utc_now()
    downloaded_count = 0
    page_number = 0

    try:
        while downloaded_count < expected_count:
            payload = request_json(
                session,
                endpoint,
                params={
                    "$where": (
                        socrata_year_predicate(
                            source.date_field,
                            year,
                        )
                    ),
                    "$order": ", ".join(
                        f"{field} ASC"
                        for field
                        in source.order_fields
                    ),
                    "$limit": page_size,
                    "$offset": downloaded_count,
                },
            )

            if not isinstance(payload, list):
                raise DownloadError(
                    "Unexpected Socrata page response: "
                    f"city={source.city}, "
                    f"year={year}"
                )

            if not payload:
                raise DownloadError(
                    "Socrata pagination ended before "
                    "the expected count was reached: "
                    f"city={source.city}, "
                    f"year={year}, "
                    f"expected={expected_count}, "
                    f"downloaded={downloaded_count}"
                )

            records = [
                record
                for record in payload
                if isinstance(
                    record,
                    Mapping,
                )
            ]

            if len(records) != len(payload):
                raise DownloadError(
                    "Socrata returned a non-object row: "
                    f"city={source.city}, "
                    f"year={year}"
                )

            frame = create_page_frame(
                records=records,
                source_fields=source_fields,
                source_city=source.city,
                source_dataset_id=(
                    source.dataset_id
                ),
                source_dataset_kind=(
                    "socrata"
                ),
                source_dataset=(
                    source.dataset_id
                ),
                source_url=endpoint,
                occurrence_year=year,
                record_id_field=(
                    source.record_id_field
                ),
                occurred_at_field=(
                    source.date_field
                ),
                occurred_at_end_field=None,
                latitude_field=(
                    source.latitude_field
                ),
                longitude_field=(
                    source.longitude_field
                ),
                downloaded_at_utc=(
                    downloaded_at
                ),
            )

            page_path = (
                staging
                / f"part-{page_number:05d}.parquet"
            )

            write_page(
                frame,
                page_path,
            )

            downloaded_count += frame.height
            page_number += 1

            LOGGER.info(
                "%s %s: downloaded %s/%s rows",
                source.city,
                year,
                f"{downloaded_count:,}",
                f"{expected_count:,}",
            )

            if pause_seconds > 0:
                time.sleep(pause_seconds)

        if downloaded_count != expected_count:
            raise DownloadError(
                "Socrata count mismatch: "
                f"city={source.city}, "
                f"year={year}, "
                f"expected={expected_count}, "
                f"downloaded={downloaded_count}"
            )

        finalize_partition(
            staging=staging,
            destination=destination,
            manifest={
                "source_city": source.city,
                "source_dataset_id": (
                    source.dataset_id
                ),
                "source_dataset_kind": "socrata",
                "source_dataset": (
                    source.dataset_id
                ),
                "source_url": endpoint,
                "metadata_url": metadata_url,
                "occurrence_year": year,
                "date_field": source.date_field,
                "record_id_field": (
                    source.record_id_field
                ),
                "record_count": downloaded_count,
                "page_count": page_number,
                "source_fields": source_fields,
                "downloaded_at_utc": downloaded_at,
                "compression": "zstd",
            },
        )

    except Exception:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise


def arcgis_layer_metadata(
    session: Session,
    layer_url: str,
) -> Mapping[str, Any]:
    payload = request_json(
        session,
        layer_url,
        params={"f": "json"},
    )

    if not isinstance(payload, Mapping):
        raise DownloadError(
            f"Invalid ArcGIS layer metadata: {layer_url}"
        )

    fields = payload.get("fields")

    if not isinstance(fields, list):
        raise DownloadError(
            f"ArcGIS layer has no field metadata: {layer_url}"
        )

    return payload


def resolve_feature_layer_url(
    session: Session,
    service_url: str,
) -> str:
    stripped = service_url.rstrip("/")

    final_component = stripped.rsplit(
        "/",
        maxsplit=1,
    )[-1]

    if final_component.isdigit():
        return stripped

    metadata = request_json(
        session,
        stripped,
        params={"f": "json"},
    )

    if not isinstance(metadata, Mapping):
        raise DownloadError(
            f"Invalid ArcGIS service metadata: {service_url}"
        )

    layers = metadata.get("layers")

    if not isinstance(layers, list):
        raise DownloadError(
            f"ArcGIS service has no layers: {service_url}"
        )

    feature_layers = [
        layer
        for layer in layers
        if (
            isinstance(layer, Mapping)
            and str(
                layer.get("id", "")
            ).isdigit()
        )
    ]

    if not feature_layers:
        raise DownloadError(
            f"ArcGIS service has no feature layer: {service_url}"
        )

    layer_id = int(
        feature_layers[0]["id"]
    )

    return f"{stripped}/{layer_id}"


def discover_dc_layer(
    session: Session,
    *,
    year: int,
) -> tuple[str, str]:
    title = f"Crime Incidents in {year}"

    search_payload = request_json(
        session,
        f"{ARCGIS_SHARING_URL}/search",
        params={
            "q": (
                f'title:"{title}" '
                'AND owner:DCGIS '
                'AND type:"Feature Service"'
            ),
            "num": 100,
            "f": "json",
        },
    )

    if not isinstance(
        search_payload,
        Mapping,
    ):
        raise DownloadError(
            "Invalid ArcGIS sharing-search response"
        )

    results = search_payload.get("results")

    if not isinstance(results, list):
        raise DownloadError(
            "ArcGIS sharing search returned no results array"
        )

    exact_matches = [
        item
        for item in results
        if (
            isinstance(item, Mapping)
            and str(
                item.get("title", "")
            ).casefold()
            == title.casefold()
            and str(
                item.get("owner", "")
            ).casefold()
            == "dcgis"
        )
    ]

    if not exact_matches:
        raise DownloadError(
            f"Could not find official DCGIS item: {title}"
        )

    exact_matches.sort(
        key=lambda item: int(
            item.get("modified", 0)
        ),
        reverse=True,
    )

    item = exact_matches[0]
    item_id = str(item["id"])
    service_url = item.get("url")

    if not service_url:
        item_payload = request_json(
            session,
            (
                f"{ARCGIS_SHARING_URL}/"
                f"content/items/{item_id}"
            ),
            params={"f": "json"},
        )

        if not isinstance(
            item_payload,
            Mapping,
        ):
            raise DownloadError(
                f"Invalid ArcGIS item metadata: {item_id}"
            )

        service_url = item_payload.get("url")

    if not service_url:
        raise DownloadError(
            f"DCGIS item has no service URL: {item_id}"
        )

    layer_url = resolve_feature_layer_url(
        session,
        str(service_url),
    )

    return item_id, layer_url


def arcgis_count(
    session: Session,
    layer_url: str,
    *,
    where: str,
) -> int:
    payload = request_json(
        session,
        f"{layer_url}/query",
        params={
            "where": where,
            "returnCountOnly": "true",
            "f": "json",
        },
    )

    if (
        not isinstance(payload, Mapping)
        or "count" not in payload
    ):
        raise DownloadError(
            f"Unexpected ArcGIS count response: {layer_url}"
        )

    return int(payload["count"])


def arcgis_feature_records(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    features = payload.get("features")

    if not isinstance(features, list):
        raise DownloadError(
            "ArcGIS page contains no features array"
        )

    records: list[dict[str, Any]] = []

    for feature in features:
        if not isinstance(feature, Mapping):
            raise DownloadError(
                "ArcGIS returned a non-object feature"
            )

        attributes = feature.get(
            "attributes",
            {},
        )

        if not isinstance(
            attributes,
            Mapping,
        ):
            raise DownloadError(
                "ArcGIS feature has invalid attributes"
            )

        record = dict(attributes)
        geometry = feature.get("geometry")

        if isinstance(geometry, Mapping):
            record["geometry_x"] = geometry.get("x")
            record["geometry_y"] = geometry.get("y")
            record["geometry_json"] = geometry

        records.append(record)

    return records


def arcgis_source_fields(
    metadata: Mapping[str, Any],
) -> list[str]:
    fields_payload = metadata["fields"]

    fields = [
        str(field["name"])
        for field in fields_payload
        if (
            isinstance(field, Mapping)
            and field.get("name")
        )
    ]

    return [
        *fields,
        "geometry_x",
        "geometry_y",
        "geometry_json",
    ]


def download_arcgis_year(
    session: Session,
    *,
    city: str,
    output_directory: str,
    source_dataset: str,
    source_dataset_id: str,
    layer_url: str,
    year: int,
    where: str,
    occurred_at_field: str,
    occurred_at_end_field: str | None,
    latitude_field: str,
    longitude_field: str,
    output_root: Path,
    page_size: int,
    pause_seconds: float,
    overwrite: bool,
) -> None:
    destination = partition_directory(
        output_root,
        output_directory=output_directory,
        source_dataset=source_dataset,
        year=year,
    )

    staging = prepare_staging_directory(
        destination,
        overwrite=overwrite,
    )

    if staging is None:
        LOGGER.info(
            "Skipping verified partition: %s",
            destination,
        )
        return

    metadata = arcgis_layer_metadata(
        session,
        layer_url,
    )

    object_id_field = str(
        metadata.get(
            "objectIdField",
            "",
        )
    )

    if not object_id_field:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise DownloadError(
            f"ArcGIS layer has no objectIdField: {layer_url}"
        )

    fields = arcgis_source_fields(
        metadata
    )

    service_limit = int(
        metadata.get(
            "maxRecordCount",
            page_size,
        )
    )

    effective_page_size = min(
        page_size,
        service_limit,
    )

    expected_count = arcgis_count(
        session,
        layer_url,
        where=where,
    )

    downloaded_at = utc_now()
    downloaded_count = 0
    page_number = 0

    try:
        while downloaded_count < expected_count:
            payload = request_json(
                session,
                f"{layer_url}/query",
                params={
                    "where": where,
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "orderByFields": (
                        f"{object_id_field} ASC"
                    ),
                    "resultOffset": (
                        downloaded_count
                    ),
                    "resultRecordCount": (
                        effective_page_size
                    ),
                    "f": "json",
                },
            )

            if not isinstance(payload, Mapping):
                raise DownloadError(
                    f"Invalid ArcGIS page: {layer_url}"
                )

            records = arcgis_feature_records(
                payload
            )

            if not records:
                raise DownloadError(
                    "ArcGIS pagination ended before "
                    "the expected count was reached: "
                    f"city={city}, "
                    f"year={year}, "
                    f"expected={expected_count}, "
                    f"downloaded={downloaded_count}"
                )

            frame = create_page_frame(
                records=records,
                source_fields=fields,
                source_city=city,
                source_dataset_id=(
                    source_dataset_id
                ),
                source_dataset_kind="arcgis",
                source_dataset=source_dataset,
                source_url=layer_url,
                occurrence_year=year,
                record_id_field=(
                    object_id_field
                ),
                occurred_at_field=(
                    occurred_at_field
                ),
                occurred_at_end_field=(
                    occurred_at_end_field
                ),
                latitude_field=(
                    latitude_field
                ),
                longitude_field=(
                    longitude_field
                ),
                downloaded_at_utc=downloaded_at,
            )

            page_path = (
                staging
                / f"part-{page_number:05d}.parquet"
            )

            write_page(
                frame,
                page_path,
            )

            downloaded_count += frame.height
            page_number += 1

            LOGGER.info(
                "%s %s: downloaded %s/%s rows",
                city,
                year,
                f"{downloaded_count:,}",
                f"{expected_count:,}",
            )

            if pause_seconds > 0:
                time.sleep(pause_seconds)

        if downloaded_count != expected_count:
            raise DownloadError(
                "ArcGIS count mismatch: "
                f"city={city}, "
                f"year={year}, "
                f"expected={expected_count}, "
                f"downloaded={downloaded_count}"
            )

        finalize_partition(
            staging=staging,
            destination=destination,
            manifest={
                "source_city": city,
                "source_dataset_id": (
                    source_dataset_id
                ),
                "source_dataset_kind": "arcgis",
                "source_dataset": source_dataset,
                "source_url": layer_url,
                "occurrence_year": year,
                "where": where,
                "record_id_field": (
                    object_id_field
                ),
                "occurred_at_field": (
                    occurred_at_field
                ),
                "occurred_at_end_field": (
                    occurred_at_end_field
                ),
                "record_count": downloaded_count,
                "page_count": page_number,
                "page_size": effective_page_size,
                "source_fields": fields,
                "downloaded_at_utc": downloaded_at,
                "compression": "zstd",
            },
        )

    except Exception:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise


def baltimore_where(
    year: int,
) -> str:
    return (
        "CrimeDateTime >= "
        f"DATE '{year:04d}-01-01' "
        "AND CrimeDateTime < "
        f"DATE '{year + 1:04d}-01-01'"
    )


def download_selected_cities(
    *,
    session: Session,
    cities: Sequence[str],
    start_year: int,
    end_year: int,
    output_root: Path,
    socrata_page_size: int,
    arcgis_page_size: int,
    pause_seconds: float,
    overwrite: bool,
) -> None:
    for city in cities:
        if city in SOCRATA_SOURCES:
            source = SOCRATA_SOURCES[
                city
            ]

            for year in range(
                start_year,
                end_year + 1,
            ):
                download_socrata_year(
                    session,
                    source,
                    year=year,
                    output_root=output_root,
                    page_size=(
                        socrata_page_size
                    ),
                    pause_seconds=(
                        pause_seconds
                    ),
                    overwrite=overwrite,
                )

        elif city == "baltimore":
            first_year = max(
                start_year,
                2022,
            )

            if end_year < 2022:
                LOGGER.warning(
                    "Skipping Baltimore because "
                    "the selected range ends before 2022"
                )
                continue

            for year in range(
                first_year,
                end_year + 1,
            ):
                download_arcgis_year(
                    session,
                    city="baltimore",
                    output_directory=(
                        "baltimore_crime"
                    ),
                    source_dataset=(
                        "baltimore_nibrs_group_a"
                    ),
                    source_dataset_id=(
                        "204beefe92a645d79fdf0969957bbdf8"
                    ),
                    layer_url=(
                        BALTIMORE_LAYER_URL
                    ),
                    year=year,
                    where=baltimore_where(
                        year
                    ),
                    occurred_at_field=(
                        "CrimeDateTime"
                    ),
                    occurred_at_end_field=None,
                    latitude_field="Latitude",
                    longitude_field="Longitude",
                    output_root=output_root,
                    page_size=arcgis_page_size,
                    pause_seconds=(
                        pause_seconds
                    ),
                    overwrite=overwrite,
                )

        elif city == "washington_dc":
            for year in range(
                start_year,
                end_year + 1,
            ):
                item_id, layer_url = (
                    discover_dc_layer(
                        session,
                        year=year,
                    )
                )

                download_arcgis_year(
                    session,
                    city="washington_dc",
                    output_directory=(
                        "washington_dc_crime"
                    ),
                    source_dataset=(
                        "dc_crime_incidents"
                    ),
                    source_dataset_id=(
                        item_id
                    ),
                    layer_url=layer_url,
                    year=year,
                    where="1=1",
                    occurred_at_field=(
                        "START_DATE"
                    ),
                    occurred_at_end_field=(
                        "END_DATE"
                    ),
                    latitude_field=(
                        "LATITUDE"
                    ),
                    longitude_field=(
                        "LONGITUDE"
                    ),
                    output_root=output_root,
                    page_size=arcgis_page_size,
                    pause_seconds=(
                        pause_seconds
                    ),
                    overwrite=overwrite,
                )

        else:
            raise ValueError(
                f"Unsupported city: {city}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download San Francisco, Seattle, "
            "Baltimore, and Washington DC crime "
            "data into partitioned Parquet."
        )
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        choices=CITY_CHOICES,
        default=list(CITY_CHOICES),
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--socrata-page-size",
        type=int,
        default=25_000,
    )

    parser.add_argument(
        "--arcgis-page-size",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError(
            "--start-year cannot exceed --end-year"
        )

    if args.socrata_page_size <= 0:
        raise ValueError(
            "--socrata-page-size must be positive"
        )

    if args.arcgis_page_size <= 0:
        raise ValueError(
            "--arcgis-page-size must be positive"
        )

    if args.pause_seconds < 0:
        raise ValueError(
            "--pause-seconds cannot be negative"
        )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = build_session()

    try:
        download_selected_cities(
            session=session,
            cities=args.cities,
            start_year=args.start_year,
            end_year=args.end_year,
            output_root=output_root,
            socrata_page_size=(
                args.socrata_page_size
            ),
            arcgis_page_size=(
                args.arcgis_page_size
            ),
            pause_seconds=(
                args.pause_seconds
            ),
            overwrite=args.overwrite,
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
