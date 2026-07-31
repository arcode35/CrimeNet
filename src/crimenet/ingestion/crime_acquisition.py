"""Download supported city crime data into raw-normalized Parquet.

Supported sources:
- Dallas Open Data (Socrata)
- Fort Worth Open Data (ArcGIS)
- Chicago Data Portal (Socrata)
- NYC Open Data / NYPD complaints (Socrata)
- San Francisco Open Data (Socrata)
- Seattle Open Data (Socrata)
- Baltimore Open Data (ArcGIS)
- Washington, DC Open Data (annual ArcGIS services)

The module deliberately does not deduplicate and does not apply the CrimeNet
2014 analytical cutoff. Every successful acquisition run is published under a
new ``acquisition_run_id=...`` directory. Bronze may therefore ingest repeated
source records across runs; canonical Silver is responsible for source-key
upserts/deduplication and date eligibility.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crimenet.observability.logging import get_logger


LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public source identities
# ---------------------------------------------------------------------------

DALLAS_DOMAIN = "https://www.dallasopendata.com"
DALLAS_DATASET_ID = "qv6i-rri7"

CHICAGO_DOMAIN = "https://data.cityofchicago.org"
CHICAGO_DATASET_ID = "ijzp-q8t2"

NYC_DOMAIN = "https://data.cityofnewyork.us"
NYC_HISTORIC_DATASET_ID = "qgea-i56i"
NYC_CURRENT_DATASET_ID = "5uac-w243"

SF_DOMAIN = "https://data.sfgov.org"
SF_DATASET_ID = "wg3w-h783"

SEATTLE_DOMAIN = "https://data.seattle.gov"
SEATTLE_DATASET_ID = "tazs-3rd5"

FORT_WORTH_ITEM_ID = "5445d088d9f143f1a3c671ee93a5f1a1"
FORT_WORTH_SOURCE_DATASET = "fort_worth_crime_data"

BALTIMORE_LAYER_URL = (
    "https://services1.arcgis.com/UWYHeuuJISiGmgXx/"
    "arcgis/rest/services/NIBRS_GroupA_Crime_Data/FeatureServer/0"
)
BALTIMORE_SOURCE_DATASET = "baltimore_nibrs_group_a"
BALTIMORE_SOURCE_DATASET_ID = "204beefe92a645d79fdf0969957bbdf8"

ARCGIS_SHARING_URL = "https://www.arcgis.com/sharing/rest"
DC_SOURCE_DATASET = "dc_crime_incidents"

CITY_CHOICES = (
    "dallas",
    "fort_worth",
    "chicago",
    "new_york",
    "san_francisco",
    "seattle",
    "baltimore",
    "washington_dc",
)

# These are source-availability floors, not the Silver analytical cutoff.
SOURCE_MINIMUM_YEARS = {
    "dallas": 2014,
    "fort_worth": 2014,
    "chicago": 2001,
    "new_york": 2006,
    "san_francisco": 2018,
    "seattle": 2008,
    "baltimore": 2022,
    "washington_dc": 2014,
}

# Preserve the explicit source contracts from the original Chicago and NYC
# acquisition scripts. Other Socrata sources retain every published field.
CHICAGO_COLUMNS = (
    "id",
    "case_number",
    "date",
    "block",
    "iucr",
    "primary_type",
    "description",
    "location_description",
    "arrest",
    "domestic",
    "beat",
    "district",
    "ward",
    "community_area",
    "fbi_code",
    "x_coordinate",
    "y_coordinate",
    "year",
    "updated_on",
    "latitude",
    "longitude",
)

NYC_COLUMNS = (
    "cmplnt_num",
    "cmplnt_fr_dt",
    "cmplnt_fr_tm",
    "cmplnt_to_dt",
    "cmplnt_to_tm",
    "rpt_dt",
    "ky_cd",
    "ofns_desc",
    "pd_cd",
    "pd_desc",
    "crm_atpt_cptd_cd",
    "law_cat_cd",
    "boro_nm",
    "addr_pct_cd",
    "loc_of_occur_desc",
    "prem_typ_desc",
    "juris_desc",
    "jurisdiction_code",
    "parks_nm",
    "hadevelopt",
    "housing_psa",
    "transit_district",
    "patrol_boro",
    "station_name",
    "latitude",
    "longitude",
    "x_coord_cd",
    "y_coord_cd",
)

LINEAGE_COLUMNS = (
    "source_city",
    "source_dataset_id",
    "source_dataset_kind",
    "source_dataset",
    "source_record_id",
    "occurred_at_raw",
    "occurred_at_end_raw",
    "latitude_raw",
    "longitude_raw",
    "occurrence_year",
    "downloaded_at_utc",
    "source_url",
    "acquisition_run_id",
)


@dataclass(frozen=True)
class SocrataSource:
    """Configuration for one Socrata-backed city source."""

    city: str
    domain: str
    dataset_id: str
    date_field: str
    record_id_field: str
    latitude_field: str
    longitude_field: str
    order_fields: tuple[str, ...]
    filter_kind: str = "timestamp"
    selected_columns: tuple[str, ...] | None = None
    cursor_field: str | None = None


@dataclass(frozen=True)
class CrimeAcquisitionSummary:
    """Aggregate result for one acquisition execution."""

    attempted_partitions: int
    completed_partitions: int
    failed_partitions: int
    downloaded_rows: int
    failures: tuple[str, ...]


SOCRATA_SOURCES: dict[str, SocrataSource] = {
    "dallas": SocrataSource(
        city="dallas",
        domain=DALLAS_DOMAIN,
        dataset_id=DALLAS_DATASET_ID,
        date_field="date1",
        record_id_field="servnumid",
        latitude_field="geocoded_column",
        longitude_field="geocoded_column",
        order_fields=("date1", "servnumid"),
    ),
    "chicago": SocrataSource(
        city="chicago",
        domain=CHICAGO_DOMAIN,
        dataset_id=CHICAGO_DATASET_ID,
        date_field="year",
        record_id_field="id",
        latitude_field="latitude",
        longitude_field="longitude",
        order_fields=("id",),
        filter_kind="year",
        selected_columns=CHICAGO_COLUMNS,
        cursor_field="id",
    ),
    "san_francisco": SocrataSource(
        city="san_francisco",
        domain=SF_DOMAIN,
        dataset_id=SF_DATASET_ID,
        date_field="incident_datetime",
        record_id_field="row_id",
        latitude_field="latitude",
        longitude_field="longitude",
        order_fields=("incident_datetime", "row_id"),
    ),
    "seattle": SocrataSource(
        city="seattle",
        domain=SEATTLE_DOMAIN,
        dataset_id=SEATTLE_DATASET_ID,
        date_field="offense_date",
        record_id_field="offense_id",
        latitude_field="latitude",
        longitude_field="longitude",
        order_fields=("offense_date", "offense_id"),
    ),
}


class CrimeDownloadError(RuntimeError):
    """Raised when a source response cannot be safely landed."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_acquisition_run_id() -> str:
    return (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid4().hex[:12]
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def build_crime_session(*, socrata_app_token: str = "") -> Session:
    retry = Retry(
        total=7,
        connect=7,
        read=7,
        status=7,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=1.0,
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "CrimeNet crime acquisition/1.0",
        }
    )

    normalized_token = socrata_app_token.strip()
    if normalized_token:
        session.headers["X-App-Token"] = normalized_token

    return session


def request_json(
    session: Session,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout: tuple[int, int] = (30, 180),
) -> Any:
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise CrimeDownloadError(f"Request timed out: {url}") from exc
    except requests.RequestException as exc:
        status_code: int | None = None
        response_text = ""

        if getattr(exc, "response", None) is not None:
            status_code = exc.response.status_code
            response_text = exc.response.text[:1000]

        raise CrimeDownloadError(
            "Request failed: "
            f"url={url}, status={status_code}, response={response_text!r}"
        ) from exc

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise CrimeDownloadError(
            "Source returned invalid JSON: "
            f"url={response.url}, status={response.status_code}, "
            f"response={response.text[:1000]!r}"
        ) from exc

    if isinstance(payload, Mapping) and payload.get("error"):
        raise CrimeDownloadError(
            f"Source returned an API error: {payload['error']!r}"
        )

    return payload


# ---------------------------------------------------------------------------
# Raw-record conversion and volume publication
# ---------------------------------------------------------------------------


def value_to_raw_string(value: Any) -> str | None:
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


def nested_coordinate(
    record: Mapping[str, Any],
    *,
    field_name: str,
    coordinate: str,
) -> str | None:
    """Extract latitude/longitude from a Socrata location object when present."""
    value = case_insensitive_value(record, field_name)
    if not isinstance(value, Mapping):
        return None

    direct = value.get(coordinate)
    if direct is not None:
        return value_to_raw_string(direct)

    coordinates = value.get("coordinates")
    if isinstance(coordinates, Sequence) and not isinstance(coordinates, str):
        if len(coordinates) >= 2:
            index = 1 if coordinate == "latitude" else 0
            return value_to_raw_string(coordinates[index])

    return None


def create_page_dataframe(
    spark: SparkSession,
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
    acquisition_run_id: str,
) -> DataFrame:
    duplicate_columns = set(source_fields) & set(LINEAGE_COLUMNS)
    if duplicate_columns:
        raise CrimeDownloadError(
            "Source columns conflict with lineage columns: "
            + ", ".join(sorted(duplicate_columns))
        )

    if len(set(source_fields)) != len(source_fields):
        raise CrimeDownloadError("Source metadata contains duplicate fields")

    schema = StructType(
        [
            *[
                StructField(field, StringType(), True)
                for field in source_fields
            ],
            StructField("source_city", StringType(), False),
            StructField("source_dataset_id", StringType(), True),
            StructField("source_dataset_kind", StringType(), False),
            StructField("source_dataset", StringType(), False),
            StructField("source_record_id", StringType(), True),
            StructField("occurred_at_raw", StringType(), True),
            StructField("occurred_at_end_raw", StringType(), True),
            StructField("latitude_raw", StringType(), True),
            StructField("longitude_raw", StringType(), True),
            StructField("occurrence_year", IntegerType(), False),
            StructField("downloaded_at_utc", StringType(), False),
            StructField("source_url", StringType(), False),
            StructField("acquisition_run_id", StringType(), False),
        ]
    )

    rows: list[dict[str, Any]] = []

    for record in records:
        row = {
            field: value_to_raw_string(record.get(field))
            for field in source_fields
        }

        latitude_raw = value_to_raw_string(
            case_insensitive_value(record, latitude_field)
        )
        longitude_raw = value_to_raw_string(
            case_insensitive_value(record, longitude_field)
        )

        # Dallas exposes a Socrata location object rather than independent
        # latitude/longitude fields in some schema revisions.
        if latitude_raw is None or latitude_raw.startswith("{"):
            latitude_raw = nested_coordinate(
                record,
                field_name=latitude_field,
                coordinate="latitude",
            ) or value_to_raw_string(record.get("geometry_y"))

        if longitude_raw is None or longitude_raw.startswith("{"):
            longitude_raw = nested_coordinate(
                record,
                field_name=longitude_field,
                coordinate="longitude",
            ) or value_to_raw_string(record.get("geometry_x"))

        row.update(
            {
                "source_city": source_city,
                "source_dataset_id": source_dataset_id,
                "source_dataset_kind": source_dataset_kind,
                "source_dataset": source_dataset,
                "source_record_id": value_to_raw_string(
                    case_insensitive_value(record, record_id_field)
                ),
                "occurred_at_raw": value_to_raw_string(
                    case_insensitive_value(record, occurred_at_field)
                ),
                "occurred_at_end_raw": value_to_raw_string(
                    case_insensitive_value(record, occurred_at_end_field)
                ),
                "latitude_raw": latitude_raw,
                "longitude_raw": longitude_raw,
                "occurrence_year": occurrence_year,
                "downloaded_at_utc": downloaded_at_utc,
                "source_url": source_url,
                "acquisition_run_id": acquisition_run_id,
            }
        )

        rows.append(row)

    return spark.createDataFrame(rows, schema=schema)


def get_partition_parent(
    output_root: Path,
    *,
    city: str,
    source_dataset: str,
    occurrence_year: int,
) -> Path:
    return (
        output_root
        / city
        / f"source_dataset={source_dataset}"
        / f"occurrence_year={occurrence_year}"
    )


def prepare_run_directories(
    output_root: Path,
    *,
    city: str,
    source_dataset: str,
    occurrence_year: int,
    acquisition_run_id: str,
) -> tuple[Path, Path]:
    parent = get_partition_parent(
        output_root,
        city=city,
        source_dataset=source_dataset,
        occurrence_year=occurrence_year,
    )
    parent.mkdir(parents=True, exist_ok=True)

    staging = parent / f".acquisition_run_id={acquisition_run_id}.tmp"
    destination = parent / f"acquisition_run_id={acquisition_run_id}"

    if staging.exists():
        shutil.rmtree(staging)

    if destination.exists():
        raise CrimeDownloadError(
            f"Acquisition destination already exists: {destination}"
        )

    staging.mkdir(parents=True, exist_ok=False)
    return staging, destination


def write_page(dataframe: DataFrame, staging: Path) -> None:
    (
        dataframe.coalesce(1)
        .write.format("parquet")
        .mode("append")
        .option("compression", "zstd")
        .save(str(staging))
    )


def count_parquet_rows(spark: SparkSession, directory: Path) -> int:
    parquet_paths = sorted(directory.glob("part-*.parquet"))
    if not parquet_paths:
        return 0

    return spark.read.parquet(str(directory)).count()


def write_manifest(
    *,
    staging: Path,
    manifest: Mapping[str, Any],
) -> None:
    manifest_path = staging / "_manifest.json"
    temporary_path = staging / "._manifest.json.tmp"

    temporary_path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)


def finalize_partition(
    spark: SparkSession,
    *,
    staging: Path,
    destination: Path,
    manifest: Mapping[str, Any],
) -> None:
    expected_rows = int(manifest["record_count"])
    actual_rows = count_parquet_rows(spark, staging)

    if actual_rows != expected_rows:
        raise CrimeDownloadError(
            "Partition row-count validation failed: "
            f"path={staging}, expected={expected_rows}, actual={actual_rows}"
        )

    write_manifest(
        staging=staging,
        manifest={
            **manifest,
            "validated_record_count": actual_rows,
            "status": "complete",
        },
    )

    os.replace(staging, destination)


# ---------------------------------------------------------------------------
# Socrata
# ---------------------------------------------------------------------------


def socrata_metadata(
    session: Session,
    *,
    domain: str,
    dataset_id: str,
) -> tuple[Mapping[str, Any], list[str], str]:
    metadata_url = f"{domain}/api/views/{dataset_id}.json"
    payload = request_json(session, metadata_url)

    if not isinstance(payload, Mapping):
        raise CrimeDownloadError(f"Invalid Socrata metadata: {metadata_url}")

    columns_payload = payload.get("columns")
    if not isinstance(columns_payload, list):
        raise CrimeDownloadError(
            f"Socrata metadata has no columns array: {metadata_url}"
        )

    fields = [
        str(column["fieldName"])
        for column in columns_payload
        if isinstance(column, Mapping) and column.get("fieldName")
    ]

    if not fields:
        raise CrimeDownloadError(
            f"Socrata metadata returned zero fields: {metadata_url}"
        )

    return payload, fields, metadata_url


def select_socrata_fields(
    source: SocrataSource,
    available_fields: Sequence[str],
) -> list[str]:
    available = set(available_fields)

    required = {
        source.date_field,
        source.record_id_field,
        *source.order_fields,
    }

    missing_required = required - available
    if missing_required:
        raise CrimeDownloadError(
            "Socrata source is missing required fields: "
            f"city={source.city}, fields={sorted(missing_required)}"
        )

    if source.selected_columns is None:
        return list(available_fields)

    selected = [
        field for field in source.selected_columns if field in available
    ]
    missing_selected_required = required - set(selected)

    if missing_selected_required:
        raise CrimeDownloadError(
            "Selected Socrata contract omitted required fields: "
            f"city={source.city}, fields={sorted(missing_selected_required)}"
        )

    return selected


def socrata_year_predicate(source: SocrataSource, year: int) -> str:
    if source.filter_kind == "year":
        return f"{source.date_field} = {year}"

    if source.filter_kind != "timestamp":
        raise ValueError(
            f"Unsupported Socrata filter kind: {source.filter_kind!r}"
        )

    return (
        f"{source.date_field} >= '{year:04d}-01-01T00:00:00.000' "
        f"AND {source.date_field} < '{year + 1:04d}-01-01T00:00:00.000'"
    )


def socrata_expected_count(
    session: Session,
    source: SocrataSource,
    *,
    year: int,
) -> int:
    endpoint = f"{source.domain}/resource/{source.dataset_id}.json"
    payload = request_json(
        session,
        endpoint,
        params={
            "$select": "count(*) AS record_count",
            "$where": socrata_year_predicate(source, year),
        },
    )

    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], Mapping)
        or "record_count" not in payload[0]
    ):
        raise CrimeDownloadError(
            "Unexpected Socrata count response: "
            f"city={source.city}, year={year}, payload={payload!r}"
        )

    return int(payload[0]["record_count"])


def resolve_nyc_source(year: int) -> SocrataSource:
    current_year = datetime.now(UTC).year
    dataset_id = (
        NYC_CURRENT_DATASET_ID
        if year == current_year
        else NYC_HISTORIC_DATASET_ID
    )

    return SocrataSource(
        city="new_york",
        domain=NYC_DOMAIN,
        dataset_id=dataset_id,
        date_field="cmplnt_fr_dt",
        record_id_field="cmplnt_num",
        latitude_field="latitude",
        longitude_field="longitude",
        order_fields=("cmplnt_num",),
        selected_columns=NYC_COLUMNS,
    )


def download_socrata_year(
    spark: SparkSession,
    session: Session,
    source: SocrataSource,
    *,
    year: int,
    output_root: Path,
    acquisition_run_id: str,
    page_size: int,
    pause_seconds: float,
) -> int:
    metadata, available_fields, metadata_url = socrata_metadata(
        session,
        domain=source.domain,
        dataset_id=source.dataset_id,
    )
    source_fields = select_socrata_fields(source, available_fields)
    expected_count = socrata_expected_count(session, source, year=year)

    endpoint = f"{source.domain}/resource/{source.dataset_id}.json"
    staging, destination = prepare_run_directories(
        output_root,
        city=source.city,
        source_dataset=source.dataset_id,
        occurrence_year=year,
        acquisition_run_id=acquisition_run_id,
    )

    downloaded_at = utc_now()
    downloaded_count = 0
    page_number = 0
    last_cursor: int | None = None

    try:
        while downloaded_count < expected_count:
            where = socrata_year_predicate(source, year)

            if source.cursor_field is not None and last_cursor is not None:
                where += f" AND {source.cursor_field} > {last_cursor}"

            params: dict[str, Any] = {
                "$select": ",".join(source_fields),
                "$where": where,
                "$order": ", ".join(
                    f"{field} ASC" for field in source.order_fields
                ),
                "$limit": page_size,
            }

            if source.cursor_field is None:
                params["$offset"] = downloaded_count

            payload = request_json(session, endpoint, params=params)

            if not isinstance(payload, list):
                raise CrimeDownloadError(
                    "Unexpected Socrata page response: "
                    f"city={source.city}, year={year}"
                )

            if not payload:
                raise CrimeDownloadError(
                    "Socrata pagination ended before the expected count: "
                    f"city={source.city}, year={year}, "
                    f"expected={expected_count}, downloaded={downloaded_count}"
                )

            records = [
                record for record in payload if isinstance(record, Mapping)
            ]
            if len(records) != len(payload):
                raise CrimeDownloadError(
                    f"Socrata returned a non-object row: city={source.city}"
                )

            if source.cursor_field is not None:
                cursor_values = [
                    int(record[source.cursor_field])
                    for record in records
                    if record.get(source.cursor_field) is not None
                ]

                if len(cursor_values) != len(records):
                    raise CrimeDownloadError(
                        f"Socrata cursor field contains nulls: {source.cursor_field}"
                    )

                if cursor_values != sorted(cursor_values):
                    raise CrimeDownloadError(
                        f"Socrata cursor page is not ordered: {source.cursor_field}"
                    )

                if last_cursor is not None and cursor_values[0] <= last_cursor:
                    raise CrimeDownloadError(
                        f"Socrata cursor pagination repeated values: {source.cursor_field}"
                    )

                last_cursor = cursor_values[-1]

            dataframe = create_page_dataframe(
                spark,
                records=records,
                source_fields=source_fields,
                source_city=source.city,
                source_dataset_id=source.dataset_id,
                source_dataset_kind="socrata",
                source_dataset=source.dataset_id,
                source_url=endpoint,
                occurrence_year=year,
                record_id_field=source.record_id_field,
                occurred_at_field=source.date_field,
                occurred_at_end_field=None,
                latitude_field=source.latitude_field,
                longitude_field=source.longitude_field,
                downloaded_at_utc=downloaded_at,
                acquisition_run_id=acquisition_run_id,
            )

            write_page(dataframe, staging)

            downloaded_count += len(records)
            page_number += 1

            LOGGER.info(
                "Downloaded Socrata crime page",
                city=source.city,
                year=year,
                page_number=page_number,
                downloaded_rows=downloaded_count,
                expected_rows=expected_count,
            )

            if pause_seconds > 0:
                time.sleep(pause_seconds)

        if downloaded_count != expected_count:
            raise CrimeDownloadError(
                "Socrata row-count mismatch: "
                f"city={source.city}, year={year}, "
                f"expected={expected_count}, downloaded={downloaded_count}"
            )

        ending_count = socrata_expected_count(session, source, year=year)
        if ending_count != expected_count:
            raise CrimeDownloadError(
                "Socrata source changed while downloading: "
                f"city={source.city}, year={year}, "
                f"starting_count={expected_count}, ending_count={ending_count}"
            )

        finalize_partition(
            spark,
            staging=staging,
            destination=destination,
            manifest={
                "source_city": source.city,
                "source_dataset_id": source.dataset_id,
                "source_dataset_kind": "socrata",
                "source_dataset": source.dataset_id,
                "source_url": endpoint,
                "metadata_url": metadata_url,
                "source_rows_updated_at": metadata.get("rowsUpdatedAt"),
                "occurrence_year": year,
                "record_count": downloaded_count,
                "page_count": page_number,
                "page_size": page_size,
                "source_fields": source_fields,
                "where": socrata_year_predicate(source, year),
                "downloaded_at_utc": downloaded_at,
                "acquisition_run_id": acquisition_run_id,
                "format": "parquet",
                "compression": "zstd",
            },
        )

        return downloaded_count

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# ArcGIS
# ---------------------------------------------------------------------------


def resolve_arcgis_item_layer_url(
    session: Session,
    *,
    item_id: str,
) -> str:
    item_payload = request_json(
        session,
        f"{ARCGIS_SHARING_URL}/content/items/{item_id}",
        params={"f": "json"},
    )

    if not isinstance(item_payload, Mapping):
        raise CrimeDownloadError(f"Invalid ArcGIS item metadata: {item_id}")

    service_url = item_payload.get("url")
    if not service_url:
        raise CrimeDownloadError(f"ArcGIS item has no service URL: {item_id}")

    return resolve_feature_layer_url(session, str(service_url))


def resolve_feature_layer_url(session: Session, service_url: str) -> str:
    stripped = service_url.rstrip("/")
    final_component = stripped.rsplit("/", maxsplit=1)[-1]

    if final_component.isdigit():
        return stripped

    metadata = request_json(session, stripped, params={"f": "json"})
    if not isinstance(metadata, Mapping):
        raise CrimeDownloadError(
            f"Invalid ArcGIS service metadata: {service_url}"
        )

    layers = metadata.get("layers")
    if not isinstance(layers, list):
        raise CrimeDownloadError(
            f"ArcGIS service has no layers array: {service_url}"
        )

    feature_layers = [
        layer
        for layer in layers
        if isinstance(layer, Mapping) and str(layer.get("id", "")).isdigit()
    ]

    if not feature_layers:
        raise CrimeDownloadError(
            f"ArcGIS service has no feature layer: {service_url}"
        )

    return f"{stripped}/{int(feature_layers[0]['id'])}"


def arcgis_layer_metadata(
    session: Session,
    layer_url: str,
) -> Mapping[str, Any]:
    payload = request_json(session, layer_url, params={"f": "json"})

    if not isinstance(payload, Mapping):
        raise CrimeDownloadError(f"Invalid ArcGIS metadata: {layer_url}")

    if not isinstance(payload.get("fields"), list):
        raise CrimeDownloadError(f"ArcGIS layer has no fields: {layer_url}")

    return payload


def arcgis_source_fields(metadata: Mapping[str, Any]) -> list[str]:
    fields = [
        str(field["name"])
        for field in metadata["fields"]
        if isinstance(field, Mapping) and field.get("name")
    ]

    return [*fields, "geometry_x", "geometry_y", "geometry_json"]


def resolve_field_name(
    available_fields: Sequence[str],
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    lookup = {field.casefold(): field for field in available_fields}

    for candidate in candidates:
        resolved = lookup.get(candidate.casefold())
        if resolved is not None:
            return resolved

    if required:
        raise CrimeDownloadError(
            "Could not resolve required ArcGIS field. "
            f"Candidates={list(candidates)!r}, available={list(available_fields)!r}"
        )

    return None


def arcgis_date_where(date_field: str, year: int) -> str:
    return (
        f"{date_field} >= DATE '{year:04d}-01-01' "
        f"AND {date_field} < DATE '{year + 1:04d}-01-01'"
    )


def arcgis_object_ids(
    session: Session,
    layer_url: str,
    *,
    where: str,
) -> list[int]:
    payload = request_json(
        session,
        f"{layer_url}/query",
        params={
            "where": where,
            "returnIdsOnly": "true",
            "f": "json",
        },
    )

    if not isinstance(payload, Mapping):
        raise CrimeDownloadError(f"Invalid ArcGIS ID response: {layer_url}")

    object_ids = payload.get("objectIds")
    if object_ids is None:
        return []

    if not isinstance(object_ids, list):
        raise CrimeDownloadError(
            f"ArcGIS objectIds is not a list: {layer_url}"
        )

    return sorted(int(value) for value in object_ids)


def arcgis_feature_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    features = payload.get("features")
    if not isinstance(features, list):
        raise CrimeDownloadError("ArcGIS page contains no features array")

    records: list[dict[str, Any]] = []

    for feature in features:
        if not isinstance(feature, Mapping):
            raise CrimeDownloadError("ArcGIS returned a non-object feature")

        attributes = feature.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise CrimeDownloadError("ArcGIS feature has invalid attributes")

        record = dict(attributes)
        geometry = feature.get("geometry")

        if isinstance(geometry, Mapping):
            record["geometry_x"] = geometry.get("x")
            record["geometry_y"] = geometry.get("y")
            record["geometry_json"] = dict(geometry)

        records.append(record)

    return records


def chunked(values: Sequence[int], size: int) -> Sequence[Sequence[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def download_arcgis_year(
    spark: SparkSession,
    session: Session,
    *,
    city: str,
    source_dataset: str,
    source_dataset_id: str,
    layer_url: str,
    year: int,
    where: str,
    occurred_at_field_candidates: Sequence[str],
    occurred_at_end_field_candidates: Sequence[str] = (),
    latitude_field_candidates: Sequence[str] = ("Latitude", "LATITUDE"),
    longitude_field_candidates: Sequence[str] = ("Longitude", "LONGITUDE"),
    output_root: Path,
    acquisition_run_id: str,
    page_size: int,
    pause_seconds: float,
) -> int:
    metadata = arcgis_layer_metadata(session, layer_url)
    object_id_field = str(metadata.get("objectIdField", ""))
    if not object_id_field:
        raise CrimeDownloadError(f"ArcGIS layer has no objectIdField: {layer_url}")

    source_fields = arcgis_source_fields(metadata)
    metadata_fields = source_fields[:-3]

    occurred_at_field = resolve_field_name(
        metadata_fields,
        occurred_at_field_candidates,
    )
    occurred_at_end_field = resolve_field_name(
        metadata_fields,
        occurred_at_end_field_candidates,
        required=False,
    )
    latitude_field = resolve_field_name(
        metadata_fields,
        latitude_field_candidates,
        required=False,
    ) or "geometry_y"
    longitude_field = resolve_field_name(
        metadata_fields,
        longitude_field_candidates,
        required=False,
    ) or "geometry_x"

    object_ids = arcgis_object_ids(session, layer_url, where=where)
    expected_count = len(object_ids)

    service_limit = int(metadata.get("maxRecordCount", page_size))
    effective_page_size = min(page_size, service_limit)

    staging, destination = prepare_run_directories(
        output_root,
        city=city,
        source_dataset=source_dataset,
        occurrence_year=year,
        acquisition_run_id=acquisition_run_id,
    )

    downloaded_at = utc_now()
    downloaded_count = 0
    page_number = 0

    try:
        for object_id_page in chunked(object_ids, effective_page_size):
            payload = request_json(
                session,
                f"{layer_url}/query",
                params={
                    "objectIds": ",".join(str(value) for value in object_id_page),
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "orderByFields": f"{object_id_field} ASC",
                    "f": "json",
                },
            )

            if not isinstance(payload, Mapping):
                raise CrimeDownloadError(f"Invalid ArcGIS page: {layer_url}")

            records = arcgis_feature_records(payload)
            if len(records) != len(object_id_page):
                raise CrimeDownloadError(
                    "ArcGIS page count mismatch: "
                    f"city={city}, year={year}, "
                    f"requested={len(object_id_page)}, returned={len(records)}"
                )

            dataframe = create_page_dataframe(
                spark,
                records=records,
                source_fields=source_fields,
                source_city=city,
                source_dataset_id=source_dataset_id,
                source_dataset_kind="arcgis",
                source_dataset=source_dataset,
                source_url=layer_url,
                occurrence_year=year,
                record_id_field=object_id_field,
                occurred_at_field=occurred_at_field,
                occurred_at_end_field=occurred_at_end_field,
                latitude_field=latitude_field,
                longitude_field=longitude_field,
                downloaded_at_utc=downloaded_at,
                acquisition_run_id=acquisition_run_id,
            )

            write_page(dataframe, staging)
            downloaded_count += len(records)
            page_number += 1

            LOGGER.info(
                "Downloaded ArcGIS crime page",
                city=city,
                year=year,
                page_number=page_number,
                downloaded_rows=downloaded_count,
                expected_rows=expected_count,
            )

            if pause_seconds > 0:
                time.sleep(pause_seconds)

        if downloaded_count != expected_count:
            raise CrimeDownloadError(
                "ArcGIS row-count mismatch: "
                f"city={city}, year={year}, "
                f"expected={expected_count}, downloaded={downloaded_count}"
            )

        finalize_partition(
            spark,
            staging=staging,
            destination=destination,
            manifest={
                "source_city": city,
                "source_dataset_id": source_dataset_id,
                "source_dataset_kind": "arcgis",
                "source_dataset": source_dataset,
                "source_url": layer_url,
                "occurrence_year": year,
                "record_count": downloaded_count,
                "page_count": page_number,
                "page_size": effective_page_size,
                "source_fields": source_fields,
                "where": where,
                "occurred_at_field": occurred_at_field,
                "occurred_at_end_field": occurred_at_end_field,
                "record_id_field": object_id_field,
                "downloaded_at_utc": downloaded_at,
                "acquisition_run_id": acquisition_run_id,
                "format": "parquet",
                "compression": "zstd",
            },
        )

        return downloaded_count

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Fort Worth, Baltimore, and DC adapters
# ---------------------------------------------------------------------------


def download_fort_worth_year(
    spark: SparkSession,
    session: Session,
    *,
    year: int,
    output_root: Path,
    acquisition_run_id: str,
    page_size: int,
    pause_seconds: float,
) -> int:
    layer_url = resolve_arcgis_item_layer_url(
        session,
        item_id=FORT_WORTH_ITEM_ID,
    )
    metadata = arcgis_layer_metadata(session, layer_url)
    fields = arcgis_source_fields(metadata)[:-3]
    date_field = resolve_field_name(
        fields,
        ("From_Date", "FromDate", "Reported_Date", "ReportedDate"),
    )

    return download_arcgis_year(
        spark,
        session,
        city="fort_worth",
        source_dataset=FORT_WORTH_SOURCE_DATASET,
        source_dataset_id=FORT_WORTH_ITEM_ID,
        layer_url=layer_url,
        year=year,
        where=arcgis_date_where(date_field, year),
        occurred_at_field_candidates=(
            "From_Date",
            "FromDate",
            "Reported_Date",
            "ReportedDate",
        ),
        latitude_field_candidates=("Latitude", "LATITUDE", "Y"),
        longitude_field_candidates=("Longitude", "LONGITUDE", "X"),
        output_root=output_root,
        acquisition_run_id=acquisition_run_id,
        page_size=page_size,
        pause_seconds=pause_seconds,
    )


def download_baltimore_year(
    spark: SparkSession,
    session: Session,
    *,
    year: int,
    output_root: Path,
    acquisition_run_id: str,
    page_size: int,
    pause_seconds: float,
) -> int:
    return download_arcgis_year(
        spark,
        session,
        city="baltimore",
        source_dataset=BALTIMORE_SOURCE_DATASET,
        source_dataset_id=BALTIMORE_SOURCE_DATASET_ID,
        layer_url=BALTIMORE_LAYER_URL,
        year=year,
        where=arcgis_date_where("CrimeDateTime", year),
        occurred_at_field_candidates=("CrimeDateTime",),
        latitude_field_candidates=("Latitude",),
        longitude_field_candidates=("Longitude",),
        output_root=output_root,
        acquisition_run_id=acquisition_run_id,
        page_size=page_size,
        pause_seconds=pause_seconds,
    )


OFFICIAL_DC_OWNERS = frozenset({"dcgisopendata", "dcgis"})
OFFICIAL_DC_TAGS = frozenset(
    {"district of columbia", "washington dc", "mpd", "cdw"}
)


def discover_dc_layer(
    session: Session,
    *,
    year: int,
) -> tuple[str, str]:
    title = f"Crime Incidents in {year}"
    search_queries = (
        f'title:"{title}" AND owner:DCGISopendata AND type:"Feature Service"',
        f'title:"{title}" AND type:"Feature Service"',
    )

    candidates: dict[str, Mapping[str, Any]] = {}

    for query in search_queries:
        payload = request_json(
            session,
            f"{ARCGIS_SHARING_URL}/search",
            params={"q": query, "num": 100, "f": "json"},
        )

        if not isinstance(payload, Mapping):
            raise CrimeDownloadError("Invalid ArcGIS sharing-search response")

        results = payload.get("results")
        if not isinstance(results, list):
            raise CrimeDownloadError(
                "ArcGIS sharing search returned no results array"
            )

        for item in results:
            if not isinstance(item, Mapping):
                continue
            item_id = str(item.get("id", ""))
            if item_id:
                candidates[item_id] = item

        if candidates:
            break

    exact_matches = [
        item
        for item in candidates.values()
        if str(item.get("title", "")).casefold() == title.casefold()
        and str(item.get("type", "")).casefold() == "feature service"
    ]

    if not exact_matches:
        raise CrimeDownloadError(
            f"Could not find an official DC Feature Service titled {title!r}"
        )

    def score(item: Mapping[str, Any]) -> tuple[int, int, int]:
        owner = str(item.get("owner", "")).casefold()
        tags = {str(tag).casefold() for tag in item.get("tags", [])}
        return (
            int(owner in OFFICIAL_DC_OWNERS),
            len(tags & OFFICIAL_DC_TAGS),
            int(item.get("modified", 0)),
        )

    exact_matches.sort(key=score, reverse=True)
    selected = exact_matches[0]

    owner = str(selected.get("owner", "")).casefold()
    tags = {str(tag).casefold() for tag in selected.get("tags", [])}
    if owner not in OFFICIAL_DC_OWNERS and not (tags & OFFICIAL_DC_TAGS):
        raise CrimeDownloadError(
            "Exact-title DC item lacked official ownership or tags: "
            f"owner={selected.get('owner')!r}, tags={selected.get('tags')!r}"
        )

    item_id = str(selected["id"])
    layer_url = resolve_arcgis_item_layer_url(session, item_id=item_id)
    return item_id, layer_url


def download_dc_year(
    spark: SparkSession,
    session: Session,
    *,
    year: int,
    output_root: Path,
    acquisition_run_id: str,
    page_size: int,
    pause_seconds: float,
) -> int:
    item_id, layer_url = discover_dc_layer(session, year=year)

    return download_arcgis_year(
        spark,
        session,
        city="washington_dc",
        source_dataset=DC_SOURCE_DATASET,
        source_dataset_id=item_id,
        layer_url=layer_url,
        year=year,
        where="1=1",
        occurred_at_field_candidates=("START_DATE", "Start_Date"),
        occurred_at_end_field_candidates=("END_DATE", "End_Date"),
        latitude_field_candidates=("LATITUDE", "Latitude"),
        longitude_field_candidates=("LONGITUDE", "Longitude"),
        output_root=output_root,
        acquisition_run_id=acquisition_run_id,
        page_size=page_size,
        pause_seconds=pause_seconds,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def acquire_crime_data(
    spark: SparkSession,
    *,
    session: Session,
    cities: Sequence[str],
    output_root: Path,
    start_year: int,
    end_year: int,
    socrata_page_size: int,
    arcgis_page_size: int,
    pause_seconds: float,
    acquisition_run_id: str | None = None,
) -> CrimeAcquisitionSummary:
    if start_year > end_year:
        raise ValueError("start_year cannot exceed end_year")

    if socrata_page_size <= 0:
        raise ValueError("socrata_page_size must be positive")

    if arcgis_page_size <= 0:
        raise ValueError("arcgis_page_size must be positive")

    if pause_seconds < 0:
        raise ValueError("pause_seconds cannot be negative")

    normalized_cities = tuple(
        dict.fromkeys(
            city.strip().lower()
            for city in cities
            if city.strip()
        )
    )

    unsupported = set(normalized_cities) - set(CITY_CHOICES)
    if unsupported:
        raise ValueError(
            "Unsupported cities: " + ", ".join(sorted(unsupported))
        )

    run_id = acquisition_run_id or create_acquisition_run_id()
    attempted_partitions = 0
    completed_partitions = 0
    failed_partitions = 0
    downloaded_rows = 0
    failures: list[str] = []

    for city in normalized_cities:
        first_year = max(start_year, SOURCE_MINIMUM_YEARS[city])

        for year in range(first_year, end_year + 1):
            attempted_partitions += 1

            try:
                if city in SOCRATA_SOURCES:
                    rows = download_socrata_year(
                        spark,
                        session,
                        SOCRATA_SOURCES[city],
                        year=year,
                        output_root=output_root,
                        acquisition_run_id=run_id,
                        page_size=socrata_page_size,
                        pause_seconds=pause_seconds,
                    )
                elif city == "new_york":
                    rows = download_socrata_year(
                        spark,
                        session,
                        resolve_nyc_source(year),
                        year=year,
                        output_root=output_root,
                        acquisition_run_id=run_id,
                        page_size=socrata_page_size,
                        pause_seconds=pause_seconds,
                    )
                elif city == "fort_worth":
                    rows = download_fort_worth_year(
                        spark,
                        session,
                        year=year,
                        output_root=output_root,
                        acquisition_run_id=run_id,
                        page_size=arcgis_page_size,
                        pause_seconds=pause_seconds,
                    )
                elif city == "baltimore":
                    rows = download_baltimore_year(
                        spark,
                        session,
                        year=year,
                        output_root=output_root,
                        acquisition_run_id=run_id,
                        page_size=arcgis_page_size,
                        pause_seconds=pause_seconds,
                    )
                elif city == "washington_dc":
                    rows = download_dc_year(
                        spark,
                        session,
                        year=year,
                        output_root=output_root,
                        acquisition_run_id=run_id,
                        page_size=arcgis_page_size,
                        pause_seconds=pause_seconds,
                    )
                else:
                    raise ValueError(f"Unsupported city: {city}")

                completed_partitions += 1
                downloaded_rows += rows

                LOGGER.info(
                    "Completed crime partition acquisition",
                    city=city,
                    year=year,
                    downloaded_rows=rows,
                    acquisition_run_id=run_id,
                )

            except Exception as exc:
                failed_partitions += 1
                failure = (
                    f"city={city}, year={year}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                failures.append(failure)

                LOGGER.exception(
                    "Crime partition acquisition failed",
                    city=city,
                    year=year,
                    acquisition_run_id=run_id,
                )

    return CrimeAcquisitionSummary(
        attempted_partitions=attempted_partitions,
        completed_partitions=completed_partitions,
        failed_partitions=failed_partitions,
        downloaded_rows=downloaded_rows,
        failures=tuple(failures),
    )
