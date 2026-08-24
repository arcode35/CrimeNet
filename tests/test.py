from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from crimenet_data.assets.crime.canonical import (
    CANONICAL_CRIME_SCHEMA,
    SOURCE_PROJECTION_SCHEMA,
)
from crimenet_data.assets.crime.ingestion import (
    prepare_bronze_source,
    read_source_pattern,
)
from crimenet_data.assets.crime.silver import build_silver
from crimenet_data.assets.crime.sources import (
    SOURCE_KEYS,
    SOURCES,
    AdapterContext,
    get_source,
)
from crimenet_data.assets.crime.sources.chandler_az import (
    EXPECTED_COLUMNS as CHANDLER_COLUMNS,
)
from crimenet_data.assets.crime.sources.montgomery_county_md import (
    EXPECTED_COLUMNS as MONTGOMERY_COLUMNS,
)
from crimenet_data.resources.crime_lake import CrimeLakeResources
from crimenet_data.resources.duckdb import DuckDBResource

EXPECTED_SOURCES = {
    "atlanta",
    "baltimore",
    "baton_rouge",
    "boston",
    "chandler_az",
    "chicago",
    "dallas",
    "denver",
    "east_baton_rouge_parish_sheriff_la",
    "fort_worth",
    "gainesville_fl",
    "los_angeles_county_sheriff",
    "marin_county_sheriff_ca",
    "montgomery_county_md",
    "new_york",
    "san_francisco",
    "seattle",
    "sonoma_county_sheriff_ca",
    "washington_dc",
}

FIXTURE_SOURCES = (
    "baltimore",
    "chicago",
    "fort_worth",
    "new_york",
    "san_francisco",
    "seattle",
    "washington_dc",
)


def local_lake() -> CrimeLakeResources:
    return CrimeLakeResources(bucket="/tmp/crimenet-test-lake")


def bronze(source_key: str, frame: pl.DataFrame) -> pl.LazyFrame:
    return prepare_bronze_source(
        frame.lazy().with_columns(
            pl.lit(f"/landing/{source_key}/sample").alias("_source_file_uri")
        ),
        get_source(source_key),
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


SOURCE_SAMPLES = {
    "atlanta": {
        "reportnumber": "1",
        "occurdate": "2024-01-02",
        "occurtime": "0304",
        "reportdate": "2024-01-03",
        "ucrliteral": "THEFT",
        "latitude": "33.75",
        "longitude": "-84.39",
        "location": "Main",
        "beat": "1",
        "neighborhood": "N",
    },
    "baltimore": {
        "RowID": "1",
        "CrimeDateTime": "2024-01-02 03:04:00",
        "CrimeCode": "4E",
        "Description": "THEFT",
        "Latitude": "39.29",
        "Longitude": "-76.61",
        "Location": "Main",
        "PremiseType": "Street",
        "New_District": "C",
        "Neighborhood": "N",
        "occurred_at_end_raw": "2024-01-02 04:00:00",
    },
    "baton_rouge": {
        "charge_id": "1",
        "incident_number": "I1",
        "charge_date": "2024-01-02 03:04:00",
        "report_date": "2024-01-03",
        "approved_date": "2024-01-04",
        "nibrs_code": "23A",
        "statute_category": "THEFT",
        "offense_description": "THEFT",
        "latitude": "30.45",
        "longitude": "-91.15",
        "street": "Main",
        "district": "1",
        "neighborhood": "N",
    },
    "boston": {
        "INCIDENT_NUMBER": "I1",
        "OFFENSE_CODE": "100",
        "OFFENSE_CODE_GROUP": "THEFT",
        "OFFENSE_DESCRIPTION": "THEFT",
        "OCCURRED_ON_DATE": "2024-01-02T03:04:00.000",
        "Lat": "42.35",
        "Long": "-71.06",
        "STREET": "Main",
        "DISTRICT": "A1",
        "REPORTING_AREA": "1",
    },
    "chandler_az": {
        "id": "1",
        "report_id": "10",
        "reported_date": "2024-01-03",
        "report_event_date_time": "2024-01-02 03:04:00",
        "report_event_end_date_time": "2024-01-02 04:00:00",
        "report_primary_offense_code": "23A",
        "report_primary_offense_description": "THEFT",
        "report_summary_offense_description": "LARCENY",
        "report_latitude": "33.30",
        "report_longitude": "-111.84",
        "report_address": "Main",
        "report_location_type": "Street",
        "report_district": "1",
        "report_beat": "A",
    },
    "chicago": {
        "id": "1",
        "date": "2024-01-02 03:04:00",
        "iucr": "0820",
        "primary_type": "THEFT",
        "description": "THEFT",
        "latitude": "41.88",
        "longitude": "-87.63",
        "block": "Main",
        "location_description": "Street",
        "district": "1",
        "community_area": "2",
    },
    "denver": {
        "OBJECTID": 1,
        "OFFENSE_ID": "O1",
        "OFFENSE_CODE": "23A",
        "OFFENSE_CATEGORY_ID": "theft",
        "OFFENSE_TYPE_ID": "larceny",
        "FIRST_OCCURRENCE_DATE": 1704164645000,
        "LAST_OCCURRENCE_DATE": None,
        "REPORTED_DATE": 1704251045000,
        "INCIDENT_ADDRESS": "Main",
        "GEO_LAT": 39.74,
        "GEO_LON": -104.99,
        "DISTRICT_ID": "1",
        "NEIGHBORHOOD_ID": "N",
        "geometry_type": "Point",
        "geometry_json": '{"type":"Point","coordinates":[-104.99,39.74]}',
    },
    "east_baton_rouge_parish_sheriff_la": {
        "charge_id": "1",
        "incident_number": "I1",
        "charge_date": "2024-01-02 03:04:00",
        "report_date": "2024-01-03",
        "approved_date": "2024-01-04",
        "nibrs_code": "23A",
        "statute_category": "THEFT",
        "offense_description": "THEFT",
        "latitude": "30.45",
        "longitude": "-91.15",
        "street": "Main",
        "district": "1",
        "neighborhood": "N",
    },
    "fort_worth": {
        "case_no_offense": "1",
        "occurrence_timestamp": "2024-01-02 03:04:00",
        "reported_date": "2024-01-03",
        "offense": "23A",
        "offense_desc": "THEFT",
        "latitude": "32.75",
        "longitude": "-97.33",
        "block_address": "Main",
        "locationtypedescription": "Street",
        "division": "C",
    },
    "gainesville_fl": {
        "id": "1",
        "narrative": "THEFT",
        "report_date": "2024-01-03",
        "report_hour_of_day": "5",
        "offense_date": "2024-01-02",
        "offense_hour_of_day": "3",
        "address": "Main",
        "latitude": "29.65",
        "longitude": "-82.32",
        "location": "Street",
    },
    "los_angeles_county_sheriff": {
        "LURN_SAK": "1",
        "INCIDENT_ID": "I1",
        "INCIDENT_DATE": "2024-01-02 03:04:00",
        "INCIDENT_REPORTED_DATE": "2024-01-03",
        "CATEGORY": "THEFT",
        "STAT": "23A",
        "STAT_DESC": "THEFT",
        "ADDRESS": "1 Main",
        "REPORTING_DISTRICT": "1",
        "UNIT_NAME": "Unit",
        "LATITUDE": "34.05",
        "LONGITUDE": "-118.25",
        "PART_CATEGORY": "1",
    },
    "marin_county_sheriff_ca": {
        "unique_id": "1",
        "incident_date_time": "2024-01-02 03:04:00",
        "crime": "THEFT",
        "crime_class": "LARCENY",
        "incident_street_address": "Main",
        "incident_city_town": "Marin",
        "jurisdiction": "Sheriff",
        "latitude": "38.0",
        "longitude": "-122.5",
    },
    "montgomery_county_md": {
        "incident_id": "I1",
        "case_number": "C1",
        "offence_code": "23A",
        "nibrs_code": "23A",
        "date": "2024-01-03",
        "start_date": "2024-01-02 03:04:00",
        "end_date": "2024-01-02 04:00:00",
        "crimename1": "Crime",
        "crimename2": "THEFT",
        "crimename3": "THEFT FROM AUTO",
        "district": "1",
        "location": "Main",
        "place": "Street",
        "sector": "A",
        "latitude": "39.1",
        "longitude": "-77.2",
    },
    "new_york": {
        "cmplnt_num": "1",
        "cmplnt_fr_dt": "01/02/2024",
        "cmplnt_fr_tm": "03:04:00",
        "cmplnt_to_dt": "01/02/2024",
        "cmplnt_to_tm": "04:00:00",
        "rpt_dt": "01/03/2024",
        "pd_cd": "100",
        "ky_cd": "200",
        "ofns_desc": "THEFT",
        "pd_desc": "THEFT",
        "latitude": "40.71",
        "longitude": "-74.00",
        "prem_typ_desc": "Street",
        "addr_pct_cd": "1",
        "boro_nm": "MANHATTAN",
    },
    "san_francisco": {
        "row_id": "1",
        "incident_datetime": "2024-01-02 03:04:00",
        "occurred_at_end_raw": "2024-01-02 04:00:00",
        "report_datetime": "2024-01-03",
        "incident_code": "100",
        "incident_category": "THEFT",
        "incident_description": "THEFT",
        "latitude": "37.77",
        "longitude": "-122.42",
        "intersection": "Main",
        "police_district": "C",
        "analysis_neighborhood": "N",
    },
    "seattle": {
        "offense_id": "1",
        "offense_date": "2024-01-02 03:04:00",
        "occurred_at_end_raw": "2024-01-02 04:00:00",
        "report_date_time": "2024-01-03",
        "nibrs_offense_code": "23A",
        "offense_category": "THEFT",
        "nibrs_offense_code_description": "THEFT",
        "latitude": "47.61",
        "longitude": "-122.33",
        "block_address": "Main",
        "precinct": "C",
        "neighborhood": "N",
    },
    "sonoma_county_sheriff_ca": {
        "id": "1",
        "incident_number": "I1",
        "date_time": "2024-01-02 03:04:00",
        "incident_type": "THEFT",
        "location_type": "Street",
        "city": "Sonoma",
        "location": "Main",
        "agency": "Sheriff",
    },
    "washington_dc": {
        "OBJECTID": "1",
        "START_DATE": "2024-01-02 03:04:00",
        "END_DATE": "2024-01-02 04:00:00",
        "REPORT_DAT": "2024-01-03",
        "OFFENSE": "THEFT",
        "LATITUDE": "38.90",
        "LONGITUDE": "-77.03",
        "BLOCK": "Main",
        "DISTRICT": "1",
        "NEIGHBORHOOD_CLUSTER": "N",
    },
}


def test_registry_contains_exactly_the_active_sources() -> None:
    assert set(SOURCE_KEYS) == EXPECTED_SOURCES
    assert set(SOURCES) == EXPECTED_SOURCES
    assert len(SOURCE_KEYS) == len(set(SOURCE_KEYS))
    assert get_source("denver") is SOURCES["denver"]
    with pytest.raises(KeyError, match="Unknown crime source"):
        get_source("philadelphia")


def test_registry_supports_all_transport_formats() -> None:
    formats = {
        pattern.format
        for source in SOURCES.values()
        for pattern in source.config.patterns
    }
    assert formats == {"csv", "parquet", "geojson"}


def test_native_csv_and_parquet_readers_retain_object_provenance(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "source.csv"
    parquet_path = tmp_path / "source.parquet"
    csv_path.write_text("value\n1\n")
    pl.DataFrame({"value": [1]}).write_parquet(parquet_path)

    csv_result = read_source_pattern(
        str(csv_path),
        get_source("atlanta").config.patterns[0],
    ).collect()
    parquet_result = read_source_pattern(
        str(parquet_path),
        get_source("baltimore").config.patterns[0],
    ).collect()

    assert csv_result["__landing_object_uri"].item() == str(csv_path)
    assert parquet_result["__landing_object_uri"].item() == str(parquet_path)


def test_active_architecture_has_no_category_registry_or_legacy_import() -> None:
    crime_root = Path(__file__).parents[1] / "src" / "crimenet_data" / "crime"
    active_files = [
        path
        for path in crime_root.rglob("*.py")
        if "legacy" not in path.relative_to(crime_root).parts
    ]
    active_text = "\n".join(path.read_text() for path in active_files)
    assert "crime.legacy" not in active_text
    assert "LEGACY_ADAPTERS" not in active_text
    assert "HIGH_ROI_ADAPTERS" not in active_text
    assert not (crime_root / "sources" / "high_roi.py").exists()
    assert not (crime_root / "sources" / "legacy.py").exists()


def test_all_sources_begin_in_landing_and_preserve_output_roots() -> None:
    lake = local_lake()
    for source_key in SOURCE_KEYS:
        assert lake.source_root(source_key).endswith(f"/raw_files/landing/{source_key}")
        assert lake.resolve_source_path(source_key, "bronze").endswith(
            f"/bronze/crime/{source_key}"
        )
        assert lake.resolve_source_path(source_key, "silver").endswith(
            f"/silver/crime/{source_key}"
        )


def test_bronze_has_one_normalized_provenance_contract() -> None:
    result = bronze("atlanta", pl.DataFrame([SOURCE_SAMPLES["atlanta"]])).collect()
    assert result["source_city"].item() == "atlanta"
    assert result["source_file_uri"].item() == "/landing/atlanta/sample"
    assert result["ingestion_run_id"].item() == "test-run"
    assert result.schema["ingested_at_utc"] == pl.Datetime("us", "UTC")
    assert "_source_file_uri" not in result.columns
    assert "_ingestion_run_id" not in result.columns


@pytest.mark.parametrize("source_key", sorted(SOURCE_SAMPLES))
def test_source_adapters_share_one_projection_contract(source_key: str) -> None:
    source = get_source(source_key)
    prepared = bronze(source_key, pl.DataFrame([SOURCE_SAMPLES[source_key]]))
    result = source.adapt_to_silver(prepared, AdapterContext()).collect()
    assert result.schema == SOURCE_PROJECTION_SCHEMA
    assert result.height == 1
    assert result["source_record_id"].item() is not None
    assert result["occurrence_year"].item() == 2024
    canonical = build_silver(
        prepared,
        local_lake().get_crosswalk_fixture(),
        source_key=source_key,
        adapter_context=AdapterContext(),
    ).collect()
    assert canonical.schema == CANONICAL_CRIME_SCHEMA
    assert "occurrence_end_timestamp" not in canonical.columns


def test_chandler_cp1252_reader_preserves_schema_and_event_end(tmp_path: Path) -> None:
    path = tmp_path / "general_offenses.csv"
    values = {column: "" for column in CHANDLER_COLUMNS}
    values.update(
        {
            "id": "1",
            "report_id": "10",
            "report_event_date_time": "2024-01-02 03:04:00",
            "report_event_end_date_time": "2024-01-02 04:05:00",
            "report_latitude": "33.30",
            "report_longitude": "-111.84",
            "report_primary_offense_description": "Victim’s property",
        }
    )
    body = ",".join(CHANDLER_COLUMNS) + "\n" + ",".join(values.values()) + "\n"
    path.write_bytes(body.encode("cp1252"))

    source = get_source("chandler_az")
    read = read_source_pattern(str(path), source.config.patterns[0])
    result = source.prepare_bronze(read).collect()

    assert set(CHANDLER_COLUMNS) <= set(result.columns)
    assert result["report_event_end_date_time"].item() == "2024-01-02 04:05:00"
    assert result["report_primary_offense_description"].item() == "Victim’s property"
    assert result.schema["report_latitude"] == pl.Float64
    assert result.schema["report_longitude"] == pl.Float64


def test_denver_geojson_flattens_features_and_retains_geometry(tmp_path: Path) -> None:
    path = tmp_path / "part.geojson"
    properties = dict(SOURCE_SAMPLES["denver"])
    properties.pop("geometry_type")
    properties.pop("geometry_json")
    second = dict(properties)
    second["OBJECTID"] = 2
    second["OFFENSE_ID"] = "O2"
    second["LAST_OCCURRENCE_DATE"] = 1704168245000
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": {"type": "Point", "coordinates": [-104.99, 39.74]},
                    },
                    {
                        "type": "Feature",
                        "properties": second,
                        "geometry": {"type": "Point", "coordinates": [-104.98, 39.75]},
                    },
                ],
            }
        )
    )

    source = get_source("denver")
    result = source.prepare_bronze(
        read_source_pattern(str(path), source.config.patterns[0])
    ).collect()

    assert result.height == 2
    assert result.schema["last_occurrence_date"] == pl.Int64
    assert result["last_occurrence_date"].null_count() == 1
    assert result["geometry_type"].to_list() == ["Point", "Point"]
    assert result["geometry_json"].str.contains("coordinates").all()
    assert result.schema["geo_lat"] == pl.Float64
    assert result.schema["geo_lon"] == pl.Float64
    timestamp = source.occurrence_timestamp(result.lazy()).first().alias("ts")
    assert result.lazy().select(timestamp).collect()["ts"].item().year == 2024


def test_montgomery_malformed_row_is_repaired_deterministically(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "part.csv"
    values = {column: "" for column in MONTGOMERY_COLUMNS}
    values.update(
        {
            "incident_id": "I1",
            "start_date": "2024-01-02 03:04:00",
            "crimename1": 'THEFT "FROM AUTO", WITH FORCE',
            "latitude": "39.1",
            "longitude": "-77.2",
        }
    )
    path.write_text(
        ",".join(MONTGOMERY_COLUMNS) + "\n" + ",".join(values.values()) + "\n"
    )

    source = get_source("montgomery_county_md")
    with caplog.at_level("WARNING"):
        result = read_source_pattern(str(path), source.config.patterns[0]).collect()

    assert result.height == 1
    assert result["crimename1"].item() == 'THEFT "FROM AUTO", WITH FORCE'
    assert result["latitude"].item() == "39.1"
    assert "csv_records_repaired" in caplog.text


def test_dallas_adapter_requires_context_and_preserves_spatial_transform() -> None:
    lake = local_lake()
    source = get_source("dallas")
    prepared = prepare_bronze_source(
        lake.get_source_fixture("dallas").head(3),
        source,
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="AdapterContext.duckdb"):
        source.adapt_to_silver(prepared, AdapterContext())

    with DuckDBResource(enable_spatial=True).get_connection() as connection:
        result = source.adapt_to_silver(
            prepared,
            AdapterContext(duckdb=connection),
        ).collect()
    transformed = result.filter(pl.col("latitude").is_not_null())
    assert transformed.height == 2
    assert transformed["latitude"].is_between(32.6, 33.1).all()
    assert transformed["longitude"].is_between(-97.1, -96.4).all()
    with DuckDBResource(enable_spatial=True).get_connection() as connection:
        canonical = build_silver(
            prepared,
            lake.get_crosswalk_fixture(),
            source_key="dallas",
            adapter_context=AdapterContext(duckdb=connection),
        ).collect()
    assert canonical.schema == CANONICAL_CRIME_SCHEMA
    registry_text = (
        Path(__file__).parents[1] / "src/crimenet_data/assets/crime/sources/registry.py"
    ).read_text()
    assert 'source_key == "dallas"' not in registry_text


def test_new_york_split_date_time_composes_exactly() -> None:
    source = get_source("new_york")
    result = source.adapt_to_silver(
        bronze("new_york", pl.DataFrame([SOURCE_SAMPLES["new_york"]])),
        AdapterContext(),
    ).collect()
    timestamp = result["occurrence_timestamp"].item()
    assert (timestamp.hour, timestamp.minute, timestamp.second) == (3, 4, 0)


def test_gainesville_preserves_hour_level_truth() -> None:
    source = get_source("gainesville_fl")
    result = source.adapt_to_silver(
        bronze("gainesville_fl", pl.DataFrame([SOURCE_SAMPLES["gainesville_fl"]])),
        AdapterContext(),
    ).collect()
    timestamp = result["occurrence_timestamp"].item()
    assert (timestamp.hour, timestamp.minute, timestamp.second) == (3, 0, 0)


def test_sonoma_does_not_fabricate_coordinates() -> None:
    source = get_source("sonoma_county_sheriff_ca")
    result = source.adapt_to_silver(
        bronze(
            "sonoma_county_sheriff_ca",
            pl.DataFrame([SOURCE_SAMPLES["sonoma_county_sheriff_ca"]]),
        ),
        AdapterContext(),
    ).collect()
    assert result["latitude"].item() is None
    assert result["longitude"].item() is None


@pytest.mark.parametrize("source_key", FIXTURE_SOURCES)
def test_established_sources_preserve_exact_canonical_schema(source_key: str) -> None:
    lake = local_lake()
    source = get_source(source_key)
    prepared = prepare_bronze_source(
        lake.get_source_fixture(source_key),
        source,
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    result = build_silver(
        prepared,
        lake.get_crosswalk_fixture(),
        source_key=source_key,
        adapter_context=AdapterContext(),
    ).collect()
    assert result.schema == CANONICAL_CRIME_SCHEMA
    assert "occurrence_end_timestamp" not in result.columns
    assert result.height > 0


def test_b2_configuration_reports_missing_variable_names_only(monkeypatch) -> None:
    for name in ("B2_ENDPOINT_URL", "B2_KEY_ID", "B2_APPLICATION_KEY", "B2_REGION"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as error:
        _ = CrimeLakeResources().storage_options
    assert "B2_APPLICATION_KEY" in str(error.value)
    assert "B2_ENDPOINT_URL" in str(error.value)
