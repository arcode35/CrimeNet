from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from crimenet_data.assets.crime.canonical import (
    CANONICAL_CRIME_SCHEMA,
    CANONICAL_MAPPING_VERSION,
    SOURCE_PROJECTION_SCHEMA,
    normalize_crosswalk_nulls,
    validate_canonical_crosswalk,
)
from crimenet_data.assets.crime.ingestion import (
    prepare_bronze_source,
    read_source_pattern,
)
from crimenet_data.assets.crime.silver import (
    adapt_silver_source,
    build_silver,
    build_unified_silver,
    crime_silver_assets,
    silver_mapping_summary,
    silver_unmapped_key_counts,
    validate_montgomery_bronze_row_count,
)
from crimenet_data.assets.crime.sources import (
    SILVER_SOURCE_KEYS,
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

EXPECTED_SILVER_CROSSWALK_KEYS = {
    "atlanta": ("source_offense_description",),
    "baltimore": ("source_offense_code", "source_offense_description"),
    "chandler_az": ("source_offense_code",),
    "chicago": ("source_offense_category", "source_offense_description"),
    "dallas": ("source_offense_category", "source_offense_description"),
    "denver": ("source_offense_description",),
    "fort_worth": ("source_offense_code", "source_offense_description"),
    "los_angeles_county_sheriff": ("source_offense_description",),
    "marin_county_sheriff_ca": ("source_offense_description",),
    "montgomery_county_md": (
        "source_offense_code",
        "source_offense_category",
        "source_offense_description",
        "source_auxiliary",
    ),
    "new_york": ("source_offense_category", "source_offense_description"),
    "san_francisco": (
        "source_offense_category",
        "source_offense_description",
    ),
    "seattle": ("source_offense_description",),
    "sonoma_county_sheriff_ca": ("source_offense_description",),
    "washington_dc": ("source_offense_category",),
}


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
        "report_number": "1",
        "occurred_from_date": "1704182640000",
        "report_date": "1704272400000",
        "nibrs_ucr_code": "13A",
        "crime_against": "Person",
        "nibrs_offense": "Aggravated Assault",
        "nibrs_bucket": "Aggravated Assault",
        "part": "Part I",
        "geometry_y": "33.75",
        "geometry_x": "-84.39",
        "street_address": "Main",
        "beat": "1",
        "nhood_name": "N",
    },
    "baltimore": {
        "RowID": "1",
        "CrimeDateTime": "1704164640000",
        "CrimeCode": "26B",
        "Description": "FRAUD",
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
        "report_primary_offense_code": "0001-0",
        "report_primary_offense_description": "Death Investigation",
        "report_summary_offense_description": "OTHER",
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
        "iucr": "0110",
        "primary_type": "ARSON",
        "description": "AGGRAVATED",
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
        "OFFENSE_CODE": "0902",
        "OFFENSE_CATEGORY_ID": "murder",
        "OFFENSE_TYPE_ID": "homicide-family",
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
        "offense": "EDU",
        "offense_desc": "000000 School Offense-class c",
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
        "CATEGORY": "CRIMINAL HOMICIDE",
        "STAT": "011",
        "STAT_DESC": "CRIMINAL HOMICIDE: Murder",
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
        "crime": "243E1",
        "crime_class": "",
        "incident_street_address": "Main",
        "incident_city_town": "Marin",
        "jurisdiction": "Sheriff",
        "latitude": "38.0",
        "longitude": "-122.5",
    },
    "montgomery_county_md": {
        "incident_id": "I1",
        "case_number": "C1",
        "offence_code": "0999",
        "nibrs_code": "09A",
        "date": "2024-01-03",
        "start_date": "2024-01-02 03:04:00",
        "end_date": "2024-01-02 04:00:00",
        "crimename1": "Crime Against Person",
        "crimename2": "Murder and Nonnegligent Manslaughter",
        "crimename3": "HOMICIDE (DESCRIBE OFFENSE)",
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
        "ofns_desc": "(null)",
        "pd_desc": "CRIM POS WEAP 4",
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
        "incident_category": "Arson",
        "incident_description": "Arson",
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
        "nibrs_offense_code": "90Z",
        "offense_category": "ALL OTHER",
        "nibrs_offense_code_description": "-",
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
        "incident_type": "Aggravated Assault",
        "location_type": "Street",
        "city": "Sonoma",
        "location": "(38.511152, -122.781156)",
        "agency": "Sheriff",
    },
    "washington_dc": {
        "OBJECTID": "1",
        "START_DATE": "2024-01-02 03:04:00",
        "occurred_at_raw": "1704164640000",
        "END_DATE": "2024-01-02 04:00:00",
        "REPORT_DAT": "2024-01-03",
        "OFFENSE": "ARSON",
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


def test_silver_registry_and_crosswalk_keys_are_authoritative() -> None:
    assert SILVER_SOURCE_KEYS == tuple(EXPECTED_SILVER_CROSSWALK_KEYS)
    assert {
        source_key: get_source(source_key).config.crosswalk_keys
        for source_key in SILVER_SOURCE_KEYS
    } == EXPECTED_SILVER_CROSSWALK_KEYS
    assert len(crime_silver_assets) == 1
    assert crime_silver_assets[0].key.to_user_string() == "silver_crime_offenses"


def test_v15_crosswalk_defaults_and_invariants() -> None:
    remote_lake = CrimeLakeResources()
    local_lake_resource = local_lake()

    assert remote_lake.canonical_crosswalk_uri == (
        "s3://crimenet-data/raw_files/landing/reference/"
        "canonical_crime_crosswalk_v1_5.csv"
    )
    assert remote_lake.silver_crime_offenses_uri == (
        "s3://crimenet-data/silver/crime_offenses"
    )
    crosswalk = validate_canonical_crosswalk(
        local_lake_resource.get_crosswalk_fixture()
    )
    assert crosswalk["mapping_version"].unique().to_list() == [
        CANONICAL_MAPPING_VERSION
    ]
    assert set(crosswalk["source_city"]) == set(SILVER_SOURCE_KEYS)
    assert not crosswalk.select(
        pl.any_horizontal(
            pl.col("source_offense_code") == "null",
            pl.col("source_offense_category") == "null",
            pl.col("source_offense_description") == "null",
            pl.col("source_auxiliary") == "null",
            pl.col("source_severity") == "null",
        ).any()
    ).item()

    local_path = (
        Path(__file__).parents[1]
        / "src/crimenet_data/artifacts/canonical_crime_crosswalk_v1_5.csv"
    )
    overridden = CrimeLakeResources(crosswalk_path=str(local_path))
    assert validate_canonical_crosswalk(overridden.resolve_crosswalk()).height == 7760


def test_v15_crosswalk_rejects_duplicate_source_keys() -> None:
    crosswalk = local_lake().get_crosswalk_fixture().collect()
    atlanta = crosswalk.filter(pl.col("source_city") == "atlanta")
    duplicate = pl.concat([crosswalk, atlanta.head(1)])

    with pytest.raises(ValueError, match="not unique for 'atlanta'"):
        validate_canonical_crosswalk(duplicate.lazy(), source_keys=("atlanta",))


def test_crosswalk_literal_nulls_are_normalized_exactly() -> None:
    crosswalk = pl.DataFrame(
        {
            "source_offense_code": ["null", "NULL", " null "],
            "source_offense_category": ["null", "value", None],
            "source_offense_description": ["value", "null", "value"],
            "source_auxiliary": [None, None, "null"],
            "source_severity": ["value", "value", "null"],
        }
    )

    normalized = normalize_crosswalk_nulls(crosswalk.lazy()).collect()

    assert normalized.row(0) == (None, None, "value", None, "value")
    assert normalized.row(1) == ("NULL", "value", None, None, "value")
    assert normalized.row(2) == (" null ", None, "value", None, None)


def test_v15_crosswalk_allows_unique_null_key_components() -> None:
    crosswalk = local_lake().get_crosswalk_fixture().collect()
    dallas = crosswalk.filter(
        (pl.col("source_city") == "dallas")
        & pl.col("source_offense_category").is_null()
    )
    assert dallas.height == 614

    validated = validate_canonical_crosswalk(
        crosswalk.lazy(), source_keys=("dallas",)
    )
    assert validated.filter(
        (pl.col("source_city") == "dallas")
        & pl.col("source_offense_category").is_null()
    ).height == 614

    duplicate = pl.concat([crosswalk, dallas.head(1)])
    with pytest.raises(ValueError, match="not unique for 'dallas'"):
        validate_canonical_crosswalk(duplicate.lazy(), source_keys=("dallas",))


def test_v15_crosswalk_rejects_other_or_missing_versions() -> None:
    crosswalk = local_lake().get_crosswalk_fixture().collect()
    for invalid_version in ("crime_canonical_v1_3", None):
        invalid = crosswalk.with_columns(
            pl.lit(invalid_version, dtype=pl.String).alias("mapping_version")
        )
        with pytest.raises(ValueError, match="version mismatch"):
            validate_canonical_crosswalk(invalid.lazy(), source_keys=("atlanta",))


def test_known_stale_montgomery_bronze_is_blocked_before_silver() -> None:
    validate_montgomery_bronze_row_count(505_251)
    with pytest.raises(RuntimeError, match="known stale 1,515,753-row snapshot"):
        validate_montgomery_bronze_row_count(1_515_753)


@pytest.mark.parametrize(
    "source_key",
    [key for key in SILVER_SOURCE_KEYS if key != "dallas"],
)
def test_every_sampled_silver_adapter_joins_v15(source_key: str) -> None:
    result = build_silver(
        bronze(source_key, pl.DataFrame([SOURCE_SAMPLES[source_key]])),
        local_lake().get_crosswalk_fixture(),
        source_key=source_key,
        adapter_context=AdapterContext(),
    ).collect()

    assert result.schema == CANONICAL_CRIME_SCHEMA
    assert result.height == 1
    assert result["canonical_mapping_found"].item()
    assert result["mapping_version"].item() == CANONICAL_MAPPING_VERSION


def test_fort_worth_adapter_preserves_crosswalk_key_fields() -> None:
    lake = local_lake()
    source = get_source("fort_worth")
    prepared = prepare_bronze_source(
        lake.get_source_fixture("fort_worth"),
        source,
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    adapted = source.adapt_to_silver(prepared, AdapterContext()).collect()
    raw = prepared.select("offense", "offense_desc").collect()

    assert adapted["source_offense_code"].to_list() == raw["offense"].cast(
        pl.String
    ).to_list()
    assert adapted["source_offense_description"].to_list() == raw[
        "offense_desc"
    ].cast(pl.String).to_list()
    assert adapted["source_offense_category"].null_count() == adapted.height
    assert adapted["source_auxiliary"].null_count() == adapted.height
    assert adapted["source_severity"].null_count() == adapted.height


def test_fort_worth_all_taxonomy_keys_exist_in_v15() -> None:
    lake = local_lake()
    source = get_source("fort_worth")
    prepared = prepare_bronze_source(
        lake.get_source_fixture("fort_worth"),
        source,
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    adapted = source.adapt_to_silver(prepared, AdapterContext())
    crosswalk = (
        lake.get_crosswalk_fixture()
        .filter(pl.col("source_city") == "fort_worth")
        .select("source_offense_code", "source_offense_description")
        .unique()
    )

    missing = (
        adapted.select(
            "source_offense_code", "source_offense_description"
        )
        .unique()
        .join(
            crosswalk,
            on=["source_offense_code", "source_offense_description"],
            how="anti",
            nulls_equal=True,
        )
        .collect()
    )

    assert missing.is_empty(), missing


def test_montgomery_v15_taxonomy_projection_is_exact() -> None:
    source = get_source("montgomery_county_md")
    result = source.adapt_to_silver(
        bronze(
            "montgomery_county_md",
            pl.DataFrame([SOURCE_SAMPLES["montgomery_county_md"]]),
        ),
        AdapterContext(),
    ).collect()

    assert result.select(
        "source_offense_code",
        "source_offense_category",
        "source_offense_description",
        "source_auxiliary",
        "source_severity",
    ).row(0) == (
        "09A",
        "Crime Against Person",
        "Murder and Nonnegligent Manslaughter",
        "HOMICIDE (DESCRIBE OFFENSE)",
        "0999",
    )


def test_mapping_dispositions_and_unmapped_rows_remain_auditable() -> None:
    crosswalk = local_lake().get_crosswalk_fixture()
    expected_actions = {
        "atlanta": ("map", True),
        "chandler_az": ("exclude_non_criminal", False),
        "seattle": ("drop", False),
    }
    for source_key, expected in expected_actions.items():
        result = build_silver(
            bronze(source_key, pl.DataFrame([SOURCE_SAMPLES[source_key]])),
            crosswalk,
            source_key=source_key,
            adapter_context=AdapterContext(),
        ).collect()
        assert result.select("mapping_action", "include_in_model").row(0) == expected
        assert result["canonical_mapping_found"].item()

    unknown = dict(SOURCE_SAMPLES["atlanta"])
    unknown["nibrs_offense"] = "NOT PRESENT IN V1.5"
    unmapped = build_silver(
        bronze("atlanta", pl.DataFrame([unknown])),
        crosswalk,
        source_key="atlanta",
        adapter_context=AdapterContext(),
    ).collect()
    assert unmapped.height == 1
    assert unmapped["crime_id"].item() == "atlanta:1:13A"
    assert not unmapped["canonical_mapping_found"].item()
    assert unmapped["mapping_version"].item() is None
    assert unmapped["canonical_subtype_code"].item() is None
    assert unmapped["mapping_action"].item() is None
    assert silver_mapping_summary(unmapped.lazy(), "atlanta").collect()[
        "unexpected_unmapped_rows"
    ].item() == 1

    blank_sonoma = dict(SOURCE_SAMPLES["sonoma_county_sheriff_ca"])
    blank_sonoma["incident_type"] = ""
    blank = build_silver(
        bronze("sonoma_county_sheriff_ca", pl.DataFrame([blank_sonoma])),
        crosswalk,
        source_key="sonoma_county_sheriff_ca",
        adapter_context=AdapterContext(),
    ).collect()
    assert not blank["canonical_mapping_found"].item()
    assert silver_mapping_summary(
        blank.lazy(), "sonoma_county_sheriff_ca"
    ).collect()["unexpected_unmapped_rows"].item() == 0


def test_crosswalk_join_matches_null_key_components() -> None:
    sample = dict(SOURCE_SAMPLES["san_francisco"])
    sample["incident_category"] = None
    sample["incident_description"] = "Theft, Phone Booth, $200-$950"
    result = build_silver(
        bronze("san_francisco", pl.DataFrame([sample])),
        local_lake().get_crosswalk_fixture(),
        source_key="san_francisco",
        adapter_context=AdapterContext(),
    ).collect()

    assert result["source_offense_category"].item() is None
    assert result["canonical_mapping_found"].item()


def test_unmapped_key_diagnostic_uses_exact_source_key() -> None:
    unknown = dict(SOURCE_SAMPLES["fort_worth"])
    unknown["offense"] = "unexpected-code"
    unknown["offense_desc"] = "unexpected-description"
    unknown["nature_of_call"] = "must-not-be-used"
    result = build_silver(
        bronze("fort_worth", pl.DataFrame([unknown, unknown])),
        local_lake().get_crosswalk_fixture(),
        source_key="fort_worth",
        adapter_context=AdapterContext(),
    ).collect()
    counts = silver_unmapped_key_counts(
        result.lazy(), "fort_worth"
    ).collect()

    assert counts.to_dicts() == [
        {
            "source_offense_code": "unexpected-code",
            "source_offense_description": "unexpected-description",
            "row_count": 1,
        }
    ]


def test_unified_builder_has_one_schema_and_multiple_sources() -> None:
    unified = build_unified_silver(
        {
            source_key: bronze(
                source_key,
                pl.DataFrame([SOURCE_SAMPLES[source_key]]),
            )
            for source_key in ("atlanta", "seattle")
        },
        local_lake().get_crosswalk_fixture(),
    ).collect()

    assert unified.schema == CANONICAL_CRIME_SCHEMA
    assert set(unified["source_city"]) == {"atlanta", "seattle"}
    assert unified["crime_id"].to_list() == ["atlanta:1:13A", "seattle:1"]


def test_known_stale_montgomery_bronze_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="known stale 1,515,753-row snapshot"):
        validate_montgomery_bronze_row_count(1_515_753)
    validate_montgomery_bronze_row_count(505_251)


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


def test_all_sources_begin_in_landing_and_share_one_silver_output() -> None:
    lake = local_lake()
    for source_key in SOURCE_KEYS:
        assert lake.source_root(source_key).endswith(f"/raw_files/landing/{source_key}")
        assert lake.resolve_source_path(source_key, "bronze").endswith(
            f"/bronze/crime/{source_key}"
        )
        assert lake.resolve_source_path(source_key, "silver") == (
            lake.silver_crime_offenses_uri
        )
    assert lake.silver_crime_offenses_uri.endswith("/silver/crime_offenses")


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


def test_montgomery_unquoted_comma_row_is_rejected_without_guessing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "part.csv"
    valid_values = {column: "" for column in MONTGOMERY_COLUMNS}
    valid_values.update(
        {
            "incident_id": "valid",
            "start_date": "2024-01-02 03:04:00",
            "crimename1": "THEFT FROM AUTO",
            "latitude": "39.1",
            "longitude": "-77.2",
        }
    )
    malformed_values = dict(valid_values)
    malformed_values.update(
        {
            "incident_id": "malformed",
            "crimename1": 'THEFT "FROM AUTO", WITH FORCE',
        }
    )
    path.write_text(
        ",".join(MONTGOMERY_COLUMNS)
        + "\n"
        + ",".join(valid_values.values())
        + "\n"
        + ",".join(malformed_values.values())
        + "\n"
    )

    source = get_source("montgomery_county_md")
    with caplog.at_level("INFO"):
        result = read_source_pattern(str(path), source.config.patterns[0]).collect()

    assert result.height == 1
    assert result.select(
        "incident_id", "crimename1", "latitude", "longitude"
    ).row(0) == ("valid", "THEFT FROM AUTO", "39.1", "-77.2")
    summary = next(
        record
        for record in caplog.records
        if record.message == "csv_record_width_summary"
    )
    assert summary.total_records == 2
    assert summary.correct_width_records == 1
    assert summary.long_records == 1
    assert summary.repaired_records == 0
    assert summary.rejected_records == 1


def test_dallas_normalization_requires_context_and_transforms_coordinates() -> None:
    lake = local_lake()
    source = get_source("dallas")
    prepared = prepare_bronze_source(
        lake.get_source_fixture("dallas").head(3),
        source,
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # The source adapter is only a projection and does not own spatial setup.
    raw_projection = source.adapt_to_silver(prepared, AdapterContext()).collect()
    assert raw_projection["latitude"].is_null().all()
    assert raw_projection["longitude"].is_null().all()

    with pytest.raises(ValueError, match="Dallas normalization requires"):
        adapt_silver_source(
            prepared,
            source_key="dallas",
            adapter_context=AdapterContext(),
        )

    with DuckDBResource(enable_spatial=True).get_connection() as connection:
        expected_taxonomy = prepared.select(
            "nibrs_crime", "type_of_incident"
        ).collect()
        result = adapt_silver_source(
            prepared,
            source_key="dallas",
            adapter_context=AdapterContext(duckdb=connection),
        )
        result = result.collect()
    assert result.schema == SOURCE_PROJECTION_SCHEMA
    assert sorted(
        result.select(
            "source_offense_category", "source_offense_description"
        ).rows()
    ) == sorted(expected_taxonomy.rows())
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
    assert canonical["canonical_mapping_found"].all()
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


@pytest.mark.parametrize("source_key", (*FIXTURE_SOURCES, "dallas"))
def test_v15_does_not_reduce_original_fixture_mapping_coverage(
    source_key: str,
) -> None:
    lake = local_lake()
    source = get_source(source_key)
    prepared = prepare_bronze_source(
        lake.get_source_fixture(source_key),
        source,
        run_id="test-run",
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    v13 = pl.scan_csv(
        Path(__file__).parent
        / "fixtures/data/references/canonical_crime_crosswalk_v1_3.csv"
    )
    v15 = lake.get_crosswalk_fixture()

    def mapped_rows(
        crosswalk: pl.LazyFrame,
        context: AdapterContext,
    ) -> int:
        result = build_silver(
            prepared,
            crosswalk,
            source_key=source_key,
            adapter_context=context,
        )
        return result.select(pl.col("canonical_mapping_found").sum()).collect().item()

    if source_key == "dallas":
        with DuckDBResource(enable_spatial=True).get_connection() as connection:
            context = AdapterContext(duckdb=connection)
            assert mapped_rows(v15, context) >= mapped_rows(v13, context)
    else:
        context = AdapterContext()
        assert mapped_rows(v15, context) >= mapped_rows(v13, context)


def test_b2_configuration_reports_missing_variable_names_only(monkeypatch) -> None:
    for name in ("B2_ENDPOINT_URL", "B2_KEY_ID", "B2_APPLICATION_KEY", "B2_REGION"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as error:
        _ = CrimeLakeResources().storage_options
    assert "B2_APPLICATION_KEY" in str(error.value)
    assert "B2_ENDPOINT_URL" in str(error.value)
