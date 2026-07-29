from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests

import crimenet.boundaries.tiger_line as tiger_line
from crimenet.boundaries.tiger_line import (
    BOUNDARY_DEFINITION_VERSION,
    BoundaryIssue,
    TigerArchive,
    TigerArchiveError,
    boundary_issues_to_dataframe,
    boundary_record_id,
    build_tiger_tract_url,
    create_boundary_dataframe,
    download_tiger_archive,
    land_tiger_archives,
    normalization_failure_issue,
    normalize_tiger_archive,
    tiger_tract_landing_path,
    validate_boundary_dataframe,
    validate_state_fips,
    validate_tiger_archive,
)


def _archive_bytes(
    *,
    year: int,
    state_fips: str,
    omit_extension: str | None = None,
) -> bytes:
    buffer = io.BytesIO()
    stem = f"tl_{year}_{state_fips}_tract"
    with zipfile.ZipFile(buffer, mode="w") as archive:
        for extension in (".shp", ".shx", ".dbf", ".prj"):
            if extension == omit_extension:
                continue
            archive.writestr(
                f"{stem}{extension}",
                f"{stem}:{extension}".encode(),
            )
    return buffer.getvalue()


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status_code: int = 200,
        stream_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.stream_error = stream_error
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(
                f"HTTP {self.status_code}",
            )
            error.response = self  # type: ignore[assignment]
            raise error

    def iter_content(self, *, chunk_size: int) -> Any:
        del chunk_size
        midpoint = max(1, len(self.payload) // 2)
        yield self.payload[:midpoint]
        if self.stream_error is not None:
            raise self.stream_error
        yield self.payload[midpoint:]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.request_count = 0
        self.closed = False

    def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
        response = self.responses[self.request_count]
        self.request_count += 1
        return response

    def close(self) -> None:
        self.closed = True


def test_tiger_url_and_landing_path_are_deterministic(
    tmp_path: Path,
) -> None:
    url = build_tiger_tract_url(year=2024, state_fips="48")
    path = tiger_tract_landing_path(
        tmp_path,
        year=2024,
        state_fips="48",
    )

    assert url.endswith("/TIGER2024/TRACT/tl_2024_48_tract.zip")
    assert path.relative_to(tmp_path).as_posix() == (
        "tiger_line/tract/year=2024/state=48/tl_2024_48_tract.zip"
    )
    with pytest.raises(ValueError, match="two digits"):
        validate_state_fips("8")


def test_boundary_record_id_changes_with_source_or_definition() -> None:
    common = {
        "year": 2024,
        "state_fips": "48",
        "geoid": "48113000100",
    }
    first = boundary_record_id(
        **common,
        source_archive_sha256="a" * 64,
    )
    replay = boundary_record_id(
        **common,
        source_archive_sha256="a" * 64,
    )
    corrected_source = boundary_record_id(
        **common,
        source_archive_sha256="b" * 64,
    )
    revised_definition = boundary_record_id(
        **common,
        source_archive_sha256="a" * 64,
        definition_version="tiger_line_tract_wgs84_v2",
    )

    assert first == replay
    assert corrected_source != first
    assert revised_definition != first


def test_download_is_atomic_and_reuses_only_valid_cache(
    tmp_path: Path,
) -> None:
    response = FakeResponse(_archive_bytes(year=2024, state_fips="48"))
    session = FakeSession([response])

    first = download_tiger_archive(
        year=2024,
        state_fips="48",
        landing_directory=tmp_path,
        session=session,  # type: ignore[arg-type]
    )
    second = download_tiger_archive(
        year=2024,
        state_fips="48",
        landing_directory=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    assert first.sha256 == second.sha256
    assert first.archive_id == second.archive_id
    assert session.request_count == 1
    assert response.closed is True
    manifest = Path(f"{first.path}.manifest.json")
    assert json.loads(manifest.read_text())["sha256"] == first.sha256
    assert not list(first.path.parent.glob("*.part-*"))


def test_interrupted_download_never_replaces_the_final_path(
    tmp_path: Path,
) -> None:
    response = FakeResponse(
        _archive_bytes(year=2024, state_fips="48"),
        stream_error=requests.ConnectionError("connection lost"),
    )
    session = FakeSession([response])
    destination = tiger_tract_landing_path(
        tmp_path,
        year=2024,
        state_fips="48",
    )

    with pytest.raises(
        tiger_line.TigerDownloadError,
        match="response stream failed",
    ):
        download_tiger_archive(
            year=2024,
            state_fips="48",
            landing_directory=tmp_path,
            session=session,  # type: ignore[arg-type]
        )

    assert not destination.exists()
    assert not list(destination.parent.glob("*.part-*"))


def test_archive_validation_rejects_missing_sidecar(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tl_2024_48_tract.zip"
    path.write_bytes(
        _archive_bytes(
            year=2024,
            state_fips="48",
            omit_extension=".prj",
        )
    )

    with pytest.raises(TigerArchiveError, match=r"\.prj"):
        validate_tiger_archive(
            path,
            year=2024,
            state_fips="48",
        )


def test_landing_collects_404_as_auditable_issue(
    tmp_path: Path,
) -> None:
    session = FakeSession(
        [
            FakeResponse(
                b"not found",
                status_code=404,
            )
        ]
    )
    result = land_tiger_archives(
        years=[2024],
        state_fips="48",
        landing_directory=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    assert result.archives == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "TIGER_ARCHIVE_NOT_FOUND"
    assert len(result.issues[0].quarantine_id) == 64


def test_quarantine_identity_ignores_paths_and_endpoint_mirrors(
    tmp_path: Path,
) -> None:
    official = land_tiger_archives(
        years=[2024],
        state_fips="48",
        landing_directory=tmp_path / "official",
        base_url="https://www2.census.gov/geo/tiger",
        session=FakeSession([FakeResponse(b"not found", status_code=404)]),  # type: ignore[arg-type]
    ).issues[0]
    mirror = land_tiger_archives(
        years=[2024],
        state_fips="48",
        landing_directory=tmp_path / "mirror",
        base_url="https://mirror.example.test/tiger",
        session=FakeSession([FakeResponse(b"not found", status_code=404)]),  # type: ignore[arg-type]
    ).issues[0]
    first_archive = TigerArchive(
        year=2024,
        state_fips="48",
        source_url="https://one.example/archive.zip",
        path=tmp_path / "one" / "archive.zip",
        sha256="a" * 64,
        size_bytes=100,
        archive_id="b" * 64,
    )
    moved_archive = TigerArchive(
        year=2024,
        state_fips="48",
        source_url="https://two.example/archive.zip",
        path=tmp_path / "two" / "renamed.zip",
        sha256="a" * 64,
        size_bytes=100,
        archive_id="c" * 64,
    )

    assert official.quarantine_id == mirror.quarantine_id
    assert (
        normalization_failure_issue(
            first_archive,
            RuntimeError("bad archive"),
        ).quarantine_id
        == normalization_failure_issue(
            moved_archive,
            RuntimeError("bad archive"),
        ).quarantine_id
    )


@dataclass
class FakeGeometry:
    wkt: str
    is_empty: bool = False
    is_valid: bool = True


class FakeGeometryColumn:
    name = "geometry"


class FakeGeoDataFrame:
    columns = ("GEOID", "NAME", "geometry")
    geometry = FakeGeometryColumn()
    crs = "EPSG:4269"
    empty = False

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.requested_epsg: int | None = None

    def to_crs(self, *, epsg: int) -> FakeGeoDataFrame:
        self.requested_epsg = epsg
        return self

    def iterrows(self) -> Any:
        return iter(enumerate(self.rows))


class FakeGeoPandas:
    def __init__(self, dataframe: FakeGeoDataFrame) -> None:
        self.dataframe = dataframe

    def read_file(self, _path: str) -> FakeGeoDataFrame:
        return self.dataframe


def test_normalization_preserves_valid_rows_and_quarantines_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataframe = FakeGeoDataFrame(
        [
            {
                "GEOID": "48113000100",
                "NAME": "1",
                "geometry": FakeGeometry("POLYGON ((0 0, 1 0, 0 0))"),
            },
            {
                "GEOID": "not-a-geoid",
                "NAME": "bad",
                "geometry": FakeGeometry("POLYGON EMPTY"),
            },
            {
                "GEOID": "06113000100",
                "NAME": "wrong state",
                "geometry": FakeGeometry("POLYGON ((0 0, 1 0, 0 0))"),
            },
            {
                "GEOID": "48113000200",
                "NAME": "empty",
                "geometry": FakeGeometry(
                    "POLYGON EMPTY",
                    is_empty=True,
                ),
            },
            {
                "GEOID": "48113000300",
                "NAME": "invalid",
                "geometry": FakeGeometry(
                    "POLYGON ((0 0, 1 0, 0 0))",
                    is_valid=False,
                ),
            },
            {
                "GEOID": "48113000400",
                "NAME": "duplicate one",
                "geometry": FakeGeometry("POLYGON ((0 0, 1 0, 0 0))"),
            },
            {
                "GEOID": "48113000400",
                "NAME": "duplicate two",
                "geometry": FakeGeometry("POLYGON ((0 0, 2 0, 0 0))"),
            },
        ]
    )
    monkeypatch.setattr(
        tiger_line,
        "_import_geopandas",
        lambda: FakeGeoPandas(dataframe),
    )
    archive = TigerArchive(
        year=2024,
        state_fips="48",
        source_url="https://example.test/archive.zip",
        path=tmp_path / "archive.zip",
        sha256="a" * 64,
        size_bytes=100,
        archive_id="b" * 64,
    )

    result = normalize_tiger_archive(
        archive,
        definition_version=BOUNDARY_DEFINITION_VERSION,
    )

    assert dataframe.requested_epsg == 4326
    assert [record["geoid"] for record in result.records] == ["48113000100"]
    assert len(result.records[0]["boundary_record_id"]) == 64
    assert {issue.reason_code for issue in result.issues} == {
        "INVALID_TRACT_GEOID",
        "TRACT_STATE_MISMATCH",
        "EMPTY_TRACT_GEOMETRY",
        "INVALID_TRACT_GEOMETRY",
        "DUPLICATE_TRACT_GEOID",
    }


@pytest.fixture(scope="module")
def boundary_spark() -> object:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import IntegerType, StringType

    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-tiger-boundaries")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.udf.register(
        "ST_GeomFromWKT",
        lambda wkt, _srid: wkt,
        StringType(),
    )
    session.udf.register(
        "ST_SRID",
        lambda geometry: None if geometry == "UNKNOWN_SRID" else 4326,
        IntegerType(),
    )
    yield session
    session.stop()


def test_boundary_dataframe_contract_and_quarantine_schema(
    boundary_spark: object,
) -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    assert isinstance(boundary_spark, SparkSession)
    record = {
        "geoid": "48113000100",
        "boundary_vintage": 2024,
        "state_fips": "48",
        "wkt_geometry": "POLYGON ((0 0, 1 0, 0 0))",
        "boundary_definition_version": (BOUNDARY_DEFINITION_VERSION),
        "source_archive_sha256": "a" * 64,
        "source_archive_id": "b" * 64,
        "boundary_record_id": "c" * 64,
    }
    dataframe = create_boundary_dataframe(
        boundary_spark,
        [record],
    )

    validate_boundary_dataframe(
        dataframe,
        expected_years=[2024],
        state_fips="48",
        minimum_tracts_per_vintage=1,
    )
    assert dataframe.first()["tract_geometry"].startswith("POLYGON")

    for column_name, invalid_value in (
        ("geoid", None),
        ("state_fips", None),
        ("boundary_record_id", None),
        ("source_archive_sha256", None),
        ("tract_geometry", "UNKNOWN_SRID"),
    ):
        invalid_dataframe = dataframe.withColumn(
            column_name,
            F.lit(invalid_value).cast(
                dataframe.schema[column_name].dataType
            ),
        )
        with pytest.raises(ValueError, match="invalid GEOID"):
            validate_boundary_dataframe(
                invalid_dataframe,
                expected_years=[2024],
                state_fips="48",
                minimum_tracts_per_vintage=1,
            )

    issue = BoundaryIssue(
        quarantine_id="d" * 64,
        reason_code="INVALID_TRACT_GEOID",
        reason="bad geoid",
        source_file="/landing/archive.zip",
        source_row_hash="e" * 64,
        raw_payload='{"GEOID":"bad"}',
        boundary_vintage=2024,
        state_fips="48",
        geoid="bad",
        source_archive_sha256="a" * 64,
    )
    quarantine = boundary_issues_to_dataframe(
        boundary_spark,
        [issue, issue],
        pipeline_run_id="run_1",
    )
    quarantined = quarantine.first()

    assert quarantine.count() == 1
    assert quarantined["pipeline_run_id"] == "run_1"
    assert quarantined["boundary_definition_version"] == BOUNDARY_DEFINITION_VERSION
