from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from crimenet_data.assets.crime.ingestion.readers import read_source_pattern
from crimenet_data.assets.crime.sources.base import SourcePattern
from crimenet_data.assets.crime.sources.montgomery_county_md import (
    EXPECTED_COLUMNS as MONTGOMERY_COLUMNS,
)


def _pattern(columns: tuple[str, ...] = ("a", "b", "c")) -> SourcePattern:
    return SourcePattern(
        "*.csv",
        "csv",
        {
            "strategy": "python_tolerant",
            "expected_columns": columns,
        },
    )


def _read(path: Path):
    return read_source_pattern(str(path), _pattern()).collect()


def test_python_csv_reader_parses_ordinary_csv(tmp_path: Path) -> None:
    path = tmp_path / "ordinary.csv"
    path.write_text("a,b,c\n1,2,3\n")

    result = _read(path)

    assert result.select("a", "b", "c").row(0) == ("1", "2", "3")


def test_python_csv_reader_preserves_quoted_comma(tmp_path: Path) -> None:
    path = tmp_path / "quoted-comma.csv"
    path.write_text('a,b,c\n1,"foo,bar",3\n')

    result = _read(path)

    assert result.select("a", "b", "c").row(0) == ("1", "foo,bar", "3")


def test_python_csv_reader_preserves_multiline_field(tmp_path: Path) -> None:
    path = tmp_path / "multiline.csv"
    path.write_text('a,b,c\n1,"foo\nbar",3\n')

    result = _read(path)

    assert result.height == 1
    assert result.select("a", "b", "c").row(0) == ("1", "foo\nbar", "3")


def test_python_csv_reader_preserves_escaped_quote(tmp_path: Path) -> None:
    path = tmp_path / "escaped-quote.csv"
    path.write_text('a,b,c\n1,"foo ""bar"" baz",3\n')

    result = _read(path)

    assert result.select("a", "b", "c").row(0) == (
        "1",
        'foo "bar" baz',
        "3",
    )


def test_python_csv_reader_preserves_strict_header_validation(tmp_path: Path) -> None:
    path = tmp_path / "wrong-header.csv"
    path.write_text("a,c,b\n1,3,2\n")

    with pytest.raises(ValueError, match="CSV header mismatch"):
        _read(path)


@pytest.mark.parametrize(
    ("body", "width_kind", "width_delta"),
    [
        ("a,b,c\n1,2\n", "short_records", -1),
        ("a,b,c\n1,2,3,4\n", "long_records", 1),
    ],
)
def test_python_csv_reader_rejects_ragged_records_without_coercion(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    body: str,
    width_kind: str,
    width_delta: int,
) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text(body)

    with caplog.at_level("DEBUG"):
        result = _read(path)

    assert result.height == 0
    summary = next(
        record
        for record in caplog.records
        if record.message == "csv_record_width_summary"
    )
    assert summary.total_records == 1
    assert summary.correct_width_records == 0
    assert getattr(summary, width_kind) == 1
    assert summary.repaired_records == 0
    assert summary.rejected_records == 1
    assert summary.width_deltas == {width_delta: 1}
    examples = next(
        record.malformed_records
        for record in caplog.records
        if record.message == "csv_malformed_record_examples"
    )
    assert examples == [
        {
            "source_file_uri": str(path),
            "line_number": 2,
            "expected_width": 3,
            "actual_width": 3 + width_delta,
            "width_delta": width_delta,
            "raw_parsed_row": body.splitlines()[1].split(","),
        }
    ]


def test_python_csv_reader_bounds_malformed_debug_samples(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "many-ragged.csv"
    path.write_text("a,b,c\n" + "1,2\n" * 30)

    with caplog.at_level("DEBUG"):
        result = _read(path)

    assert result.height == 0
    summary = next(
        record
        for record in caplog.records
        if record.message == "csv_record_width_summary"
    )
    assert summary.total_records == 30
    assert summary.short_records == 30
    assert summary.rejected_records == 30
    examples = next(
        record.malformed_records
        for record in caplog.records
        if record.message == "csv_malformed_record_examples"
    )
    assert len(examples) == 25


def test_montgomery_csv_records_remain_semantically_aligned(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "montgomery.csv"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MONTGOMERY_COLUMNS)
    writer.writeheader()
    rows = [
        {
            "incident_id": "clean",
            "offence_code": "2308",
            "case_number": "C1",
            "start_date": "2024-01-02T03:04:00.000",
            "nibrs_code": "23D",
            "crimename1": "Crime Against Property",
            "crimename2": "Theft from Building",
            "crimename3": "LARCENY - FROM BLDG",
            "district": "BETHESDA",
            "location": "100 BLK MAIN ST",
            "city": "BETHESDA",
            "state": "MD",
            "latitude": "38.98",
            "longitude": "-77.08",
            "geolocation": "(38.98, -77.08)",
        },
        {
            "incident_id": "quoted-comma",
            "offence_code": "2699",
            "case_number": "C2",
            "start_date": "2024-02-03T04:05:00.000",
            "nibrs_code": "26A",
            "crimename1": "Crime Against Property",
            "crimename2": "False Pretenses, Swindle",
            "crimename3": "FRAUD (DESCRIBE OFFENSE)",
            "district": "ROCKVILLE",
            "location": "1 BLK COURTHOUSE SQ",
            "city": "ROCKVILLE",
            "state": "MD",
            "latitude": "39.08",
            "longitude": "-77.15",
            "geolocation": "(39.08, -77.15)",
        },
        {
            "incident_id": "multiline",
            "offence_code": "1103",
            "case_number": "C3",
            "start_date": "2024-03-04T05:06:00.000",
            "nibrs_code": "11A",
            "crimename1": "Crime Against Person",
            "crimename2": "Forcible Rape",
            "crimename3": "RAPE - STRONG-ARM",
            "district": "WHEATON",
            "location": "2000 BLK MAYFLOWER DR",
            "city": "SILVER SPRING",
            "state": "MD",
            "latitude": "39.10",
            "longitude": "-76.97",
            "geolocation": "\n,  \n(39.10, -76.97)",
        },
    ]
    for row in rows:
        writer.writerow(row)
    writer.writer.writerow(
        [rows[0].get(column, "") for column in MONTGOMERY_COLUMNS]
        + ["unrepairable-extra-field"]
    )
    path.write_text(stream.getvalue())

    pattern = SourcePattern(
        "*.csv",
        "csv",
        {
            "strategy": "python_tolerant",
            "encoding": "utf-8-sig",
            "expected_columns": MONTGOMERY_COLUMNS,
        },
    )
    with caplog.at_level("INFO"):
        result = read_source_pattern(str(path), pattern).collect()

    assert result.height == 3
    semantic_columns = (
        "incident_id",
        "offence_code",
        "case_number",
        "nibrs_code",
        "crimename1",
        "crimename2",
        "crimename3",
        "district",
        "location",
        "city",
        "state",
        "latitude",
        "longitude",
    )
    assert result.select(semantic_columns).rows(named=True) == [
        {column: row.get(column) for column in semantic_columns} for row in rows
    ]
    assert result.filter(result["incident_id"] == "multiline")[
        "geolocation"
    ].item() == "\n,  \n(39.10, -76.97)"
    summary = next(
        record
        for record in caplog.records
        if record.message == "csv_record_width_summary"
    )
    assert summary.total_records == 4
    assert summary.correct_width_records == 3
    assert summary.long_records == 1
    assert summary.repaired_records == 0
    assert summary.rejected_records == 1
