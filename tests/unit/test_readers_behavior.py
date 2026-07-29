from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
from pyspark.sql import SparkSession

from crimenet.contracts.bronze import (
    ACS5_TRACT_RESPONSE_SCHEMA,
    DALLAS_SCHEMA,
    FORT_WORTH_SCHEMA,
    HOUSTON_SCHEMA,
    OPEN_METEO_RESPONSE_SCHEMA,
    get_source_contract,
    validate_contract_columns,
)
from crimenet.ingestion import readers
from crimenet.ingestion.readers import (
    read_acs5_tract_raw,
    read_dallas_raw,
    read_fort_worth_raw,
    read_houston_raw,
    read_weather_raw,
)


def _csv_text(headers: list[str], values: list[str]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerow(values)
    return stream.getvalue()


def test_batch_readers_apply_explicit_contracts_and_lineage(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    dallas_path = tmp_path / "dallas"
    dallas_path.mkdir()
    dallas_headers = [
        field.name for field in DALLAS_SCHEMA.fields if field.name != "_corrupt_record"
    ]
    dallas_values = ["value"] * len(dallas_headers)
    dallas_values[dallas_headers.index("Service Number ID")] = "D-100"
    (dallas_path / "records.csv").write_text(
        _csv_text(dallas_headers, dallas_values),
        encoding="utf-8",
    )

    houston_path = tmp_path / "houston"
    houston_path.mkdir()
    houston_headers = [
        field.name
        for field in HOUSTON_SCHEMA.fields
        if field.name != "_corrupt_record"
    ]
    houston_values = ["value"] * len(houston_headers)
    houston_values[houston_headers.index("Incident")] = "H-100"
    (houston_path / "records.csv").write_text(
        _csv_text(houston_headers, houston_values),
        encoding="utf-8",
    )

    fort_worth_path = tmp_path / "fort_worth"
    fort_worth_path.mkdir()
    fort_worth_payload = {
        field.name: "value"
        for field in FORT_WORTH_SCHEMA.fields
        if field.name != "_corrupt_record"
    }
    fort_worth_payload["Case_No_Offense"] = "FW-100-A"
    (fort_worth_path / "records.jsonl").write_text(
        json.dumps(fort_worth_payload) + "\n",
        encoding="utf-8",
    )

    dallas = read_dallas_raw(spark, str(dallas_path)).first()
    houston = read_houston_raw(spark, str(houston_path)).first()
    fort_worth = read_fort_worth_raw(spark, str(fort_worth_path)).first()

    assert dallas["Service Number ID"] == "D-100"
    assert dallas["_source_file"].endswith("records.csv")
    assert houston["Incident"] == "H-100"
    assert houston["_source_file"].endswith("records.csv")
    assert fort_worth["Case_No_Offense"] == "FW-100-A"
    assert fort_worth["_source_file"].endswith("records.jsonl")


class _RecordingReader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def format(self, value: str) -> _RecordingReader:
        self.calls.append(("format", value))
        return self

    def schema(self, value: object) -> _RecordingReader:
        self.calls.append(("schema", value))
        return self

    def option(self, key: str, value: object) -> _RecordingReader:
        self.calls.append((key, value))
        return self

    def load(self, path: str) -> dict[str, str]:
        self.calls.append(("load", path))
        return {"loaded": path}


class _FakeSpark:
    def __init__(self) -> None:
        self.readStream = _RecordingReader()


@pytest.mark.parametrize(
    ("reader", "expected_schema", "glob"),
    [
        (read_weather_raw, OPEN_METEO_RESPONSE_SCHEMA, "*.json"),
        (read_acs5_tract_raw, ACS5_TRACT_RESPONSE_SCHEMA, "*.jsonl"),
    ],
)
def test_streaming_readers_pin_schema_and_rescue_evolution(
    reader: Any,
    expected_schema: object,
    glob: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spark = _FakeSpark()
    monkeypatch.setattr(readers, "_with_source_file", lambda dataframe: dataframe)
    result = reader(
        spark,  # type: ignore[arg-type]
        "/Volumes/raw/input",
        schema_path="/Volumes/state/schema",
    )
    assert result == {"loaded": "/Volumes/raw/input"}
    assert ("format", "cloudFiles") in spark.readStream.calls
    assert ("schema", expected_schema) in spark.readStream.calls
    assert ("cloudFiles.schemaLocation", "/Volumes/state/schema") in (
        spark.readStream.calls
    )
    assert ("cloudFiles.schemaEvolutionMode", "rescue") in spark.readStream.calls
    assert ("rescuedDataColumn", "_rescued_data") in spark.readStream.calls
    assert ("pathGlobFilter", glob) in spark.readStream.calls


def test_contract_lookup_and_required_column_failures() -> None:
    contract = get_source_contract("dallas")
    validate_contract_columns(
        list(contract.required_normalized_columns),
        contract,
    )
    with pytest.raises(ValueError, match="missing required normalized columns"):
        validate_contract_columns(["service_number_id"], contract)
    with pytest.raises(ValueError, match="No municipal source contract"):
        get_source_contract("unknown")
