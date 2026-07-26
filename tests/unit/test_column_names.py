from crimenet.ingestion.column_names import (
    normalize_column_name,
    normalized_column_names,
)


def test_normalize_column_name() -> None:
    assert normalize_column_name("Call (911) Problem") == "call_911_problem"
    assert normalize_column_name("Date/Time") == "date_time"
    assert normalize_column_name("\ufeff ZIP Code ") == "zip_code"


def test_collision_suffixes_are_deterministic() -> None:
    assert normalized_column_names(
        ["Latitude", "_latitude", "Longitude", "_longitude"],
        overrides={
            "Latitude": "latitude",
            "_latitude": "alternate_latitude",
            "Longitude": "longitude",
            "_longitude": "alternate_longitude",
        },
    ) == [
        "latitude",
        "alternate_latitude",
        "longitude",
        "alternate_longitude",
    ]
