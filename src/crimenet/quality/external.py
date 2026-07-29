"""Shared quarantine construction for external source records."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def split_external_quarantine(
    dataframe: DataFrame,
    *,
    reason_codes_column: str,
    reason_messages: dict[str, str],
    source_system: str,
    pipeline_run_id: str,
) -> tuple[DataFrame, DataFrame]:
    """Split annotated source records and create stable per-reason rejects."""
    valid = dataframe.filter(F.size(reason_codes_column) == 0).drop(
        reason_codes_column
    )
    message_items = [
        item
        for code, message in reason_messages.items()
        for item in (F.lit(code), F.lit(message))
    ]
    messages = F.create_map(*message_items)
    payload_columns = [
        name
        for name in dataframe.columns
        if name not in {reason_codes_column, "source_file"}
    ]
    source_hash = F.coalesce(
        F.col("source_row_hash"),
        F.sha2(
            F.to_json(
                F.struct(*[F.col(name) for name in payload_columns]),
                options={"ignoreNullFields": "false"},
            ),
            256,
        ),
    )
    quarantine = (
        dataframe.filter(F.size(reason_codes_column) > 0)
        .withColumn(
            "quarantine_reason_code",
            F.explode(F.col(reason_codes_column)),
        )
        .withColumn(
            "quarantine_reason",
            messages[F.col("quarantine_reason_code")],
        )
        .withColumn("source_system", F.lit(source_system))
        .withColumn("pipeline_run_id", F.lit(pipeline_run_id))
        .withColumn("quarantined_at", F.current_timestamp())
        .withColumn(
            "raw_payload",
            F.to_json(
                F.struct(*[F.col(name) for name in payload_columns]),
                options={"ignoreNullFields": "false"},
            ),
        )
        .withColumn(
            "validation_fields",
            F.to_json(
                F.struct(
                    *[
                        F.col(name)
                        for name in payload_columns
                        if name
                        in {
                            "request_id",
                            "geoid",
                            "acs_vintage",
                            "weather_query_cell_id",
                        }
                    ]
                ),
                options={"ignoreNullFields": "false"},
            ),
        )
        .withColumn(
            "quarantine_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.lit(source_system),
                    source_hash,
                    F.col("quarantine_reason_code"),
                ),
                256,
            ),
        )
        .select(
            "quarantine_id",
            "source_system",
            "source_file",
            "source_row_hash",
            "raw_payload",
            "quarantine_reason_code",
            "quarantine_reason",
            "pipeline_run_id",
            "quarantined_at",
            "validation_fields",
        )
    )
    return valid, quarantine
