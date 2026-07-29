"""Land Census ACS responses as JSON Lines files."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from crimenet.socioeconomic.acs_client import (
    CensusApiError,
    build_census_session,
    fetch_acs5_tracts,
)

logger = logging.getLogger(__name__)


def get_acs5_tract_path(
    landing_directory: str | Path,
    *,
    vintage: int,
    state_fips: str,
) -> Path:
    return (
        Path(landing_directory)
        / "acs5"
        / "tract"
        / f"vintage={vintage}"
        / f"state={state_fips}"
        / "acs5_tract.jsonl"
    )


def write_json_lines(
    records: Iterable[dict[str, Any]],
    destination: Path,
) -> int:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    record_count = 0
    try:
        with temporary_path.open(
            "x",
            encoding="utf-8",
        ) as output_file:
            for record in records:
                output_file.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                )
                output_file.write("\n")
                record_count += 1

            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(
            temporary_path,
            destination,
        )
    finally:
        temporary_path.unlink(missing_ok=True)

    return record_count


def is_valid_acs_cache(
    path: Path,
    *,
    vintage: int,
    state_fips: str,
    minimum_record_count: int = 1,
) -> bool:
    """Reject empty, partial, or wrong-partition ACS landing files."""
    if minimum_record_count < 1:
        raise ValueError("minimum_record_count must be positive.")
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        record_count = 0
        geoids: set[str] = set()
        with path.open(encoding="utf-8") as input_file:
            for line in input_file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if (
                    not isinstance(payload, dict)
                    or int(payload.get("acs_vintage", -1)) != vintage
                    or str(payload.get("state", "")) != state_fips
                    or not isinstance(payload.get("geoid"), str)
                ):
                    return False
                geoid = payload["geoid"].strip()
                if (
                    len(geoid) != 11
                    or not geoid.isdigit()
                    or not geoid.startswith(state_fips)
                    or geoid in geoids
                ):
                    return False
                geoids.add(geoid)
                record_count += 1
        return record_count >= minimum_record_count
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def ingest_acs5_tract_vintages(
    *,
    landing_directory: str | Path,
    start_vintage: int,
    end_vintage: int,
    state_fips: str,
    api_key: str | None,
    overwrite: bool = False,
    minimum_record_count: int = 1,
) -> None:
    if start_vintage > end_vintage:
        raise ValueError(
            "start_vintage cannot be greater "
            "than end_vintage."
        )
    if minimum_record_count < 1:
        raise ValueError("minimum_record_count must be positive.")

    session = build_census_session()
    try:
        for vintage in range(
            start_vintage,
            end_vintage + 1,
        ):
            destination = get_acs5_tract_path(
                landing_directory,
                vintage=vintage,
                state_fips=state_fips,
            )

            if (
                not overwrite
                and is_valid_acs_cache(
                    destination,
                    vintage=vintage,
                    state_fips=state_fips,
                    minimum_record_count=minimum_record_count,
                )
            ):
                logger.info(
                    "Skipping valid existing ACS file: %s",
                    destination,
                )
                continue

            logger.info(
                "Fetching ACS 5-year tract data: "
                "vintage=%s state=%s",
                vintage,
                state_fips,
            )

            records = fetch_acs5_tracts(
                vintage=vintage,
                state_fips=state_fips,
                api_key=api_key,
                session=session,
            )
            if len(records) < minimum_record_count:
                raise CensusApiError(
                    "Census API returned fewer tract records than required: "
                    f"vintage={vintage}, state={state_fips}, "
                    f"records={len(records)}, minimum={minimum_record_count}"
                )

            record_count = write_json_lines(
                records,
                destination,
            )

            logger.info(
                "Landed %s ACS tract records at %s",
                record_count,
                destination,
            )
    finally:
        session.close()
