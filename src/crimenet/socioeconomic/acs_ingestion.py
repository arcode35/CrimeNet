"""Land Census ACS responses as JSON Lines files."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

from crimenet.socioeconomic.acs_client import (
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

    temporary_path = destination.with_suffix(
        ".jsonl.tmp"
    )

    record_count = 0

    with temporary_path.open(
        "w",
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

    os.replace(
        temporary_path,
        destination,
    )

    return record_count


def ingest_acs5_tract_vintages(
    *,
    landing_directory: str | Path,
    start_vintage: int,
    end_vintage: int,
    state_fips: str,
    api_key: str,
    overwrite: bool = False,
) -> None:
    if start_vintage > end_vintage:
        raise ValueError(
            "start_vintage cannot be greater "
            "than end_vintage."
        )

    session = build_census_session()

    for vintage in range(
        start_vintage,
        end_vintage + 1,
    ):
        destination = get_acs5_tract_path(
            landing_directory,
            vintage=vintage,
            state_fips=state_fips,
        )

        if destination.exists() and not overwrite:
            logger.info(
                "Skipping existing ACS file: %s",
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

        record_count = write_json_lines(
            records,
            destination,
        )

        logger.info(
            "Landed %s ACS tract records at %s",
            record_count,
            destination,
        )