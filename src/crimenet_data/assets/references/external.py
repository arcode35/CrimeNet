# assets/reference/external.py

import dagster as dg

canonical_crime_crosswalk = dg.AssetSpec(
    key=["reference", "canonical_crime_crosswalk"],
    group_name="reference",
    description=(
        "Manually curated city-to-canonical crime mapping "
        "stored in the CrimeNet B2 lake."
    ),
    metadata={
        "crime_lake_property": "canonical_crosswalk_uri",
        "mapping_version": "crime_canonical_v1_5",
    },
    kinds={"s3", "csv"},
)
