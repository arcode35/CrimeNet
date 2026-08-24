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
        "path": (
            "s3://crimenet-data/raw_files/landing/reference/"
            "canonical_crime_crosswalk_v1_3.csv"
        ),
        "mapping_version": "crime_canonical_v1_3",
    },
    kinds={"s3", "csv"},
)
