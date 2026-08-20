from crimenet_data.resources.crime_lake import CrimeLakeResources, CITIES
import dagster as dg


@dg.asset(
    deps=[f"{city}_quarantine" for city in CITIES]
)
def quarantine(
    crime_lake: CrimeLakeResources,
) -> dg.MaterializeResult:
    """
    Represents the complete CrimeNet quarantine dataset.
    """

    return dg.MaterializeResult(
        metadata={
            "path": f"{crime_lake.quality_root}/quarantine",
        }
    )