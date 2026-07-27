import urllib.error
from functools import reduce

import geopandas as gpd
from pyspark.databricks.sql import functions as dbf
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# Include 2012 and 2013 because early-2014 crimes can select those
# ACS vintages under the release-aware mapping.
years = [str(year) for year in range(2012, 2025)]

table_name = (
    "crimenet_dev.silver.census_tract_boundaries"
)

spark_dfs: list[DataFrame] = []


for year in years:
    url = (
        "https://www2.census.gov/geo/tiger/"
        f"TIGER{year}/TRACT/"
        f"tl_{year}_48_tract.zip"
    )

    print(f"Fetching {year} boundaries...")

    try:
        gdf = gpd.read_file(url)

        # Standardize coordinates to WGS84.
        gdf = gdf.to_crs(epsg=4326)

        # Ordinary WKT does not preserve CRS/SRID metadata.
        gdf["wkt_geometry"] = gdf.geometry.to_wkt()

        df_filtered = gdf[
            ["GEOID", "wkt_geometry"]
        ].copy()

        spark_df = spark.createDataFrame(
            df_filtered
        )

        processed_df = (
            spark_df
            .withColumn(
                "tract_geometry",
                dbf.st_geomfromwkt(
                    F.col("wkt_geometry"),
                    4326,
                ),
            )
            .withColumn(
                "boundary_vintage",
                F.lit(int(year)),
            )
            .select(
                F.col("GEOID").alias("geoid"),
                F.col("boundary_vintage"),
                F.col("tract_geometry"),
            )
        )

        spark_dfs.append(processed_df)

        print(
            f"  -> Successfully processed {year}."
        )

    except urllib.error.HTTPError as error:
        if error.code == 404:
            print(
                f"  -> Skipping {year}: "
                "TIGER/Line file was not found."
            )
        else:
            raise

    except Exception as error:
        print(
            f"  -> Unexpected error processing "
            f"{year}: {error}"
        )


if not spark_dfs:
    raise RuntimeError(
        "No TIGER/Line tract files were processed."
    )


print(
    "\nUnioning all years and writing "
    "to the Delta table..."
)

final_df = reduce(
    lambda left, right: left.unionByName(right),
    spark_dfs,
)


# Validate before replacing the existing table.
invalid_srids = (
    final_df
    .filter(
        dbf.st_srid("tract_geometry") != 4326
    )
    .count()
)

if invalid_srids:
    raise ValueError(
        f"Found {invalid_srids:,} geometries "
        "without SRID 4326."
    )


(
    final_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(table_name)
)

spark.sql(
    f"""
    OPTIMIZE {table_name}
    ZORDER BY (boundary_vintage)
    """
)

row_count = final_df.count()

print(
    f"Pipeline complete. Wrote "
    f"{row_count:,} rows to {table_name}."
)

display(
    spark.table(table_name)
    .groupBy(
        "boundary_vintage",
        dbf.st_srid(
            "tract_geometry"
        ).alias("srid"),
    )
    .count()
    .orderBy("boundary_vintage")
)