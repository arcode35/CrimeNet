import polars as pl 
import datetime as datetime
df = pl.read_parquet("fixtures/data/baltimore.parquet")

def add_ingestion_metadata(bronze_df: pl.LazyFrame):
    bronze_df.with_columns
    (
        pl.lit(pl.datetime)
    )