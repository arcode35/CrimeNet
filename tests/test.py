import polars as pl 

df = pl.read_parquet("fixtures/data/baltimore.parquet")

print(df.collect_schema())