import os
import polars as pl

name = "train"
path = f"./dataset/v5.0/{name}.parquet"
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pl.read_parquet(path).with_columns(
        pl.col("era").cast(pl.Int64).alias("era_int")
        )


for remainder in range(4):
    filtererd_df = df.filter((df["era_int"] % 4) == remainder)

    output_filename = os.path.join(output_dir, f"{name}_group_{remainder}.parquet")
    filtererd_df.write_parquet(output_filename, compression="gzip")
    print(f"Saved {output_filename}")

    break
