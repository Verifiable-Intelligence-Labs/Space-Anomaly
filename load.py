import pandas as pd

# # Path to your manually downloaded Parquet file
# parquet_file_path = "./train-00000-of-00027.parquet"  # Replace with the actual file path

# # Load the Parquet file into a Pandas DataFrame
# df = pd.read_parquet(parquet_file_path)

# # Take the first 10,000 rows
# df_first_10k = df.head(10000)

# # Save to CSV
# csv_file_path = "jwst_dataset.csv"  # Output CSV file name
# df_first_10k.to_csv(csv_file_path, index=False)

# print(f"First 10,000 rows saved as {csv_file_path}")

csvpath = './jwst_dataset.csv'

df_first_1k = pd.read_csv(csvpath, nrows=1000)
df_first_1k.to_csv("jwst_dataset_1k.csv",index=False)
