import pandas as pd
from datetime import datetime

user_df = pd.read_csv("data/processed/user_features.csv")
item_df = pd.read_csv("data/processed/item_features.csv")

user_df["event_timestamp"] = datetime.now()
item_df["event_timestamp"] = datetime.now()

user_df.to_parquet("data/processed/user_features_ts.parquet", index=False)
item_df.to_parquet("data/processed/item_features_ts.parquet", index=False)

print("Wrote parquet sources with event_timestamp for Feast.")