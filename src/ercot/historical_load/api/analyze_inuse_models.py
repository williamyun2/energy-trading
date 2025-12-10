import pandas as pd
from pathlib import Path

# Path to your CSV
csv_path = Path(r"C:\code\energy_trading\data_extracted\load_forecast\load_forecast_historical.csv")

print(f"Loading CSV from: {csv_path}")

# Load only the columns we need to save RAM
cols = ["model", "inUseFlag"]
df = pd.read_csv(csv_path, usecols=cols)

# Fix booleans (CSV stores TRUE/FALSE as strings)
df["inUseFlag"] = df["inUseFlag"].astype(str).str.upper().isin(["TRUE", "1"])

# Count how many rows are in use
total_in_use = df["inUseFlag"].sum()

print("\n===== IN USE SUMMARY =====")
print(f"Total rows where inUseFlag = TRUE: {total_in_use:,}")

# Count in-use per model
model_counts = df[df["inUseFlag"] == True]["model"].value_counts()

print("\n===== MODEL USAGE (inUseFlag = TRUE) =====")
for model, count in model_counts.items():
    print(f"Model {model}: {count:,} times")
