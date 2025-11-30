import pandas as pd
from pathlib import Path

# Path to CSV
csv_path = Path(r"C:\code\energy_trading\data_extracted\load_forecast\load_forecast_historical.csv")

print(f"Loading CSV from: {csv_path}")

# Load CSV
df = pd.read_csv(csv_path)

# --- Print head and tail ---
print("\n=== DataFrame Head (first 5 rows) ===")
print(df.head())

print("\n=== DataFrame Tail (last 5 rows) ===")
print(df.tail())

# --- Extract and process the dates ---
# Convert to datetime
df["deliveryDate"] = pd.to_datetime(df["deliveryDate"], errors="coerce")

# Get unique sorted dates
unique_dates = sorted(df["deliveryDate"].dropna().unique())

# Convert to strings for saving
date_strings = [d.strftime("%Y-%m-%d") for d in unique_dates]

# --- Save to text file in current directory ---
output_path = Path.cwd() / "delivery_dates.txt"

with open(output_path, "w") as f:
    for d in date_strings:
        f.write(d + "\n")

print(f"\nSaved {len(date_strings):,} dates to: {output_path}\n")

# Preview first and last 10 dates
print("First 10 dates:", date_strings[:10])
print("Last 10 dates:", date_strings[-10:])
