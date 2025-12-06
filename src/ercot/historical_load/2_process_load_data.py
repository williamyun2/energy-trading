import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

print("=" * 70)
print("ERCOT Load Data Processor")
print("=" * 70)

base_dir = Path(__file__).parent.parent
load_zip_dir = base_dir / "input_data" / "load" / "Actual_System_Load_by_Weather_Zone"
output_dir = base_dir / "data_extracted" / "load"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Extract all ZIP files (may contain nested ZIPs)
zip_files = list(load_zip_dir.glob("*.zip"))
print(f"\nFound {len(zip_files)} ZIP files")

print("\nExtracting (handling nested ZIPs)...")
for zip_path in tqdm(zip_files):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

# Step 2: Check for nested ZIPs in extracted folder
nested_zips = list(output_dir.glob("*.zip"))
if nested_zips:
    print(f"\nFound {len(nested_zips)} nested ZIP files, extracting...")
    for zip_path in tqdm(nested_zips):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        zip_path.unlink()  # Delete the nested ZIP after extraction

# Step 3: Find all data files (CSV or Excel)
csv_files = list(output_dir.glob("*.csv"))
xlsx_files = list(output_dir.glob("*.xlsx"))
xls_files = list(output_dir.glob("*.xls"))

print("\nFound data files:")
print(f"  CSV:  {len(csv_files)}")
print(f"  XLSX: {len(xlsx_files)}")
print(f"  XLS:  {len(xls_files)}")

if not (csv_files or xlsx_files or xls_files):
    print("\nNo data files found!")
    print(f"Contents of {output_dir}:")
    for f in output_dir.iterdir():
        print(f"  {f.name}")
    raise SystemExit(1)

# Step 4: Read all files
print("\nReading load data...")
all_load = []

for file in tqdm(csv_files, desc="CSV files"):
    df = pd.read_csv(file)
    all_load.append(df)

for file in tqdm(xlsx_files, desc="XLSX files"):
    df = pd.read_excel(file)
    all_load.append(df)

for file in tqdm(xls_files, desc="XLS files"):
    df = pd.read_excel(file)
    all_load.append(df)

combined = pd.concat(all_load, ignore_index=True)

# Normalize column names once
combined.columns = [c.strip() for c in combined.columns]

print(f"\nTotal load records: {len(combined):,}")
print(f"\nColumns: {combined.columns.tolist()}")
print("\nSample data:")
print(combined.head(10))

# Process datetime
print("\nProcessing datetime...")
if "Hour Ending" in combined.columns or "HourEnding" in combined.columns:
    hour_col = "HourEnding" if "HourEnding" in combined.columns else "Hour Ending"
    date_col = None
    if "Delivery Date" in combined.columns:
        date_col = "Delivery Date"
    elif "Oper Day" in combined.columns:
        date_col = "Oper Day"
    elif "OperDay" in combined.columns:
        date_col = "OperDay"
    if date_col is None:
        raise ValueError("Could not find a date column for hour-ending data")
    combined[hour_col] = combined[hour_col].astype(str).str.replace("24:00", "00:00")
    combined["datetime"] = pd.to_datetime(
        combined[date_col] + " " + combined[hour_col],
        errors="coerce",
    )
    combined.loc[combined[hour_col] == "00:00", "datetime"] += pd.Timedelta(days=1)
else:
    # fallback: try parsing first column as datetime
    combined["datetime"] = pd.to_datetime(combined.iloc[:, 0], errors="coerce")

combined = combined.dropna(subset=["datetime"])

# Calculate system load and keep all weather zone columns
print("\nCalculating ERCOT system load (keeping all zones)...")
zone_cols_preference = ["COAST", "EAST", "FAR_WEST", "NORTH", "NORTH_C", "SOUTHERN", "SOUTH_C", "WEST"]
zone_cols = [c for c in zone_cols_preference if c in combined.columns]

# Build output frame with original numeric columns + datetime
keep_cols = ["datetime"] + zone_cols
if "TOTAL" in combined.columns:
    keep_cols.append("TOTAL")
if "DSTFlag" in combined.columns:
    keep_cols.append("DSTFlag")

system_load = combined[keep_cols].copy()

# Derive system_load_mw
if "TOTAL" in system_load.columns:
    system_load["system_load_mw"] = system_load["TOTAL"]
elif zone_cols:
    system_load["system_load_mw"] = system_load[zone_cols].sum(axis=1)
else:
    print("Could not find load columns!")
    raise SystemExit(1)

# Remove duplicates (keep first occurrence)
print(f"\nBefore deduplication: {len(system_load):,} records")
system_load = system_load.drop_duplicates(subset="datetime", keep="first")
print(f"After deduplication:  {len(system_load):,} records")

print(f"\nSystem load records: {len(system_load):,}")
print(f"Date range: {system_load['datetime'].min()} to {system_load['datetime'].max()}")
print("\nLoad statistics:")
print(f"  Mean: {system_load['system_load_mw'].mean():.0f} MW")
print(f"  Min:  {system_load['system_load_mw'].min():.0f} MW")
print(f"  Max:  {system_load['system_load_mw'].max():.0f} MW")

# Save
output_file = base_dir / "data_processed" / "system_load.csv"
system_load.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")
