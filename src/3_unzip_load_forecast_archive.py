import zipfile
import pandas as pd
from pathlib import Path
import re
from datetime import datetime

print("=" * 70)
print("Unzip and Process Load Forecast Archives")
print("=" * 70)

INPUT_DIR = Path(r"C:\code\energy_trading\input_data\load\SevenDay_Load_Forecast_by_Model_and_Weather_Zone")
OUTPUT_FILE = Path(r"C:\code\energy_trading\data_processed\load_forecast_historical_archive.csv")
TEMP_DIR = INPUT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 10000  # Write to disk every 10k CSVs



def extract_run_datetime(name: str):
    m = re.search(r'(20\d{6})\.(\d{6})', name)  # ← Changed this line
    if not m:
        return pd.NaT
    date_str, time_str = m.groups()
    try:
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return pd.Timestamp(dt).floor('min')
    except Exception:
        return pd.NaT
    



def standardize_columns(df):
    col_rename = {
        'DeliveryDate': 'deliveryDate',
        'HourEnding': 'hourEnding',
        'Coast': 'coast',
        'East': 'east',
        'FarWest': 'farWest',
        'North': 'north',
        'NorthCentral': 'northCentral',
        'SouthCentral': 'southCentral',
        'Southern': 'southern',
        'West': 'west',
        'SystemTotal': 'systemTotal',
        'Model': 'model',
        'InUseFlag': 'inUseFlag',
        'DSTFlag': 'DSTFlag'
    }
    for old, new in col_rename.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df

print(f"\n1. Finding zip files in: {INPUT_DIR}")
zip_files = list(INPUT_DIR.glob("*.zip"))
print(f"   ✓ Found {len(zip_files)} outer zip files")

chunk_data = []
total_csvs = 0
total_rows_written = 0
first_write = True

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

for i, outer_zip in enumerate(zip_files, 1):
    print(f"\n2. Processing {outer_zip.name} ({i}/{len(zip_files)})...")

    try:
        with zipfile.ZipFile(outer_zip, 'r') as outer:
            inner_files = outer.namelist()
            inner_zips = [f for f in inner_files if f.endswith('.zip')]
            print(f"   Found {len(inner_zips)} inner zips")

            for inner_name in inner_zips:
                outer.extract(inner_name, TEMP_DIR)
                inner_path = TEMP_DIR / inner_name

                run_dt = extract_run_datetime(inner_name)
                if pd.isna(run_dt):
                    run_dt = extract_run_datetime(outer_zip.name)

                try:
                    with zipfile.ZipFile(inner_path, 'r') as inner:
                        csv_files = [f for f in inner.namelist() if f.endswith('.csv')]

                        for csv_name in csv_files:
                            with inner.open(csv_name) as f:
                                df = pd.read_csv(f)
                            
                            # Skip non-forecast CSVs
                            if 'InUseFlag' not in df.columns or 'DeliveryDate' not in df.columns or 'HourEnding' not in df.columns:
                                continue
                            
                            # Filter to InUseFlag='Y' EARLY
                            df['InUseFlag'] = df['InUseFlag'].astype(str).str.upper()
                            df = df[df['InUseFlag'] == 'Y']
                            if df.empty:
                                continue
                            
                            # Add runDatetime
                            df['runDatetime'] = run_dt
                            
                            # Process datetime EARLY (while df is small)
                            df['HourEnding'] = df['HourEnding'].astype(str).str.strip()
                            df['DeliveryDate'] = df['DeliveryDate'].astype(str).str.strip()
                            
                            he = df['HourEnding']
                            is_24 = he == '24:00'
                            he_fixed = he.where(~is_24, '00:00')
                            df['datetime'] = pd.to_datetime(df['DeliveryDate'] + ' ' + he_fixed, errors='coerce')
                            df.loc[is_24, 'datetime'] += pd.Timedelta(days=1)
                            
                            # Drop bad datetimes EARLY
                            df = df.dropna(subset=['datetime'])
                            if df.empty:
                                continue
                            
                            # Standardize column names
                            df = standardize_columns(df)
                            
                            chunk_data.append(df)
                            total_csvs += 1

                            # Write chunk to disk periodically
                            if len(chunk_data) >= CHUNK_SIZE:
                                combined = pd.concat(chunk_data, ignore_index=True)
                                combined = combined.sort_values(['datetime', 'runDatetime']).reset_index(drop=True)
                                combined.to_csv(
                                    OUTPUT_FILE,
                                    mode='a' if not first_write else 'w',
                                    header=first_write,
                                    index=False
                                )
                                total_rows_written += len(combined)
                                first_write = False
                                print(f"   ✓ Wrote chunk: {total_rows_written:,} total rows")
                                chunk_data = []

                except Exception as e:
                    print(f"   ✗ Error reading {inner_name}: {e}")

                try:
                    inner_path.unlink()
                except FileNotFoundError:
                    pass

    except Exception as e:
        print(f"   ✗ Error with {outer_zip.name}: {e}")

# Process remaining data
if chunk_data:
    print(f"\n3. Processing final chunk ({len(chunk_data)} CSVs)...")
    combined = pd.concat(chunk_data, ignore_index=True)
    combined = combined.sort_values(['datetime', 'runDatetime']).reset_index(drop=True)
    combined.to_csv(
        OUTPUT_FILE,
        mode='a' if not first_write else 'w',
        header=first_write,
        index=False
    )
    total_rows_written += len(combined)
    print(f"   ✓ Wrote final chunk: {total_rows_written:,} total rows")

if TEMP_DIR.exists():
    try:
        TEMP_DIR.rmdir()
    except OSError:
        pass

print(f"\n✅ COMPLETE")
print(f"   Total CSVs processed: {total_csvs:,}")
print(f"   Total rows written: {total_rows_written:,}")
print(f"   Output file: {OUTPUT_FILE}")
print("=" * 70)