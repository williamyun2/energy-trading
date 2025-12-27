"""
Comprehensive Data Inspection - All Input Datasets
Inspects DAM prices, natural gas, weather, and load forecasts
Saves detailed analysis to text file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Redirect output to file (in current merge_dataset directory)
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILE = SCRIPT_DIR / "data_inspection_report.txt"

# Open output file
output = open(OUTPUT_FILE, 'w', encoding='utf-8')
sys.stdout = output

# Base paths
BASE_DIR = Path(r"D:\Users\williamyun\proj\power_trading")
PROCESSED_DIR = BASE_DIR / "data_processed"

print("="*80)
print("COMPREHENSIVE DATA INSPECTION REPORT")
print("="*80)
print(f"Generated: {datetime.now()}")
print(f"Output file: {OUTPUT_FILE}")
print()

# ============================================================================
# 1. DAM PRICES
# ============================================================================
print("="*80)
print("1. DAM PRICES (Day-Ahead Market)")
print("="*80)

dam_path = PROCESSED_DIR / "ercot" / "combined_dam_prices.csv"
print(f"\nFile: {dam_path}")
print(f"File size: {dam_path.stat().st_size / 1024**2:.1f} MB")
print(f"File exists: {dam_path.exists()}")

if dam_path.exists():
    # Load sample
    df_dam = pd.read_csv(dam_path, nrows=10000)
    print(f"\nSample loaded: 10,000 rows")
    print(f"\nColumns ({len(df_dam.columns)}):")
    for col in df_dam.columns:
        print(f"  - {col}")

    print(f"\nData types:")
    print(df_dam.dtypes)

    print(f"\nFirst 10 rows:")
    print(df_dam.head(10))

    print(f"\nUnique values:")
    print(f"  Settlement Points: {df_dam['Settlement Point'].nunique()}")
    print(f"  Unique Settlement Points:")
    for sp in sorted(df_dam['Settlement Point'].unique()):
        count = (df_dam['Settlement Point'] == sp).sum()
        print(f"    {sp}: {count:,} records")

    # Check for duplicates in sample
    df_dam['Delivery Date'] = pd.to_datetime(df_dam['Delivery Date'])
    df_dam['HE'] = df_dam['Hour Ending'].str.split(':').str[0].astype(int)
    df_dam['datetime'] = df_dam['Delivery Date'] + pd.to_timedelta(df_dam['HE'] - 1, unit='h')
    df_dam.loc[df_dam['HE'] == 24, 'datetime'] = df_dam.loc[df_dam['HE'] == 24, 'Delivery Date'] + pd.Timedelta(days=1)

    # Filter to HB_BUSAVG
    df_dam_avg = df_dam[df_dam['Settlement Point'] == 'HB_BUSAVG']
    duplicates = df_dam_avg.duplicated(subset=['datetime']).sum()
    print(f"\n  HB_BUSAVG duplicates in sample: {duplicates}")

    print(f"\n  Price statistics (sample):")
    print(df_dam_avg['Settlement Point Price'].describe())

print()

# ============================================================================
# 2. NATURAL GAS PRICES
# ============================================================================
print("="*80)
print("2. NATURAL GAS PRICES (Yahoo Finance Futures)")
print("="*80)

ng_path = PROCESSED_DIR / "fuel" / "NG" / "ng_futures_yahoo_daily_2010-12-01_2025-12-26.csv"
print(f"\nFile: {ng_path}")
print(f"File size: {ng_path.stat().st_size / 1024:.1f} KB")
print(f"File exists: {ng_path.exists()}")

if ng_path.exists():
    df_ng = pd.read_csv(ng_path)
    print(f"\nTotal rows: {len(df_ng):,}")

    print(f"\nColumns ({len(df_ng.columns)}):")
    for col in df_ng.columns:
        print(f"  - {col}")

    print(f"\nData types:")
    print(df_ng.dtypes)

    print(f"\nFirst 10 rows:")
    print(df_ng.head(10))

    print(f"\nLast 10 rows:")
    print(df_ng.tail(10))

    # Check for duplicates
    duplicates = df_ng.duplicated(subset=['Date']).sum()
    print(f"\nDuplicate dates: {duplicates}")

    if duplicates > 0:
        print(f"\nSample duplicates:")
        dup_dates = df_ng[df_ng.duplicated(subset=['Date'], keep=False)].sort_values('Date')
        print(dup_dates.head(20))

    # Statistics
    print(f"\nPrice statistics (Close):")
    print(df_ng['Close'].describe())

print()

# ============================================================================
# 3. WEATHER DATA (HRRR)
# ============================================================================
print("="*80)
print("3. WEATHER DATA (HRRR Forecasts)")
print("="*80)

weather_dir = PROCESSED_DIR / "weather" / "hrrr"
print(f"\nDirectory: {weather_dir}")
print(f"Directory exists: {weather_dir.exists()}")

if weather_dir.exists():
    weather_files = sorted(weather_dir.glob("hrrr_texas_*.parquet"))
    print(f"Total files: {len(weather_files):,}")

    if len(weather_files) > 0:
        # Load first file
        first_file = weather_files[0]
        print(f"\nInspecting first file: {first_file.name}")
        df_weather = pd.read_parquet(first_file)

        print(f"\nRows in first file: {len(df_weather):,}")
        print(f"\nColumns ({len(df_weather.columns)}):")
        for col in df_weather.columns:
            print(f"  - {col}")

        print(f"\nData types:")
        print(df_weather.dtypes)

        print(f"\nFirst 10 rows:")
        print(df_weather.head(10))

        print(f"\nUnique zones:")
        for zone in sorted(df_weather['zone'].unique()):
            count = (df_weather['zone'] == zone).sum()
            print(f"  {zone}: {count:,} records")

        # Check for duplicates
        duplicates = df_weather.duplicated(subset=['datetime', 'zone']).sum()
        print(f"\nDuplicates (datetime + zone): {duplicates}")

        # Variable statistics
        print(f"\nWeather variable statistics:")
        for var in ['temp_f', 'wind_speed_mph', 'solar_radiation_wm2', 'relative_humidity']:
            if var in df_weather.columns:
                print(f"\n  {var}:")
                print(f"    {df_weather[var].describe()}")

print()

# ============================================================================
# 4. LOAD FORECASTS
# ============================================================================
print("="*80)
print("4. LOAD FORECASTS (ERCOT Historical)")
print("="*80)

load_path = PROCESSED_DIR / "ercot" / "historical_load" / "load_forecast_complete.csv"
print(f"\nFile: {load_path}")
print(f"File size: {load_path.stat().st_size / 1024**3:.2f} GB")
print(f"File exists: {load_path.exists()}")

if load_path.exists():
    # Load sample (first 50,000 rows)
    print(f"\nLoading sample (50,000 rows)...")
    df_load = pd.read_csv(load_path, nrows=50000)

    print(f"\nSample loaded: {len(df_load):,} rows")
    print(f"\nColumns ({len(df_load.columns)}):")
    for col in df_load.columns:
        print(f"  - {col}")

    print(f"\nData types:")
    print(df_load.dtypes)

    print(f"\nFirst 10 rows:")
    print(df_load.head(10))

    # Check for datetime column
    datetime_cols = [c for c in df_load.columns if 'date' in c.lower() or 'time' in c.lower()]
    print(f"\nDateTime-related columns: {datetime_cols}")

    # Check for model column
    if 'model' in df_load.columns:
        print(f"\nUnique models:")
        model_counts = df_load['model'].value_counts()
        for model, count in model_counts.items():
            print(f"  {model}: {count:,} records")

    # Check for deliveryDate column
    if 'deliveryDate' in df_load.columns:
        df_load['datetime'] = pd.to_datetime(df_load['deliveryDate'])
        print(f"\nDate range (sample): {df_load['datetime'].min()} to {df_load['datetime'].max()}")

        # Check for duplicates
        duplicates = df_load.duplicated(subset=['datetime']).sum()
        print(f"Duplicate datetimes in sample: {duplicates:,}")

        if duplicates > 0:
            print(f"\nDuplicate percentage: {duplicates/len(df_load)*100:.1f}%")

            # Show sample duplicates
            dup_rows = df_load[df_load.duplicated(subset=['datetime'], keep=False)].sort_values('datetime')
            print(f"\nSample duplicates (first 20):")
            print(dup_rows.head(20))

            # Analyze duplicates by model
            if 'model' in df_load.columns:
                print(f"\nDuplicates by model:")
                dup_by_model = dup_rows.groupby('model').size()
                for model, count in dup_by_model.items():
                    print(f"  {model}: {count:,} duplicate records")

    # Check zone columns
    zone_cols = ['coast', 'east', 'farWest', 'north', 'northCentral',
                 'southCentral', 'southern', 'west', 'systemTotal']
    available_zones = [c for c in zone_cols if c in df_load.columns]

    if available_zones:
        print(f"\nAvailable zone columns: {available_zones}")
        print(f"\nZone data statistics (first zone: {available_zones[0]}):")
        print(df_load[available_zones[0]].describe())

        # Check for non-numeric values
        for zone in available_zones[:3]:  # Check first 3 zones
            non_numeric = pd.to_numeric(df_load[zone], errors='coerce').isna().sum()
            if non_numeric > 0:
                print(f"\n  ⚠️  {zone} has {non_numeric:,} non-numeric values")

print()

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)

print("\nData files status:")
print(f"  ✓ DAM prices: {dam_path.exists()}")
print(f"  ✓ Natural gas: {ng_path.exists()}")
print(f"  ✓ Weather: {weather_dir.exists()} ({len(weather_files) if weather_dir.exists() else 0} files)")
print(f"  ✓ Load forecasts: {load_path.exists()}")

print("\nKey findings:")
print(f"  - Load forecast file is {load_path.stat().st_size / 1024**3:.2f} GB")
print(f"  - Sample contains {duplicates:,} duplicate datetimes" if load_path.exists() else "")
print(f"  - Multiple models per datetime likely causing duplicates" if load_path.exists() and duplicates > 0 else "")

print("\nRecommendations:")
if load_path.exists() and duplicates > 0:
    print("  1. Deduplicate load forecasts by keeping first occurrence per datetime")
    print("     df.drop_duplicates(subset=['datetime'], keep='first')")
    print("  2. Or filter to a single forecast model before merging")
    print("  3. Re-run baseline model after cleaning")

print()
print("="*80)
print("INSPECTION COMPLETE")
print("="*80)

# Close output file
output.close()

# Print confirmation to actual terminal
import sys
sys.stdout = sys.__stdout__
print(f"✓ Data inspection complete!")
print(f"✓ Report saved to: {OUTPUT_FILE}")
print(f"\nTo view the report:")
print(f"  notepad {OUTPUT_FILE}")
print(f"  or")
print(f"  cat {OUTPUT_FILE}")
