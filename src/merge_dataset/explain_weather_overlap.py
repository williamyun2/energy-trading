"""
Demonstrate how weather forecast overlap works in the loader
"""
import pandas as pd
from pathlib import Path

# Load a few weather files to show the overlap
weather_dir = Path(r"D:\Users\williamyun\proj\power_trading\data_processed\weather\hrrr")
weather_files = sorted(weather_dir.glob("hrrr_texas_*.parquet"))[:3]  # Just first 3 files

print("="*80)
print("WEATHER FORECAST OVERLAP EXPLANATION")
print("="*80)

for i, file in enumerate(weather_files):
    print(f"\n{'='*80}")
    print(f"File {i+1}: {file.name}")
    print('='*80)

    df = pd.read_parquet(file)

    # Show file metadata
    print(f"Total rows: {len(df):,}")
    print(f"Zones: {df['zone'].unique()}")
    print(f"Forecast issued: {df['forecast_issued'].iloc[0]}")
    print(f"Forecast hours: {df['forecast_hour'].min()} to {df['forecast_hour'].max()}")
    print(f"Valid times: {df['datetime'].min()} to {df['datetime'].max()}")

    # Show first few rows for one zone
    print(f"\nFirst 10 rows (HOUSTON zone):")
    houston = df[df['zone'] == 'HOUSTON'].sort_values('datetime').head(10)
    print(houston[['datetime', 'forecast_issued', 'forecast_hour', 'temp_f', 'solar_radiation_wm2']])

# Now demonstrate the overlap
print("\n" + "="*80)
print("DEMONSTRATING OVERLAP")
print("="*80)

# Load first two files completely
df1 = pd.read_parquet(weather_files[0])
df2 = pd.read_parquet(weather_files[1])

print(f"\nFile 1: {weather_files[0].name}")
print(f"  Forecast issued: {df1['forecast_issued'].iloc[0]}")
print(f"  Valid times: {df1['datetime'].min()} to {df1['datetime'].max()}")
print(f"  Total rows: {len(df1):,}")

print(f"\nFile 2: {weather_files[1].name}")
print(f"  Forecast issued: {df2['forecast_issued'].iloc[0]}")
print(f"  Valid times: {df2['datetime'].min()} to {df2['datetime'].max()}")
print(f"  Total rows: {len(df2):,}")

# Check overlap
df1_datetimes = set(df1[df1['zone'] == 'HOUSTON']['datetime'])
df2_datetimes = set(df2[df2['zone'] == 'HOUSTON']['datetime'])
overlap = df1_datetimes & df2_datetimes

print(f"\n🔍 OVERLAP ANALYSIS (HOUSTON zone only):")
print(f"  File 1 unique datetimes: {len(df1_datetimes)}")
print(f"  File 2 unique datetimes: {len(df2_datetimes)}")
print(f"  Overlapping datetimes: {len(overlap)}")

if overlap:
    print(f"\n  Example overlapping datetimes (first 5):")
    for dt in sorted(overlap)[:5]:
        # Get forecasts for this datetime from both files
        f1_row = df1[(df1['datetime'] == dt) & (df1['zone'] == 'HOUSTON')].iloc[0]
        f2_row = df2[(df2['datetime'] == dt) & (df2['zone'] == 'HOUSTON')].iloc[0]

        print(f"\n  Datetime: {dt}")
        print(f"    File 1: issued={f1_row['forecast_issued']}, fh={f1_row['forecast_hour']:2d}, temp={f1_row['temp_f']:.1f}°F")
        print(f"    File 2: issued={f2_row['forecast_issued']}, fh={f2_row['forecast_hour']:2d}, temp={f2_row['temp_f']:.1f}°F")

# Show what happens when combined
print("\n" + "="*80)
print("WHAT LOADER.PY DOES TO HANDLE OVERLAP")
print("="*80)

df_combined = pd.concat([df1, df2], ignore_index=True)
print(f"\nBefore deduplication: {len(df_combined):,} rows")

# Dedup like loader.py does
df_dedup = df_combined.drop_duplicates(subset=['datetime', 'zone'], keep='first')
print(f"After deduplication: {len(df_dedup):,} rows")
print(f"Removed: {len(df_combined) - len(df_dedup):,} duplicate datetime/zone pairs")

print("\n✅ Keeps FIRST forecast (earliest issued) for each datetime/zone")
print("   This means: use the forecast that was issued earliest")
print("   Example: For 2024-07-07 10:00, keeps the forecast issued 2024-07-07 01:00")
print("            instead of the one issued 2024-07-08 01:00")
