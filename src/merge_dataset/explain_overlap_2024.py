"""
Demonstrate weather forecast overlap for 48-hour forecasts (2024 data)
"""
import pandas as pd
from pathlib import Path

weather_dir = Path(r"D:\Users\williamyun\proj\power_trading\data_processed\weather\hrrr")

# Get two consecutive 2024 files
weather_files_2024 = sorted([f for f in weather_dir.glob("hrrr_texas_2024*.parquet")])[:2]

if len(weather_files_2024) < 2:
    print("Need at least 2 files from 2024")
    exit()

print("="*80)
print("WEATHER FORECAST OVERLAP - 48 HOUR FORECASTS")
print("="*80)

# Load first two files
df1 = pd.read_parquet(weather_files_2024[0])
df2 = pd.read_parquet(weather_files_2024[1])

print(f"\nFile 1: {weather_files_2024[0].name}")
print(f"  Forecast issued: {df1['forecast_issued'].iloc[0]}")
print(f"  Forecast hours: {df1['forecast_hour'].min()} to {df1['forecast_hour'].max()}")
print(f"  Valid times: {df1['datetime'].min()} to {df1['datetime'].max()}")
print(f"  Total rows: {len(df1):,} ({len(df1)//4} hours x 4 zones)")

print(f"\nFile 2: {weather_files_2024[1].name}")
print(f"  Forecast issued: {df2['forecast_issued'].iloc[0]}")
print(f"  Forecast hours: {df2['forecast_hour'].min()} to {df2['forecast_hour'].max()}")
print(f"  Valid times: {df2['datetime'].min()} to {df2['datetime'].max()}")
print(f"  Total rows: {len(df2):,} ({len(df2)//4} hours x 4 zones)")

# Check overlap for HOUSTON zone
df1_houston = df1[df1['zone'] == 'HOUSTON'].sort_values('datetime')
df2_houston = df2[df2['zone'] == 'HOUSTON'].sort_values('datetime')

df1_datetimes = set(df1_houston['datetime'])
df2_datetimes = set(df2_houston['datetime'])
overlap = sorted(df1_datetimes & df2_datetimes)

print(f"\n{'='*80}")
print("OVERLAP ANALYSIS (HOUSTON zone)")
print('='*80)
print(f"File 1 covers: {len(df1_datetimes)} hours")
print(f"File 2 covers: {len(df2_datetimes)} hours")
print(f"Overlapping hours: {len(overlap)}")

if overlap:
    print(f"\nOverlapping time range: {min(overlap)} to {max(overlap)}")
    print(f"\nExample: Same datetime predicted by TWO different forecasts")
    print("="*80)

    # Show 3 examples from the overlap
    for dt in overlap[:3]:
        f1_row = df1_houston[df1_houston['datetime'] == dt].iloc[0]
        f2_row = df2_houston[df2_houston['datetime'] == dt].iloc[0]

        print(f"\nTarget datetime: {dt}")
        print(f"  Forecast 1: issued {f1_row['forecast_issued']} (f{f1_row['forecast_hour']:02d})")
        print(f"    temp={f1_row['temp_f']:.1f}F, solar={f1_row['solar_radiation_wm2']:.1f} W/m2")
        print(f"  Forecast 2: issued {f2_row['forecast_issued']} (f{f2_row['forecast_hour']:02d})")
        print(f"    temp={f2_row['temp_f']:.1f}F, solar={f2_row['solar_radiation_wm2']:.1f} W/m2")
        print(f"  Difference: temp={abs(f1_row['temp_f']-f2_row['temp_f']):.1f}F, solar={abs(f1_row['solar_radiation_wm2']-f2_row['solar_radiation_wm2']):.1f} W/m2")

print(f"\n{'='*80}")
print("HOW LOADER.PY HANDLES THIS")
print('='*80)
print("\nloader.py uses: drop_duplicates(subset=['datetime', 'zone'], keep='first')")
print("\nThis means:")
print("  - For overlapping datetimes, KEEPS the forecast issued EARLIER")
print("  - Example: For overlapping hours, uses forecast from Day 1 (longer horizon)")
print("  - Discards forecast from Day 2 (shorter horizon for those hours)")

# Show what gets kept
print(f"\nCombining these 2 files:")
df_combined = pd.concat([df1, df2], ignore_index=True)
print(f"  Before dedup: {len(df_combined):,} rows")

df_dedup = df_combined.drop_duplicates(subset=['datetime', 'zone'], keep='first')
print(f"  After dedup:  {len(df_dedup):,} rows")
print(f"  Removed:      {len(df_combined) - len(df_dedup):,} duplicate rows ({(len(df_combined) - len(df_dedup))/len(df_combined)*100:.1f}%)")

print("\n" + "="*80)
print("WHAT THIS MEANS FOR YOUR MODEL")
print("="*80)
print("\nPros:")
print("  + Consistent: Each datetime has ONE weather forecast")
print("  + Realistic: Uses forecast available at that time (no future leakage)")
print("  + Simple: No need to choose between forecasts")
print("\nCons:")
print("  - Variable forecast horizon: Some datetimes use f48 forecast, others use f00")
print("  - Forecast uncertainty: Longer horizon = less accurate")
print("  - Could be improved: Could use the CLOSEST forecast instead of FIRST")
