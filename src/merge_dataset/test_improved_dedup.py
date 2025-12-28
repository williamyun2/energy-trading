"""
Test the improved weather deduplication logic
Compare old (keep first issued) vs new (keep shortest forecast_hour)
"""
import pandas as pd
from pathlib import Path

weather_dir = Path(r"D:\Users\williamyun\proj\power_trading\data_processed\weather\hrrr")
weather_files_2024 = sorted([f for f in weather_dir.glob("hrrr_texas_2024*.parquet")])[:3]

print("="*80)
print("TEST: OLD vs NEW DEDUPLICATION STRATEGY")
print("="*80)

# Load 3 consecutive days to have overlap
dfs = [pd.read_parquet(f) for f in weather_files_2024[:3]]
df_combined = pd.concat(dfs, ignore_index=True)
df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])

print(f"\nLoaded {len(weather_files_2024[:3])} files")
print(f"Combined rows: {len(df_combined):,}")
print(f"Date range: {df_combined['datetime'].min()} to {df_combined['datetime'].max()}")

# OLD METHOD: Keep first (by file order, which is by forecast_issued date)
print("\n" + "="*80)
print("OLD METHOD: Keep first forecast (earliest issued)")
print("="*80)
df_old = df_combined.drop_duplicates(subset=['datetime', 'zone'], keep='first').reset_index(drop=True)
print(f"Rows after dedup: {len(df_old):,}")
print(f"Removed: {len(df_combined) - len(df_old):,} rows")

# Check what forecast horizons we kept
houston_old = df_old[df_old['zone'] == 'HOUSTON'].sort_values('datetime')
print(f"\nForecast horizon distribution (HOUSTON zone):")
print(f"  Min: f{houston_old['forecast_hour'].min():02d}")
print(f"  Max: f{houston_old['forecast_hour'].max():02d}")
print(f"  Mean: f{houston_old['forecast_hour'].mean():.1f}")
print(f"  Median: f{houston_old['forecast_hour'].median():.0f}")

# NEW METHOD: Keep shortest forecast_hour
print("\n" + "="*80)
print("NEW METHOD: Keep shortest forecast horizon (most accurate)")
print("="*80)
df_new = df_combined.sort_values('forecast_hour').reset_index(drop=True)
df_new = df_new.drop_duplicates(subset=['datetime', 'zone'], keep='first').reset_index(drop=True)
print(f"Rows after dedup: {len(df_new):,}")
print(f"Removed: {len(df_combined) - len(df_new):,} rows")

houston_new = df_new[df_new['zone'] == 'HOUSTON'].sort_values('datetime')
print(f"\nForecast horizon distribution (HOUSTON zone):")
print(f"  Min: f{houston_new['forecast_hour'].min():02d}")
print(f"  Max: f{houston_new['forecast_hour'].max():02d}")
print(f"  Mean: f{houston_new['forecast_hour'].mean():.1f}")
print(f"  Median: f{houston_new['forecast_hour'].median():.0f}")

# Compare specific examples
print("\n" + "="*80)
print("COMPARISON: Same datetimes, different forecasts kept")
print("="*80)

# Find overlapping datetimes
overlap_datetimes = sorted(set(houston_old['datetime']) & set(houston_new['datetime']))
print(f"\nTotal overlapping datetimes: {len(overlap_datetimes)}")

# Show differences for first 5 overlapping times
print("\nExamples where methods differ:")
print("="*80)

diff_count = 0
for dt in overlap_datetimes[:20]:  # Check first 20
    old_row = houston_old[houston_old['datetime'] == dt].iloc[0]
    new_row = houston_new[houston_new['datetime'] == dt].iloc[0]

    # Check if they picked different forecasts
    if old_row['forecast_hour'] != new_row['forecast_hour']:
        diff_count += 1
        if diff_count <= 5:  # Show first 5 differences
            print(f"\nDatetime: {dt}")
            print(f"  OLD: f{old_row['forecast_hour']:02d} | temp={old_row['temp_f']:.1f}F, solar={old_row['solar_radiation_wm2']:.1f}")
            print(f"  NEW: f{new_row['forecast_hour']:02d} | temp={new_row['temp_f']:.1f}F, solar={new_row['solar_radiation_wm2']:.1f}")
            print(f"  Improvement: Used forecast {old_row['forecast_hour'] - new_row['forecast_hour']} hours closer")

print(f"\n{diff_count} datetimes had different forecasts selected")

# Calculate improvement
print("\n" + "="*80)
print("IMPROVEMENT SUMMARY")
print("="*80)
print(f"OLD method average forecast horizon: f{houston_old['forecast_hour'].mean():.1f}")
print(f"NEW method average forecast horizon: f{houston_new['forecast_hour'].mean():.1f}")
improvement = houston_old['forecast_hour'].mean() - houston_new['forecast_hour'].mean()
print(f"Average improvement: {improvement:.1f} hours closer to target time")
print(f"\nBENEFIT: Forecasts are on average {improvement:.1f}hr more recent")
print(f"         = More accurate weather predictions for your model!")
