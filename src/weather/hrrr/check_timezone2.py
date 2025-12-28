"""
Verify that the final parquet files have correct Central Time timestamps
"""
import pandas as pd
from datetime import datetime

# Load processed parquet file
parquet_file = r"D:\Users\williamyun\proj\power_trading\data_processed\weather\hrrr\hrrr_texas_20240707_0600.parquet"
df = pd.read_parquet(parquet_file)

print(f"Loaded: {parquet_file}\n")

# Check forecast issued time
print("Forecast issued time:")
print(f"  Raw value: {df['forecast_issued'].iloc[0]}")
print(f"  Expected: 2024-07-07 01:00:00 (6am UTC = 1am CDT)")
print()

# Check valid times for HOUSTON zone
houston = df[df['zone'] == 'HOUSTON'].sort_values('datetime')
print("First few datetime values (HOUSTON zone):")
print(houston[['datetime', 'forecast_hour', 'solar_radiation_wm2']].head(10))
print()

# Key question: When should sunrise be?
# July 7, 2024 in Houston, TX: sunrise ~6:30am CDT
# If our times are in CDT, we should see solar radiation increase around 6-7am
print("Solar radiation around sunrise (should be ~6-7am CDT for Houston in July):")
sunrise_window = houston[
    (houston['datetime'] >= '2024-07-07 05:00:00') &
    (houston['datetime'] <= '2024-07-07 09:00:00')
][['datetime', 'solar_radiation_wm2']]
print(sunrise_window)
print()

# The forecast was issued at 6am UTC = 1am CDT
# So forecast_hour=0 should be 1am CDT
# forecast_hour=6 should be 7am CDT (sunrise time)
print("Checking if forecast times align with Central Time:")
print(f"  forecast_hour=0: {houston[houston['forecast_hour']==0]['datetime'].iloc[0]} (should be ~1am)")
print(f"  forecast_hour=6: {houston[houston['forecast_hour']==6]['datetime'].iloc[0]} (should be ~7am)")
print(f"  Solar at hour=0: {houston[houston['forecast_hour']==0]['solar_radiation_wm2'].iloc[0]:.1f} W/m² (should be 0, nighttime)")
print(f"  Solar at hour=6: {houston[houston['forecast_hour']==6]['solar_radiation_wm2'].iloc[0]:.1f} W/m² (should be >0, sunrise)")
print()

print("✓ If times shown are 1am, 7am, etc., then timezone conversion is working correctly!")
print("✓ Solar radiation should be 0 at night (1am) and >0 at sunrise (7am)")
