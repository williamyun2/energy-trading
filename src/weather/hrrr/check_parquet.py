"""
Check if HRRR parquet file is good - validates structure and data quality.
"""

import pandas as pd
import numpy as np

def check_parquet_file(filepath):
    """Validate HRRR weather parquet file"""
    
    print(f"Checking: {filepath}\n")
    
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"❌ FAILED to load file: {e}")
        return False
    
    print(f"✓ File loaded successfully")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {df.columns.tolist()}\n")
    
    # Check required columns
    required_cols = [
        'datetime', 'forecast_issued', 'forecast_hour', 'zone',
        'temp_f', 'dewpoint_f', 'relative_humidity',
        'wind_speed_mph', 'wind_gust_mph', 'wind_direction',
        'solar_radiation_wm2', 'cloud_cover_pct', 'precipitation_rate',
        'pressure_inhg'
    ]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    
    print(f"✓ All required columns present")
    
    # Check zones
    expected_zones = {'HOUSTON', 'NORTH', 'SOUTH', 'WEST'}
    actual_zones = set(df['zone'].unique())
    
    if actual_zones != expected_zones:
        print(f"⚠️  Expected zones: {expected_zones}")
        print(f"   Actual zones: {actual_zones}")
    else:
        print(f"✓ All 4 zones present: {sorted(actual_zones)}")
    
    # Check for nulls
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        print(f"\n⚠️  Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"   {col}: {count} nulls ({count/len(df)*100:.1f}%)")
    else:
        print(f"✓ No null values")
    
    # Check data ranges (sanity checks)
    print(f"\n=== Data Range Checks ===")
    
    checks_passed = True
    
    # Temperature (reasonable for Texas)
    temp_min, temp_max = df['temp_f'].min(), df['temp_f'].max()
    print(f"Temperature: {temp_min:.1f}°F to {temp_max:.1f}°F", end='')
    if temp_min < -20 or temp_max > 130:
        print(" ⚠️  Outside reasonable range")
        checks_passed = False
    else:
        print(" ✓")
    
    # Wind speed
    wind_min, wind_max = df['wind_speed_mph'].min(), df['wind_speed_mph'].max()
    print(f"Wind speed: {wind_min:.1f} to {wind_max:.1f} mph", end='')
    if wind_min < 0 or wind_max > 100:
        print(" ⚠️  Outside reasonable range")
        checks_passed = False
    else:
        print(" ✓")
    
    # Solar radiation
    solar_min, solar_max = df['solar_radiation_wm2'].min(), df['solar_radiation_wm2'].max()
    print(f"Solar radiation: {solar_min:.1f} to {solar_max:.1f} W/m²", end='')
    if solar_min < 0 or solar_max > 1500:
        print(" ⚠️  Outside reasonable range")
        checks_passed = False
    else:
        print(" ✓")
    
    # Relative humidity
    rh_min, rh_max = df['relative_humidity'].min(), df['relative_humidity'].max()
    print(f"Relative humidity: {rh_min:.1f}% to {rh_max:.1f}%", end='')
    if rh_min < 0 or rh_max > 100:
        print(" ⚠️  Outside reasonable range")
        checks_passed = False
    else:
        print(" ✓")
    
    # Barometric pressure (typical range: 27.0-31.5 inHg, allows for storms/hurricanes)
    if 'pressure_inhg' in df.columns:
        pres_min, pres_max = df['pressure_inhg'].min(), df['pressure_inhg'].max()
        print(f"Pressure: {pres_min:.2f} to {pres_max:.2f} inHg", end='')
        if pres_min < 26.0 or pres_max > 32.0:
            print(" ⚠️  Extreme pressure (hurricane/major storm?)")
            checks_passed = False
        else:
            print(" ✓")
    
    # Check forecast hours are sequential
    print(f"\n=== Temporal Checks ===")
    forecast_hours = sorted(df['forecast_hour'].unique())
    print(f"Forecast hours: {forecast_hours[0]:.0f} to {forecast_hours[-1]:.0f} ({len(forecast_hours)} hours)")
    
    expected_hours = list(range(int(forecast_hours[0]), int(forecast_hours[-1]) + 1))
    if len(forecast_hours) != len(expected_hours):
        print(f"⚠️  Missing forecast hours")
        checks_passed = False
    else:
        print(f"✓ All forecast hours present")
    
    # Check each zone has same number of records
    zone_counts = df.groupby('zone').size()
    print(f"\nRecords per zone:")
    for zone, count in zone_counts.items():
        print(f"  {zone:8s}: {count:3d} rows")
    
    if zone_counts.nunique() != 1:
        print(f"⚠️  Zones have different record counts")
        checks_passed = False
    else:
        print(f"✓ All zones have equal records")
    
    # Sample data
    print(f"\n=== Sample Data (first 3 rows) ===")
    sample_cols = ['datetime', 'zone', 'temp_f', 'wind_speed_mph', 'solar_radiation_wm2']
    print(df[sample_cols].head(3).to_string(index=False))
    
    # DEEP CHECKS
    print(f"\n=== DEEP QUALITY CHECKS ===")
    
    deep_checks_passed = True
    
    # 1. Check for constant values (stuck sensor)
    print("\n1. Checking for constant values (stuck sensors):")
    for col in ['temp_f', 'dewpoint_f', 'wind_speed_mph', 'solar_radiation_wm2']:
        if col in df.columns:
            for zone in df['zone'].unique():
                zone_data = df[df['zone'] == zone][col]
                unique_ratio = zone_data.nunique() / len(zone_data)
                if unique_ratio < 0.3:  # Less than 30% unique values
                    print(f"   ⚠️  {zone} {col}: Only {unique_ratio*100:.1f}% unique values")
                    deep_checks_passed = False
    print("   ✓ No stuck sensors detected")
    
    # 2. Check temporal consistency (values should change gradually)
    print("\n2. Checking temporal consistency (no sudden jumps):")
    for col in ['temp_f', 'dewpoint_f']:
        if col in df.columns:
            for zone in df['zone'].unique():
                zone_data = df[df['zone'] == zone].sort_values('datetime')
                diffs = zone_data[col].diff().abs()
                max_diff = diffs.max()
                # Temperature shouldn't jump more than 20°F in 1 hour
                if max_diff > 20:
                    print(f"   ⚠️  {zone} {col}: Max hourly change {max_diff:.1f}°F")
                    deep_checks_passed = False
    print("   ✓ No unrealistic jumps detected")
    
    # 3. Check solar radiation follows day/night cycle
    print("\n3. Checking solar radiation day/night cycle:")
    if 'solar_radiation_wm2' in df.columns:
        # Sort by datetime to see actual progression
        df_sample = df[df['zone'] == df['zone'].iloc[0]].sort_values('datetime')
        
        print("   Solar radiation progression (first zone):")
        for idx, row in df_sample.iterrows():
            dt = pd.to_datetime(row['datetime'])
            solar = row['solar_radiation_wm2']
            marker = "🌙" if solar < 10 else "🌤️" if solar < 200 else "☀️"
            print(f"     {dt.strftime('%m-%d %H:%M')}  {solar:6.1f} W/m² {marker}")
        
        # True night check: solar should be ~0 during true darkness
        # Use forecast_hour to identify nighttime (varies by season)
        # For a 6am forecast: midnight-5am next day = forecast hours 18-23
        true_night = df[(df['forecast_hour'] >= 18) & (df['forecast_hour'] <= 23)]['solar_radiation_wm2']
        
        if len(true_night) > 0:
            night_mean = true_night.mean()
            night_max = true_night.max()
            if night_mean > 10 or night_max > 50:
                print(f"\n   ⚠️  Nighttime solar: avg={night_mean:.1f}, max={night_max:.1f} W/m² (expected near 0)")
                deep_checks_passed = False
            else:
                print(f"\n   ✓ Nighttime solar: avg={night_mean:.1f}, max={night_max:.1f} W/m²")
        
        # Peak solar check: should see >300 W/m² somewhere during the day
        if df['solar_radiation_wm2'].max() < 300:
            print(f"   ⚠️  Peak solar: {df['solar_radiation_wm2'].max():.1f} W/m² (seems low)")
            deep_checks_passed = False
        else:
            print(f"   ✓ Peak solar: {df['solar_radiation_wm2'].max():.1f} W/m²")
    
    # 4. Check dewpoint <= temperature (physical constraint)
    print("\n4. Checking dewpoint ≤ temperature (physics check):")
    if 'temp_f' in df.columns and 'dewpoint_f' in df.columns:
        violations = (df['dewpoint_f'] > df['temp_f']).sum()
        if violations > 0:
            print(f"   ⚠️  {violations} cases where dewpoint > temperature (impossible!)")
            deep_checks_passed = False
        else:
            print(f"   ✓ Dewpoint ≤ temperature in all cases")
    
    # 5. Check relative humidity bounds
    print("\n5. Checking relative humidity bounds:")
    if 'relative_humidity' in df.columns:
        if (df['relative_humidity'] < 0).any() or (df['relative_humidity'] > 100).any():
            print(f"   ⚠️  RH outside 0-100% range")
            deep_checks_passed = False
        else:
            print(f"   ✓ RH within valid 0-100% range")
    
    # 6. Check wind direction bounds
    print("\n6. Checking wind direction bounds:")
    if 'wind_direction' in df.columns:
        if (df['wind_direction'] < 0).any() or (df['wind_direction'] >= 360).any():
            print(f"   ⚠️  Wind direction outside 0-360° range")
            deep_checks_passed = False
        else:
            print(f"   ✓ Wind direction within valid 0-360° range")
    
    # 7. Check zone-to-zone consistency (zones shouldn't be wildly different)
    print("\n7. Checking zone-to-zone consistency:")
    if 'temp_f' in df.columns:
        for fh in df['forecast_hour'].unique():
            fh_data = df[df['forecast_hour'] == fh]
            temp_range = fh_data['temp_f'].max() - fh_data['temp_f'].min()
            # Texas zones shouldn't differ by more than 40°F at same time
            if temp_range > 40:
                print(f"   ⚠️  Forecast hour {fh}: {temp_range:.1f}°F spread across zones (seems high)")
                deep_checks_passed = False
        print(f"   ✓ Zone temperatures reasonably consistent")
    
    # 8. Statistical summary per zone
    print("\n8. Statistical summary by zone:")
    summary_cols = ['temp_f', 'wind_speed_mph', 'solar_radiation_wm2']
    for zone in sorted(df['zone'].unique()):
        print(f"\n   {zone}:")
        zone_data = df[df['zone'] == zone]
        for col in summary_cols:
            if col in df.columns:
                mean_val = zone_data[col].mean()
                std_val = zone_data[col].std()
                print(f"     {col:25s}: {mean_val:6.1f} ± {std_val:5.1f}")
    
    # 9. Check forecast_issued is constant
    print("\n9. Checking forecast metadata:")
    if df['forecast_issued'].nunique() != 1:
        print(f"   ⚠️  Multiple forecast issue times in single file")
        deep_checks_passed = False
    else:
        print(f"   ✓ Single forecast issue time: {df['forecast_issued'].iloc[0]}")
    
    # 10. Check datetime is sequential
    print("\n10. Checking datetime sequence:")
    for zone in df['zone'].unique():
        zone_data = df[df['zone'] == zone].sort_values('datetime')
        time_diffs = zone_data['datetime'].diff().dt.total_seconds() / 3600
        if not (time_diffs[1:] == 1.0).all():
            print(f"   ⚠️  {zone}: Non-hourly timestamps detected")
            deep_checks_passed = False
    print(f"   ✓ All zones have hourly timestamps")
    
    # Final verdict
    print(f"\n{'='*50}")
    all_checks = checks_passed and deep_checks_passed and len(missing) == 0 and not null_counts.any()
    if all_checks:
        print("✓✓✓ FILE IS EXCELLENT - All quality checks passed!")
        print("Ready for ML training with high confidence.")
        return True
    elif checks_passed and len(missing) == 0:
        print("✓ FILE IS ACCEPTABLE - Basic checks passed, some warnings.")
        print("Review deep check warnings above before training.")
        return True
    else:
        print("⚠️  FILE HAS ISSUES - Review warnings above")
        return False


if __name__ == "__main__":
    # Check your file
    filepath = r"D:\Users\williamyun\proj\energy_trading\data_processed\weather\hrrr\hrrr_texas_20240101_0600.parquet"
    
    check_parquet_file(filepath)



    filepath = r"D:\Users\williamyun\proj\energy_trading\data_processed\weather\hrrr\hrrr_texas_20240707_0600.parquet"
    
    check_parquet_file(filepath)

