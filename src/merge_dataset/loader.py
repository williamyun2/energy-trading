"""
Data Loader - ERCOT Energy Trading Project
Loads and merges all datasets with caching support

FIXED: Proper deduplication to prevent data leakage

Usage:
    from merge_dataset.loader import load_clean_data
    df = load_clean_data()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project paths
BASE_DIR = Path(r"D:\Users\williamyun\proj\power_trading")
PROCESSED_DIR = BASE_DIR / "data_processed"
CACHE_DIR = PROCESSED_DIR / "merged_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "master_training_data.parquet"


def load_training_data(use_cache=True, rebuild_cache=False, date_range=None, verbose=True):
    """Load merged training dataset"""
    if verbose:
        print("="*80)
        print("LOADING TRAINING DATA")
        print("="*80)
    
    if use_cache and CACHE_FILE.exists() and not rebuild_cache:
        if verbose:
            print(f"\nLoading from cache: {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
        if verbose:
            print(f"  Loaded {len(df):,} records")
    else:
        if verbose:
            print("\nMerging fresh data...")
        df = merge_all_datasets(verbose=verbose)
        if use_cache:
            df.to_parquet(CACHE_FILE, index=False)
            if verbose:
                print(f"\nSaved to cache")
    
    if date_range:
        start, end = date_range
        original_len = len(df)
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
        if verbose:
            print(f"\nFiltered to {start} - {end}: {len(df):,} records ({original_len - len(df):,} dropped)")
    
    return df


def merge_all_datasets(verbose=True):
    """Merge all 4 datasets"""
    
    # 1. DAM Prices
    if verbose:
        print("\n[1/4] Loading DAM Prices...")
    dam_path = PROCESSED_DIR / "ercot" / "combined_dam_prices.csv"
    df_dam = pd.read_csv(dam_path)
    df_dam['Delivery Date'] = pd.to_datetime(df_dam['Delivery Date'])
    df_dam['HE'] = df_dam['Hour Ending'].str.split(':').str[0].astype(int)
    df_dam['datetime'] = df_dam['Delivery Date'] + pd.to_timedelta(df_dam['HE'] - 1, unit='h')
    df_dam.loc[df_dam['HE'] == 24, 'datetime'] = df_dam.loc[df_dam['HE'] == 24, 'Delivery Date'] + pd.Timedelta(days=1)
    df_dam = df_dam[df_dam['Settlement Point'] == 'HB_BUSAVG'].copy()
    df_dam = df_dam[['datetime', 'Settlement Point Price']].rename(columns={'Settlement Point Price': 'dam_price'})
    
    # FIX 1: Deduplicate DAM prices (keep first occurrence)
    original_dam_len = len(df_dam)
    df_dam = df_dam.drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)
    if verbose:
        print(f"   Loaded {len(df_dam):,} hourly prices")
        if original_dam_len > len(df_dam):
            print(f"   WARNING: Removed {original_dam_len - len(df_dam):,} duplicate datetimes")

    # 2. Natural Gas - Yahoo
    if verbose:
        print("\n[2/4] Loading Natural Gas (Yahoo)...")
    ng_path = PROCESSED_DIR / "fuel" / "NG" / "ng_futures_yahoo_daily_2010-12-01_2025-12-26.csv"
    df_ng = pd.read_csv(ng_path, parse_dates=['Date'])

    # Remove timezone if present (Yahoo data includes timezone)
    if pd.api.types.is_datetime64tz_dtype(df_ng['Date']):
        df_ng['Date'] = df_ng['Date'].dt.tz_convert(None)
    elif pd.api.types.is_datetime64_any_dtype(df_ng['Date']):
        # Already timezone-naive datetime
        pass
    else:
        # Not a datetime, try to convert
        df_ng['Date'] = pd.to_datetime(df_ng['Date'], utc=True).dt.tz_localize(None)

    df_ng = df_ng.rename(columns={'Close': 'ng_price'})[['Date', 'ng_price']].copy()

    # FIX 2: Deduplicate natural gas prices
    original_ng_len = len(df_ng)
    df_ng = df_ng.drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)
    if verbose:
        print(f"   Loaded {len(df_ng):,} daily prices")
        if original_ng_len > len(df_ng):
            print(f"   WARNING: Removed {original_ng_len - len(df_ng):,} duplicate dates")

    # 3. Weather
    if verbose:
        print("\n[3/4] Loading Weather...")
    weather_dir = PROCESSED_DIR / "weather" / "hrrr"
    weather_files = sorted(weather_dir.glob("hrrr_texas_*.parquet"))
    weather_dfs = []
    for i, file in enumerate(weather_files):
        if verbose and i % 500 == 0:
            print(f"      File {i+1}/{len(weather_files)}...")
        weather_dfs.append(pd.read_parquet(file))
    df_weather = pd.concat(weather_dfs, ignore_index=True)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

    # FIX 3 (IMPROVED): Keep shortest forecast horizon (most accurate) for each datetime/zone
    # Strategy: For overlapping forecasts, shorter horizon = more recent = more accurate
    original_weather_len = len(df_weather)

    # Sort by forecast_hour (ascending) so shortest horizon comes first
    df_weather = df_weather.sort_values('forecast_hour').reset_index(drop=True)

    # Keep first (shortest horizon) for each datetime/zone
    df_weather = df_weather.drop_duplicates(subset=['datetime', 'zone'], keep='first').reset_index(drop=True)

    if verbose:
        print(f"   Loaded {len(df_weather):,} weather records")
        if original_weather_len > len(df_weather):
            print(f"   Removed {original_weather_len - len(df_weather):,} duplicate datetime/zone combinations")
            print(f"   Kept SHORTEST forecast horizon (most accurate) for each datetime")

    # 4. Load Forecasts
    if verbose:
        print("\n[4/4] Loading Load Forecasts...")
    load_path = PROCESSED_DIR / "ercot" / "historical_load" / "load_forecast_complete.csv"
    df_load = pd.read_csv(load_path)
    df_load['datetime'] = pd.to_datetime(df_load['datetime'])
    
    # FIX 4: CRITICAL - Deduplicate load forecasts (keep first forecast for each datetime)
    # This is the main cause of the 25M duplicate rows!
    original_load_len = len(df_load)
    df_load = df_load.drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)
    if verbose:
        print(f"   Loaded {len(df_load):,} forecasts")
        if original_load_len > len(df_load):
            print(f"   WARNING: Removed {original_load_len - len(df_load):,} duplicate datetimes (CRITICAL FIX!)")

    # 5. Merge
    if verbose:
        print("\n[5/5] Merging datasets...")
    df_merged = df_dam.copy()
    
    # Add natural gas
    df_merged['date'] = df_merged['datetime'].dt.date
    df_ng['date'] = df_ng['Date'].dt.date
    df_merged = df_merged.merge(df_ng[['date', 'ng_price']], on='date', how='left')
    df_merged['ng_price'] = df_merged['ng_price'].fillna(method='ffill')
    
    # Add weather
    weather_pivot = df_weather.pivot_table(
        index='datetime', columns='zone',
        values=['temp_f', 'wind_speed_mph', 'solar_radiation_wm2', 'relative_humidity'],
        aggfunc='mean'
    )
    weather_pivot.columns = ['_'.join(col) for col in weather_pivot.columns.values]
    weather_pivot = weather_pivot.reset_index()
    df_merged = df_merged.merge(weather_pivot, on='datetime', how='left')
    
    # Add load
    if 'systemTotal' in df_load.columns:
        df_load['system_load_forecast'] = pd.to_numeric(df_load['systemTotal'], errors='coerce')
    df_merged = df_merged.merge(df_load[['datetime', 'system_load_forecast']], on='datetime', how='left')
    
    # FIX 5: Final safety check - ensure no duplicates in merged data
    original_merged_len = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)
    if verbose and original_merged_len > len(df_merged):
        print(f"   WARNING: Final dedup removed {original_merged_len - len(df_merged):,} duplicate rows")

    df_merged = df_merged.drop('date', axis=1).sort_values('datetime').reset_index(drop=True)

    if verbose:
        print(f"\nMerge complete! Shape: {df_merged.shape}")
        print(f"   Expected ~{8.5*365*24:,.0f} rows for 8.5 years of hourly data")
        print(f"   Actual: {len(df_merged):,} rows")

        # Verify no duplicates
        n_duplicates = df_merged.duplicated(subset=['datetime']).sum()
        if n_duplicates > 0:
            print(f"   ERROR: Still have {n_duplicates:,} duplicate datetimes!")
        else:
            print(f"   OK: No duplicate datetimes")
        
        missing = df_merged.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing values:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]:,} ({missing[col]/len(df_merged)*100:.1f}%)")
    
    return df_merged

def load_clean_data(verbose=True):
    """Load clean data ready for modeling with smart missing data handling"""
    if verbose:
        print("="*80)
        print("LOADING CLEAN DATA FOR MODELING")
        print("="*80)
    
    df = load_training_data(use_cache=True, date_range=('2017-06-01', '2025-11-23'), verbose=verbose)
    
    if verbose:
        print("\n" + "="*80)
        print("CLEANING DATA")
        print("="*80)
        print(f"Before: {len(df):,} rows")
    
    # Smart missing data handling instead of aggressive dropna()
    # 1. Drop rows missing critical columns (target and key features)
    critical_cols = ['dam_price', 'ng_price', 'system_load_forecast']
    df_clean = df.dropna(subset=critical_cols)
    
    if verbose:
        print(f"After dropping rows missing critical columns: {len(df_clean):,} rows")
        print(f"  Dropped: {len(df) - len(df_clean):,} rows with missing dam_price, ng_price, or load_forecast")
    
    # 2. Impute weather data (forward fill, then backward fill for any remaining)
    weather_cols = [c for c in df_clean.columns if any(x in c for x in ['temp_f', 'wind_speed', 'solar_radiation', 'relative_humidity'])]
    
    if weather_cols:
        # Count missing before imputation
        missing_before = df_clean[weather_cols].isnull().sum().sum()
        
        # Forward fill (use previous hour's weather)
        df_clean[weather_cols] = df_clean[weather_cols].fillna(method='ffill')
        
        # Backward fill for any remaining (start of dataset)
        df_clean[weather_cols] = df_clean[weather_cols].fillna(method='bfill')
        
        missing_after = df_clean[weather_cols].isnull().sum().sum()
        
        if verbose and missing_before > 0:
            print(f"  Imputed {missing_before - missing_after:,} missing weather values using forward/backward fill")
    
    # 3. Final check - drop any rows still missing data (should be very few)
    rows_before_final = len(df_clean)
    df_clean = df_clean.dropna()
    rows_dropped_final = rows_before_final - len(df_clean)
    
    if verbose:
        if rows_dropped_final > 0:
            print(f"  Final cleanup: dropped {rows_dropped_final:,} rows still missing data")
        print(f"\nAfter all cleaning: {len(df_clean):,} rows")
        print(f"Total dropped: {len(df) - len(df_clean):,} rows ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")
        
        # Final duplicate check
        n_duplicates = df_clean.duplicated(subset=['datetime']).sum()
        if n_duplicates > 0:
            print(f"   ERROR: Clean data has {n_duplicates:,} duplicate datetimes!")
        else:
            print(f"   OK: No duplicate datetimes in clean data")

        if df_clean.isnull().sum().sum() == 0:
            print("\nCLEAN DATA READY!")
            print(f"   {len(df_clean):,} rows x {len(df_clean.columns)} columns")
            print(f"   {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
            print(f"   Data retention: {len(df_clean)/len(df)*100:.1f}%")
    
    return df_clean



def clear_cache():
    """Delete cache"""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print(f"Deleted cache: {CACHE_FILE}")
    else:
        print("No cache to delete")


if __name__ == "__main__":
    df = load_clean_data()
    print(f"\nSUCCESS: {len(df):,} clean rows ready for modeling")