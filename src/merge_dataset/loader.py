"""
Data Loader - ERCOT Energy Trading Project
Loads and merges all datasets with caching support

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
            print(f"\n✓ Loading from cache: {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
        if verbose:
            print(f"  Loaded {len(df):,} records")
    else:
        if verbose:
            print("\n🔄 Merging fresh")
        df = merge_all_datasets(verbose=verbose)
        if use_cache:
            df.to_parquet(CACHE_FILE, index=False)
            if verbose:
                print(f"\n💾 Saved to cache")
    
    if date_range:
        start, end = date_range
        original_len = len(df)
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
        if verbose:
            print(f"\n📅 Filtered to {start} - {end}: {len(df):,} records ({original_len - len(df):,} dropped)")
    
    return df


def merge_all_datasets(verbose=True):
    """Merge all 4 datasets"""
    
    # 1. DAM Prices
    if verbose:
        print("\n1️⃣ Loading DAM Prices...")
    dam_path = PROCESSED_DIR / "ercot" / "combined_dam_prices.csv"
    df_dam = pd.read_csv(dam_path)
    df_dam['Delivery Date'] = pd.to_datetime(df_dam['Delivery Date'])
    df_dam['HE'] = df_dam['Hour Ending'].str.split(':').str[0].astype(int)
    df_dam['datetime'] = df_dam['Delivery Date'] + pd.to_timedelta(df_dam['HE'] - 1, unit='h')
    df_dam.loc[df_dam['HE'] == 24, 'datetime'] = df_dam.loc[df_dam['HE'] == 24, 'Delivery Date'] + pd.Timedelta(days=1)
    df_dam = df_dam[df_dam['Settlement Point'] == 'HB_BUSAVG'].copy()
    df_dam = df_dam[['datetime', 'Settlement Point Price']].rename(columns={'Settlement Point Price': 'dam_price'})
    if verbose:
        print(f"   ✓ Loaded {len(df_dam):,} hourly prices")
    
    # 2. Natural Gas - Yahoo
    if verbose:
        print("\n2️⃣ Loading Natural Gas (Yahoo)...")
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
    if verbose:
        print(f"   ✓ Loaded {len(df_ng):,} daily prices")
    
    # 3. Weather
    if verbose:
        print("\n3️⃣ Loading Weather...")
    weather_dir = PROCESSED_DIR / "weather" / "hrrr"
    weather_files = sorted(weather_dir.glob("hrrr_texas_*.parquet"))
    weather_dfs = []
    for i, file in enumerate(weather_files):
        if verbose and i % 500 == 0:
            print(f"      File {i+1}/{len(weather_files)}...")
        weather_dfs.append(pd.read_parquet(file))
    df_weather = pd.concat(weather_dfs, ignore_index=True)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    if verbose:
        print(f"   ✓ Loaded {len(df_weather):,} weather records")
    
    # 4. Load Forecasts
    if verbose:
        print("\n4️⃣ Loading Load Forecasts...")
    load_path = PROCESSED_DIR / "ercot" / "historical_load" / "load_forecast_complete.csv"
    df_load = pd.read_csv(load_path)
    df_load['datetime'] = pd.to_datetime(df_load['deliveryDate'])
    if verbose:
        print(f"   ✓ Loaded {len(df_load):,} forecasts")
    
    # 5. Merge
    if verbose:
        print("\n5️⃣ Merging...")
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
    
    df_merged = df_merged.drop('date', axis=1).sort_values('datetime').reset_index(drop=True)
    
    if verbose:
        print(f"\n✓ Merge complete! Shape: {df_merged.shape}")
        missing = df_merged.isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]:,} ({missing[col]/len(df_merged)*100:.1f}%)")
    
    return df_merged


def load_clean_data(verbose=True):
    """Load clean data ready for modeling"""
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
    
    df_clean = df.dropna()
    
    if verbose:
        print(f"After dropna(): {len(df_clean):,} rows")
        print(f"Dropped: {len(df) - len(df_clean):,} rows")
        if df_clean.isnull().sum().sum() == 0:
            print("\n✅ CLEAN DATA READY!")
            print(f"   {len(df_clean):,} rows × {len(df_clean.columns)} columns")
            print(f"   {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
    
    return df_clean


def clear_cache():
    """Delete cache"""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print(f"✓ Deleted cache")
    else:
        print("No cache to delete")


if __name__ == "__main__":
    df = load_clean_data()
    print(f"\n✅ SUCCESS: {len(df):,} clean rows ready for modeling")