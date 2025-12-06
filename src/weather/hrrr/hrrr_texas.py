"""
Download HRRR weather forecasts and aggregate to ERCOT load zones.
Smart mode: checks AWS for new files, then uses local cache for re-runs.
Automatically deletes GRIB files after processing to save space.
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ERCOT load zones (lat/lon bounds, overestimate)
LOAD_ZONES = {
    'HOUSTON': {'lat': (28.0, 31.0), 'lon': (-96.5, -94.0)},
    'NORTH': {'lat': (31.5, 34.5), 'lon': (-98.0, -95.0)},
    'SOUTH': {'lat': (26.5, 30.5), 'lon': (-100.0, -96.5)},
    'WEST': {'lat': (30.5, 33.5), 'lon': (-104.0, -99.5)}
}


def file_exists_locally(date, fxx, save_dir):
    """Check if GRIB file exists locally (fast, no network)"""
    date_str = date.strftime('%Y%m%d')
    hour_str = date.strftime('%H')
    
    # Herbie's file path pattern
    filepath = os.path.join(
        save_dir, 'hrrr', date_str,
        f"hrrr.t{hour_str}z.wrfsfcf{fxx:02d}.grib2"
    )
    
    return os.path.exists(filepath)


def cleanup_grib_files(date, save_dir):
    """Delete GRIB files for a specific forecast issue time to save space"""
    date_str = date.strftime('%Y%m%d')
    grib_dir = Path(save_dir) / 'hrrr' / date_str
    
    if not grib_dir.exists():
        return
    
    try:
        # Delete the entire day's directory
        shutil.rmtree(grib_dir)
        print(f"  🗑️  Cleaned up GRIB files for {date_str}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not delete GRIB files: {e}")


def download_hrrr_hour(date, fxx, save_dir, force_download=False):
    """
    Download single HRRR forecast hour.
    
    Args:
        date: forecast issue datetime
        fxx: forecast hour
        save_dir: cache directory
        force_download: if True, always check AWS (initial download)
                       if False, skip AWS if file exists locally (fast mode)
    """
    try:
        # Fast mode: skip AWS check if file exists locally
        if not force_download and file_exists_locally(date, fxx, save_dir):
            H = Herbie(date, model='hrrr', product='sfc', fxx=fxx, save_dir=save_dir)
            # File exists, just load it (no AWS check)
        else:
            # Initial download: check AWS
            H = Herbie(date, model='hrrr', product='sfc', fxx=fxx, save_dir=save_dir)
            H.download(verbose=False)
        
        # Download variables one at a time
        datasets = []
        
        vars_to_get = [
            ':TMP:2 m',           # Temperature
            ':DPT:2 m',           # Dewpoint  
            ':RH:2 m',            # Relative humidity
            ':UGRD:10 m',         # U wind component 10m
            ':VGRD:10 m',         # V wind component 10m
            ':GUST:surface',      # Wind gust
            ':DSWRF:surface',     # Solar radiation
            ':TCDC:entire',       # Total cloud cover
            ':PRATE:surface',     # Precipitation rate
            ':PRES:surface',      # Barometric pressure
        ]
        
        for var in vars_to_get:
            try:
                ds_var = H.xarray(var, remove_grib=False)
                if isinstance(ds_var, list):
                    ds_var = ds_var[0]
                datasets.append(ds_var)
            except:
                pass  # Skip missing variables
        
        if not datasets:
            return None
            
        # Merge with override
        ds = xr.merge(datasets, compat='override')
        
        return ds
        
    except Exception as e:
        print(f"Error downloading {date} f{fxx:02d}: {e}")
        return None


def process_weather(ds):
    """Convert units - use actual variable names from HRRR"""
    result = {}
    
    # Map HRRR variable names directly
    var_mapping = {
        't2m': 'temp',           # 2m temperature
        'd2m': 'dewpoint',       # 2m dewpoint
        'r2': 'rh',              # 2m relative humidity
        'u10': 'u10',            # 10m U wind
        'v10': 'v10',            # 10m V wind
        'gust': 'gust',          # Wind gust
        'sdswrf': 'solar',       # Solar radiation
        'tcc': 'clouds',         # Cloud cover
        'prate': 'precip',       # Precipitation rate
        'sp': 'pressure',        # Surface pressure
    }
    
    # Extract variables
    for hrrr_name, short_name in var_mapping.items():
        if hrrr_name in ds.data_vars:
            result[short_name] = ds[hrrr_name]
    
    # Temperature conversions (K to F)
    if 'temp' in result:
        result['temp_f'] = (result['temp'] - 273.15) * 9/5 + 32
        del result['temp']
    
    if 'dewpoint' in result:
        result['dewpoint_f'] = (result['dewpoint'] - 273.15) * 9/5 + 32
        del result['dewpoint']
    
    # Relative humidity (already in %)
    if 'rh' in result:
        result['relative_humidity'] = result['rh']
        del result['rh']
    
    # Wind calculations
    if 'u10' in result and 'v10' in result:
        result['wind_speed_mph'] = np.sqrt(result['u10']**2 + result['v10']**2) * 2.237
        result['wind_direction'] = (np.arctan2(-result['u10'], -result['v10']) * 180/np.pi) % 360
        del result['u10']
        del result['v10']
    
    # Wind gust (m/s to mph)
    if 'gust' in result:
        result['wind_gust_mph'] = result['gust'] * 2.237
        del result['gust']
    
    # Solar radiation (already in W/m2)
    if 'solar' in result:
        result['solar_radiation_wm2'] = result['solar']
        del result['solar']
    
    # Cloud cover (already in %)
    if 'clouds' in result:
        result['cloud_cover_pct'] = result['clouds']
        del result['clouds']
    
    # Precipitation rate (kg/m2/s to mm/hr)
    if 'precip' in result:
        result['precipitation_rate'] = result['precip'] * 3600
        del result['precip']
    
    # Barometric pressure (Pa to inHg, then relative to sea level)
    if 'pressure' in result:
        result['pressure_inhg'] = result['pressure'] * 0.0002953  # Pa to inHg
        del result['pressure']
    
    # Convert dict to dataset
    ds_out = xr.Dataset(result)
    
    return ds_out


def aggregate_to_zones(ds):
    """Aggregate to ERCOT load zones using lat/lon coordinates"""
    zone_data = []
    
    # Get lat/lon from the dataset
    if 'latitude' in ds.coords:
        lats = ds.latitude
        lons = ds.longitude
        
        # Convert longitude from 0-360 to -180-180 if needed
        if float(lons.max()) > 180:
            lons = xr.where(lons > 180, lons - 360, lons)
    else:
        print("Warning: No latitude/longitude coordinates found")
        return None
    
    for zone_name, bounds in LOAD_ZONES.items():
        # Create mask for this zone
        mask = (
            (lats >= bounds['lat'][0]) & 
            (lats <= bounds['lat'][1]) &
            (lons >= bounds['lon'][0]) & 
            (lons <= bounds['lon'][1])
        )
        
        # Apply mask and average
        zone_ds = ds.where(mask, drop=True)
        
        if zone_ds.sizes.get('x', 0) == 0 or zone_ds.sizes.get('y', 0) == 0:
            print(f"  Warning: No data points in zone {zone_name}")
            continue
        
        # Average across spatial dimensions
        zone_avg = zone_ds.mean(dim=['y', 'x'])
        
        # Convert to dataframe
        df = zone_avg.to_dataframe().reset_index()
        df['zone'] = zone_name
        
        zone_data.append(df)
    
    if not zone_data:
        return None
        
    combined = pd.concat(zone_data, ignore_index=True)
    
    # Rename time column
    if 'valid_time' in combined.columns:
        combined.rename(columns={'valid_time': 'datetime_utc'}, inplace=True)
    elif 'time' in combined.columns:
        combined.rename(columns={'time': 'datetime_utc'}, inplace=True)
    
    # Convert UTC to Central Time (ERCOT operates on Central Time)
    combined['datetime'] = pd.to_datetime(combined['datetime_utc']).dt.tz_localize('UTC').dt.tz_convert('US/Central')
    # Remove timezone info but keep Central Time values
    combined['datetime'] = combined['datetime'].dt.tz_localize(None)
    combined = combined.drop(columns=['datetime_utc'])
    
    # Drop extra coordinate columns that aren't weather data
    drop_cols = ['time', 'step', 'heightAboveGround', 'gribfile_projection', 'surface', 'atmosphere']
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])
    
    return combined


def download_hrrr_day(date, forecast_hours, save_dir, output_dir, force_download=None, keep_grib=False):
    """
    Download HRRR forecasts for a single issue time.
    
    Args:
        date: datetime of forecast issue time
        forecast_hours: list of forecast hours (e.g., range(0, 25))
        save_dir: where to cache GRIB files
        output_dir: where to save parquet files
        force_download: True=always check AWS, False=local only, None=auto-detect (recommended)
        keep_grib: if True, keep GRIB files after processing (default False to save space)
    """
    # Auto-detect if not specified
    if force_download is None:
        # Check if ANY forecast hour exists locally
        force_download = not file_exists_locally(date, forecast_hours[0], save_dir)
        mode_msg = "first download" if force_download else "using cache"
    else:
        mode_msg = "checking AWS" if force_download else "cache only"
    
    print(f"\nDownloading HRRR issued {date.strftime('%Y-%m-%d %H:00')} ({mode_msg})")
    
    ds_list = []
    for fxx in forecast_hours:
        print(f"  f{fxx:02d}...", end='', flush=True)
        ds = download_hrrr_hour(date, fxx, save_dir, force_download)
        if ds is not None:
            ds_list.append(ds)
            print("✓", end='', flush=True)
        else:
            print("✗", end='', flush=True)
    print()  # New line
    
    if not ds_list:
        print(f"No data downloaded for {date}")
        return
    
    print(f"  Combining {len(ds_list)} forecast hours...")
    ds_combined = xr.concat(ds_list, dim='valid_time')
    ds_combined = ds_combined.sortby('valid_time')
    
    print("  Processing variables...")
    ds_processed = process_weather(ds_combined)
    
    print("  Aggregating to zones...")
    df_zones = aggregate_to_zones(ds_processed)
    
    if df_zones is None:
        print(f"Failed to aggregate zones for {date}")
        return
    
    # Add forecast metadata
    df_zones['forecast_issued'] = date
    # Calculate forecast_hour properly (hours since forecast was issued, in local time)
    df_zones['forecast_hour'] = ((pd.to_datetime(df_zones['datetime']) - pd.to_datetime(date).tz_localize('UTC').tz_convert('US/Central').tz_localize(None)).dt.total_seconds() / 3600).astype(int)
    
    # Reorder columns
    base_cols = ['datetime', 'forecast_issued', 'forecast_hour', 'zone']
    data_cols = [c for c in df_zones.columns if c not in base_cols]
    df_zones = df_zones[base_cols + data_cols]
    
    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, 
        f"hrrr_texas_{date.strftime('%Y%m%d_%H00')}.parquet"
    )
    df_zones.to_parquet(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    print(f"    Rows: {len(df_zones)}, Zones: {df_zones['zone'].nunique()}")
    
    # Delete GRIB files to save space (unless keep_grib=True)
    if not keep_grib:
        cleanup_grib_files(date, save_dir)


def download_hrrr_historical(start_date, end_date, issue_hours, forecast_hours, 
                             save_dir, output_dir, force_download=None, max_workers=4, keep_grib=False):
    """
    Download historical HRRR forecasts with parallel processing.
    
    Args:
        start_date: start date
        end_date: end date
        issue_hours: hours to issue forecasts (e.g., [6] for 6am only)
        forecast_hours: forecast horizon (e.g., range(0, 25) for 0-24hr)
        save_dir: GRIB cache directory
        output_dir: parquet output directory
        force_download: True for initial download, False for re-processing cached data
        max_workers: number of parallel downloads (default 4, recommend 4-8)
        keep_grib: if True, keep GRIB files after processing (default False)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Build list of all dates to download
    dates_to_download = []
    current = start_date
    while current <= end_date:
        for hour in issue_hours:
            forecast_time = current.replace(hour=hour, minute=0, second=0)
            dates_to_download.append(forecast_time)
        current += timedelta(days=1)
    
    total = len(dates_to_download)
    print(f"Downloading HRRR for {total} forecast times using {max_workers} workers...")
    if not keep_grib:
        print("🗑️  GRIB files will be deleted after processing to save space")
    
    # Download in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_hrrr_day, date, forecast_hours, save_dir, output_dir, force_download, keep_grib
            ): date for date in dates_to_download
        }
        
        for future in as_completed(futures):
            completed += 1
            date = futures[future]
            try:
                future.result()
                print(f"[{completed}/{total}] Completed {date.strftime('%Y-%m-%d %H:00')}")
            except Exception as e:
                print(f"[{completed}/{total}] FAILED {date.strftime('%Y-%m-%d %H:00')}: {e}")
    
    print(f"\n✓ Download complete! {completed}/{total} files processed.")


def combine_parquet_files(input_dir, output_file):
    """Combine multiple daily parquet files"""
    parquet_files = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.parquet') and f.startswith('hrrr_texas_')
    ])
    
    if not parquet_files:
        print("No parquet files found")
        return
    
    print(f"Combining {len(parquet_files)} files...")
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined = df_combined.sort_values(['forecast_issued', 'datetime', 'zone'])
    df_combined.to_parquet(output_file, index=False)
    print(f"✓ Combined: {output_file} ({len(df_combined):,} rows)")


if __name__ == "__main__":
    SAVE_DIR = r"D:\Users\williamyun\proj\energy_trading\input_data\weather"  # Herbie adds /hrrr/
    OUTPUT_DIR = r"D:\Users\williamyun\proj\energy_trading\data_processed\weather\hrrr"
    
    # Single day test
    # force_download: None=auto (recommended), True=always check AWS, False=cache only
    # keep_grib: False=delete after processing (saves space), True=keep GRIB files
    # download_hrrr_day(
    #     date=datetime(2024, 7, 7, 6, 0),
    #     forecast_hours=range(0, 25),
    #     save_dir=SAVE_DIR,
    #     output_dir=OUTPUT_DIR,
    #     force_download=None,
    #     keep_grib=False  # Delete GRIB after processing
    # )
    
    # Historical download
    # Estimated: ~50GB parquet for 8.5 years (GRIB files deleted automatically)
    download_hrrr_historical(
        start_date=datetime(2017, 6, 1),
        end_date=datetime(2025, 12, 1),
        issue_hours=[6],
        forecast_hours=range(0, 19),    
        save_dir=SAVE_DIR,
        output_dir=OUTPUT_DIR,
        force_download=None,
        max_workers=6,
        keep_grib=False  # Delete GRIB files after processing to save space
    )
    
    # Combine files after historical download
    # combine_parquet_files(OUTPUT_DIR, f"{OUTPUT_DIR}/hrrr_texas_2017_2025_full.parquet")  