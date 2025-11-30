import zipfile
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 70)
print("Inspect HRRR Weather Data")
print("=" * 70)

# Path to one month of HRRR data
hrrr_file = Path(r"C:\code\energy_trading\input_data\weather\hrrr\CONUS_2025_10_28.zip")

print(f"\n1. Inspecting: {hrrr_file.name}")
print(f"   File size: {hrrr_file.stat().st_size / (1024**3):.2f} GB")

# List contents of zip
print("\n2. Contents of zip file:")
with zipfile.ZipFile(hrrr_file, 'r') as z:
    file_list = z.namelist()
    print(f"   Total files: {len(file_list)}")
    print("\n   First 10 files:")
    for f in file_list[:10]:
        info = z.getinfo(f)
        print(f"   - {f} ({info.file_size / (1024**2):.1f} MB)")
    
    if len(file_list) > 10:
        print(f"   ... and {len(file_list) - 10} more files")
    
    # Try to open first file to see structure
    print("\n3. Examining first data file...")
    first_file = file_list[0]
    
    with z.open(first_file) as f:
        # Check if it's NetCDF/GRIB
        if first_file.endswith('.nc') or first_file.endswith('.nc4'):
            print(f"   Format: NetCDF")
            ds = xr.open_dataset(f)
            print(f"\n   Dimensions: {dict(ds.dims)}")
            print(f"\n   Coordinates:")
            for coord in ds.coords:
                print(f"      {coord}: {ds.coords[coord].shape}")
            print(f"\n   Variables:")
            for var in ds.data_vars:
                print(f"      {var}: {ds[var].shape} - {ds[var].attrs.get('long_name', 'N/A')}")
            
            # Check coordinate ranges
            if 'latitude' in ds.coords or 'lat' in ds.coords:
                lat = ds['latitude'] if 'latitude' in ds.coords else ds['lat']
                lon = ds['longitude'] if 'longitude' in ds.coords else ds['lon']
                print(f"\n   Spatial coverage:")
                print(f"      Lat: {lat.min().values:.2f} to {lat.max().values:.2f}")
                print(f"      Lon: {lon.min().values:.2f} to {lon.max().values:.2f}")
            
            # Check time dimension
            if 'time' in ds.coords:
                print(f"\n   Time coverage:")
                print(f"      Start: {pd.Timestamp(ds.time.values[0])}")
                print(f"      End: {pd.Timestamp(ds.time.values[-1])}")
                print(f"      Steps: {len(ds.time)}")
            
            ds.close()
            
        elif first_file.endswith('.grib') or first_file.endswith('.grb') or first_file.endswith('.grib2'):
            print(f"   Format: GRIB")
            print(f"   (Need to use cfgrib library to read)")
            
        else:
            # Try to read as text/csv
            content = f.read(1000)
            print(f"   First 1000 bytes:")
            print(content)

print("\n4. Texas bounding box for filtering:")
print("   Lat: 25.8°N to 36.5°N")
print("   Lon: -106.6°W to -93.5°W (-106.6 to -93.5)")

print("\n" + "=" * 70)
print("Next: Extract only Texas data to reduce size")
print("=" * 70)