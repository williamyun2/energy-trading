import pandas as pd
import xarray as xr
from herbie import Herbie
from datetime import datetime

# Load one raw HRRR file to check timezone
date = datetime(2024, 7, 7, 6, 0)
save_dir = r"D:\Users\williamyun\proj\energy_trading\input_data\weather"

print("Loading raw HRRR data...")
H = Herbie(date, model='hrrr', product='sfc', fxx=18, save_dir=save_dir)
ds = H.xarray(':DSWRF:surface', remove_grib=False)

if isinstance(ds, list):
    ds = ds[0]

print(f"\nRaw HRRR timestamp info:")
print(f"valid_time: {ds.valid_time.values}")
print(f"Time type: {type(ds.valid_time.values)}")

# Check what time this actually represents
import numpy as np
vt = pd.Timestamp(ds.valid_time.values)
print(f"\nParsed as pandas Timestamp: {vt}")
print(f"Timezone info: {vt.tz}")
print(f"UTC offset: {vt.tz if vt.tz else 'No timezone (assumed UTC)'}")

# What time is this in Texas?
if vt.tz is None:
    print(f"\nAssuming UTC, in Texas (CDT, UTC-5) this is: {vt - pd.Timedelta(hours=5)}")
    print(f"In West Texas (MDT, UTC-6) this is: {vt - pd.Timedelta(hours=6)}")