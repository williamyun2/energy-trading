import zipfile
import pandas as pd
from pathlib import Path
import re
from datetime import datetime

INPUT_DIR = Path(r"C:\code\energy_trading\input_data\load\SevenDay_Load_Forecast_by_Model_and_Weather_Zone")

def extract_run_datetime(name: str):
    print(f"  Testing: {name}")
    # Look for YYYYMMDD.HHMMSS pattern (year 2000-2099)
    m = re.search(r'(20\d{6})\.(\d{6})', name)
    if not m:
        print(f"    ✗ No match found")
        return pd.NaT
    date_str, time_str = m.groups()
    print(f"    ✓ Found: {date_str}.{time_str}")
    try:
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        result = pd.Timestamp(dt).floor('min')
        print(f"    ✓ Parsed: {result}")
        return result
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return pd.NaT

# Test on first zip file
zip_files = sorted(INPUT_DIR.glob("*.zip"))
if len(zip_files) > 0:
    print(f"Testing first zip: {zip_files[0].name}")
    print("-" * 70)
    
    with zipfile.ZipFile(zip_files[0], 'r') as outer:
        inner_files = outer.namelist()
        inner_zips = [f for f in inner_files if f.endswith('.zip')]
        
        print(f"\nFound {len(inner_zips)} inner zips")
        print("\nTesting first 3 inner zips:")
        print("-" * 70)
        
        for inner_name in inner_zips[:3]:
            run_dt = extract_run_datetime(inner_name)
            print(f"    Result: {run_dt}\n")