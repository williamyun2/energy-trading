import pandas as pd
from pathlib import Path

print("=" * 70)
print("Process API Load Forecast Data")
print("=" * 70)

INPUT_FILE = Path(r"C:\code\energy_trading\data_extracted\load_forecast\load_forecast_historical.csv")
OUTPUT_FILE = Path(r"C:\code\energy_trading\data_processed\load_forecast_api.csv")

print(f"\n1. Loading data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"   ✓ Loaded {len(df):,} rows")

print("\n2. Processing columns...")

# Rename postedDatetime → runDatetime
df.rename(columns={'postedDatetime': 'runDatetime'}, inplace=True)

# Parse runDatetime (when forecast was published)
df['runDatetime'] = pd.to_datetime(df['runDatetime'])

# Remove seconds (floor to minute)
df['runDatetime'] = df['runDatetime'].dt.floor('min')

# Process delivery datetime
df['hourEnding'] = df['hourEnding'].astype(str).str.strip()
df['deliveryDate'] = df['deliveryDate'].astype(str).str.strip()

# Handle 24:00 edge case
he = df['hourEnding']
is_24 = (he == '24:00')
he_fixed = he.where(~is_24, '00:00')

df['datetime'] = pd.to_datetime(df['deliveryDate'] + ' ' + he_fixed, errors='coerce')
df.loc[is_24, 'datetime'] += pd.Timedelta(days=1)

# Drop bad datetimes
df = df.dropna(subset=['datetime'])

# Standardize column names
df.rename(columns={
    'deliveryDate': 'deliveryDate',
    'hourEnding': 'hourEnding',
    'coast': 'coast',
    'east': 'east',
    'farWest': 'farWest',
    'north': 'north',
    'northCent': 'northCentral',
    'southCent': 'southCentral',
    'southern': 'southern',
    'west': 'west',
    'systemTot': 'systemTotal',
    'model': 'model',
    'inUseFlag': 'inUseFlag',
    'DSTFlag': 'DSTFlag'
}, inplace=True)

# Convert inUseFlag
df['inUseFlag'] = df['inUseFlag'].map({True: 'Y', False: 'N'})

# Convert DSTFlag
df['DSTFlag'] = df['DSTFlag'].map({True: 'Y', False: 'N'})

print(f"   ✓ Processed {len(df):,} rows")

print("\n3. Sorting...")
df = df.sort_values(['datetime', 'runDatetime']).reset_index(drop=True)

print("\n4. Saving...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"   ✓ Saved to: {OUTPUT_FILE}")

print("\n5. Summary:")
print(f"   Total rows: {len(df):,}")
print(f"   Date range (delivery): {df['datetime'].min()} to {df['datetime'].max()}")
print(f"   Run datetime range: {df['runDatetime'].min()} to {df['runDatetime'].max()}")
print(f"   Models: {sorted(df['model'].unique())}")
print(f"   InUseFlag='Y': {(df['inUseFlag']=='Y').sum():,}")

print("\n" + "=" * 70)
print("✅ COMPLETE")
print("=" * 70)