import pandas as pd
from pathlib import Path

print("=" * 70)
print("Merge Load Forecast Data (API + Archive)")
print("=" * 70)

API_FILE = Path(r"C:\code\energy_trading\data_processed\load_forecast_api.csv")
ARCHIVE_FILE = Path(r"C:\code\energy_trading\data_processed\load_forecast_archive_filtered.csv")  # Already filtered
OUTPUT_FILE = Path(r"C:\code\energy_trading\data_processed\load_forecast_clean.csv")

# Load API data
print("\n1. Loading API data...")
api_df = pd.read_csv(API_FILE)
print(f"   ✓ {len(api_df):,} rows")
print(f"   Range: {api_df['deliveryDate'].min()} to {api_df['deliveryDate'].max()}")

# Load archive data (already filtered to inUseFlag=Y)
print("\n2. Loading filtered archive data...")
archive_df = pd.read_csv(ARCHIVE_FILE)
print(f"   ✓ {len(archive_df):,} rows (already filtered)")
print(f"   Range: {archive_df['deliveryDate'].min()} to {archive_df['deliveryDate'].max()}")

# Combine
print("\n3. Combining datasets...")
combined = pd.concat([api_df, archive_df], ignore_index=True)
print(f"   ✓ {len(combined):,} rows combined")

# Process datetime if needed
if 'datetime' not in combined.columns:
    print("\n4. Creating datetime column...")
    combined['hourEnding'] = combined['hourEnding'].astype(str).str.replace('24:00', '00:00')
    combined['datetime'] = pd.to_datetime(combined['deliveryDate'] + ' ' + combined['hourEnding'])
    combined.loc[combined['hourEnding'] == '00:00', 'datetime'] += pd.Timedelta(days=1)
else:
    combined['datetime'] = pd.to_datetime(combined['datetime'])

# Deduplicate
print("\n5. Removing duplicates...")
combined = combined.sort_values(['datetime', 'model'])
combined = combined.drop_duplicates(subset=['datetime', 'model'], keep='first')
print(f"   ✓ {len(combined):,} rows after dedup")

# Sort
combined = combined.sort_values('datetime')

# Save
print("\n6. Saving merged data...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT_FILE, index=False)
print(f"   ✓ Saved: {OUTPUT_FILE}")

print(f"\nFinal Summary:")
print(f"  Rows: {len(combined):,}")
print(f"  Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")

if 'model' in combined.columns:
    model_counts = combined['model'].value_counts()
    print(f"\n  Model distribution:")
    for model, count in model_counts.head(10).items():
        print(f"    {model}: {count:,}")

print("\n" + "=" * 70)
print("✅ MERGE COMPLETE")
print("=" * 70)
print("\nUse this file for modeling:")
print(f"  {OUTPUT_FILE}")