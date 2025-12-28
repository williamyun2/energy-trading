"""
Rebuild the cache with improved weather deduplication
"""
from loader import load_clean_data, clear_cache

print("="*80)
print("REBUILDING CACHE WITH IMPROVED WEATHER DEDUPLICATION")
print("="*80)

# Clear old cache
print("\nStep 1: Clearing old cache...")
clear_cache()

# Load data with new logic (will rebuild cache)
print("\nStep 2: Loading data with new deduplication logic...")
print("This will take a few minutes to load 3000+ weather files...")
df = load_clean_data(verbose=True)

print("\n" + "="*80)
print("CACHE REBUILT SUCCESSFULLY!")
print("="*80)
print(f"Final dataset: {len(df):,} rows x {len(df.columns)} columns")
print(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"\nColumns: {list(df.columns)}")

# Show some stats
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)
print(df.describe())
