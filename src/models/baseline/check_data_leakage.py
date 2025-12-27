"""
Data Quality Check - Investigate Suspiciously Good Baseline Results

The persistence model achieved MAE=$0.03/MWh and R²=0.995, which is unrealistically good.
This script investigates potential data leakage issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root and src directory to path
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from merge_dataset.loader import load_clean_data

print("="*80)
print("DATA QUALITY CHECK - INVESTIGATING SUSPICIOUSLY GOOD RESULTS")
print("="*80)
print()

# Load data
print("Loading clean data...")
df = load_clean_data(verbose=False)
print(f"✓ Loaded {len(df):,} records")
print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print()

# ============================================================================
# 1. CHECK FOR DUPLICATE HOURS
# ============================================================================
print("="*80)
print("1. CHECKING FOR DUPLICATE DATETIMES")
print("="*80)

duplicates = df[df.duplicated(subset=['datetime'], keep=False)]
print(f"Duplicate datetime entries: {len(duplicates):,}")

if len(duplicates) > 0:
    print("\n⚠️  WARNING: Found duplicate datetime entries!")
    print("\nSample duplicates:")
    print(duplicates[['datetime', 'dam_price']].head(20))

    # Check if duplicates have same price
    dup_grouped = duplicates.groupby('datetime')['dam_price'].agg(['count', 'nunique', 'mean', 'std'])
    same_price_dups = dup_grouped[dup_grouped['nunique'] == 1]
    print(f"\nDuplicates with IDENTICAL price: {len(same_price_dups):,}")
    print(f"Duplicates with DIFFERENT prices: {len(dup_grouped) - len(same_price_dups):,}")
else:
    print("✓ No duplicate datetimes found")

print()

# ============================================================================
# 2. CHECK FOR CONSECUTIVE IDENTICAL PRICES
# ============================================================================
print("="*80)
print("2. CHECKING FOR CONSECUTIVE IDENTICAL PRICES")
print("="*80)

df_sorted = df.sort_values('datetime').reset_index(drop=True)
consecutive_same = (df_sorted['dam_price'] == df_sorted['dam_price'].shift(1)).sum()
print(f"Consecutive identical prices: {consecutive_same:,} / {len(df_sorted):,} ({consecutive_same/len(df_sorted)*100:.1f}%)")

# Check for long runs of identical prices
df_sorted['price_change'] = df_sorted['dam_price'] != df_sorted['dam_price'].shift(1)
df_sorted['price_run_id'] = df_sorted['price_change'].cumsum()
run_lengths = df_sorted.groupby('price_run_id').size()

print(f"\nLongest run of identical prices: {run_lengths.max()} hours")
print(f"Median run length: {run_lengths.median():.0f} hours")
print(f"Mean run length: {run_lengths.mean():.1f} hours")

if run_lengths.max() > 24:
    print(f"\n⚠️  WARNING: Found price runs longer than 24 hours!")
    long_runs = run_lengths[run_lengths > 24].sort_values(ascending=False).head(10)
    print("\nTop 10 longest runs:")
    print(long_runs)

print()

# ============================================================================
# 3. CHECK TEST SET SPECIFICALLY
# ============================================================================
print("="*80)
print("3. TEST SET ANALYSIS (Last 6 months)")
print("="*80)

max_date = df_sorted['datetime'].max()
split_date = max_date - pd.DateOffset(months=6)
df_test = df_sorted[df_sorted['datetime'] >= split_date].copy()

print(f"Test set size: {len(df_test):,} records")
print(f"Test date range: {df_test['datetime'].min()} to {df_test['datetime'].max()}")
print()

print("Test set price statistics:")
print(df_test['dam_price'].describe())
print()

print(f"Unique prices in test set: {df_test['dam_price'].nunique():,}")
print(f"Unique hours in test set: {df_test['datetime'].nunique():,}")

# Check for duplicates in test set
test_duplicates = df_test[df_test.duplicated(subset=['datetime'], keep=False)]
print(f"Duplicate datetimes in test set: {len(test_duplicates):,}")

if len(test_duplicates) > 0:
    print("\n⚠️  WARNING: Test set has duplicate datetimes!")

print()

# ============================================================================
# 4. CHECK 24-HOUR LAG CORRELATION
# ============================================================================
print("="*80)
print("4. 24-HOUR LAG CORRELATION ANALYSIS")
print("="*80)

df_test['price_lag_24h'] = df_test['dam_price'].shift(24)
df_test_lag = df_test.dropna(subset=['price_lag_24h'])

correlation = df_test_lag['dam_price'].corr(df_test_lag['price_lag_24h'])
print(f"Correlation between price(t) and price(t-24h): {correlation:.6f}")

# Calculate how often 24h lag is exactly the same
exact_matches = (df_test_lag['dam_price'] == df_test_lag['price_lag_24h']).sum()
print(f"Exact matches (price(t) == price(t-24h)): {exact_matches:,} / {len(df_test_lag):,} ({exact_matches/len(df_test_lag)*100:.2f}%)")

if exact_matches / len(df_test_lag) > 0.5:
    print("\n⚠️  WARNING: More than 50% of prices are EXACTLY the same as 24h ago!")
    print("This suggests possible data duplication or forward-filling issues.")

print()

# ============================================================================
# 5. CHECK FOR MISSING HOURS / DATA GAPS
# ============================================================================
print("="*80)
print("5. CHECKING FOR TIME GAPS")
print("="*80)

df_sorted['time_diff'] = df_sorted['datetime'].diff()
gaps = df_sorted[df_sorted['time_diff'] > pd.Timedelta(hours=1)]

print(f"Expected hourly intervals: {len(df_sorted)-1:,}")
print(f"Gaps found (time_diff > 1 hour): {len(gaps):,}")

if len(gaps) > 0:
    print("\nSample gaps:")
    print(gaps[['datetime', 'time_diff']].head(10))

print()

# ============================================================================
# 6. CHECK LOAD FORECAST DATA
# ============================================================================
print("="*80)
print("6. CHECKING LOAD FORECAST COLUMN")
print("="*80)

if 'system_load_forecast' in df.columns:
    missing_load = df['system_load_forecast'].isnull().sum()
    print(f"Missing load forecast values: {missing_load:,} / {len(df):,} ({missing_load/len(df)*100:.1f}%)")

    # Check for duplicate load forecasts
    duplicate_loads = df.groupby('datetime')['system_load_forecast'].nunique()
    multi_forecasts = duplicate_loads[duplicate_loads > 1]

    if len(multi_forecasts) > 0:
        print(f"\n⚠️  WARNING: {len(multi_forecasts):,} datetimes have multiple different load forecasts!")
        print("This suggests duplicate records with different forecast values.")
else:
    print("system_load_forecast column not found in dataset")

print()

# ============================================================================
# 7. SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

issues = []

if len(duplicates) > 0:
    issues.append(f"Found {len(duplicates):,} duplicate datetime entries")

if exact_matches / len(df_test_lag) > 0.5:
    issues.append(f"Abnormally high exact matches between t and t-24h ({exact_matches/len(df_test_lag)*100:.1f}%)")

if run_lengths.max() > 100:
    issues.append(f"Found extremely long runs of identical prices ({run_lengths.max()} hours)")

if len(issues) > 0:
    print("\n⚠️  ISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\nRECOMMENDED ACTIONS:")
    if len(duplicates) > 0:
        print("  1. Remove duplicate datetimes (keep='first' or aggregate)")
        print("     df_clean = df.drop_duplicates(subset=['datetime'], keep='first')")

    print("  2. Check the data merge process in loader.py")
    print("  3. Verify load forecast data doesn't have duplicates")
    print("  4. Re-run baseline model after cleaning")
else:
    print("\n✓ No obvious data quality issues detected")
    print("The high performance might be legitimate if prices are very stable.")

print()
print("="*80)
print("Check complete! Review the output above for data quality issues.")
print("="*80)
