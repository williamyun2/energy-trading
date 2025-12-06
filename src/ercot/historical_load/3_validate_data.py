import pandas as pd
from pathlib import Path

print("="*70)
print("DATA VALIDATION CHECKS")
print("="*70)

# Load the output
OUTPUT_FILE = Path(r"C:\code\energy_trading\data_processed\load_forecast_part2.csv")
df = pd.read_csv(OUTPUT_FILE)

print(f"\nLoaded file: {OUTPUT_FILE}")
print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**3):.2f} GB")

# 1. Basic Info
print("\n" + "="*70)
print("1. BASIC INFO")
print("="*70)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

# 2. Date Range (Column 16: datetime - delivery hours)
print("\n" + "="*70)
print("2. DATETIME RANGE (Delivery Hours)")
print("="*70)
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"Start: {df['datetime'].min()}")
print(f"End:   {df['datetime'].max()}")
span_days = (df['datetime'].max() - df['datetime'].min()).days
print(f"Span:  {span_days:,} days ({span_days/365:.1f} years)")

# 3. Run Datetime Range (Column 15: forecast publication times)
print("\n" + "="*70)
print("3. RUN DATETIME RANGE (Forecast Publication Times)")
print("="*70)
df['runDatetime'] = pd.to_datetime(df['runDatetime'])
print(f"First forecast: {df['runDatetime'].min()}")
print(f"Last forecast:  {df['runDatetime'].max()}")
unique_runs = df['runDatetime'].nunique()
print(f"Unique forecast publications: {unique_runs:,}")
print(f"Average rows per forecast: {len(df) / unique_runs:.0f}")

# 4. InUseFlag Check (should be 100% 'Y')
print("\n" + "="*70)
print("4. INUSEFLAG CHECK (Should be 100% 'Y')")
print("="*70)
flag_counts = df['inUseFlag'].value_counts()
print(flag_counts)
if 'Y' in flag_counts.index:
    print(f"'Y' percentage: {flag_counts['Y']/len(df)*100:.2f}%")
if 'N' in flag_counts.index:
    print(f"⚠️  WARNING: Found {flag_counts['N']:,} rows with inUseFlag='N'")

# 5. Model Distribution
print("\n" + "="*70)
print("5. MODELS PRESENT")
print("="*70)
model_counts = df['model'].value_counts().sort_index()
print(model_counts)
print(f"\nUnique models: {df['model'].nunique()}")

# 6. Duplicates Check
print("\n" + "="*70)
print("6. DUPLICATES CHECK")
print("="*70)
# Check for exact duplicate rows
exact_dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {exact_dupes:,}")

# Check for duplicate (datetime, runDatetime, model) combinations
key_dupes = df.duplicated(subset=['datetime', 'runDatetime', 'model']).sum()
print(f"Duplicate (datetime, runDatetime, model): {key_dupes:,}")

if key_dupes > 0:
    print("\n⚠️  WARNING: Found duplicate forecasts!")
    print("Sample duplicates:")
    dupe_mask = df.duplicated(subset=['datetime', 'runDatetime', 'model'], keep=False)
    print(df[dupe_mask][['datetime', 'runDatetime', 'model', 'systemTotal']].head(10))

# 7. NULL Check
print("\n" + "="*70)
print("7. NULL/MISSING VALUES CHECK")
print("="*70)
critical_cols = ['datetime', 'runDatetime', 'systemTotal', 'model', 'inUseFlag']
null_counts = df[critical_cols].isna().sum()
print(null_counts)
if null_counts.sum() > 0:
    print("\n⚠️  WARNING: Found null values in critical columns!")

# 8. Data Type Check
print("\n" + "="*70)
print("8. DATA TYPES")
print("="*70)
print(df.dtypes)

# 9. Sample Data
print("\n" + "="*70)
print("9. SAMPLE DATA (First 10 Rows)")
print("="*70)
sample_cols = ['datetime', 'runDatetime', 'model', 'systemTotal', 'inUseFlag']
print(df[sample_cols].head(10).to_string())

# 10. Statistical Summary
print("\n" + "="*70)
print("10. STATISTICAL SUMMARY (systemTotal)")
print("="*70)
print(df['systemTotal'].describe())

# 11. Time Distribution
print("\n" + "="*70)
print("11. FORECAST PUBLICATIONS PER YEAR")
print("="*70)
df['year'] = df['runDatetime'].dt.year
year_counts = df.groupby('year').size()
print(year_counts)

# 12. Rows per runDatetime distribution
print("\n" + "="*70)
print("12. ROWS PER FORECAST PUBLICATION (Distribution)")
print("="*70)
rows_per_run = df.groupby('runDatetime').size()
print(f"Min rows per forecast: {rows_per_run.min()}")
print(f"Max rows per forecast: {rows_per_run.max()}")
print(f"Mean rows per forecast: {rows_per_run.mean():.0f}")
print(f"Median rows per forecast: {rows_per_run.median():.0f}")

# 13. Final Summary
print("\n" + "="*70)
print("✅ VALIDATION SUMMARY")
print("="*70)
print(f"✓ Total rows: {len(df):,}")
print(f"✓ Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
print(f"✓ Unique forecasts: {unique_runs:,}")
print(f"✓ InUseFlag='Y': {(df['inUseFlag']=='Y').sum():,} rows")
print(f"✓ Duplicates: {key_dupes:,}")
print(f"✓ Null values: {null_counts.sum():,}")

if key_dupes == 0 and null_counts.sum() == 0 and (df['inUseFlag']=='Y').all():
    print("\n🎉 DATA LOOKS GOOD!")
else:
    print("\n⚠️  WARNINGS DETECTED - Review above")

print("="*70)