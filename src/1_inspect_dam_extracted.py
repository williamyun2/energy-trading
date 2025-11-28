import pandas as pd
from pathlib import Path

print("=" * 70)
print("ERCOT Data Inspector")
print("=" * 70)

base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data_extracted" / "dam_prices"

xlsx_files = sorted(data_dir.glob("*.xlsx"))

if not xlsx_files:
    print("No files found!")
    exit(1)

print(f"\nFound {len(xlsx_files)} files\n")

total_rows = 0
file_info = []

for file in xlsx_files:
    df = pd.read_excel(file)
    year = file.stem.split('_')[-1]
    
    # Count unique dates and hours
    df['datetime'] = pd.to_datetime(df['Delivery Date'] + ' ' + df['Hour Ending'].str.replace('24:00', '00:00'))
    unique_dates = df['datetime'].dt.date.nunique()
    unique_hours = df.groupby(df['datetime'].dt.date).size()
    
    hubs = df['Settlement Point'].nunique()
    total_rows += len(df)
    
    file_info.append({
        'Year': year,
        'Rows': len(df),
        'Dates': unique_dates,
        'Avg Hours/Day': unique_hours.mean(),
        'Hubs': hubs,
        'Date Range': f"{df['datetime'].min().date()} to {df['datetime'].max().date()}"
    })

print("File Summary:")
print("-" * 70)
for info in file_info:
    print(f"{info['Year']}: {info['Rows']:>6} rows | {info['Dates']:>3} days | "
          f"{info['Avg Hours/Day']:>5.1f} hrs/day | {info['Hubs']:>2} hubs | {info['Date Range']}")

print("-" * 70)
print(f"Total: {total_rows:,} rows")

# Expected vs Actual
expected_per_year = 365 * 24 * 15  # 365 days × 24 hours × ~15 hubs
print(f"\nExpected per year: ~{expected_per_year:,} rows (365 days × 24 hrs × 15 hubs)")
print(f"Actual per year avg: {total_rows // len(xlsx_files):,} rows")

# Check one file in detail
print("\n" + "=" * 70)
print("Detailed Look at 2023:")
print("=" * 70)
df_2023 = pd.read_excel([f for f in xlsx_files if '2023' in f.name][0])
df_2023['datetime'] = pd.to_datetime(df_2023['Delivery Date'] + ' ' + df_2023['Hour Ending'].str.replace('24:00', '00:00'))

print(f"\nTotal rows: {len(df_2023):,}")
print(f"Unique dates: {df_2023['datetime'].dt.date.nunique()}")
print(f"Date range: {df_2023['datetime'].min()} to {df_2023['datetime'].max()}")
print(f"\nSettlement Points:")
for sp in sorted(df_2023['Settlement Point'].unique()):
    count = len(df_2023[df_2023['Settlement Point'] == sp])
    print(f"  {sp:15s}: {count:>4} rows")

print(f"\nRows per hour (should be ~15 for all hubs):")
hourly_counts = df_2023.groupby('datetime').size()
print(hourly_counts.describe())

print(f"\nSample data:")
print(df_2023[['Delivery Date', 'Hour Ending', 'Settlement Point', 'Settlement Point Price']].head(30))