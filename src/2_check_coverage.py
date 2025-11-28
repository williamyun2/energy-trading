import calendar
from pathlib import Path

import pandas as pd

base_dir = Path(__file__).parent.parent
load_file = base_dir / "data_processed" / "system_load.csv"

df = pd.read_csv(load_file)
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

print("=" * 70)
print("Load Data Coverage Analysis")
print("=" * 70)

print(f"\nTotal hours: {len(df):,}")
print(f"Unique days: {df['date'].nunique():,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Calculate expected vs actual
start_date = df['date'].min()
end_date = df['date'].max()
days_in_range = (end_date - start_date).days + 1

print(f"\nExpected days in range: {days_in_range:,}")
print(f"Actual days with data:  {df['date'].nunique():,}")
print(f"Missing days:           {days_in_range - df['date'].nunique():,}")
print(f"Coverage:               {df['date'].nunique()/days_in_range*100:.1f}%")

# Year by year breakdown
df['year'] = df['datetime'].dt.year
yearly = df.groupby('year')['date'].nunique()

print("\nDays of data per year:")
for year, days in yearly.items():
    year_days = 366 if calendar.isleap(year) else 365
    print(f"  {year}: {days:>3} days ({days/year_days*100:>5.1f}% of year)")

# Find large gaps
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
actual_dates = pd.DatetimeIndex(pd.to_datetime(df['date'].unique()))
missing_dates = all_dates.difference(actual_dates)
print(f"\nTotal missing days: {len(missing_dates):,}")

# Find consecutive gaps
if len(missing_dates) > 0:
    gaps = []
    current_gap_start = missing_dates[0]
    current_gap_end = missing_dates[0]
    
    for i in range(1, len(missing_dates)):
        if (missing_dates[i] - missing_dates[i-1]).days == 1:
            current_gap_end = missing_dates[i]
        else:
            gaps.append((current_gap_start, current_gap_end, (current_gap_end - current_gap_start).days + 1))
            current_gap_start = missing_dates[i]
            current_gap_end = missing_dates[i]
    
    gaps.append((current_gap_start, current_gap_end, (current_gap_end - current_gap_start).days + 1))
    
    # Sort by gap size
    gaps.sort(key=lambda x: x[2], reverse=True)
    
    print("\nLargest data gaps:")
    for start, end, days in gaps[:10]:
        print(f"  {start.date()} to {end.date()}: {days} days")
