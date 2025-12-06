import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"C:\code\energy_trading\data_processed\load_forecast_historical_archive.csv")

# Convert to datetime
df['runDatetime'] = pd.to_datetime(df['runDatetime'])
df['year'] = df['runDatetime'].dt.year
df['year_month'] = df['runDatetime'].dt.to_period('M')

print("="*70)
print("MODEL USAGE ANALYSIS")
print("="*70)

# 1. Model usage by year
print("\n1. ROWS PER MODEL BY YEAR")
print("-"*70)
model_by_year = df.groupby(['year', 'model']).size().unstack(fill_value=0)
print(model_by_year)

# 2. Percentage by year
print("\n2. MODEL PERCENTAGE BY YEAR")
print("-"*70)
model_pct_by_year = df.groupby(['year', 'model']).size().unstack(fill_value=0)
model_pct_by_year = model_pct_by_year.div(model_pct_by_year.sum(axis=1), axis=0) * 100
print(model_pct_by_year.round(1))

# 3. First and last appearance of each model
print("\n3. FIRST AND LAST APPEARANCE OF EACH MODEL")
print("-"*70)
for model in sorted(df['model'].unique()):
    model_data = df[df['model'] == model]
    first = model_data['runDatetime'].min()
    last = model_data['runDatetime'].max()
    count = len(model_data)
    print(f"{model:3s}: {first} → {last} ({count:,} rows)")

# 4. Model usage by month (for trend)
print("\n4. MODEL USAGE TREND (Monthly)")
print("-"*70)
monthly_counts = df.groupby(['year_month', 'model']).size().unstack(fill_value=0)
print(monthly_counts.tail(24))  # Last 2 years

# 5. Check if A3/A6 still used recently
print("\n5. A3/A6 USAGE IN RECENT YEARS")
print("-"*70)
recent_cutoff = pd.Timestamp('2022-01-01')
recent_data = df[df['runDatetime'] >= recent_cutoff]
a3_a6_recent = recent_data[recent_data['model'].isin(['A3', 'A6'])]
print(f"A3/A6 rows after 2022: {len(a3_a6_recent):,}")
print(f"Total rows after 2022: {len(recent_data):,}")
if len(a3_a6_recent) > 0:
    print(f"A3/A6 percentage: {len(a3_a6_recent)/len(recent_data)*100:.2f}%")
    print(f"Last A3 forecast: {recent_data[recent_data['model']=='A3']['runDatetime'].max()}")
    print(f"Last A6 forecast: {recent_data[recent_data['model']=='A6']['runDatetime'].max()}")
else:
    print("A3/A6 NOT USED after 2022")

# 6. Visualize model usage over time
print("\n6. Generating visualization...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Absolute counts by year
model_by_year.plot(kind='bar', stacked=False, ax=ax1, width=0.8)
ax1.set_title('Model Usage Over Time (Absolute Counts)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Rows')
ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Percentage by year
model_pct_by_year.plot(kind='bar', stacked=True, ax=ax2, width=0.8)
ax2.set_title('Model Usage Over Time (Percentage)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Percentage (%)')
ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\code\energy_trading\data_processed\model_usage_over_time.png", dpi=300, bbox_inches='tight')
print("✓ Saved chart: model_usage_over_time.png")

# 7. Monthly trend for last 3 years
print("\n7. Generating monthly trend chart...")
fig2, ax3 = plt.subplots(figsize=(16, 8))
last_3yr = df[df['year'] >= 2021].copy()
monthly_trend = last_3yr.groupby(['year_month', 'model']).size().unstack(fill_value=0)
monthly_trend.plot(ax=ax3, marker='o')
ax3.set_title('Model Usage Trend (Monthly, Last 3 Years)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Number of Rows')
ax3.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"C:\code\energy_trading\data_processed\model_usage_monthly_trend.png", dpi=300, bbox_inches='tight')
print("✓ Saved chart: model_usage_monthly_trend.png")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)