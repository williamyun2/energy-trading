"""
Initial EDA - ERCOT Energy Trading Project
Tests data loading and creates basic visualizations

Run on server with: python 01_initial_eda.py
Saves plots to: data_processed/eda_outputs/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12

# Project paths
BASE_DIR = Path(r"D:\Users\williamyun\proj\power_trading")
PROCESSED_DIR = BASE_DIR / "data_processed"
OUTPUT_DIR = PROCESSED_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("ERCOT ENERGY TRADING PROJECT - INITIAL EDA")
print("="*80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Timestamp: {datetime.now()}")
print()

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================

print("="*80)
print("LOADING DATASETS")
print("="*80)

# 1.1 Natural Gas Prices
print("\n1. Loading Natural Gas Prices (FRED)...")
ng_path = PROCESSED_DIR / "fuel" / "NG" / "ng_henry_hub_spot_fred_2017-01-01_2025-12-06.csv"
df_ng = pd.read_csv(ng_path)
df_ng['Date'] = pd.to_datetime(df_ng['Date'])
df_ng = df_ng.sort_values('Date').reset_index(drop=True)
print(f"   ✓ Loaded {len(df_ng):,} records")
print(f"   Date range: {df_ng['Date'].min().date()} to {df_ng['Date'].max().date()}")
print(f"   Price range: ${df_ng['Price'].min():.2f} to ${df_ng['Price'].max():.2f}/MMBtu")

# 1.2 DAM Prices
print("\n2. Loading ERCOT DAM Prices...")
dam_path = PROCESSED_DIR / "ercot" / "combined_dam_prices.csv"
df_dam = pd.read_csv(dam_path)
print(f"   ✓ Loaded {len(df_dam):,} records")
print(f"   Columns: {df_dam.columns.tolist()}")

# Create datetime column and filter for HB_BUSAVG (system average)
# Convert 'Hour Ending' from format like '01:00', '24:00' to datetime
# HE 24:00 means end of day (midnight next day), so we need special handling
df_dam['Delivery Date'] = pd.to_datetime(df_dam['Delivery Date'])

# Extract hour from 'Hour Ending' (format: '01:00', '02:00', ..., '24:00')
df_dam['HE'] = df_dam['Hour Ending'].str.split(':').str[0].astype(int)

# For HE=24, it means midnight of next day, so use hour 0 and add 1 day
# For HE=1-23, use the hour as-is
df_dam['Datetime'] = df_dam['Delivery Date'] + pd.to_timedelta(df_dam['HE'] - 1, unit='h')
df_dam.loc[df_dam['HE'] == 24, 'Datetime'] = df_dam.loc[df_dam['HE'] == 24, 'Delivery Date'] + pd.Timedelta(days=1)
df_dam_avg = df_dam[df_dam['Settlement Point'] == 'HB_BUSAVG'].copy()
df_dam_avg = df_dam_avg.sort_values('Datetime').reset_index(drop=True)

print(f"   Filtered to HB_BUSAVG (system average): {len(df_dam_avg):,} records")
print(f"   Date range: {df_dam_avg['Datetime'].min()} to {df_dam_avg['Datetime'].max()}")
print(f"   Price range: ${df_dam_avg['Settlement Point Price'].min():.2f} to ${df_dam_avg['Settlement Point Price'].max():.2f}/MWh")

# 1.3 Load Forecasts
print("\n3. Loading ERCOT Load Forecasts...")
load_path = PROCESSED_DIR / "ercot" / "historical_load" / "load_forecast_complete.csv"
if load_path.exists():
    print(f"   Loading sample (first 100k rows)...")
    df_load = pd.read_csv(load_path, nrows=100000)
    print(f"   ✓ Loaded {len(df_load):,} records (sample)")
    print(f"   Columns: {df_load.columns.tolist()[:5]}...")  # Show first 5 columns
    
    # Try to find datetime column
    for col in ['DeliveryDate', 'Delivery Date', 'datetime', 'DateTime', 'Date']:
        if col in df_load.columns:
            df_load[col] = pd.to_datetime(df_load[col])
            print(f"   Date range: {df_load[col].min()} to {df_load[col].max()}")
            break
else:
    print(f"   ⚠ File not found: {load_path}")
    df_load = None

# 1.4 HRRR Weather - Sample one file
print("\n4. Loading HRRR Weather (sample)...")
weather_dir = PROCESSED_DIR / "weather" / "hrrr"
weather_files = sorted(weather_dir.glob("hrrr_texas_*.parquet"))
print(f"   Found {len(weather_files):,} weather files")

df_weather_sample = pd.read_parquet(weather_files[0])
print(f"   ✓ Sample file loaded: {weather_files[0].name}")
print(f"   Rows: {len(df_weather_sample)}, Columns: {len(df_weather_sample.columns)}")
print(f"   Zones: {df_weather_sample['zone'].unique()}")

# ============================================================================
# 2. DATA SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("DATA SUMMARY STATISTICS")
print("="*80)

# Natural Gas Summary
print("\n📊 NATURAL GAS PRICES")
print("-" * 80)
print(df_ng['Price'].describe())
print(f"\nExtreme events:")
feb_2021 = df_ng[df_ng['Date'].between('2021-02-01', '2021-02-28')]
if len(feb_2021) > 0:
    print(f"  Feb 2021 peak: ${feb_2021['Price'].max():.2f}/MMBtu")
pandemic = df_ng[df_ng['Date'].between('2020-03-01', '2020-06-30')]
if len(pandemic) > 0:
    print(f"  Pandemic low:  ${pandemic['Price'].min():.2f}/MMBtu")

# DAM Prices Summary
print("\n⚡ DAM PRICES (HB_BUSAVG - System Average)")
print("-" * 80)
print(df_dam_avg['Settlement Point Price'].describe())
print(f"\nExtreme prices:")
print(f"  > $1000/MWh: {(df_dam_avg['Settlement Point Price'] > 1000).sum():,} hours")
print(f"  > $100/MWh:  {(df_dam_avg['Settlement Point Price'] > 100).sum():,} hours")
print(f"  < $0/MWh:    {(df_dam_avg['Settlement Point Price'] < 0).sum():,} hours")

# Feb 2021 winter storm
feb_2021_dam = df_dam_avg[df_dam_avg['Datetime'].between('2021-02-01', '2021-02-28')]
if len(feb_2021_dam) > 0:
    print(f"\n  Feb 2021 Winter Storm:")
    print(f"    Peak: ${feb_2021_dam['Settlement Point Price'].max():.2f}/MWh")
    print(f"    Mean: ${feb_2021_dam['Settlement Point Price'].mean():.2f}/MWh")

# Weather Summary
print("\n🌡️ WEATHER VARIABLES (Sample file)")
print("-" * 80)
for var in ['temp_f', 'wind_speed_mph', 'solar_radiation_wm2', 'relative_humidity']:
    if var in df_weather_sample.columns:
        print(f"\n{var}:")
        print(df_weather_sample[var].describe())

# ============================================================================
# 3. TIME SERIES PLOTS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Natural Gas Price Over Time
print("\n1. Plotting Natural Gas prices...")
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(df_ng['Date'], df_ng['Price'], linewidth=1, alpha=0.8)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Price ($/MMBtu)', fontsize=14)
ax.set_title('Henry Hub Natural Gas Spot Prices (2017-2025)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Highlight Feb 2021
feb_2021 = df_ng[df_ng['Date'].between('2021-02-01', '2021-02-28')]
if len(feb_2021) > 0:
    ax.scatter(feb_2021['Date'], feb_2021['Price'], color='red', s=50, zorder=5, label='Feb 2021 Winter Storm')
    ax.legend()

plt.tight_layout()
output_path = OUTPUT_DIR / "01_natural_gas_prices.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# Plot 2: Natural Gas Price Distribution
print("\n2. Plotting Natural Gas distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.hist(df_ng['Price'], bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Price ($/MMBtu)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Natural Gas Price Distribution', fontsize=14, fontweight='bold')
ax1.axvline(df_ng['Price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df_ng["Price"].mean():.2f}')
ax1.axvline(df_ng['Price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df_ng["Price"].median():.2f}')
ax1.legend()

ax2.boxplot(df_ng['Price'], vert=True)
ax2.set_ylabel('Price ($/MMBtu)', fontsize=12)
ax2.set_title('Natural Gas Price Box Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUTPUT_DIR / "02_natural_gas_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# Plot 3: DAM Prices Over Time
print("\n3. Plotting DAM prices...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Full time series
ax1.plot(df_dam_avg['Datetime'], df_dam_avg['Settlement Point Price'], linewidth=0.5, alpha=0.7)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price ($/MWh)', fontsize=12)
ax1.set_title('ERCOT Day-Ahead Market Prices (Full Range)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Highlight extreme events
extreme = df_dam_avg[df_dam_avg['Settlement Point Price'] > 1000]
if len(extreme) > 0:
    ax1.scatter(extreme['Datetime'], extreme['Settlement Point Price'], color='red', s=10, alpha=0.5, label=f'Price > $1000/MWh ({len(extreme)} hours)')
    ax1.legend()

# Zoomed view (capped at $500 to see normal variation)
ax2.plot(df_dam_avg['Datetime'], df_dam_avg['Settlement Point Price'].clip(upper=500), linewidth=0.5, alpha=0.7)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Price ($/MWh)', fontsize=12)
ax2.set_title('ERCOT DAM Prices (Capped at $500 for visibility)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 500)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUTPUT_DIR / "03_dam_prices.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# Plot 4: DAM Price Distribution
print("\n4. Plotting DAM price distribution...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Full distribution
axes[0,0].hist(df_dam_avg['Settlement Point Price'], bins=100, edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Price ($/MWh)', fontsize=12)
axes[0,0].set_ylabel('Frequency', fontsize=12)
axes[0,0].set_title('DAM Price Distribution (All Prices)', fontsize=14, fontweight='bold')
axes[0,0].axvline(df_dam_avg['Settlement Point Price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df_dam_avg["Settlement Point Price"].mean():.2f}')
axes[0,0].axvline(df_dam_avg['Settlement Point Price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df_dam_avg["Settlement Point Price"].median():.2f}')
axes[0,0].legend()

# Normal range (< $200)
normal_prices = df_dam_avg[df_dam_avg['Settlement Point Price'] < 200]
axes[0,1].hist(normal_prices['Settlement Point Price'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0,1].set_xlabel('Price ($/MWh)', fontsize=12)
axes[0,1].set_ylabel('Frequency', fontsize=12)
axes[0,1].set_title(f'DAM Price Distribution (< $200/MWh) - {len(normal_prices):,} hours', fontsize=14, fontweight='bold')
axes[0,1].axvline(normal_prices['Settlement Point Price'].mean(), color='red', linestyle='--', linewidth=2)

# Log scale
axes[1,0].hist(df_dam_avg['Settlement Point Price'], bins=100, edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Price ($/MWh)', fontsize=12)
axes[1,0].set_ylabel('Frequency (log scale)', fontsize=12)
axes[1,0].set_yscale('log')
axes[1,0].set_title('DAM Price Distribution (Log Scale)', fontsize=14, fontweight='bold')

# Box plot
axes[1,1].boxplot([normal_prices['Settlement Point Price'], df_dam_avg['Settlement Point Price']], labels=['< $200/MWh', 'All Prices'])
axes[1,1].set_ylabel('Price ($/MWh)', fontsize=12)
axes[1,1].set_title('DAM Price Box Plots', fontsize=14, fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUTPUT_DIR / "04_dam_price_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# Plot 5: Weather Variables (Sample day)
print("\n5. Plotting weather variables...")
houston = df_weather_sample[df_weather_sample['zone'] == 'HOUSTON'].sort_values('datetime')

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(f'HRRR Weather Forecast - Houston Zone\n{weather_files[0].stem}', fontsize=16, fontweight='bold')

# Temperature
axes[0,0].plot(houston['datetime'], houston['temp_f'], marker='o', linewidth=2)
axes[0,0].set_ylabel('Temperature (°F)', fontsize=12)
axes[0,0].set_title('Temperature')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='x', rotation=45)

# Wind Speed
axes[0,1].plot(houston['datetime'], houston['wind_speed_mph'], marker='o', linewidth=2, color='green')
axes[0,1].set_ylabel('Wind Speed (mph)', fontsize=12)
axes[0,1].set_title('Wind Speed')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].tick_params(axis='x', rotation=45)

# Solar Radiation
axes[1,0].plot(houston['datetime'], houston['solar_radiation_wm2'], marker='o', linewidth=2, color='orange')
axes[1,0].set_ylabel('Solar Radiation (W/m²)', fontsize=12)
axes[1,0].set_title('Solar Radiation')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].tick_params(axis='x', rotation=45)

# Relative Humidity
axes[1,1].plot(houston['datetime'], houston['relative_humidity'], marker='o', linewidth=2, color='blue')
axes[1,1].set_ylabel('Relative Humidity (%)', fontsize=12)
axes[1,1].set_title('Relative Humidity')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].tick_params(axis='x', rotation=45)

# Pressure
axes[2,0].plot(houston['datetime'], houston['pressure_inhg'], marker='o', linewidth=2, color='purple')
axes[2,0].set_ylabel('Pressure (inHg)', fontsize=12)
axes[2,0].set_title('Barometric Pressure')
axes[2,0].grid(True, alpha=0.3)
axes[2,0].tick_params(axis='x', rotation=45)

# Cloud Cover
axes[2,1].plot(houston['datetime'], houston['cloud_cover_pct'], marker='o', linewidth=2, color='gray')
axes[2,1].set_ylabel('Cloud Cover (%)', fontsize=12)
axes[2,1].set_title('Cloud Cover')
axes[2,1].grid(True, alpha=0.3)
axes[2,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
output_path = OUTPUT_DIR / "05_weather_variables_sample.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# Plot 6: Monthly Natural Gas Patterns
print("\n6. Plotting monthly patterns...")
df_ng['Year'] = df_ng['Date'].dt.year
df_ng['Month'] = df_ng['Date'].dt.month

fig, ax = plt.subplots(figsize=(16, 8))
for year in sorted(df_ng['Year'].unique())[-5:]:  # Last 5 years
    year_data = df_ng[df_ng['Year'] == year]
    monthly_avg = year_data.groupby('Month')['Price'].mean()
    ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, label=str(year))

ax.set_xlabel('Month', fontsize=14)
ax.set_ylabel('Average Price ($/MMBtu)', fontsize=14)
ax.set_title('Natural Gas Seasonal Patterns (Last 5 Years)', fontsize=16, fontweight='bold')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUTPUT_DIR / "06_natural_gas_seasonal.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# Plot 7: Gas vs DAM Correlation
print("\n7. Creating gas vs electricity correlation...")

# Align dates (daily natural gas to daily average DAM)
df_dam_avg['Date'] = pd.to_datetime(df_dam_avg['Datetime']).dt.date
df_ng['DateOnly'] = df_ng['Date'].dt.date

dam_daily = df_dam_avg.groupby('Date')['Settlement Point Price'].mean().reset_index()
dam_daily.columns = ['Date', 'DAM_Price_Avg']

merged = pd.merge(df_ng[['DateOnly', 'Price']], dam_daily, left_on='DateOnly', right_on='Date', how='inner')

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(merged['Price'], merged['DAM_Price_Avg'], alpha=0.3, s=10)
ax.set_xlabel('Natural Gas Price ($/MMBtu)', fontsize=14)
ax.set_ylabel('DAM Price ($/MWh, daily avg)', fontsize=14)
ax.set_title('Natural Gas vs Electricity Prices', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add correlation coefficient
corr = merged['Price'].corr(merged['DAM_Price_Avg'])
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
        fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = OUTPUT_DIR / "07_gas_vs_electricity.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 4. SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

summary = f"""
ERCOT Energy Trading Project - Initial EDA Summary
Generated: {datetime.now()}

DATASETS LOADED:
+ Natural Gas Prices: {len(df_ng):,} records ({df_ng['Date'].min().date()} to {df_ng['Date'].max().date()})
+ HRRR Weather Files: {len(weather_files):,} files
+ DAM Prices: {len(df_dam_avg):,} records (HB_BUSAVG system average)
+ Load Forecasts: {len(df_load):,} records (sample) if df_load is not None else 'Not loaded'

NATURAL GAS STATISTICS:
  Mean:   ${df_ng['Price'].mean():.2f}/MMBtu
  Median: ${df_ng['Price'].median():.2f}/MMBtu
  Std:    ${df_ng['Price'].std():.2f}/MMBtu
  Min:    ${df_ng['Price'].min():.2f}/MMBtu (pandemic low)
  Max:    ${df_ng['Price'].max():.2f}/MMBtu (Feb 2021 storm)

DAM PRICE STATISTICS (HB_BUSAVG):
  Mean:   ${df_dam_avg['Settlement Point Price'].mean():.2f}/MWh
  Median: ${df_dam_avg['Settlement Point Price'].median():.2f}/MWh
  Std:    ${df_dam_avg['Settlement Point Price'].std():.2f}/MWh
  Max:    ${df_dam_avg['Settlement Point Price'].max():.2f}/MWh
  Spikes >$1000/MWh: {(df_dam_avg['Settlement Point Price'] > 1000).sum():,} hours

GAS vs ELECTRICITY CORRELATION:
  Pearson correlation: {merged['Price'].corr(merged['DAM_Price_Avg']):.3f}
  (This is strong! Natural gas is a key price driver)

VISUALIZATIONS CREATED:
  1. Natural gas price time series
  2. Natural gas price distribution
  3. DAM prices time series (full + zoomed)
  4. DAM price distribution (4 analyses)
  5. Weather variables (sample forecast - 6 subplots)
  6. Natural gas seasonal patterns
  7. Gas vs electricity correlation scatter

All plots saved to: {OUTPUT_DIR}

NEXT STEPS:
1. Review generated plots
2. Examine extreme events (Feb 2021)
3. Create merge script to combine all datasets
4. Build full correlation matrix
5. Feature engineering

GPU INFO:
System has access to 2x NVIDIA H100 GPUs
These will be valuable for:
  - Deep learning models (Phase 2C)
  - Large-scale hyperparameter tuning
  - Ensemble model training
Note: Initial models (XGBoost/LightGBM) run on CPU efficiently
"""

print(summary)

# Save summary to file (with UTF-8 encoding)
summary_path = OUTPUT_DIR / "00_eda_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)
print(f"\n+ Summary saved to: {summary_path}")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print(f"\nCheck outputs in: {OUTPUT_DIR}")
print("\nKey Findings:")
print(f"  - Gas-electricity correlation: {merged['Price'].corr(merged['DAM_Price_Avg']):.3f}")
print(f"  - Feb 2021 gas peak: ${feb_2021['Price'].max():.2f}/MMBtu")
print(f"  - Feb 2021 DAM peak: ${feb_2021_dam['Settlement Point Price'].max():.2f}/MWh")
print(f"  - Extreme price events (>$1000/MWh): {(df_dam_avg['Settlement Point Price'] > 1000).sum():,} hours")