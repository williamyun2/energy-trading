"""
Feature Engineering for ERCOT DAM Price Prediction

Creates:
1. Lag features (previous prices, load, weather)
2. Temporal features (hour, day, month, season)
3. Weather-derived features (temp extremes, wind power potential)
4. Domain-specific features (price volatility, load forecast error proxies)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def create_lag_features(df, verbose=True):
    """Create lag features for time series prediction"""
    
    if verbose:
        print("\n" + "="*80)
        print("CREATING LAG FEATURES")
        print("="*80)
    
    # Sort by datetime to ensure proper lag calculation
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Price lags (critical for time series)
    if verbose:
        print("\n1️⃣ Price Lag Features:")
    
    lag_hours = [24, 48, 72, 168]  # 1 day, 2 days, 3 days, 1 week
    for lag in lag_hours:
        col_name = f'price_lag_{lag}h'
        df[col_name] = df['dam_price'].shift(lag)
        if verbose:
            print(f"   Created: {col_name}")
    
    # Rolling price statistics
    for window in [24, 168]:  # 1 day, 1 week
        df[f'price_rolling_mean_{window}h'] = df['dam_price'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'price_rolling_std_{window}h'] = df['dam_price'].shift(1).rolling(window=window, min_periods=1).std()
        if verbose:
            print(f"   Created: price_rolling_mean_{window}h, price_rolling_std_{window}h")
    
    # Price changes
    df['price_change_24h'] = df['dam_price'].shift(1) - df['dam_price'].shift(25)
    if verbose:
        print(f"   Created: price_change_24h")
    
    # Load forecast lags
    if verbose:
        print("\n2️⃣ Load Forecast Lag Features:")
    
    for lag in [24, 48, 168]:
        col_name = f'load_lag_{lag}h'
        df[col_name] = df['system_load_forecast'].shift(lag)
        if verbose:
            print(f"   Created: {col_name}")
    
    # Load change (day-over-day)
    df['load_change_24h'] = df['system_load_forecast'] - df['system_load_forecast'].shift(24)
    if verbose:
        print(f"   Created: load_change_24h")
    
    # Natural gas price lags
    if verbose:
        print("\n3️⃣ Natural Gas Price Lag Features:")
    
    for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month (daily data)
        # Since ng_price is daily, we need to shift by hours
        col_name = f'ng_price_lag_{lag}d'
        df[col_name] = df['ng_price'].shift(lag * 24)
        if verbose:
            print(f"   Created: {col_name}")
    
    df['ng_price_change_7d'] = df['ng_price'] - df['ng_price'].shift(168)
    if verbose:
        print(f"   Created: ng_price_change_7d")
    
    # Weather lags (using Houston as primary, it's the largest load zone)
    if verbose:
        print("\n4️⃣ Weather Lag Features:")
    
    weather_vars = ['temp_f_HOUSTON', 'wind_speed_mph_HOUSTON']
    for var in weather_vars:
        if var in df.columns:
            for lag in [24, 48]:
                col_name = f'{var}_lag_{lag}h'
                df[col_name] = df[var].shift(lag)
                if verbose:
                    print(f"   Created: {col_name}")
    
    return df


def create_temporal_features(df, verbose=True):
    """Create temporal features (hour, day, month, season)"""
    
    if verbose:
        print("\n" + "="*80)
        print("CREATING TEMPORAL FEATURES")
        print("="*80)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Hour of day (0-23)
    df['hour'] = df['datetime'].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Month (1-12)
    df['month'] = df['datetime'].dt.month
    
    # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df['season'] = df['month'].map({
        12: 1, 1: 1, 2: 1,  # Winter
        3: 2, 4: 2, 5: 2,   # Spring
        6: 3, 7: 3, 8: 3,   # Summer
        9: 4, 10: 4, 11: 4  # Fall
    })
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Peak hours (high demand periods in ERCOT: 6am-10pm)
    df['is_peak_hour'] = ((df['hour'] >= 6) & (df['hour'] <= 22)).astype(int)
    
    # Super peak (2pm-7pm, highest demand)
    df['is_super_peak'] = ((df['hour'] >= 14) & (df['hour'] <= 19)).astype(int)
    
    # Cyclical encoding for hour (preserves 23→0 continuity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if verbose:
        print(f"\n   Created temporal features:")
        print(f"   - hour, day_of_week, month, season")
        print(f"   - is_weekend, is_peak_hour, is_super_peak")
        print(f"   - Cyclical encodings: hour_sin/cos, day_sin/cos, month_sin/cos")
    
    return df


def create_weather_features(df, verbose=True):
    """Create weather-derived features"""
    
    if verbose:
        print("\n" + "="*80)
        print("CREATING WEATHER-DERIVED FEATURES")
        print("="*80)
    
    # Average temperature across all zones
    temp_cols = [c for c in df.columns if c.startswith('temp_f_')]
    if temp_cols:
        df['temp_f_avg'] = df[temp_cols].mean(axis=1)
        df['temp_f_max'] = df[temp_cols].max(axis=1)
        df['temp_f_min'] = df[temp_cols].min(axis=1)
        df['temp_f_range'] = df['temp_f_max'] - df['temp_f_min']
        if verbose:
            print(f"   Created: temp_f_avg, temp_f_max, temp_f_min, temp_f_range")
    
    # Heating/Cooling degree days approximation
    # Cooling degree days: max(0, temp - 65)
    # Heating degree days: max(0, 65 - temp)
    if 'temp_f_avg' in df.columns:
        df['cooling_degree_hours'] = np.maximum(0, df['temp_f_avg'] - 65)
        df['heating_degree_hours'] = np.maximum(0, 65 - df['temp_f_avg'])
        if verbose:
            print(f"   Created: cooling_degree_hours, heating_degree_hours")
    
    # Extreme temperature indicators
    if 'temp_f_avg' in df.columns:
        df['is_extreme_heat'] = (df['temp_f_avg'] > 95).astype(int)  # >95°F
        df['is_extreme_cold'] = (df['temp_f_avg'] < 32).astype(int)  # <32°F
        if verbose:
            print(f"   Created: is_extreme_heat, is_extreme_cold")
    
    # Wind speed features
    wind_cols = [c for c in df.columns if c.startswith('wind_speed_mph_')]
    if wind_cols:
        df['wind_speed_avg'] = df[wind_cols].mean(axis=1)
        df['wind_speed_max'] = df[wind_cols].max(axis=1)
        if verbose:
            print(f"   Created: wind_speed_avg, wind_speed_max")
    
    # Solar radiation features
    solar_cols = [c for c in df.columns if c.startswith('solar_radiation_wm2_')]
    if solar_cols:
        df['solar_radiation_avg'] = df[solar_cols].mean(axis=1)
        # High solar = potential for solar generation reducing net load
        df['high_solar'] = (df['solar_radiation_avg'] > 500).astype(int)
        if verbose:
            print(f"   Created: solar_radiation_avg, high_solar")
    
    return df


def create_interaction_features(df, verbose=True):
    """Create interaction features between key variables"""
    
    if verbose:
        print("\n" + "="*80)
        print("CREATING INTERACTION FEATURES")
        print("="*80)
    
    # Temperature × Load (AC demand proxy)
    if 'temp_f_avg' in df.columns and 'system_load_forecast' in df.columns:
        df['temp_load_interaction'] = df['temp_f_avg'] * df['system_load_forecast'] / 1000
        if verbose:
            print(f"   Created: temp_load_interaction")
    
    # Natural Gas × Load (fuel cost impact)
    if 'ng_price' in df.columns and 'system_load_forecast' in df.columns:
        df['gas_load_interaction'] = df['ng_price'] * df['system_load_forecast'] / 1000
        if verbose:
            print(f"   Created: gas_load_interaction")
    
    # Peak hour × Temperature (peak demand during heat)
    if 'is_peak_hour' in df.columns and 'temp_f_avg' in df.columns:
        df['peak_temp_interaction'] = df['is_peak_hour'] * df['temp_f_avg']
        if verbose:
            print(f"   Created: peak_temp_interaction")
    
    # Extreme heat × Load (critical stress condition)
    if 'is_extreme_heat' in df.columns and 'system_load_forecast' in df.columns:
        df['extreme_heat_load'] = df['is_extreme_heat'] * df['system_load_forecast']
        if verbose:
            print(f"   Created: extreme_heat_load")
    
    return df


def engineer_all_features(df, verbose=True):
    """
    Apply all feature engineering steps
    
    Args:
        df: DataFrame with raw data (must have datetime sorted)
        verbose: Print progress
    
    Returns:
        DataFrame with engineered features
    """
    
    if verbose:
        print("="*80)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*80)
        print(f"Input shape: {df.shape}")
    
    # Apply feature engineering steps
    df = create_lag_features(df, verbose=verbose)
    df = create_temporal_features(df, verbose=verbose)
    df = create_weather_features(df, verbose=verbose)
    df = create_interaction_features(df, verbose=verbose)
    
    if verbose:
        print("\n" + "="*80)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*80)
        print(f"Output shape: {df.shape}")
        print(f"New features created: {df.shape[1] - 20}")  # Original had 20 columns
        
        # Show feature categories
        lag_features = [c for c in df.columns if 'lag' in c or 'rolling' in c or 'change' in c]
        temporal_features = [c for c in df.columns if any(x in c for x in ['hour', 'day', 'month', 'season', 'weekend', 'peak', 'sin', 'cos'])]
        weather_features = [c for c in df.columns if any(x in c for x in ['cooling', 'heating', 'extreme', 'solar', 'wind_speed_avg', 'temp_f_avg'])]
        interaction_features = [c for c in df.columns if 'interaction' in c or 'extreme_heat_load' in c]
        
        print(f"\nFeature breakdown:")
        print(f"  Lag features: {len(lag_features)}")
        print(f"  Temporal features: {len(temporal_features)}")
        print(f"  Weather-derived features: {len(weather_features)}")
        print(f"  Interaction features: {len(interaction_features)}")
        print(f"  Original features: 20")
        print(f"  Total: {df.shape[1]}")
    
    return df


if __name__ == "__main__":
    """Test feature engineering on sample data"""
    
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))
    
    from merge_dataset.loader import load_clean_data
    
    print("Loading data...")
    df = load_clean_data(verbose=False)
    print(f"✓ Loaded {len(df):,} records")
    
    print("\nApplying feature engineering...")
    df_features = engineer_all_features(df, verbose=True)
    
    print("\n" + "="*80)
    print("SAMPLE FEATURES")
    print("="*80)
    print("\nFirst 5 rows (selected columns):")
    sample_cols = ['datetime', 'dam_price', 'price_lag_24h', 'hour', 'is_peak_hour', 
                   'temp_f_avg', 'cooling_degree_hours', 'temp_load_interaction']
    available_cols = [c for c in sample_cols if c in df_features.columns]
    print(df_features[available_cols].head())
    
    print(f"\n✓ Feature engineering test complete!")
    print(f"✓ {df_features.shape[1]} total features ready for modeling")