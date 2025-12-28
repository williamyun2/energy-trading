"""
Linear Regression Model - ERCOT DAM Price Forecasting
ML Model #1: Establishes ML baseline with engineered features

Goal: Beat persistence baseline (MAE < $10.97/MWh)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root and src directory to path
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from merge_dataset.loader import load_clean_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import feature engineering
from models.feature_engineering import engineer_all_features

# Output directory
RESULTS_DIR = Path(r"D:\Users\williamyun\proj\power_trading\results\linear_regression")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def select_features(df):
    """
    Select features for modeling
    Excludes: datetime, target variable, and highly correlated duplicates
    """
    
    # Exclude these columns from features
    exclude_cols = [
        'datetime',           # Not a feature
        'dam_price',          # Target variable
    ]
    
    # Also exclude original raw weather columns (we use derived features instead)
    raw_weather = [c for c in df.columns if any(x in c for x in [
        'temp_f_HOUSTON', 'temp_f_NORTH', 'temp_f_SOUTH', 'temp_f_WEST',
        'wind_speed_mph_HOUSTON', 'wind_speed_mph_NORTH', 'wind_speed_mph_SOUTH', 'wind_speed_mph_WEST',
        'solar_radiation_wm2_HOUSTON', 'solar_radiation_wm2_NORTH', 'solar_radiation_wm2_SOUTH', 'solar_radiation_wm2_WEST',
        'relative_humidity_HOUSTON', 'relative_humidity_NORTH', 'relative_humidity_SOUTH', 'relative_humidity_WEST'
    ])]
    
    exclude_cols.extend(raw_weather)
    
    # Get all feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return feature_cols


def train_test_split(df, test_months=6):
    """Split data into train/test sets (temporal split)"""
    
    max_date = df['datetime'].max()
    split_date = max_date - pd.DateOffset(months=test_months)
    
    df_train = df[df['datetime'] < split_date].copy()
    df_test = df[df['datetime'] >= split_date].copy()
    
    return df_train, df_test, split_date


def train_linear_regression(X_train, y_train, verbose=True):
    """Train linear regression model with feature scaling"""
    
    if verbose:
        print("\nTraining linear regression...")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Training samples: {X_train.shape[0]:,}")
    
    # Scale features (important for linear regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    if verbose:
        print(f"  ✓ Model trained")
    
    return model, scaler


def evaluate_model(y_true, y_pred, model_name="Linear Regression"):
    """Calculate evaluation metrics"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    max_error = np.max(np.abs(y_true - y_pred))
    median_error = np.median(np.abs(y_true - y_pred))
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Max Error': max_error,
        'Median Error': median_error
    }


def plot_predictions(df_test, y_pred, results_dir):
    """Create diagnostic plots"""
    
    df_plot = df_test.copy()
    df_plot['predicted'] = y_pred
    df_plot['error'] = df_plot['dam_price'] - df_plot['predicted']
    df_plot['abs_error'] = np.abs(df_plot['error'])
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df_plot['dam_price'], df_plot['predicted'], alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(df_plot['dam_price'].min(), df_plot['predicted'].min())
    max_val = max(df_plot['dam_price'].max(), df_plot['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price ($/MWh)', fontsize=12)
    ax.set_ylabel('Predicted Price ($/MWh)', fontsize=12)
    ax.set_title('Linear Regression: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '01_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Error distribution
    axes[0, 0].hist(df_plot['error'], bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Prediction Error ($/MWh)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error over time
    axes[0, 1].plot(df_plot['datetime'], df_plot['error'], alpha=0.5, linewidth=0.5)
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Prediction Error ($/MWh)')
    axes[0, 1].set_title('Errors Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Absolute error over time
    axes[1, 0].plot(df_plot['datetime'], df_plot['abs_error'], alpha=0.5, linewidth=0.5, color='orange')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Absolute Error ($/MWh)')
    axes[1, 0].set_title('Absolute Errors Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Error by actual price
    axes[1, 1].scatter(df_plot['dam_price'], df_plot['abs_error'], alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Actual Price ($/MWh)')
    axes[1, 1].set_ylabel('Absolute Error ($/MWh)')
    axes[1, 1].set_title('Error vs Actual Price')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / '02_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Last 30 days
    df_recent = df_plot.tail(30 * 24)  # Last 30 days (24 hours/day)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df_recent['datetime'], df_recent['dam_price'], label='Actual', linewidth=2, alpha=0.8)
    ax.plot(df_recent['datetime'], df_recent['predicted'], label='Predicted', linewidth=2, alpha=0.8)
    ax.fill_between(df_recent['datetime'], df_recent['dam_price'], df_recent['predicted'], 
                     alpha=0.3, label='Error')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($/MWh)', fontsize=12)
    ax.set_title('Last 30 Days: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / '03_last_30_days.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Error by hour of day
    hourly_errors = df_plot.groupby(df_plot['datetime'].dt.hour).agg({
        'error': ['mean', 'std'],
        'abs_error': 'mean'
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    # Use actual hours from the data, not assuming all 24 hours exist
    hours = hourly_errors.index
    ax.errorbar(hours, hourly_errors['error']['mean'], yerr=hourly_errors['error']['std'],
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Mean Error ($/MWh)', fontsize=12)
    ax.set_title('Prediction Error by Hour of Day', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))  # Show all 24 hours on x-axis even if some missing
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '04_error_by_hour.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_feature_importance(model, feature_names, results_dir):
    """Analyze and plot feature importance (coefficients)"""
    
    # Get absolute coefficients (magnitude of impact)
    importances = np.abs(model.coef_)
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'coefficient': model.coef_
    }).sort_values('importance', ascending=False)
    
    # Save to CSV
    feature_importance.to_csv(results_dir / 'feature_importance.csv', index=False)
    
    # Plot top 20 features
    top_20 = feature_importance.head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in top_20['coefficient']]
    ax.barh(range(len(top_20)), top_20['importance'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'])
    ax.set_xlabel('Absolute Coefficient (Feature Importance)')
    ax.set_title('Top 20 Most Important Features\n(Green=Positive, Red=Negative)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(results_dir / '05_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*80)
    for i, row in feature_importance.head(10).iterrows():
        direction = "↑ increases" if row['coefficient'] > 0 else "↓ decreases"
        print(f"{row['feature']:40s}: {row['importance']:8.2f} ({direction} price)")


def main():
    """Main training and evaluation pipeline"""
    
    print("="*80)
    print("LINEAR REGRESSION MODEL - ERCOT DAM PRICE FORECASTING")
    print("="*80)
    print("Goal: Beat persistence baseline (MAE < $10.97/MWh)")
    print()
    
    # 1. Load data
    print("Loading clean data...")
    df = load_clean_data(verbose=False)
    print(f"✓ Loaded {len(df):,} records")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # 2. Engineer features
    print("\nEngineering features...")
    df_features = engineer_all_features(df, verbose=False)
    print(f"✓ Created {df_features.shape[1] - 20} new features")
    print(f"  Total features: {df_features.shape[1]}")
    
    # 3. Select features
    feature_cols = select_features(df_features)
    print(f"\n✓ Selected {len(feature_cols)} features for modeling")
    
    # 4. Handle missing values from lag features
    print("\nHandling missing values from lag features...")
    rows_before = len(df_features)
    df_features = df_features.dropna()
    rows_after = len(df_features)
    print(f"  Dropped {rows_before - rows_after:,} rows with missing lags")
    print(f"  Remaining: {rows_after:,} rows")
    
    # 5. Train/test split
    print("\nSplitting train/test (last 6 months for testing)...")
    df_train, df_test, split_date = train_test_split(df_features, test_months=6)
    print(f"✓ Train: {len(df_train):,} records ({df_train['datetime'].min().date()} to {df_train['datetime'].max().date()})")
    print(f"✓ Test:  {len(df_test):,} records ({df_test['datetime'].min().date()} to {df_test['datetime'].max().date()})")
    
    # 6. Prepare feature matrices
    X_train = df_train[feature_cols].values
    y_train = df_train['dam_price'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['dam_price'].values
    
    # 7. Train model
    model, scaler = train_linear_regression(X_train, y_train, verbose=True)
    
    # 8. Predict on test set
    print("\nPredicting on test set...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    print(f"✓ Generated {len(y_pred):,} predictions")
    
    # 9. Evaluate
    print("\nEvaluating model performance...")
    metrics = evaluate_model(y_test, y_pred, model_name='Linear Regression')
    
    # Compare to baseline
    baseline_mae = 10.97
    improvement = (baseline_mae - metrics['MAE']) / baseline_mae * 100
    
    print("\n" + "="*80)
    print("LINEAR REGRESSION RESULTS")
    print("="*80)
    print(f"Model: {metrics['Model']}")
    print(f"MAE:          ${metrics['MAE']:.2f}/MWh")
    print(f"RMSE:         ${metrics['RMSE']:.2f}/MWh")
    print(f"MAPE:         {metrics['MAPE']:.2f}%")
    print(f"R² Score:     {metrics['R2']:.4f}")
    print(f"Max Error:    ${metrics['Max Error']:.2f}/MWh")
    print(f"Median Error: ${metrics['Median Error']:.2f}/MWh")
    print()
    print("="*80)
    print("COMPARISON TO BASELINE")
    print("="*80)
    print(f"Baseline (Persistence) MAE:  ${baseline_mae:.2f}/MWh")
    print(f"Linear Regression MAE:       ${metrics['MAE']:.2f}/MWh")
    print(f"Improvement:                 {improvement:+.1f}%")
    
    if metrics['MAE'] < baseline_mae:
        print(f"\n✅ SUCCESS! Beat baseline by ${baseline_mae - metrics['MAE']:.2f}/MWh")
    else:
        print(f"\n❌ Did not beat baseline (worse by ${metrics['MAE'] - baseline_mae:.2f}/MWh)")
    
    # 10. Create plots
    print("\nCreating diagnostic plots...")
    plot_predictions(df_test, y_pred, RESULTS_DIR)
    print(f"✓ Saved plots to: {RESULTS_DIR}")
    
    # 11. Analyze feature importance
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(model, feature_cols, RESULTS_DIR)
    
    # 12. Save results
    print("\nSaving results...")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(RESULTS_DIR / "linear_regression_metrics.csv", index=False)
    
    predictions_df = df_test[['datetime', 'dam_price']].copy()
    predictions_df['predicted'] = y_pred
    predictions_df['error'] = predictions_df['dam_price'] - predictions_df['predicted']
    predictions_df.to_csv(RESULTS_DIR / "linear_regression_predictions.csv", index=False)
    
    print(f"✓ Saved metrics to: {RESULTS_DIR / 'linear_regression_metrics.csv'}")
    print(f"✓ Saved predictions to: {RESULTS_DIR / 'linear_regression_predictions.csv'}")
    print(f"✓ Saved feature importance to: {RESULTS_DIR / 'feature_importance.csv'}")
    
    print("\n" + "="*80)
    print("LINEAR REGRESSION MODEL COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - linear_regression_metrics.csv")
    print(f"  - linear_regression_predictions.csv")
    print(f"  - feature_importance.csv")
    print(f"  - 01_actual_vs_predicted.png")
    print(f"  - 02_error_analysis.png")
    print(f"  - 03_last_30_days.png")
    print(f"  - 04_error_by_hour.png")
    print(f"  - 05_feature_importance.png")
    
    return metrics, model, scaler


if __name__ == "__main__":
    metrics, model, scaler = main()