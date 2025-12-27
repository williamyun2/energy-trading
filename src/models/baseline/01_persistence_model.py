"""
Baseline Persistence Model - ERCOT DAM Price Forecasting
Simple benchmark: tomorrow's price = today's price

This establishes the baseline performance that ML models must beat.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Output directory
RESULTS_DIR = Path(r"D:\Users\williamyun\proj\power_trading\results\baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_persistence_forecast(df):
    """
    Create persistence forecast: tomorrow = today
    
    For each hour, predict the price will be the same as 24 hours ago
    """
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Create lag feature: price from 24 hours ago
    df['price_lag_24h'] = df['dam_price'].shift(24)
    
    # Drop first 24 hours (no lag data)
    df = df.dropna(subset=['price_lag_24h'])
    
    return df


def train_test_split(df, test_months=6):
    """
    Split data into train/test sets
    
    Parameters:
    -----------
    df : pd.DataFrame
    test_months : int
        Number of months to use for testing (default 6)
    
    Returns:
    --------
    df_train, df_test
    """
    
    # Calculate split date (last N months for testing)
    max_date = df['datetime'].max()
    split_date = max_date - pd.DateOffset(months=test_months)
    
    df_train = df[df['datetime'] < split_date].copy()
    df_test = df[df['datetime'] >= split_date].copy()
    
    return df_train, df_test, split_date


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate evaluation metrics"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_error = np.median(np.abs(y_true - y_pred))
    
    metrics = {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Max Error': max_error,
        'Median Error': median_error
    }
    
    return metrics


def plot_predictions(df_test, output_dir):
    """Create diagnostic plots"""
    
    # Plot 1: Actual vs Predicted (scatter)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df_test['dam_price'], df_test['price_lag_24h'], alpha=0.1, s=1)
    
    # Add perfect prediction line
    min_val = min(df_test['dam_price'].min(), df_test['price_lag_24h'].min())
    max_val = max(df_test['dam_price'].max(), df_test['price_lag_24h'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price ($/MWh)', fontsize=14)
    ax.set_ylabel('Predicted Price ($/MWh)', fontsize=14)
    ax.set_title('Persistence Model: Actual vs Predicted', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_actual_vs_predicted.png', dpi=150)
    plt.close()
    
    # Plot 2: Prediction errors over time
    df_test['error'] = df_test['dam_price'] - df_test['price_lag_24h']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Errors over time
    ax1.plot(df_test['datetime'], df_test['error'], linewidth=0.5, alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Error ($/MWh)', fontsize=12)
    ax1.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2.hist(df_test['error'], bins=100, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Error ($/MWh)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_errors_analysis.png', dpi=150)
    plt.close()
    
    # Plot 3: Time series comparison (last 30 days)
    last_30_days = df_test.tail(30 * 24)  # 30 days * 24 hours
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(last_30_days['datetime'], last_30_days['dam_price'], 
            label='Actual Price', linewidth=2, alpha=0.8)
    ax.plot(last_30_days['datetime'], last_30_days['price_lag_24h'], 
            label='Predicted Price (Persistence)', linewidth=2, alpha=0.8)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price ($/MWh)', fontsize=14)
    ax.set_title('Last 30 Days: Actual vs Predicted', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_last_30_days.png', dpi=150)
    plt.close()
    
    # Plot 4: Error by hour of day
    df_test['hour'] = df_test['datetime'].dt.hour
    hourly_errors = df_test.groupby('hour')['error'].agg(['mean', 'std'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(hourly_errors.index, hourly_errors['mean'], yerr=hourly_errors['std'], 
           capsize=5, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Mean Error ($/MWh)', fontsize=14)
    ax.set_title('Prediction Error by Hour of Day', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_error_by_hour.png', dpi=150)
    plt.close()


def main():
    """Run baseline persistence model"""
    
    print("="*80)
    print("BASELINE PERSISTENCE MODEL")
    print("="*80)
    print("Benchmark: Tomorrow's price = Today's price")
    print()
    
    # Load data
    print("Loading clean data...")
    df = load_clean_data(verbose=False)
    print(f"✓ Loaded {len(df):,} clean records")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Create persistence forecast
    print("\nCreating persistence forecast (lag 24 hours)...")
    df = create_persistence_forecast(df)
    print(f"✓ Created {len(df):,} forecasts")
    
    # Train/test split
    print("\nSplitting train/test (last 6 months for testing)...")
    df_train, df_test, split_date = train_test_split(df, test_months=6)
    print(f"✓ Train: {len(df_train):,} records ({df_train['datetime'].min().date()} to {df_train['datetime'].max().date()})")
    print(f"✓ Test:  {len(df_test):,} records ({df_test['datetime'].min().date()} to {df_test['datetime'].max().date()})")
    
    # Evaluate on test set
    print("\nEvaluating persistence model on test set...")
    metrics = evaluate_model(
        y_true=df_test['dam_price'],
        y_pred=df_test['price_lag_24h'],
        model_name='Persistence (24h lag)'
    )
    
    print("\n" + "="*80)
    print("BASELINE RESULTS")
    print("="*80)
    print(f"Model: {metrics['Model']}")
    print(f"MAE:          ${metrics['MAE']:.2f}/MWh")
    print(f"RMSE:         ${metrics['RMSE']:.2f}/MWh")
    print(f"MAPE:         {metrics['MAPE']:.2f}%")
    print(f"R² Score:     {metrics['R2']:.4f}")
    print(f"Max Error:    ${metrics['Max Error']:.2f}/MWh")
    print(f"Median Error: ${metrics['Median Error']:.2f}/MWh")
    
    # Create plots
    print("\nCreating diagnostic plots...")
    plot_predictions(df_test, RESULTS_DIR)
    print(f"✓ Saved plots to: {RESULTS_DIR}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = RESULTS_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    # Save predictions
    predictions_path = RESULTS_DIR / "baseline_predictions.csv"
    df_test[['datetime', 'dam_price', 'price_lag_24h']].to_csv(predictions_path, index=False)
    print(f"✓ Saved predictions to: {predictions_path}")
    
    print("\n" + "="*80)
    print("BASELINE MODEL COMPLETE!")
    print("="*80)
    print(f"\nKey Takeaway:")
    print(f"  ML models must achieve MAE < ${metrics['MAE']:.2f}/MWh to beat baseline")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - baseline_metrics.csv")
    print(f"  - baseline_predictions.csv")
    print(f"  - 01_actual_vs_predicted.png")
    print(f"  - 02_errors_analysis.png")
    print(f"  - 03_last_30_days.png")
    print(f"  - 04_error_by_hour.png")
    
    return metrics


if __name__ == "__main__":
    metrics = main()