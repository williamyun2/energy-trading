"""
Ensemble Model - ERCOT DAM Price Forecasting
Combines Linear Regression + XGBoost for optimal performance

Strategy: Weighted average based on validation performance
- Linear Regression: Good at stable periods
- XGBoost: Good at volatility and extreme events

Goal: Beat XGBoost alone (MAE < $5.40/MWh)
Target: MAE $4.80-5.20/MWh, R² > 0.82
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project paths - robust path resolution
project_root = Path(__file__).resolve().parents[3]  # .../power_trading
src_dir = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from merge_dataset.loader import load_clean_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Import feature engineering
from models.feature_engineering import engineer_all_features

# Output directory
RESULTS_DIR = Path(r"D:\Users\williamyun\proj\power_trading\results\ensemble")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def select_features(df):
    """Select features for modeling"""
    
    exclude_cols = ['datetime', 'dam_price']
    
    # Exclude original raw weather columns
    raw_weather = [c for c in df.columns if any(x in c for x in [
        'temp_f_HOUSTON', 'temp_f_NORTH', 'temp_f_SOUTH', 'temp_f_WEST',
        'wind_speed_mph_HOUSTON', 'wind_speed_mph_NORTH', 'wind_speed_mph_SOUTH', 'wind_speed_mph_WEST',
        'solar_radiation_wm2_HOUSTON', 'solar_radiation_wm2_NORTH', 'solar_radiation_wm2_SOUTH', 'solar_radiation_wm2_WEST',
        'relative_humidity_HOUSTON', 'relative_humidity_NORTH', 'relative_humidity_SOUTH', 'relative_humidity_WEST'
    ])]
    
    exclude_cols.extend(raw_weather)
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
    """Train Linear Regression with scaling"""
    
    if verbose:
        print("\n1️⃣ Training Linear Regression...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    if verbose:
        print(f"   ✓ Linear Regression trained")
    
    return model, scaler


def train_xgboost(X_train, y_train, X_val, y_val, verbose=True):
    """Train XGBoost with early stopping"""
    
    if verbose:
        print("\n2️⃣ Training XGBoost...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'alpha': 0.1,
        'lambda': 1.0,
        'eval_metric': 'mae',
        'seed': 42
    }
    
    evals = [(dtrain, 'train'), (dval, 'validation')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    if verbose:
        print(f"   ✓ XGBoost trained (best iteration: {model.best_iteration})")
    
    return model


def optimize_ensemble_weights(y_true, pred_lr, pred_xgb, verbose=True):
    """
    Find optimal weights for ensemble using grid search
    
    Returns: (weight_lr, weight_xgb) where weight_lr + weight_xgb = 1.0
    """
    
    if verbose:
        print("\n3️⃣ Optimizing ensemble weights...")
    
    best_mae = float('inf')
    best_weights = (0.5, 0.5)
    
    # Try different weight combinations
    for w_lr in np.arange(0.0, 1.01, 0.05):
        w_xgb = 1.0 - w_lr
        
        # Weighted average
        pred_ensemble = w_lr * pred_lr + w_xgb * pred_xgb
        mae = mean_absolute_error(y_true, pred_ensemble)
        
        if mae < best_mae:
            best_mae = mae
            best_weights = (w_lr, w_xgb)
    
    if verbose:
        print(f"   ✓ Optimal weights:")
        print(f"      Linear Regression: {best_weights[0]:.2f} ({best_weights[0]*100:.0f}%)")
        print(f"      XGBoost:          {best_weights[1]:.2f} ({best_weights[1]*100:.0f}%)")
        print(f"      Validation MAE:   ${best_mae:.2f}/MWh")
    
    return best_weights


def evaluate_model(y_true, y_pred, model_name="Model"):
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


def plot_model_comparison(df_test, pred_lr, pred_xgb, pred_ensemble, results_dir):
    """Compare all models side by side"""
    
    df_plot = df_test.copy()
    df_plot['pred_lr'] = pred_lr
    df_plot['pred_xgb'] = pred_xgb
    df_plot['pred_ensemble'] = pred_ensemble
    
    # Last 30 days comparison
    df_recent = df_plot.tail(30 * 24)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Linear Regression
    axes[0].plot(df_recent['datetime'], df_recent['dam_price'], 
                 label='Actual', linewidth=2, alpha=0.8, color='blue')
    axes[0].plot(df_recent['datetime'], df_recent['pred_lr'], 
                 label='Linear Regression', linewidth=2, alpha=0.8, color='orange')
    axes[0].fill_between(df_recent['datetime'], df_recent['dam_price'], df_recent['pred_lr'], 
                         alpha=0.3)
    axes[0].set_ylabel('Price ($/MWh)', fontsize=11)
    axes[0].set_title('Linear Regression: Last 30 Days', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # XGBoost
    axes[1].plot(df_recent['datetime'], df_recent['dam_price'], 
                 label='Actual', linewidth=2, alpha=0.8, color='blue')
    axes[1].plot(df_recent['datetime'], df_recent['pred_xgb'], 
                 label='XGBoost', linewidth=2, alpha=0.8, color='green')
    axes[1].fill_between(df_recent['datetime'], df_recent['dam_price'], df_recent['pred_xgb'], 
                         alpha=0.3, color='green')
    axes[1].set_ylabel('Price ($/MWh)', fontsize=11)
    axes[1].set_title('XGBoost: Last 30 Days', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Ensemble
    axes[2].plot(df_recent['datetime'], df_recent['dam_price'], 
                 label='Actual', linewidth=2, alpha=0.8, color='blue')
    axes[2].plot(df_recent['datetime'], df_recent['pred_ensemble'], 
                 label='Ensemble', linewidth=2, alpha=0.8, color='red')
    axes[2].fill_between(df_recent['datetime'], df_recent['dam_price'], df_recent['pred_ensemble'], 
                         alpha=0.3, color='red')
    axes[2].set_xlabel('Date', fontsize=11)
    axes[2].set_ylabel('Price ($/MWh)', fontsize=11)
    axes[2].set_title('Ensemble (Combined): Last 30 Days', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(results_dir / '01_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions(df_test, y_pred, results_dir):
    """Create diagnostic plots for ensemble"""
    
    df_plot = df_test.copy()
    df_plot['predicted'] = y_pred
    df_plot['error'] = df_plot['dam_price'] - df_plot['predicted']
    df_plot['abs_error'] = np.abs(df_plot['error'])
    
    sns.set_style("whitegrid")
    
    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df_plot['dam_price'], df_plot['predicted'], alpha=0.3, s=10)
    
    min_val = min(df_plot['dam_price'].min(), df_plot['predicted'].min())
    max_val = max(df_plot['dam_price'].max(), df_plot['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price ($/MWh)', fontsize=12)
    ax.set_ylabel('Predicted Price ($/MWh)', fontsize=12)
    ax.set_title('Ensemble: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '02_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_plot['error'], bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error ($/MWh)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Ensemble: Error Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '03_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_performance_comparison(metrics_all, results_dir):
    """Bar chart comparing all models"""
    
    models = [m['Model'] for m in metrics_all]
    mae_values = [m['MAE'] for m in metrics_all]
    r2_values = [m['R2'] for m in metrics_all]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE comparison
    colors = ['gray', 'orange', 'red', 'green', 'purple']
    axes[0].bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('MAE ($/MWh)', fontsize=12)
    axes[0].set_title('Mean Absolute Error Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 0.2, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # R² comparison
    axes[1].bar(models, r2_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('R² Score Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.0])
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(r2_values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / '04_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main ensemble training and evaluation pipeline"""
    
    print("="*80)
    print("ENSEMBLE MODEL - ERCOT DAM PRICE FORECASTING")
    print("="*80)
    print("Strategy: Combine Linear Regression + XGBoost")
    print("Goal: Beat XGBoost alone (MAE < $5.40/MWh)")
    print("Target: MAE $4.80-5.20/MWh, R² > 0.82")
    print()
    
    # 1. Load data
    print("Loading clean data...")
    df = load_clean_data(verbose=False)
    print(f"✓ Loaded {len(df):,} records")
    
    # 2. Engineer features
    print("Engineering features...")
    df_features = engineer_all_features(df, verbose=False)
    print(f"✓ Created {df_features.shape[1] - 20} new features")
    
    # 3. Select features
    feature_cols = select_features(df_features)
    print(f"✓ Selected {len(feature_cols)} features")
    
    # 4. Handle missing values
    print("Handling missing values...")
    df_features = df_features.dropna()
    print(f"✓ {len(df_features):,} rows ready")
    
    # 5. Train/test split
    print("\nSplitting data...")
    df_train, df_test, split_date = train_test_split(df_features, test_months=6)
    
    # Create validation set
    val_split = df_train['datetime'].max() - pd.DateOffset(months=2)
    df_train_only = df_train[df_train['datetime'] < val_split].copy()
    df_val = df_train[df_train['datetime'] >= val_split].copy()
    
    print(f"✓ Train: {len(df_train_only):,} | Validation: {len(df_val):,} | Test: {len(df_test):,}")
    
    # 6. Prepare matrices
    X_train = df_train_only[feature_cols].values
    y_train = df_train_only['dam_price'].values
    X_val = df_val[feature_cols].values
    y_val = df_val['dam_price'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['dam_price'].values
    
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE MODELS")
    print("="*80)
    
    # 7. Train Linear Regression
    lr_model, lr_scaler = train_linear_regression(X_train, y_train)
    
    # 8. Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # 9. Get validation predictions
    print("\n" + "="*80)
    print("OPTIMIZING ENSEMBLE")
    print("="*80)
    
    # Linear Regression validation predictions
    X_val_scaled = lr_scaler.transform(X_val)
    val_pred_lr = lr_model.predict(X_val_scaled)
    
    # XGBoost validation predictions
    dval = xgb.DMatrix(X_val)
    val_pred_xgb = xgb_model.predict(dval)
    
    # 10. Optimize weights
    weights = optimize_ensemble_weights(y_val, val_pred_lr, val_pred_xgb)
    
    # 11. Test set predictions
    print("\n" + "="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)
    
    # Linear Regression
    X_test_scaled = lr_scaler.transform(X_test)
    test_pred_lr = lr_model.predict(X_test_scaled)
    
    # XGBoost
    dtest = xgb.DMatrix(X_test)
    test_pred_xgb = xgb_model.predict(dtest)
    
    # Ensemble (weighted average)
    test_pred_ensemble = weights[0] * test_pred_lr + weights[1] * test_pred_xgb
    
    print(f"✓ Generated {len(test_pred_ensemble):,} predictions for each model")
    
    # 12. Evaluate all models
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Baseline metrics (for comparison)
    metrics_persistence = {'Model': 'Persistence', 'MAE': 10.97, 'RMSE': 23.03, 
                          'MAPE': 63.22, 'R2': 0.0899, 'Max Error': 428.93, 'Median Error': 5.84}
    metrics_lstm = {'Model': 'LSTM', 'MAE': 11.87, 'RMSE': 22.86, 
                   'MAPE': 95.35, 'R2': 0.1301, 'Max Error': 422.73, 'Median Error': 6.53}
    
    # Calculate metrics for new models
    metrics_lr = evaluate_model(y_test, test_pred_lr, "Linear Regression")
    metrics_xgb = evaluate_model(y_test, test_pred_xgb, "XGBoost")
    metrics_ensemble = evaluate_model(y_test, test_pred_ensemble, "Ensemble")
    
    # Display results
    all_metrics = [metrics_persistence, metrics_lstm, metrics_lr, metrics_xgb, metrics_ensemble]
    
    print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Max Err':>10}")
    print("-" * 70)
    for m in all_metrics:
        print(f"{m['Model']:<20} ${m['MAE']:>9.2f} ${m['RMSE']:>9.2f} {m['R2']:>10.4f} ${m['Max Error']:>9.2f}")
    
    # Compare to XGBoost
    improvement = (metrics_xgb['MAE'] - metrics_ensemble['MAE']) / metrics_xgb['MAE'] * 100
    
    print("\n" + "="*80)
    print("ENSEMBLE vs XGBoost")
    print("="*80)
    print(f"XGBoost MAE:    ${metrics_xgb['MAE']:.2f}/MWh")
    print(f"Ensemble MAE:   ${metrics_ensemble['MAE']:.2f}/MWh")
    print(f"Improvement:    {improvement:+.1f}%")
    
    if metrics_ensemble['MAE'] < metrics_xgb['MAE']:
        print(f"\n✅ SUCCESS! Ensemble beat XGBoost by ${metrics_xgb['MAE'] - metrics_ensemble['MAE']:.2f}/MWh")
    else:
        print(f"\n⚠️  Ensemble did not beat XGBoost (worse by ${metrics_ensemble['MAE'] - metrics_xgb['MAE']:.2f}/MWh)")
        print("    XGBoost alone is already excellent - ensemble may not add value")
    
    # 13. Create plots
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_model_comparison(df_test, test_pred_lr, test_pred_xgb, test_pred_ensemble, RESULTS_DIR)
    plot_predictions(df_test, test_pred_ensemble, RESULTS_DIR)
    plot_performance_comparison(all_metrics, RESULTS_DIR)
    print(f"✓ Saved plots to: {RESULTS_DIR}")
    
    # 14. Save results
    print("\nSaving results...")
    
    # Predictions
    predictions_df = df_test[['datetime', 'dam_price']].copy()
    predictions_df['pred_lr'] = test_pred_lr
    predictions_df['pred_xgb'] = test_pred_xgb
    predictions_df['pred_ensemble'] = test_pred_ensemble
    predictions_df['error_ensemble'] = predictions_df['dam_price'] - predictions_df['pred_ensemble']
    predictions_df.to_csv(RESULTS_DIR / "ensemble_predictions.csv", index=False)
    
    # Metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "ensemble_metrics.csv", index=False)
    
    # Ensemble config
    config = {
        'weight_linear_regression': float(weights[0]),
        'weight_xgboost': float(weights[1]),
        'validation_mae': float(mean_absolute_error(y_val, weights[0] * val_pred_lr + weights[1] * val_pred_xgb))
    }
    with open(RESULTS_DIR / 'ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Saved results to: {RESULTS_DIR}")
    
    print("\n" + "="*80)
    print("ENSEMBLE MODEL COMPLETE!")
    print("="*80)
    print(f"\nFinal Performance:")
    print(f"  MAE:  ${metrics_ensemble['MAE']:.2f}/MWh")
    print(f"  R²:   {metrics_ensemble['R2']:.4f}")
    print(f"\nImprovement over Persistence: {(10.97 - metrics_ensemble['MAE']) / 10.97 * 100:+.1f}%")
    
    return metrics_ensemble, weights


if __name__ == "__main__":
    metrics, weights = main()