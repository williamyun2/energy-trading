"""
XGBoost Model - ERCOT DAM Price Forecasting
ML Model #2: Gradient Boosting for Tabular Data

Optimized for feature-driven prediction with:
- Non-linear relationships
- Feature interactions
- True feature importance (gain-based)
- Handles outliers well

Goal: Beat Linear Regression (MAE < $9.27/MWh)
Target: MAE $6-7/MWh, R² > 0.75
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).resolve().parents[3]  # .../power_trading
src_dir = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))
from merge_dataset.loader import load_clean_data
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
RESULTS_DIR = Path(r"D:\Users\williamyun\proj\power_trading\results\xgboost")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def select_features(df):
    """Select features for modeling"""
    
    exclude_cols = ['datetime', 'dam_price']
    
    # Exclude original raw weather columns (use derived features)
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


def train_xgboost(X_train, y_train, X_val, y_val, verbose=True):
    """Train XGBoost model with early stopping"""
    
    if verbose:
        print("\nTraining XGBoost...")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Validation samples: {X_val.shape[0]:,}")
    
    # Create DMatrix for XGBoost (optimized data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,                    # Deep trees for complex interactions
        'learning_rate': 0.05,             # Conservative learning rate
        'subsample': 0.8,                  # Use 80% of data per tree
        'colsample_bytree': 0.8,           # Use 80% of features per tree
        'min_child_weight': 3,             # Minimum samples in leaf
        'gamma': 0.1,                      # Minimum loss reduction for split
        'alpha': 0.1,                      # L1 regularization
        'lambda': 1.0,                     # L2 regularization
        'eval_metric': 'mae',              # Optimize for MAE
        'seed': 42
    }
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'validation')]
    
    if verbose:
        print(f"\n  Hyperparameters:")
        for key, value in params.items():
            print(f"    {key}: {value}")
        print()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,              # Max iterations
        evals=evals,
        early_stopping_rounds=50,          # Stop if no improvement for 50 rounds
        verbose_eval=50 if verbose else False
    )
    
    if verbose:
        print(f"\n  ✓ Model trained")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best validation MAE: ${model.best_score:.2f}/MWh")
    
    return model


def evaluate_model(y_true, y_pred, model_name="XGBoost"):
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


def plot_feature_importance(model, feature_names, results_dir):
    """Plot feature importance using multiple metrics"""
    
    # Get importance scores
    importance_gain = model.get_score(importance_type='gain')
    importance_weight = model.get_score(importance_type='weight')
    importance_cover = model.get_score(importance_type='cover')
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gain': [importance_gain.get(f'f{i}', 0) for i in range(len(feature_names))],
        'weight': [importance_weight.get(f'f{i}', 0) for i in range(len(feature_names))],
        'cover': [importance_cover.get(f'f{i}', 0) for i in range(len(feature_names))]
    })
    
    # Normalize to percentages
    importance_df['gain_pct'] = 100 * importance_df['gain'] / importance_df['gain'].sum()
    importance_df['weight_pct'] = 100 * importance_df['weight'] / importance_df['weight'].sum()
    importance_df['cover_pct'] = 100 * importance_df['cover'] / importance_df['cover'].sum()
    
    # Sort by gain (most important metric)
    importance_df = importance_df.sort_values('gain_pct', ascending=False)
    
    # Save to CSV
    importance_df.to_csv(results_dir / 'feature_importance.csv', index=False)
    
    # Plot top 20 by gain
    top_20 = importance_df.head(20)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Gain
    axes[0].barh(range(len(top_20)), top_20['gain_pct'], color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(len(top_20)))
    axes[0].set_yticklabels(top_20['feature'])
    axes[0].set_xlabel('Gain (%)', fontsize=12)
    axes[0].set_title('Feature Importance by Gain\n(Information Gain)', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Weight
    top_20_weight = importance_df.sort_values('weight_pct', ascending=False).head(20)
    axes[1].barh(range(len(top_20_weight)), top_20_weight['weight_pct'], color='coral', alpha=0.8)
    axes[1].set_yticks(range(len(top_20_weight)))
    axes[1].set_yticklabels(top_20_weight['feature'])
    axes[1].set_xlabel('Weight (%)', fontsize=12)
    axes[1].set_title('Feature Importance by Weight\n(# of Times Used)', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    # Cover
    top_20_cover = importance_df.sort_values('cover_pct', ascending=False).head(20)
    axes[2].barh(range(len(top_20_cover)), top_20_cover['cover_pct'], color='mediumseagreen', alpha=0.8)
    axes[2].set_yticks(range(len(top_20_cover)))
    axes[2].set_yticklabels(top_20_cover['feature'])
    axes[2].set_xlabel('Cover (%)', fontsize=12)
    axes[2].set_title('Feature Importance by Cover\n(# of Samples)', fontweight='bold', fontsize=14)
    axes[2].grid(True, alpha=0.3, axis='x')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(results_dir / '05_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print top 10 by gain
    print("\n" + "="*80)
    print("TOP 10 MOST IMPORTANT FEATURES (by Gain)")
    print("="*80)
    print("Gain = How much this feature improves predictions (TRUE importance)")
    print()
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:40s}: {row['gain_pct']:6.2f}% gain")
    
    return importance_df


def plot_training_history(evals_result, results_dir):
    """Plot training history"""
    
    train_mae = evals_result['train']['mae']
    val_mae = evals_result['validation']['mae']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(len(train_mae))
    
    ax.plot(epochs, train_mae, label='Training MAE', linewidth=2, alpha=0.8)
    ax.plot(epochs, val_mae, label='Validation MAE', linewidth=2, alpha=0.8)
    ax.axvline(np.argmin(val_mae), color='red', linestyle='--', 
               linewidth=2, label=f'Best Iteration ({np.argmin(val_mae)})')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MAE ($/MWh)', fontsize=12)
    ax.set_title('XGBoost Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions(df_test, y_pred, results_dir):
    """Create diagnostic plots"""
    
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
    ax.set_title('XGBoost: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '01_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(df_plot['error'], bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Prediction Error ($/MWh)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df_plot['datetime'], df_plot['error'], alpha=0.5, linewidth=0.5)
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Prediction Error ($/MWh)')
    axes[0, 1].set_title('Errors Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 0].plot(df_plot['datetime'], df_plot['abs_error'], alpha=0.5, linewidth=0.5, color='orange')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Absolute Error ($/MWh)')
    axes[1, 0].set_title('Absolute Errors Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].scatter(df_plot['dam_price'], df_plot['abs_error'], alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Actual Price ($/MWh)')
    axes[1, 1].set_ylabel('Absolute Error ($/MWh)')
    axes[1, 1].set_title('Error vs Actual Price')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / '02_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Last 30 days
    df_recent = df_plot.tail(30 * 24)
    
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
    hours = hourly_errors.index
    ax.errorbar(hours, hourly_errors['error']['mean'], yerr=hourly_errors['error']['std'],
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Mean Error ($/MWh)', fontsize=12)
    ax.set_title('Prediction Error by Hour of Day', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '04_error_by_hour.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training and evaluation pipeline"""
    
    print("="*80)
    print("XGBOOST MODEL - ERCOT DAM PRICE FORECASTING")
    print("="*80)
    print("Goal: Beat Linear Regression (MAE < $9.27/MWh)")
    print("Target: MAE $6-7/MWh, R² > 0.75")
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
    
    # 6. Create validation set (last 2 months of training data)
    val_split = df_train['datetime'].max() - pd.DateOffset(months=2)
    df_train_only = df_train[df_train['datetime'] < val_split].copy()
    df_val = df_train[df_train['datetime'] >= val_split].copy()
    
    print(f"\nCreating validation set from training data...")
    print(f"  Train only: {len(df_train_only):,} records")
    print(f"  Validation: {len(df_val):,} records")
    
    # 7. Prepare feature matrices (NO SCALING for XGBoost!)
    print("\nPreparing feature matrices...")
    X_train = df_train_only[feature_cols].values
    y_train = df_train_only['dam_price'].values
    X_val = df_val[feature_cols].values
    y_val = df_val['dam_price'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['dam_price'].values
    print(f"✓ XGBoost doesn't require feature scaling (tree-based model)")
    
    # 8. Train model
    model = train_xgboost(X_train, y_train, X_val, y_val, verbose=True)
    
    # 9. Predict on test set
    print("\nPredicting on test set...")
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    print(f"✓ Generated {len(y_pred):,} predictions")
    
    # 10. Evaluate
    print("\nEvaluating model performance...")
    metrics = evaluate_model(y_test, y_pred, model_name='XGBoost')
    
    # Compare to baselines
    baseline_mae = 10.97
    linear_mae = 9.27
    lstm_mae = 11.87
    improvement_baseline = (baseline_mae - metrics['MAE']) / baseline_mae * 100
    improvement_linear = (linear_mae - metrics['MAE']) / linear_mae * 100
    
    print("\n" + "="*80)
    print("XGBOOST RESULTS")
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
    print("COMPARISON TO BASELINES")
    print("="*80)
    print(f"Persistence Baseline MAE:    ${baseline_mae:.2f}/MWh")
    print(f"Linear Regression MAE:       ${linear_mae:.2f}/MWh")
    print(f"LSTM MAE:                    ${lstm_mae:.2f}/MWh")
    print(f"XGBoost MAE:                 ${metrics['MAE']:.2f}/MWh")
    print()
    print(f"Improvement over Persistence: {improvement_baseline:+.1f}%")
    print(f"Improvement over Linear Reg:  {improvement_linear:+.1f}%")
    
    if metrics['MAE'] < linear_mae:
        print(f"\n✅ SUCCESS! Beat Linear Regression by ${linear_mae - metrics['MAE']:.2f}/MWh")
    else:
        print(f"\n⚠️  Did not beat Linear Regression (worse by ${metrics['MAE'] - linear_mae:.2f}/MWh)")
    
    # 11. Feature importance analysis
    print("\nAnalyzing feature importance...")
    importance_df = plot_feature_importance(model, feature_cols, RESULTS_DIR)
    
    # 12. Plot training history
    print("\nCreating training history plot...")
    evals_result = model.attributes()
    # Note: XGBoost doesn't store full history in the model, so we'll skip detailed history plot
    
    # 13. Create diagnostic plots
    print("\nCreating diagnostic plots...")
    plot_predictions(df_test, y_pred, RESULTS_DIR)
    print(f"✓ Saved plots to: {RESULTS_DIR}")
    
    # 14. Save results
    print("\nSaving results...")
    
    # Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(RESULTS_DIR / "xgboost_metrics.csv", index=False)
    
    # Predictions
    predictions_df = df_test[['datetime', 'dam_price']].copy()
    predictions_df['predicted'] = y_pred
    predictions_df['error'] = predictions_df['dam_price'] - predictions_df['predicted']
    predictions_df.to_csv(RESULTS_DIR / "xgboost_predictions.csv", index=False)
    
    # Model config
    config = {
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': model.best_iteration,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror'
    }
    with open(RESULTS_DIR / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save model
    model.save_model(str(RESULTS_DIR / 'xgboost_model.json'))
    
    print(f"✓ Saved metrics to: {RESULTS_DIR / 'xgboost_metrics.csv'}")
    print(f"✓ Saved predictions to: {RESULTS_DIR / 'xgboost_predictions.csv'}")
    print(f"✓ Saved feature importance to: {RESULTS_DIR / 'feature_importance.csv'}")
    print(f"✓ Saved model to: {RESULTS_DIR / 'xgboost_model.json'}")
    
    print("\n" + "="*80)
    print("XGBOOST MODEL COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - xgboost_metrics.csv")
    print(f"  - xgboost_predictions.csv")
    print(f"  - feature_importance.csv")
    print(f"  - model_config.json")
    print(f"  - xgboost_model.json")
    print(f"  - 01_actual_vs_predicted.png")
    print(f"  - 02_error_analysis.png")
    print(f"  - 03_last_30_days.png")
    print(f"  - 04_error_by_hour.png")
    print(f"  - 05_feature_importance.png")
    
    return metrics, model, importance_df


if __name__ == "__main__":
    metrics, model, importance_df = main()