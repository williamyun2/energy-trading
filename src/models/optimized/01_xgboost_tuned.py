"""
XGBoost Hyperparameter Tuning - ERCOT DAM Price Forecasting
Uses Optuna for automatic hyperparameter optimization

Strategy:
- Bayesian optimization (smart search, not random)
- Cross-validation for robust evaluation
- Early stopping to prevent overfitting
- Prunes bad trials early (saves time)

Goal: Beat baseline XGBoost (MAE < $5.40/MWh)
Target: MAE $5.00-5.20/MWh, R² > 0.81
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import logging

# Import feature engineering
from models.feature_engineering import engineer_all_features

# Output directory
RESULTS_DIR = Path(r"D:\Users\williamyun\proj\power_trading\results\xgboost_tuned")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging to file
log_file = RESULTS_DIR / f'optimization_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Still show some output to terminal
    ]
)
logger = logging.getLogger(__name__)

# Note: We don't redirect sys.stdout/stderr to avoid conflicts with Optuna's progress bar
# All output will be captured by the logging handlers above


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


def objective(trial, X_train, y_train, n_folds=3):
    """
    Optuna objective function
    
    This function defines the hyperparameter search space and
    evaluates each combination using cross-validation
    """
    
    # Suggest hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'seed': 42,
        'verbosity': 0,
        
        # Tree structure
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        
        # Learning parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        
        # Sampling
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        
        # Regularization
        'alpha': trial.suggest_float('alpha', 0.0, 1.0),  # L1
        'lambda': trial.suggest_float('lambda', 0.0, 2.0),  # L2
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_folds)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Train model
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        # Use early stopping for faster training
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Predict and evaluate
        y_pred = model.predict(dval)
        mae = mean_absolute_error(y_fold_val, y_pred)
        cv_scores.append(mae)
        
        # Report intermediate value for pruning
        trial.report(mae, fold)
        
        # Prune trial if it's not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return average MAE across folds
    return np.mean(cv_scores)


def evaluate_model(y_true, y_pred, model_name="XGBoost Tuned"):
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


def plot_optimization_history(study, results_dir):
    """Plot optimization progress"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Optimization history
    trials_df = study.trials_dataframe()
    
    axes[0].plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6)
    axes[0].axhline(trials_df['value'].min(), color='red', linestyle='--', 
                    linewidth=2, label=f'Best: ${trials_df["value"].min():.2f}')
    axes[0].set_xlabel('Trial Number', fontsize=12)
    axes[0].set_ylabel('Cross-Validation MAE ($/MWh)', fontsize=12)
    axes[0].set_title('Optimization History', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative best
    cummin = trials_df['value'].cummin()
    axes[1].plot(trials_df['number'], cummin, linewidth=2, color='green')
    axes[1].set_xlabel('Trial Number', fontsize=12)
    axes[1].set_ylabel('Best MAE So Far ($/MWh)', fontsize=12)
    axes[1].set_title('Cumulative Best Performance', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / '01_optimization_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_param_importances(study, results_dir):
    """Plot hyperparameter importances"""
    
    # Get parameter importances
    try:
        importances = optuna.importance.get_param_importances(study)
        
        if len(importances) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            params = list(importances.keys())
            values = list(importances.values())
            
            # Sort by importance
            sorted_idx = np.argsort(values)
            params = [params[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            
            ax.barh(range(len(params)), values, color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels(params)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(results_dir / '02_param_importances.png', dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"  Note: Could not plot parameter importances ({e})")


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
    ax.set_title('XGBoost Tuned: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / '03_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Last 30 days
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
    plt.savefig(results_dir / '04_last_30_days.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main optimization and evaluation pipeline"""
    
    print("="*80)
    print("XGBOOST HYPERPARAMETER TUNING - ERCOT DAM PRICE FORECASTING")
    print("="*80)
    print("Using Optuna for Bayesian optimization")
    print("Goal: Beat baseline XGBoost (MAE < $5.40/MWh)")
    print("Target: MAE $5.00-5.20/MWh, R² > 0.81")
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
    print("\nSplitting train/test...")
    df_train, df_test, split_date = train_test_split(df_features, test_months=6)
    print(f"✓ Train: {len(df_train):,} | Test: {len(df_test):,}")
    
    # 6. Prepare matrices
    X_train = df_train[feature_cols].values
    y_train = df_train['dam_price'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['dam_price'].values
    
    # 7. Create Optuna study
    print("\n" + "="*80)
    print("RUNNING HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print("This will take 1-3 hours depending on your hardware...")
    print()
    
    # Pruner stops unpromising trials early
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    study = optuna.create_study(
        direction='minimize',  # Minimize MAE
        pruner=pruner,
        study_name='xgboost_ercot_tuning'
    )
    
    # Run optimization
    n_trials = 100  # Try 100 different parameter combinations
    
    print(f"Starting optimization with {n_trials} trials...")
    print(f"Progress updates every 10 trials:\n")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, n_folds=3),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Use 1 core (change to -1 for all cores, but may be unstable)
    )
    
    print("\n✓ Optimization complete!")
    
    # 8. Get best parameters
    best_params = study.best_params
    best_cv_mae = study.best_value
    
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*80)
    print(f"Cross-Validation MAE: ${best_cv_mae:.2f}/MWh\n")
    
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param:20s}: {value}")
    
    # 9. Train final model with best parameters
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL")
    print("="*80)
    
    final_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'seed': 42,
        **best_params
    }
    
    # Remove n_estimators from params (use it as num_boost_round)
    n_estimators = final_params.pop('n_estimators')
    
    # Create validation set from training data
    val_split = df_train['datetime'].max() - pd.DateOffset(months=2)
    df_train_only = df_train[df_train['datetime'] < val_split].copy()
    df_val = df_train[df_train['datetime'] >= val_split].copy()
    
    X_train_only = df_train_only[feature_cols].values
    y_train_only = df_train_only['dam_price'].values
    X_val = df_val[feature_cols].values
    y_val = df_val['dam_price'].values
    
    # Train
    dtrain = xgb.DMatrix(X_train_only, label=y_train_only)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    print("Training final model with best parameters...")
    final_model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    print(f"\n✓ Final model trained (best iteration: {final_model.best_iteration})")
    
    # 10. Predict on test set
    print("\nGenerating test predictions...")
    dtest = xgb.DMatrix(X_test)
    y_pred = final_model.predict(dtest)
    print(f"✓ Generated {len(y_pred):,} predictions")
    
    # 11. Evaluate
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE")
    print("="*80)
    
    metrics_tuned = evaluate_model(y_test, y_pred, "XGBoost Tuned")
    
    # Compare to baseline
    baseline_mae = 10.97
    xgb_baseline_mae = 5.40
    improvement_baseline = (baseline_mae - metrics_tuned['MAE']) / baseline_mae * 100
    improvement_xgb = (xgb_baseline_mae - metrics_tuned['MAE']) / xgb_baseline_mae * 100
    
    print(f"Model: {metrics_tuned['Model']}")
    print(f"MAE:          ${metrics_tuned['MAE']:.2f}/MWh")
    print(f"RMSE:         ${metrics_tuned['RMSE']:.2f}/MWh")
    print(f"MAPE:         {metrics_tuned['MAPE']:.2f}%")
    print(f"R² Score:     {metrics_tuned['R2']:.4f}")
    print(f"Max Error:    ${metrics_tuned['Max Error']:.2f}/MWh")
    print(f"Median Error: ${metrics_tuned['Median Error']:.2f}/MWh")
    
    print("\n" + "="*80)
    print("COMPARISON TO BASELINES")
    print("="*80)
    print(f"Persistence Baseline:    ${baseline_mae:.2f}/MWh")
    print(f"XGBoost (untuned):       ${xgb_baseline_mae:.2f}/MWh")
    print(f"XGBoost Tuned:           ${metrics_tuned['MAE']:.2f}/MWh")
    print()
    print(f"Improvement over Persistence: {improvement_baseline:+.1f}%")
    print(f"Improvement over XGBoost:     {improvement_xgb:+.1f}%")
    
    if metrics_tuned['MAE'] < xgb_baseline_mae:
        print(f"\n✅ SUCCESS! Tuning improved by ${xgb_baseline_mae - metrics_tuned['MAE']:.2f}/MWh")
    else:
        print(f"\n⚠️  Tuning did not improve performance")
        print(f"    Baseline XGBoost may already be well-configured")
    
    # 12. Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_optimization_history(study, RESULTS_DIR)
    plot_param_importances(study, RESULTS_DIR)
    plot_predictions(df_test, y_pred, RESULTS_DIR)
    print(f"✓ Saved plots to: {RESULTS_DIR}")
    
    # 13. Save results
    print("\nSaving results...")
    
    # Metrics
    metrics_df = pd.DataFrame([metrics_tuned])
    metrics_df.to_csv(RESULTS_DIR / 'tuned_metrics.csv', index=False)
    
    # Predictions
    predictions_df = df_test[['datetime', 'dam_price']].copy()
    predictions_df['predicted'] = y_pred
    predictions_df['error'] = predictions_df['dam_price'] - predictions_df['predicted']
    predictions_df.to_csv(RESULTS_DIR / 'tuned_predictions.csv', index=False)
    
    # Best parameters
    with open(RESULTS_DIR / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Study trials (full optimization history)
    study_df = study.trials_dataframe()
    study_df.to_csv(RESULTS_DIR / 'optimization_trials.csv', index=False)
    
    # Save model
    final_model.save_model(str(RESULTS_DIR / 'xgboost_tuned_model.json'))
    
    print(f"✓ Saved results to: {RESULTS_DIR}")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
    print(f"\nFinal Performance:")
    print(f"  MAE:  ${metrics_tuned['MAE']:.2f}/MWh")
    print(f"  R²:   {metrics_tuned['R2']:.4f}")
    print(f"\nBest parameters saved to: {RESULTS_DIR / 'best_params.json'}")
    print(f"Full log saved to: {log_file}")
    print(f"Use these parameters for future models!")
    
    return metrics_tuned, best_params, study


if __name__ == "__main__":
    metrics, params, study = main()