import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import json

print("=" * 70)
print("ERCOT DAM Price Forecasting - XGBoost (LEAKAGE-FREE)")
print("=" * 70)

base_dir = Path(__file__).parent.parent
data_file = base_dir / "data_processed" / "combined_dam_prices.csv"
results_dir = base_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(data_file)
print(f"   ✓ {len(df):,} total rows")

# Process datetime
print("\n2. Processing datetime...")
df['Hour Ending'] = df['Hour Ending'].str.replace('24:00', '00:00')
df['datetime'] = pd.to_datetime(df['Delivery Date'] + ' ' + df['Hour Ending'])
df.loc[df['Hour Ending'] == '00:00', 'datetime'] += pd.Timedelta(days=1)
df['price'] = pd.to_numeric(df['Settlement Point Price'], errors='coerce')

# Filter to HB_HOUSTON
print("\n3. Filtering to HB_HOUSTON hub...")
hub_data = df[df['Settlement Point'] == 'HB_HOUSTON'].copy()
hub_data = hub_data.sort_values('datetime').reset_index(drop=True)
print(f"   ✓ {len(hub_data):,} rows")
print(f"   Date range: {hub_data['datetime'].min()} to {hub_data['datetime'].max()}")

# Extract calendar features
print("\n4. Creating leakage-free features...")
hub_data['hour'] = hub_data['datetime'].dt.hour
hub_data['day_of_week'] = hub_data['datetime'].dt.dayofweek
hub_data['month'] = hub_data['datetime'].dt.month
hub_data['is_weekend'] = (hub_data['day_of_week'] >= 5).astype(int)

# CRITICAL: Only use PRIOR DAY prices (no same-day leakage)
hub_data['price_lag24'] = hub_data['price'].shift(24)
hub_data['price_lag48'] = hub_data['price'].shift(48)
hub_data['price_lag168'] = hub_data['price'].shift(168)

# Rolling stats using ONLY prior days
hub_data['price_roll_mean_168'] = hub_data['price'].shift(24).rolling(168).mean()
hub_data['price_roll_std_168'] = hub_data['price'].shift(24).rolling(168).std()

# Daily aggregates from yesterday
hub_data['price_yesterday_mean'] = hub_data['price'].shift(24).rolling(24).mean()
hub_data['price_yesterday_max'] = hub_data['price'].shift(24).rolling(24).max()
hub_data['price_yesterday_min'] = hub_data['price'].shift(24).rolling(24).min()

hub_data = hub_data.dropna()
print(f"   ✓ {len(hub_data):,} rows after feature engineering")

# Train/test split
train_size = int(len(hub_data) * 0.8)
train = hub_data[:train_size]
test = hub_data[train_size:]

print(f"\n5. Train/Test Split:")
print(f"   Train: {len(train):,} rows ({train['datetime'].min().date()} to {train['datetime'].max().date()})")
print(f"   Test:  {len(test):,} rows ({test['datetime'].min().date()} to {test['datetime'].max().date()})")

features = ['hour', 'day_of_week', 'month', 'is_weekend',
            'price_lag24', 'price_lag48', 'price_lag168',
            'price_roll_mean_168', 'price_roll_std_168',
            'price_yesterday_mean', 'price_yesterday_max', 'price_yesterday_min']

X_train, y_train = train[features], train['price']
X_test, y_test = test[features], test['price']

# Train XGBoost
print("\n6. Training XGBoost model...")
model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    objective='reg:squarederror'
)
model.fit(X_train, y_train, verbose=False)
print("   ✓ Model trained")

predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
nonzero_mask = y_test != 0
mape = np.mean(np.abs((y_test[nonzero_mask] - predictions[nonzero_mask]) / y_test[nonzero_mask])) * 100
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 70)
print("XGBOOST PERFORMANCE (LEAKAGE-FREE)")
print("=" * 70)
print(f"MAE:   ${mae:.2f}/MWh")
print(f"RMSE:  ${rmse:.2f}/MWh")
print(f"MAPE:  {mape:.2f}%")
print(f"R²:    {r2:.4f}")
print("=" * 70)

# Error by regime
test_with_preds = test.copy()
test_with_preds['prediction'] = predictions
test_with_preds['error'] = np.abs(test_with_preds['price'] - test_with_preds['prediction'])

low_price = test_with_preds[test_with_preds['price'] < 30]
mid_price = test_with_preds[(test_with_preds['price'] >= 30) & (test_with_preds['price'] < 60)]
high_price = test_with_preds[test_with_preds['price'] >= 60]

print(f"\nError by Price Regime:")
print(f"  Low (<$30):   {len(low_price):>6} samples, MAE = ${low_price['error'].mean():.2f}")
print(f"  Mid ($30-60): {len(mid_price):>6} samples, MAE = ${mid_price['error'].mean():.2f}")
print(f"  High (>$60):  {len(high_price):>6} samples, MAE = ${high_price['error'].mean():.2f}")

# Feature importance
importances = pd.DataFrame({
    'feature_name': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for _, row in importances.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature_name']:25s} {row['importance']:.4f} {bar}")

# Save results
test_with_preds[['datetime', 'price', 'prediction', 'error']].to_csv(
    results_dir / 'xgb_leakage_free_predictions.csv', index=False
)

results = {
    'model': 'XGBoost',
    'leakage_free': True,
    'mae': mae,
    'rmse': rmse,
    'mape': mape,
    'r2': r2,
    'mae_low': low_price['error'].mean(),
    'mae_mid': mid_price['error'].mean(),
    'mae_high': high_price['error'].mean(),
    'feature_importance': importances.to_dict('records')
}

with open(results_dir / 'xgb_leakage_free_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("✅ LEAKAGE-FREE XGBoost COMPLETE")
print("=" * 70)
print(f"\nResults saved to: {results_dir}")