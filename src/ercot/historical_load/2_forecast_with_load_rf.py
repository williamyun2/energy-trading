import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

print("=" * 70)
print("ERCOT DAM Forecasting - RF with COAST Load (LEAKAGE-FREE)")
print("=" * 70)

base_dir = Path(__file__).parent.parent
price_file = base_dir / "data_processed" / "combined_dam_prices.csv"
load_file = base_dir / "data_processed" / "system_load.csv"
results_dir = base_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Price data
print("\n1. Loading price data...")
df = pd.read_csv(price_file)
print(f"   ✓ {len(df):,} price records")

print("\n2. Processing price datetime...")
df["Hour Ending"] = df["Hour Ending"].str.replace("24:00", "00:00")
df["datetime"] = pd.to_datetime(df["Delivery Date"] + " " + df["Hour Ending"])
df.loc[df["Hour Ending"] == "00:00", "datetime"] += pd.Timedelta(days=1)
df["price"] = pd.to_numeric(df["Settlement Point Price"], errors="coerce")

print("\n3. Filtering to HB_HOUSTON...")
hub_data = df[df["Settlement Point"] == "HB_HOUSTON"].copy()
hub_data = hub_data.sort_values("datetime").reset_index(drop=True)
print(f"   ✓ {len(hub_data):,} rows")

# Load data
print("\n4. Loading COAST load data (Houston proxy)...")
load_data = pd.read_csv(load_file)
load_data["datetime"] = pd.to_datetime(load_data["datetime"])
load_data["houston_load_mw"] = load_data["COAST"]
print(f"   ✓ {len(load_data):,} load records")

# Merge
print("\n5. Merging price and load data...")
merged = hub_data.merge(load_data, on="datetime", how="left")
print(f"   ✓ {len(merged):,} rows after merge")
print(f"   Missing load values: {merged['houston_load_mw'].isna().sum()}")

merged = merged.dropna(subset=["houston_load_mw"])
print(f"   ✓ {len(merged):,} rows after dropping NAs")

# Features - LEAKAGE-FREE
print("\n6. Creating leakage-free features...")
merged["hour"] = merged["datetime"].dt.hour
merged["day_of_week"] = merged["datetime"].dt.dayofweek
merged["month"] = merged["datetime"].dt.month
merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)

# Price: only prior days (24h+)
merged["price_lag24"] = merged["price"].shift(24)
merged["price_lag48"] = merged["price"].shift(48)
merged["price_lag168"] = merged["price"].shift(168)
merged["price_roll_mean_168"] = merged["price"].shift(24).rolling(168).mean()
merged["price_roll_std_168"] = merged["price"].shift(24).rolling(168).std()
merged["price_yesterday_mean"] = merged["price"].shift(24).rolling(24).mean()
merged["price_yesterday_max"] = merged["price"].shift(24).rolling(24).max()
merged["price_yesterday_min"] = merged["price"].shift(24).rolling(24).min()

# Load: only prior days
merged["load_lag24"] = merged["houston_load_mw"].shift(24)
merged["load_roll_mean_168"] = merged["houston_load_mw"].shift(24).rolling(168).mean()
merged["load_roll_std_168"] = merged["houston_load_mw"].shift(24).rolling(168).std()
merged["load_yesterday_mean"] = merged["houston_load_mw"].shift(24).rolling(24).mean()

merged = merged.dropna()
print(f"   ✓ {len(merged):,} rows after feature engineering")

# Train/test
train_size = int(len(merged) * 0.8)
train = merged[:train_size]
test = merged[train_size:]

print(f"\n7. Train/Test Split:")
print(f"   Train: {len(train):,} rows")
print(f"   Test:  {len(test):,} rows")

features = [
    "hour", "day_of_week", "month", "is_weekend",
    "price_lag24", "price_lag48", "price_lag168",
    "price_roll_mean_168", "price_roll_std_168",
    "price_yesterday_mean", "price_yesterday_max", "price_yesterday_min",
    # "houston_load_mw", 
    "load_lag24", "load_roll_mean_168", "load_roll_std_168",
    "load_yesterday_mean"
]

X_train = train[features]
X_test = test[features]
y_train = train["price"]
y_test = test["price"]

# Train
print("\n8. Training RF model...")
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("   ✓ Model trained")

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
nonzero_mask = y_test != 0
mape = np.mean(np.abs((y_test[nonzero_mask] - predictions[nonzero_mask]) / y_test[nonzero_mask])) * 100
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 70)
print("RF PERFORMANCE (LEAKAGE-FREE, WITH LOAD)")
print("=" * 70)
print(f"MAE:   ${mae:.2f}/MWh")
print(f"RMSE:  ${rmse:.2f}/MWh")
print(f"MAPE:  {mape:.2f}%")
print(f"R²:    {r2:.4f}")
print("=" * 70)

# Error by regime
test_with_preds = test.copy()
test_with_preds["prediction"] = predictions
test_with_preds["error"] = np.abs(test_with_preds["price"] - test_with_preds["prediction"])

low = test_with_preds[test_with_preds["price"] < 30]
mid = test_with_preds[(test_with_preds["price"] >= 30) & (test_with_preds["price"] < 60)]
high = test_with_preds[test_with_preds["price"] >= 60]

print(f"\nError by Price Regime:")
print(f"  Low (<$30):   {len(low):>6} samples, MAE = ${low['error'].mean():.2f}")
print(f"  Mid ($30-60): {len(mid):>6} samples, MAE = ${mid['error'].mean():.2f}")
print(f"  High (>$60):  {len(high):>6} samples, MAE = ${high['error'].mean():.2f}")

# Feature importance
importances = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
importances = importances.sort_values("importance", ascending=False)

print("\nFeature Importance:")
for _, row in importances.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:25s} {row['importance']:.4f} {bar}")

# Save
test_with_preds[["datetime", "price", "prediction", "error"]].to_csv(
    results_dir / "rf_load_leakage_free_predictions.csv", index=False
)

import json
results = {
    "model": "RandomForest",
    "load_data": True,
    "leakage_free": True,
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "r2": r2,
    "mae_low": low["error"].mean(),
    "mae_mid": mid["error"].mean(),
    "mae_high": high["error"].mean(),
    "feature_importance": importances.to_dict("records")
}

with open(results_dir / "rf_load_leakage_free_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("✅ LEAKAGE-FREE RF (WITH LOAD) COMPLETE")
print("=" * 70)

print("\nCHANGES FROM OLD VERSION:")
print("  ❌ Removed: price_lag1, load_lag1, rolling_24 (same-day)")
print("  ✅ Added: lag24/48/168, yesterday aggregates, 168h rolling")