import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import json
from datetime import datetime, time

def main():
    print("=" * 70)
    print("DAM Price Forecast - NO DATA LEAKAGE")
    print("=" * 70)

    # Paths
    base_dir = Path(__file__).parent.parent
    price_file = base_dir / "data_processed" / "combined_dam_prices.csv"
    load_forecast_file = base_dir / "data_processed" / "load_forecast_complete.csv"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    hub_data = load_prices(price_file)
    load_fc = load_forecasts(load_forecast_file)
    
    # Prepare features (NO LEAKAGE)
    X, y, dates = prepare_features(hub_data, load_fc)
    
    # Train/test split
    X_train, X_test, y_train, y_test, test_dates = train_test_split_data(X, y, dates)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate
    predictions = predict(models, X_test)
    evaluate_model(y_test, predictions, X_test.columns, models)
    
    # Save results
    save_results(test_dates, y_test, predictions, results_dir)
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)

def load_prices(price_file):
    """Load DAM price data"""
    print("\n1. Loading DAM prices...")
    df = pd.read_csv(price_file)
    
    # Parse datetime
    df["Hour Ending"] = df["Hour Ending"].str.replace("24:00", "00:00")
    df["datetime"] = pd.to_datetime(df["Delivery Date"] + " " + df["Hour Ending"])
    df.loc[df["Hour Ending"] == "00:00", "datetime"] += pd.Timedelta(days=1)
    df["price"] = pd.to_numeric(df["Settlement Point Price"], errors="coerce")
    
    # Filter to Houston hub
    hub_data = df[df["Settlement Point"] == "HB_HOUSTON"].copy()
    hub_data = hub_data.sort_values("datetime").reset_index(drop=True)
    
    # Add delivery_date and hour for deduplication
    hub_data["delivery_date"] = hub_data["datetime"].dt.date
    hub_data["hour"] = hub_data["datetime"].dt.hour
    
    # Drop duplicates (keep first occurrence)
    print(f"   Before dedup: {len(hub_data):,} rows")
    hub_data = hub_data.drop_duplicates(subset=["delivery_date", "hour"], keep="first")
    print(f"   After dedup: {len(hub_data):,} rows")
    
    print(f"   ✓ Range: {hub_data['datetime'].min()} to {hub_data['datetime'].max()}")
    
    return hub_data


def load_forecasts(load_forecast_file):
    """Load load forecast data"""
    print("\n2. Loading load forecasts...")
    load_fc = pd.read_csv(load_forecast_file)
    load_fc["datetime"] = pd.to_datetime(load_fc["datetime"])
    load_fc["runDatetime"] = pd.to_datetime(load_fc["runDatetime"])
    
    print(f"   ✓ {len(load_fc):,} forecast rows")
    print(f"   ✓ Delivery range: {load_fc['datetime'].min()} to {load_fc['datetime'].max()}")
    print(f"   ✓ Publication range: {load_fc['runDatetime'].min()} to {load_fc['runDatetime'].max()}")
    
    return load_fc


def prepare_features(hub_data, load_fc):
    """
    Prepare features with NO DATA LEAKAGE.
    
    Key rule: Only use forecasts published BEFORE 10:00 AM on the day before delivery.
    """
    print("\n3. Preparing features (NO DATA LEAKAGE)...")
    
    # Reshape prices to daily
    daily_prices = hub_data.pivot(index="delivery_date", columns="hour", values="price")
    daily_prices = daily_prices.reindex(columns=range(24)).dropna()
    
    print(f"   ✓ {len(daily_prices)} days of price data")
    
    features_list = []
    targets_list = []
    dates_list = []
    
    for i in range(7, len(daily_prices)):
        delivery_date = daily_prices.index[i]
        target = daily_prices.iloc[i].values  # Prices for delivery_date
        
        # CRITICAL: Bidding deadline is 10:00 AM on (delivery_date - 1 day)
        bidding_deadline = pd.Timestamp(delivery_date) - pd.Timedelta(days=1)
        bidding_deadline = bidding_deadline.replace(hour=10, minute=0, second=0)
        
        # Get forecasts for delivery_date published BEFORE bidding deadline
        forecast_for_delivery = load_fc[
            (load_fc["datetime"].dt.date == delivery_date) &
            (load_fc["runDatetime"] < bidding_deadline)
        ]
        
        if len(forecast_for_delivery) == 0:
            continue  # No forecast available before deadline - skip this day
        
        # Use the LATEST forecast before deadline (closest to 9:30 AM)
        latest_forecast = forecast_for_delivery.loc[forecast_for_delivery["runDatetime"].idxmax()]
        
        houston_load_fc = forecast_for_delivery.groupby(
            forecast_for_delivery["datetime"].dt.hour
            )["coast"].last()  # Use .last() not .first() to get closest to deadline
        

        # Build features
        features = {}
        
        # Calendar features
        pd_date = pd.to_datetime(delivery_date)
        features["day_of_week"] = pd_date.dayofweek
        features["month"] = pd_date.month
        features["is_weekend"] = 1 if pd_date.dayofweek >= 5 else 0
        
        # Load forecast features (available before deadline)
        houston_load_fc = forecast_for_delivery.groupby(
            forecast_for_delivery["datetime"].dt.hour
        )["coast"].first()  # Use Coast as Houston proxy
        
        for h in range(24):
            features[f"load_fc_he{h}"] = houston_load_fc.get(h, np.nan)
        
        # Load forecast aggregates
        features["load_fc_mean"] = houston_load_fc.mean()
        features["load_fc_peak"] = houston_load_fc.max()
        features["load_fc_min"] = houston_load_fc.min()
        features["load_fc_std"] = houston_load_fc.std()
        
        # Historical price features (definitely available)
        yesterday_prices = daily_prices.iloc[i-1].values
        for h in range(24):
            features[f"price_d1_he{h}"] = yesterday_prices[h]
        
        features["price_d1_mean"] = yesterday_prices.mean()
        features["price_d1_max"] = yesterday_prices.max()
        features["price_d1_min"] = yesterday_prices.min()
        features["price_d1_std"] = yesterday_prices.std()
        
        # 7-day price history
        for h in range(24):
            past_week = [daily_prices.iloc[i-d, h] for d in range(1, 8)]
            features[f"price_7d_mean_he{h}"] = np.mean(past_week)
        
        # Check for missing values
        if pd.Series(features).isna().any():
            continue
        
        features_list.append(features)
        targets_list.append(target)
        dates_list.append(delivery_date)
    
    X = pd.DataFrame(features_list)
    y = np.array(targets_list)
    dates = pd.Series(dates_list)
    
    print(f"   ✓ Created {len(X):,} training examples")
    print(f"   ✓ Features: {X.shape[1]} columns")
    print(f"   ✓ Date range: {dates.min()} to {dates.max()}")
    
    return X, y, dates


def train_test_split_data(X, y, dates, train_ratio=0.8):
    """Split data chronologically (no shuffling to avoid leakage)"""
    print("\n4. Train/Test Split...")
    
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = dates[train_size:]
    
    print(f"   Train: {len(X_train):,} days ({dates.iloc[0]} to {dates.iloc[train_size-1]})")
    print(f"   Test:  {len(X_test):,} days ({dates.iloc[train_size]} to {dates.iloc[-1]})")
    
    return X_train, X_test, y_train, y_test, test_dates


def train_models(X_train, y_train):
    """Train 24 hour-specific Random Forest models"""
    print("\n5. Training 24 hour-specific models...")
    
    models = []
    for hour in range(24):
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train[:, hour])
        models.append(model)
        
        if (hour + 1) % 6 == 0:
            print(f"   ✓ Trained models for HE01-HE{hour+1:02d}")
    
    print("   ✓ All 24 models trained")
    return models


def predict(models, X_test):
    """Generate predictions for all hours"""
    predictions_by_hour = []
    for hour in range(24):
        predictions_by_hour.append(models[hour].predict(X_test))
    return np.column_stack(predictions_by_hour)


def evaluate_model(y_test, predictions, feature_names, models):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    
    y_test_flat = y_test.flatten()
    predictions_flat = predictions.flatten()
    
    # Overall metrics
    mae = mean_absolute_error(y_test_flat, predictions_flat)
    rmse = np.sqrt(mean_squared_error(y_test_flat, predictions_flat))
    r2 = r2_score(y_test_flat, predictions_flat)
    
    print(f"\nOverall:")
    print(f"  MAE:   ${mae:.2f}/MWh")
    print(f"  RMSE:  ${rmse:.2f}/MWh")
    print(f"  R²:    {r2:.4f}")
    
    # Per-hour metrics
    print(f"\nPer-Hour Performance:")
    print(f"{'Hour':<6} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 40)
    for hour in range(24):
        h_mae = mean_absolute_error(y_test[:, hour], predictions[:, hour])
        h_rmse = np.sqrt(mean_squared_error(y_test[:, hour], predictions[:, hour]))
        h_r2 = r2_score(y_test[:, hour], predictions[:, hour])
        print(f"HE{hour:02d}   ${h_mae:>9.2f} ${h_rmse:>9.2f} {h_r2:>10.4f}")
    
    # Error by price regime
    errors = np.abs(y_test_flat - predictions_flat)
    low = y_test_flat < 30
    mid = (y_test_flat >= 30) & (y_test_flat < 60)
    high = y_test_flat >= 60
    
    print(f"\nError by Price Regime:")
    print(f"  Low (<$30):   MAE = ${errors[low].mean():.2f} ({low.sum():,} hours)")
    print(f"  Mid ($30-60): MAE = ${errors[mid].mean():.2f} ({mid.sum():,} hours)")
    print(f"  High (>$60):  MAE = ${errors[high].mean():.2f} ({high.sum():,} hours)")
    
    # Feature importance (for HE14 - peak hour)
    peak_model = models[13]  # HE14 (1-2 PM)
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": peak_model.feature_importances_
    })
    imp = imp.sort_values("importance", ascending=False)
    
    print(f"\nTop 15 Features (HE14 - Peak Hour):")
    for _, row in imp.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")


def save_results(test_dates, y_test, predictions, results_dir):
    """Save predictions and metrics"""
    print("\n6. Saving results...")
    
    # Save predictions
    results_df = []
    for i, date in enumerate(test_dates):
        for hour in range(24):
            results_df.append({
                "date": date,
                "hour": hour,
                "actual": y_test[i, hour],
                "predicted": predictions[i, hour],
                "error": abs(y_test[i, hour] - predictions[i, hour])
            })
    
    pd.DataFrame(results_df).to_csv(results_dir / "predictions.csv", index=False)
    
    # Save metrics
    mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions.flatten()))
    r2 = r2_score(y_test.flatten(), predictions.flatten())
    
    summary = {
        "model": "24 hour-specific Random Forest",
        "data_leakage": "NO - only uses forecasts published before 10 AM deadline",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "test_days": int(len(test_dates))
    }
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✓ Saved to: {results_dir}")


if __name__ == "__main__":
    main()