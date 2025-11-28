import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

print("=" * 60)
print("ERCOT Price Forecasting Model")
print("=" * 60)

base_dir = Path(__file__).parent.parent
data_file = base_dir / "data_processed" / "combined_dam_prices.csv"
results_dir = base_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

print("\n1. Loading data...")
df = pd.read_csv(data_file)
print(f"   ✓ {len(df):,} rows")

print("\n2. Processing datetime...")
df['datetime'] = pd.to_datetime(df['Delivery Date'] + ' ' + df['Hour Ending'])
df['price'] = pd.to_numeric(df['Settlement Point Price'], errors='coerce')

print("\n3. Filtering to HB_HOUSTON...")
hub_data = df[df['Settlement Point'] == 'HB_HOUSTON'].copy()
hub_data = hub_data.sort_values('datetime').reset_index(drop=True)
print(f"   ✓ {len(hub_data):,} rows")

print("\n4. Creating features...")
hub_data['hour'] = hub_data['datetime'].dt.hour
hub_data['day_of_week'] = hub_data['datetime'].dt.dayofweek
hub_data['month'] = hub_data['datetime'].dt.month
hub_data['price_lag1'] = hub_data['price'].shift(1)
hub_data['price_lag24'] = hub_data['price'].shift(24)
hub_data = hub_data.dropna()

train_size = int(len(hub_data) * 0.8)
train = hub_data[:train_size]
test = hub_data[train_size:]

features = ['hour', 'day_of_week', 'month', 'price_lag1', 'price_lag24']
X_train, y_train = train[features], train['price']
X_test, y_test = test[features], test['price']

print("\n5. Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"MAE:  ${mae:.2f}/MWh")
print(f"RMSE: ${rmse:.2f}/MWh")
print("=" * 60)

plt.figure(figsize=(14, 6))
plt.plot(test['datetime'].iloc[:168], y_test.iloc[:168], label='Actual', lw=2)
plt.plot(test['datetime'].iloc[:168], predictions[:168], label='Predicted', lw=2)
plt.xlabel('Date')
plt.ylabel('Price ($/MWh)')
plt.title('ERCOT DAM Price Forecast')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(results_dir / 'forecast.png', dpi=150)
print(f"\n✅ Plot: {results_dir / 'forecast.png'}")