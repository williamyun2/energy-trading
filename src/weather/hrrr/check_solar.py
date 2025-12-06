import pandas as pd

filepath = r"D:\Users\williamyun\proj\energy_trading\data_processed\weather\hrrr\hrrr_texas_20240101_0600.parquet"

df = pd.read_parquet(filepath)

# Show HOUSTON zone full detail
houston = df[df['zone']=='HOUSTON'][['datetime', 'forecast_issued', 'forecast_hour', 'solar_radiation_wm2']]

print("HOUSTON zone - all 25 hours:")
print(houston.to_string(index=False))



filepath = r"D:\Users\williamyun\proj\energy_trading\data_processed\weather\hrrr\hrrr_texas_20240707_0600.parquet"

df = pd.read_parquet(filepath)

# Show HOUSTON zone full detail
houston = df[df['zone']=='HOUSTON'][['datetime', 'forecast_issued', 'forecast_hour', 'solar_radiation_wm2']]

print("HOUSTON zone - all 25 hours:")
print(houston.to_string(index=False))



