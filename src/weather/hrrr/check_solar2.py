import pandas as pd

filepath = r"D:\Users\williamyun\proj\energy_trading\data_processed\weather\hrrr\hrrr_texas_20240101_0600.parquet"
print(filepath)
df = pd.read_parquet(filepath)

# Check midnight solar for all zones
midnight = df[df['datetime'] == '2024-07-08 00:00:00']

print("Midnight  solar radiation by zone:")
print(midnight[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))

print("\n\n1am  solar radiation by zone:")
one_am = df[df['datetime'] == '2024-07-08 01:00:00']
print(one_am[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))

print("\n\n11pm solar radiation by zone:")
eleven_pm = df[df['datetime'] == '2024-07-07 23:00:00']
print(eleven_pm[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))

print("\n\n10pm solar radiation by zone:")
ten_pm = df[df['datetime'] == '2024-07-07 22:00:00']
print(ten_pm[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))








filepath = r"D:\Users\williamyun\proj\energy_trading\data_processed\weather\hrrr\hrrr_texas_20240707_0600.parquet"
print(filepath)
df = pd.read_parquet(filepath)

# Check midnight solar for all zones
midnight = df[df['datetime'] == '2024-07-08 00:00:00']

print("Midnight (2024-07-08 00:00) solar radiation by zone:")
print(midnight[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))

print("\n\n1am (2024-07-08 01:00) solar radiation by zone:")
one_am = df[df['datetime'] == '2024-07-08 01:00:00']
print(one_am[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))

print("\n\n11pm (2024-07-07 23:00) solar radiation by zone:")
eleven_pm = df[df['datetime'] == '2024-07-07 23:00:00']
print(eleven_pm[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))

print("\n\n10pm (2024-07-07 22:00) solar radiation by zone:")
ten_pm = df[df['datetime'] == '2024-07-07 22:00:00']
print(ten_pm[['zone', 'solar_radiation_wm2', 'cloud_cover_pct']].to_string(index=False))


