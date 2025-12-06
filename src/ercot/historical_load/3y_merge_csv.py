import pandas as pd
from pathlib import Path

print("=" * 70)
print("Merging Part 1 (2017-2023) and Part 2 (2023-2025)")
print("=" * 70)

part1_file = Path(r"C:\code\energy_trading\data_processed\load_forecast_historical_archive.csv")
part2_file = Path(r"C:\code\energy_trading\data_processed\load_forecast_part2.csv")
output_file = Path(r"C:\code\energy_trading\data_processed\load_forecast_complete.csv")

print("\n1. Loading Part 1...")
part1 = pd.read_csv(part1_file)
part1['datetime'] = pd.to_datetime(part1['datetime'])
part1['runDatetime'] = pd.to_datetime(part1['runDatetime'])
print(f"   Part 1: {len(part1):,} rows")
print(f"   Date range: {part1['datetime'].min()} to {part1['datetime'].max()}")
print(f"   Models: {sorted(part1['model'].unique())}")

print("\n2. Loading Part 2...")
part2 = pd.read_csv(part2_file)
part2['datetime'] = pd.to_datetime(part2['datetime'])
part2['runDatetime'] = pd.to_datetime(part2['runDatetime'])
print(f"   Part 2: {len(part2):,} rows")
print(f"   Date range: {part2['datetime'].min()} to {part2['datetime'].max()}")
print(f"   Models: {sorted(part2['model'].unique())}")

print("\n3. Checking overlap...")
overlap_start = max(part1['datetime'].min(), part2['datetime'].min())
overlap_end = min(part1['datetime'].max(), part2['datetime'].max())
print(f"   Overlap period: {overlap_start.date()} to {overlap_end.date()}")

overlap_part1 = part1[(part1['datetime'] >= overlap_start) & (part1['datetime'] <= overlap_end)]
overlap_part2 = part2[(part2['datetime'] >= overlap_start) & (part2['datetime'] <= overlap_end)]
print(f"   Overlapping rows in Part 1: {len(overlap_part1):,}")
print(f"   Overlapping rows in Part 2: {len(overlap_part2):,}")

print("\n4. Combining datasets...")
combined = pd.concat([part1, part2], ignore_index=True)
print(f"   Before dedup: {len(combined):,} rows")

print("\n5. Removing exact duplicates...")
combined = combined.drop_duplicates()
print(f"   After dedup: {len(combined):,} rows")
print(f"   Removed: {len(part1) + len(part2) - len(combined):,} duplicate rows")

print("\n6. Sorting...")
combined = combined.sort_values(['datetime', 'runDatetime']).reset_index(drop=True)

print("\n7. Saving merged file...")
combined.to_csv(output_file, index=False)
print(f"   ✓ Saved to: {output_file}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Total rows: {len(combined):,}")
print(f"Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
print(f"Span: {(combined['datetime'].max() - combined['datetime'].min()).days} days")
print(f"Unique forecasts: {combined['runDatetime'].nunique():,}")
print(f"Models: {sorted(combined['model'].unique())}")
print(f"File size: {output_file.stat().st_size / (1024**3):.2f} GB")

print("\nModel distribution:")
print(combined['model'].value_counts().sort_index())

print("\n✅ MERGE COMPLETE")
print("=" * 70)