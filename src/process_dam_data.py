import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("=" * 60)
print("ERCOT DAM Price Data Processor")
print("=" * 60)

base_dir = Path(__file__).parent.parent
input_dir = base_dir / "data_extracted" / "dam_prices"
output_dir = base_dir / "data_processed"
output_file = output_dir / "combined_dam_prices.csv"

xlsx_files = list(input_dir.glob("*.xlsx"))

if not xlsx_files:
    print(f"\n❌ No Excel files in: {input_dir}")
    print("Run unzip_dam_data.py first!")
    exit(1)

print(f"\nFound {len(xlsx_files)} Excel files\n")

all_data = []
for file in tqdm(xlsx_files, desc="Processing"):
    df = pd.read_excel(file)
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)

print(f"\n✓ Total rows: {len(combined):,}")
print(f"Columns: {combined.columns.tolist()}")

if 'Settlement Point' in combined.columns:
    hubs = [h for h in combined['Settlement Point'].unique() if 'HB_' in str(h)]
    print(f"\n✓ Hubs found: {hubs}")

output_dir.mkdir(parents=True, exist_ok=True)
combined.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file}")