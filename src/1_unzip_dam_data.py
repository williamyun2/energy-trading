import zipfile
from pathlib import Path
from tqdm import tqdm

print("=" * 60)
print("ERCOT Data Extractor")
print("=" * 60)

base_dir = Path(__file__).parent.parent
dam_dir = base_dir / "input_data" / "market_prices" / "historical_DAM_load_zone"
output_dir = base_dir / "data_extracted" / "dam_prices"

output_dir.mkdir(parents=True, exist_ok=True)

zip_files = list(dam_dir.glob("*.zip"))

if not zip_files:
    print(f"\n❌ No ZIP files found in: {dam_dir}")
    exit(1)

print(f"\nFound {len(zip_files)} ZIP files")
print(f"Extracting to: {output_dir}\n")

for zip_path in tqdm(zip_files, desc="Extracting"):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    except Exception as e:
        print(f"\n❌ Error: {zip_path.name}: {e}")

print(f"\n✅ Done! CSVs in: {output_dir}")