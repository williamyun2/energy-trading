import zipfile
import io
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("ERCOT DAM ZIP Inspector & Extractor")
print("=" * 70)

# ---------------------------------------------------------------------
# Resolve base directory (assuming this script is in src/, project root one level up)
# ---------------------------------------------------------------------
try:
    base_dir = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback if __file__ is not defined (e.g., in some interactive environments)
    base_dir = Path.cwd().parent

dam_dir = base_dir / "input_data" / "market_prices" / "historical_DAM_load_zone"
output_dir = base_dir / "data_extracted" / "dam_prices"

output_dir.mkdir(parents=True, exist_ok=True)

zip_files = sorted(dam_dir.glob("*.zip"))

if not zip_files:
    print(f"\n❌ No ZIP files found in: {dam_dir}")
    raise SystemExit(1)

print(f"\nFound {len(zip_files)} ZIP files in:")
print(f"  {dam_dir}\n")

# ---------------------------------------------------------------------
# Helper: recursively inspect ZIP contents (for printing / debugging)
# ---------------------------------------------------------------------
def inspect_zip(zf: zipfile.ZipFile, depth: int = 0, max_depth: int = 2):
    """
    Recursively print the structure of a ZipFile.

    depth: current recursion depth
    max_depth: how deep we go into nested ZIPs (just for readability)
    """
    indent = "   " * depth
    files_inside = zf.infolist()
    print(f"{indent}Contains {len(files_inside)} file(s):\n")

    for i, info in enumerate(files_inside, 1):
        name = info.filename
        size_mb = info.file_size / (1024 * 1024) if info.file_size else 0.0

        # Case-insensitive zip detection
        is_zip = name.lower().endswith(".zip")
        file_type = "ZIP" if is_zip else name.split(".")[-1].upper()

        print(f"{indent}{i}. {name}")
        print(f"{indent}   Type: {file_type} | Size: {size_mb:.2f} MB")

        # If it's a nested ZIP and we're allowed to go deeper, inspect it
        if is_zip and depth < max_depth:
            print(f"{indent}   >>> NESTED ZIP (depth {depth + 1}) - inspecting contents...")
            try:
                nested_data = zf.read(name)
                with zipfile.ZipFile(io.BytesIO(nested_data)) as nested_zip:
                    inspect_zip(nested_zip, depth=depth + 1, max_depth=max_depth)
            except Exception as e:
                print(f"{indent}   >>> Error reading nested ZIP: {e}")

        print()

# ---------------------------------------------------------------------
# Helper: recursively summarize ZIP contents (for final counts)
# ---------------------------------------------------------------------
def summarize_zip(zf: zipfile.ZipFile, depth: int = 0, max_depth: int = 10):
    """
    Recursively count files, nested ZIPs, and Excel files inside a ZipFile.

    Returns:
        (total_files, nested_zips, excel_files)
    """
    total_files = 0
    nested_zips = 0
    excel_files = 0

    for info in zf.infolist():
        name = info.filename
        total_files += 1

        lower_name = name.lower()

        if lower_name.endswith(".zip") and depth < max_depth:
            nested_zips += 1
            try:
                nested_data = zf.read(name)
                with zipfile.ZipFile(io.BytesIO(nested_data)) as nested_zip:
                    t, n, e = summarize_zip(nested_zip, depth=depth + 1, max_depth=max_depth)
                    total_files += t
                    nested_zips += n
                    excel_files += e
            except Exception:
                # If a nested zip is corrupted or unreadable, skip it
                pass
        elif lower_name.endswith(".xlsx"):
            excel_files += 1

    return total_files, nested_zips, excel_files

# ---------------------------------------------------------------------
# Detailed inspection of the first 3 ZIP files
# ---------------------------------------------------------------------
print("=" * 70)
print("Detailed inspection of first few ZIPs")
print("=" * 70)

for zip_path in zip_files[:3]:
    print("\n" + "=" * 70)
    print(f"File: {zip_path.name}")
    print("=" * 70)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Recursively inspect this zip (up to depth=2 for readability)
            inspect_zip(z, depth=0, max_depth=2)

    except Exception as e:
        print(f"❌ Error reading ZIP: {e}")

# ---------------------------------------------------------------------
# Summary across ALL ZIPs (recursive, all levels)
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("Summary across all ZIPs")
print("=" * 70)

grand_total_files = 0
grand_nested_zips = 0
grand_excel_files = 0

for zip_path in zip_files:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            t, n, e = summarize_zip(z, depth=0, max_depth=10)
            grand_total_files += t
            grand_nested_zips += n
            grand_excel_files += e
    except Exception as e:
        print(f"Error summarizing {zip_path.name}: {e}")

print(f"Total ZIPs: {len(zip_files)}")
print(f"Total files inside all ZIPs (including nested): {grand_total_files}")
print(f"Nested ZIP files (all levels): {grand_nested_zips}")
print(f"Excel files (.xlsx) at all levels: {grand_excel_files}")

if grand_nested_zips > 0:
    print("\n⚠️  NESTED ZIPS DETECTED!")
    print("    These nested ZIPs will be extracted as .zip files into the output")
    print("    directory; a second pass would be needed to unzip them further.\n")

# ---------------------------------------------------------------------
# Extraction step
# ---------------------------------------------------------------------
print("=" * 70)
print("Extracting all top-level ZIPs")
print("=" * 70)
print(f"\nExtracting to: {output_dir}\n")

for zip_path in tqdm(zip_files, desc="Extracting"):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    except Exception as e:
        print(f"\n❌ Error extracting {zip_path.name}: {e}")

print(f"\n✅ Done! Files extracted to: {output_dir}")
