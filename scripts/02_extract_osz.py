"""
Step 2: Extract .osz archives for multi-mapper subset.

Reads the multi-mapper index and extracts all needed .osz files
to experiment/output/data/extracted/{beatmapset_id}/.

Usage:
    uv run python scripts/02_extract_osz.py /path/to/osz/files/

Input:  experiment/output/data/multi_mapper_index.json
Output: experiment/output/data/extracted/ with subdirectories per beatmapset
"""

import argparse
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "experiment" / "output" / "data"
INDEX_PATH = DATA_DIR / "multi_mapper_index.json"
EXTRACT_DIR = DATA_DIR / "extracted"


def extract(dataset_dir: Path):
    with open(INDEX_PATH) as f:
        index = json.load(f)

    needed = {}
    for song in index:
        for bs in song["beatmapsets"]:
            needed[bs["beatmapset_id"]] = bs

    print(f"Need to extract {len(needed)} beatmapsets")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    errors = 0

    for bsid, bs_info in sorted(needed.items()):
        out_dir = EXTRACT_DIR / str(bsid)
        if out_dir.exists() and any(out_dir.glob("*.osu")):
            skipped += 1
            continue

        osz_path = dataset_dir / f"{bsid}.osz"
        if not osz_path.exists():
            print(f"  Missing: {bsid}.osz")
            errors += 1
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(osz_path, "r") as z:
                for name in z.namelist():
                    if name.endswith(".osu") or name.endswith(".mp3") or name.endswith(".ogg"):
                        z.extract(name, out_dir)
            extracted += 1
        except Exception as e:
            print(f"  Error extracting {bsid}: {e}")
            errors += 1

        if (extracted + skipped) % 100 == 0:
            print(f"  Progress: {extracted} extracted, {skipped} skipped, {errors} errors")

    print(f"\nDone: {extracted} extracted, {skipped} already existed, {errors} errors")
    print(f"Total directories: {sum(1 for d in EXTRACT_DIR.iterdir() if d.is_dir())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract .osz archives for multi-mapper subset")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing .osz files")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        parser.error(f"Not a directory: {args.dataset_dir}")
    if not INDEX_PATH.exists():
        parser.error(f"Run 01_build_index.py first — missing {INDEX_PATH}")

    extract(args.dataset_dir)
