"""
Phase 1.2: Extract .osz archives for multi-mapper subset.

Reads the multi-mapper index and extracts all needed .osz files
to data/extracted/{beatmapset_id}/.

Output: data/extracted/ with subdirectories per beatmapset
"""

import json
import zipfile
from pathlib import Path

DATASET_DIR = Path("/Users/red/Downloads/rcx25/LoopMaker/LoopMaker Notebooks/LoopBaker Local/lmp_crate/datasets/Osu2MIR/config/single_timing_point")
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "multi_mapper_index.json"
EXTRACT_DIR = DATA_DIR / "extracted"


def extract():
    with open(INDEX_PATH) as f:
        index = json.load(f)

    # Collect all needed BeatmapSetIDs
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

        osz_path = DATASET_DIR / f"{bsid}.osz"
        if not osz_path.exists():
            print(f"  Missing: {bsid}.osz")
            errors += 1
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(osz_path, 'r') as z:
                # Extract only .osu files and audio
                for name in z.namelist():
                    if name.endswith('.osu') or name.endswith('.mp3') or name.endswith('.ogg'):
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
    extract()
