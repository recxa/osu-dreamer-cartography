"""
Step 1: Build multi-mapper index from .osz dataset.

Scans all .osz files, extracts Title/Artist/Creator metadata,
and identifies songs with multiple different mappers.

Usage:
    uv run python scripts/01_build_index.py /path/to/osz/files/

Output: experiment/output/data/multi_mapper_index.json
"""

import argparse
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "experiment" / "output" / "data"


def extract_metadata(osz_path: Path) -> dict | None:
    """Extract Title, Artist, Creator from first .osu file in an .osz archive."""
    try:
        with zipfile.ZipFile(osz_path, "r") as z:
            osu_files = [n for n in z.namelist() if n.endswith(".osu")]
            if not osu_files:
                return None

            content = z.read(osu_files[0]).decode("utf-8", errors="replace")

            title = re.search(r"^Title:(.+)$", content, re.MULTILINE)
            artist = re.search(r"^Artist:(.+)$", content, re.MULTILINE)
            creator = re.search(r"^Creator:(.+)$", content, re.MULTILINE)
            mode = re.search(r"^Mode:\s*(\d+)", content, re.MULTILINE)

            if not (title and artist and creator):
                return None

            return {
                "title": title.group(1).strip(),
                "artist": artist.group(1).strip(),
                "creator": creator.group(1).strip(),
                "mode": int(mode.group(1)) if mode else 0,
                "beatmapset_id": int(osz_path.stem),
                "filename": osz_path.name,
            }
    except Exception as e:
        print(f"  Error reading {osz_path.name}: {e}")
        return None


def build_index(dataset_dir: Path):
    osz_files = sorted(dataset_dir.glob("*.osz"))
    print(f"Found {len(osz_files)} .osz files in {dataset_dir}")

    all_meta = []
    errors = 0
    for i, osz_path in enumerate(osz_files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(osz_files)}...")
        meta = extract_metadata(osz_path)
        if meta:
            all_meta.append(meta)
        else:
            errors += 1

    print(f"Successfully extracted: {len(all_meta)}, Errors: {errors}")

    # Group by (title, artist) — normalized for matching
    songs = defaultdict(list)
    for meta in all_meta:
        key = (meta["title"].lower().strip(), meta["artist"].lower().strip())
        songs[key].append(meta)

    # Filter to multi-mapper groups
    multi_mapper = []
    for (title_lower, artist_lower), entries in songs.items():
        creators = set(e["creator"] for e in entries)
        if len(creators) < 2:
            continue

        multi_mapper.append({
            "title": entries[0]["title"],
            "artist": entries[0]["artist"],
            "match_key": f"{title_lower} ||| {artist_lower}",
            "num_mappers": len(creators),
            "mappers": sorted(creators),
            "beatmapsets": [
                {
                    "beatmapset_id": e["beatmapset_id"],
                    "creator": e["creator"],
                    "filename": e["filename"],
                }
                for e in entries
            ],
        })

    multi_mapper.sort(key=lambda x: -x["num_mappers"])

    print(f"\nTotal unique songs: {len(songs)}")
    print(f"Songs with 2+ mappers: {len(multi_mapper)}")
    print(f"Songs with 3+ mappers: {sum(1 for s in multi_mapper if s['num_mappers'] >= 3)}")
    print(f"Songs with 4+ mappers: {sum(1 for s in multi_mapper if s['num_mappers'] >= 4)}")
    print(f"Songs with 5+ mappers: {sum(1 for s in multi_mapper if s['num_mappers'] >= 5)}")
    total_sets = sum(len(s["beatmapsets"]) for s in multi_mapper)
    print(f"Total beatmapsets in multi-mapper groups: {total_sets}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "multi_mapper_index.json"
    with open(output_path, "w") as f:
        json.dump(multi_mapper, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build multi-mapper song index from .osz files")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing .osz files")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        parser.error(f"Not a directory: {args.dataset_dir}")

    build_index(args.dataset_dir)
