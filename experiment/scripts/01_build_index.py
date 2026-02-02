"""
Phase 1.1: Build multi-mapper index from .osz dataset.

Scans all .osz files, extracts Title/Artist/Creator metadata,
and identifies songs with multiple different mappers.

Output: data/multi_mapper_index.json
"""

import zipfile
import os
import re
import json
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path("/Users/red/Downloads/rcx25/LoopMaker/LoopMaker Notebooks/LoopBaker Local/lmp_crate/datasets/Osu2MIR/config/single_timing_point")
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def extract_metadata(osz_path: Path) -> dict | None:
    """Extract Title, Artist, Creator from first .osu file in an .osz archive."""
    try:
        with zipfile.ZipFile(osz_path, 'r') as z:
            osu_files = [n for n in z.namelist() if n.endswith('.osu')]
            if not osu_files:
                return None

            content = z.read(osu_files[0]).decode('utf-8', errors='replace')

            title = re.search(r'^Title:(.+)$', content, re.MULTILINE)
            artist = re.search(r'^Artist:(.+)$', content, re.MULTILINE)
            creator = re.search(r'^Creator:(.+)$', content, re.MULTILINE)
            mode = re.search(r'^Mode:\s*(\d+)', content, re.MULTILINE)

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
        print(f"Error reading {osz_path.name}: {e}")
        return None


def build_index():
    osz_files = sorted(DATASET_DIR.glob("*.osz"))
    print(f"Found {len(osz_files)} .osz files")

    # Extract metadata from all files
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

    # Summary stats
    print(f"\nTotal unique songs: {len(songs)}")
    print(f"Songs with 2+ mappers: {len(multi_mapper)}")
    print(f"Songs with 3+ mappers: {sum(1 for s in multi_mapper if s['num_mappers'] >= 3)}")
    print(f"Songs with 4+ mappers: {sum(1 for s in multi_mapper if s['num_mappers'] >= 4)}")
    print(f"Songs with 5+ mappers: {sum(1 for s in multi_mapper if s['num_mappers'] >= 5)}")
    total_sets = sum(len(s["beatmapsets"]) for s in multi_mapper)
    print(f"Total beatmapsets in multi-mapper groups: {total_sets}")
    unique_mappers = set(m for s in multi_mapper for m in s["mappers"])
    print(f"Unique mappers: {len(unique_mappers)}")

    # Top songs
    print("\nTop 10 songs by mapper count:")
    for s in multi_mapper[:10]:
        print(f"  {s['num_mappers']} mappers: \"{s['title']}\" by {s['artist']}")
        for bs in s["beatmapsets"]:
            print(f"    - {bs['creator']} ({bs['beatmapset_id']})")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "multi_mapper_index.json"
    with open(output_path, 'w') as f:
        json.dump(multi_mapper, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    return multi_mapper


if __name__ == "__main__":
    build_index()
