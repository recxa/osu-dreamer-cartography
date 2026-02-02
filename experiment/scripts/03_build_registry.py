"""
Phase 1.3: Build per-beatmap registry from extracted files.

Parses every .osu file in the extracted directories, filters to
osu!standard (Mode 0), and records metadata for each individual beatmap.

Also selects one "representative" difficulty per mapper per song
for cross-mapper comparison (highest OD as a difficulty proxy).

Output: data/beatmap_registry.json
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "multi_mapper_index.json"
EXTRACT_DIR = DATA_DIR / "extracted"
OUTPUT_PATH = DATA_DIR / "beatmap_registry.json"


def parse_osu_metadata(osu_path: Path) -> dict | None:
    """Parse key metadata from a .osu file."""
    try:
        content = osu_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None

    def get_field(name):
        m = re.search(rf'^{name}:\s*(.+)$', content, re.MULTILINE)
        return m.group(1).strip() if m else None

    def get_float(name):
        m = re.search(rf'^{name}:\s*([\d.]+)', content, re.MULTILINE)
        return float(m.group(1)) if m else None

    mode = get_field("Mode")
    if mode is None or int(mode) != 0:
        return None  # Skip non-standard modes

    # Find audio file
    audio_filename = get_field("AudioFilename")
    audio_path = osu_path.parent / audio_filename if audio_filename else None

    return {
        "osu_path": str(osu_path),
        "beatmapset_id": int(osu_path.parent.name),
        "title": get_field("Title"),
        "artist": get_field("Artist"),
        "creator": get_field("Creator"),
        "version": get_field("Version"),
        "beatmap_id": get_field("BeatmapID"),
        "mode": 0,
        "cs": get_float("CircleSize"),
        "ar": get_float("ApproachRate"),
        "od": get_float("OverallDifficulty"),
        "hp": get_float("HPDrainRate"),
        "audio_path": str(audio_path) if audio_path and audio_path.exists() else None,
    }


def build_registry():
    with open(INDEX_PATH) as f:
        index = json.load(f)

    # Build song group lookup: beatmapset_id -> song group info
    song_lookup = {}
    for i, song in enumerate(index):
        for bs in song["beatmapsets"]:
            song_lookup[bs["beatmapset_id"]] = {
                "song_group_idx": i,
                "song_title": song["title"],
                "song_artist": song["artist"],
                "num_mappers": song["num_mappers"],
            }

    registry = []
    mode_filtered = 0
    parse_errors = 0

    # Walk all extracted directories
    extract_dirs = sorted(d for d in EXTRACT_DIR.iterdir() if d.is_dir())
    print(f"Processing {len(extract_dirs)} extracted directories...")

    for d in extract_dirs:
        bsid = int(d.name)
        song_info = song_lookup.get(bsid, {})

        for osu_file in sorted(d.glob("*.osu")):
            meta = parse_osu_metadata(osu_file)
            if meta is None:
                # Could be non-std mode or parse error
                mode_filtered += 1
                continue

            meta["song_group_idx"] = song_info.get("song_group_idx")
            meta["num_mappers_in_group"] = song_info.get("num_mappers")
            registry.append(meta)

    print(f"Registry entries (Mode 0): {len(registry)}")
    print(f"Filtered (non-std mode): {mode_filtered}")

    # Stats
    creators = set(r["creator"] for r in registry)
    songs = set(r["song_group_idx"] for r in registry if r["song_group_idx"] is not None)
    print(f"Unique creators: {len(creators)}")
    print(f"Unique song groups: {len(songs)}")

    # Select representatives: one per mapper per song (highest OD)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in registry:
        key = (r["song_group_idx"], r["creator"])
        groups[key].append(r)

    representatives = []
    for (song_idx, creator), beatmaps in groups.items():
        if song_idx is None:
            continue
        # Pick highest OD as difficulty proxy
        best = max(beatmaps, key=lambda b: b["od"] or 0)
        best["is_representative"] = True
        representatives.append(best)

    # Mark representatives in registry
    rep_paths = set(r["osu_path"] for r in representatives)
    for r in registry:
        r["is_representative"] = r["osu_path"] in rep_paths

    print(f"Representative beatmaps (one per mapper per song): {len(representatives)}")

    # Missing audio check
    no_audio = sum(1 for r in registry if r["audio_path"] is None)
    print(f"Beatmaps missing audio: {no_audio}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    return registry


if __name__ == "__main__":
    build_registry()
