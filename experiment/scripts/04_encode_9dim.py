"""
Phase 2.1: Encode representative beatmaps to 9-dim temporal signals.

Uses osu-dreamer's encoding functions (no model needed) to convert
each .osu beatmap to a [9, L] array of interpretable features:
  [ONSET, COMBO, SLIDE, SUSTAIN, WHISTLE, FINISH, CLAP, X, Y]

Handles audio alignment within song groups (different .osz archives
may have slightly different audio file lengths).

Output: data/encodings_9dim/{beatmapset_id}_{version}.npy
        data/encoding_manifest.json (maps filenames to metadata)

Run from osu-dreamer repo directory:
  uv run python /path/to/04_encode_9dim.py
"""

import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.data.load_audio import load_audio, get_frame_times
from osu_dreamer.data.beatmap.encode import encode_beatmap

DATA_DIR = Path("/Users/red/Downloads/rcx26/OSUxMIR/experiments/latent-cartography/data")
REGISTRY_PATH = DATA_DIR / "beatmap_registry.json"
INDEX_PATH = DATA_DIR / "multi_mapper_index.json"
OUTPUT_DIR = DATA_DIR / "encodings_9dim"
MANIFEST_PATH = DATA_DIR / "encoding_manifest.json"


def sanitize_version(version: str) -> str:
    """Make version string safe for filenames."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in version).strip()


def encode_all():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    # Filter to representatives only
    reps = [r for r in registry if r.get("is_representative")]
    print(f"Representative beatmaps to encode: {len(reps)}")

    # Group by song for audio alignment checking
    by_song = defaultdict(list)
    for r in reps:
        by_song[r["song_group_idx"]].append(r)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    encoded_count = 0
    error_count = 0
    alignment_issues = 0

    for song_idx, song_reps in sorted(by_song.items()):
        # First pass: load audio for each beatmap to get frame counts
        audio_info = {}
        for r in song_reps:
            audio_path = r.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                print(f"  Missing audio: {r['beatmapset_id']}")
                continue
            try:
                spec = load_audio(Path(audio_path))
                audio_info[r["osu_path"]] = {
                    "spec": spec,
                    "L": spec.shape[1],
                    "audio_path": audio_path,
                }
            except Exception as e:
                print(f"  Audio error ({r['beatmapset_id']}): {e}")
                error_count += 1

        if not audio_info:
            continue

        # Check alignment: all should have same frame count for same song
        frame_counts = [info["L"] for info in audio_info.values()]
        if len(set(frame_counts)) > 1:
            alignment_issues += 1
            min_L = min(frame_counts)
            max_L = max(frame_counts)
            diff_pct = (max_L - min_L) / max_L * 100
            print(f"  Alignment issue song {song_idx}: L ranges {min_L}-{max_L} ({diff_pct:.1f}% diff)")
            # Use minimum frame count (truncate longer ones)
            target_L = min_L
        else:
            target_L = frame_counts[0]

        frame_times = get_frame_times(target_L)

        # Second pass: encode beatmaps
        for r in song_reps:
            if r["osu_path"] not in audio_info:
                continue

            try:
                bm = Beatmap(Path(r["osu_path"]))
                encoded = encode_beatmap(bm, frame_times)  # [9, L]

                # Truncate if needed
                if encoded.shape[1] > target_L:
                    encoded = encoded[:, :target_L]
                elif encoded.shape[1] < target_L:
                    # Pad with zeros if beatmap is shorter
                    pad = target_L - encoded.shape[1]
                    encoded = np.pad(encoded, ((0, 0), (0, pad)), mode='constant')

                # Save
                version_safe = sanitize_version(r["version"] or "unknown")
                filename = f"{r['beatmapset_id']}_{version_safe}.npy"
                np.save(OUTPUT_DIR / filename, encoded)

                manifest.append({
                    "filename": filename,
                    "osu_path": r["osu_path"],
                    "beatmapset_id": r["beatmapset_id"],
                    "creator": r["creator"],
                    "version": r["version"],
                    "title": r["title"],
                    "artist": r["artist"],
                    "song_group_idx": r["song_group_idx"],
                    "od": r["od"],
                    "ar": r["ar"],
                    "cs": r["cs"],
                    "shape": list(encoded.shape),
                    "target_L": target_L,
                })

                encoded_count += 1
                if encoded_count % 50 == 0:
                    print(f"  Encoded {encoded_count}...")

            except Exception as e:
                print(f"  Encode error ({r['beatmapset_id']} {r['version']}): {e}")
                error_count += 1

    # Save manifest
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Encoded: {encoded_count}")
    print(f"  Errors: {error_count}")
    print(f"  Audio alignment issues: {alignment_issues}")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"  Encodings: {OUTPUT_DIR}/")


if __name__ == "__main__":
    encode_all()
