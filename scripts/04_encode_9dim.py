"""
Step 4: Encode representative beatmaps to 9-dim temporal signals.

Uses osu-dreamer's encoding functions (no model needed) to convert
each .osu beatmap to a [9, L] array of interpretable features:
  [ONSET, COMBO, SLIDE, SUSTAIN, WHISTLE, FINISH, CLAP, X, Y]

Handles audio alignment within song groups (different .osz archives
may have slightly different audio file lengths).

Usage:
    uv run python scripts/04_encode_9dim.py

Input:  experiment/output/data/beatmap_registry.json
Output: experiment/output/data/encodings_9dim/{beatmapset_id}_{version}.npy
        experiment/output/data/encoding_manifest.json
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.data.load_audio import load_audio, get_frame_times
from osu_dreamer.data.beatmap.encode import encode_beatmap

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "experiment" / "output" / "data"
REGISTRY_PATH = DATA_DIR / "beatmap_registry.json"
OUTPUT_DIR = DATA_DIR / "encodings_9dim"
MANIFEST_PATH = DATA_DIR / "encoding_manifest.json"


def sanitize_version(version: str) -> str:
    """Make version string safe for filenames."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in version).strip()


def encode_all():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    reps = [r for r in registry if r.get("is_representative")]
    print(f"Representative beatmaps to encode: {len(reps)}")

    by_song = defaultdict(list)
    for r in reps:
        by_song[r["song_group_idx"]].append(r)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    encoded_count = 0
    error_count = 0
    alignment_issues = 0

    for song_idx, song_reps in sorted(by_song.items()):
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

        frame_counts = [info["L"] for info in audio_info.values()]
        if len(set(frame_counts)) > 1:
            alignment_issues += 1
            min_L = min(frame_counts)
            max_L = max(frame_counts)
            diff_pct = (max_L - min_L) / max_L * 100
            print(f"  Alignment issue song {song_idx}: L ranges {min_L}-{max_L} ({diff_pct:.1f}% diff)")
            target_L = min_L
        else:
            target_L = frame_counts[0]

        frame_times = get_frame_times(target_L)

        for r in song_reps:
            if r["osu_path"] not in audio_info:
                continue

            try:
                bm = Beatmap(Path(r["osu_path"]))
                encoded = encode_beatmap(bm, frame_times)  # [9, L]

                if encoded.shape[1] > target_L:
                    encoded = encoded[:, :target_L]
                elif encoded.shape[1] < target_L:
                    pad = target_L - encoded.shape[1]
                    encoded = np.pad(encoded, ((0, 0), (0, pad)), mode="constant")

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

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Encoded: {encoded_count}")
    print(f"  Errors: {error_count}")
    print(f"  Audio alignment issues: {alignment_issues}")
    print(f"  Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    if not REGISTRY_PATH.exists():
        print(f"Error: Run 03_build_registry.py first — missing {REGISTRY_PATH}")
        exit(1)

    encode_all()
