"""
Latent Cartography: Full Analysis Pipeline

Runs the complete analysis on a user-provided folder of .osz beatmap files.
Combines the logic from scripts 01-08 into a single self-contained pipeline.

Usage:
    PYTHONPATH=. python experiment/run_analysis.py /path/to/osz/folder/

Pipeline:
    1. Scan .osz files, extract, build multi-mapper index
    2. Build beatmap registry (Mode 0 only, select representatives)
    3. Encode representatives to 9-dim signals
    4. Run Track A cross-mapper analysis (Pearson per dimension)
    5. Encode through pre-trained VAE (using bundled checkpoint)
    6. Run Track B cross-mapper analysis
    7. Run synthesis (32x9 correlation matrix)
    8. Generate all plots
    9. Print summary to terminal

Output is written to experiment/output/ relative to the repo root.
"""

import json
import os
import re
import sys
import time
import warnings
import zipfile
import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (relative to repo root, auto-detected)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = REPO_ROOT / "experiment" / "checkpoints" / "epoch=3-step=58000.ckpt"

# Output directory (all generated artefacts go here)
OUTPUT_DIR = REPO_ROOT / "experiment" / "output"
DATA_DIR = OUTPUT_DIR / "data"
RESULTS_DIR = OUTPUT_DIR / "results"

# Names for the 9 interpretable dimensions
DIM_NAMES = ["ONSET", "COMBO", "SLIDE", "SUSTAIN", "WHISTLE", "FINISH", "CLAP", "X", "Y"]
N_LATENT = 32

# Minimum number of multi-mapper songs required to proceed
MIN_MULTI_MAPPER_SONGS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(step: int, title: str) -> None:
    total = 9
    print(f"\n{'=' * 70}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'=' * 70}")


def _elapsed(start: float) -> str:
    dt = time.time() - start
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}min"


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D arrays."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def sanitize_version(version: str) -> str:
    """Make version string safe for filenames."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in version).strip()


# ---------------------------------------------------------------------------
# Step 1: Scan .osz files and build multi-mapper index
# ---------------------------------------------------------------------------

def step1_build_index(dataset_dir: Path) -> list:
    _banner(1, "Building multi-mapper index")
    t0 = time.time()

    osz_files = sorted(dataset_dir.glob("*.osz"))
    print(f"  Found {len(osz_files)} .osz files in {dataset_dir}")

    if not osz_files:
        print("  ERROR: No .osz files found. Check the path.")
        sys.exit(1)

    all_meta = []
    errors = 0
    for i, osz_path in enumerate(osz_files):
        if (i + 1) % 500 == 0:
            print(f"    Scanned {i + 1}/{len(osz_files)}...")
        try:
            with zipfile.ZipFile(osz_path, "r") as z:
                osu_files = [n for n in z.namelist() if n.endswith(".osu")]
                if not osu_files:
                    errors += 1
                    continue
                content = z.read(osu_files[0]).decode("utf-8", errors="replace")
                title = re.search(r"^Title:(.+)$", content, re.MULTILINE)
                artist = re.search(r"^Artist:(.+)$", content, re.MULTILINE)
                creator = re.search(r"^Creator:(.+)$", content, re.MULTILINE)
                mode = re.search(r"^Mode:\s*(\d+)", content, re.MULTILINE)
                if not (title and artist and creator):
                    errors += 1
                    continue
                all_meta.append({
                    "title": title.group(1).strip(),
                    "artist": artist.group(1).strip(),
                    "creator": creator.group(1).strip(),
                    "mode": int(mode.group(1)) if mode else 0,
                    "beatmapset_id": int(osz_path.stem) if osz_path.stem.isdigit() else hash(osz_path.stem),
                    "filename": osz_path.name,
                })
        except Exception:
            errors += 1

    print(f"  Metadata extracted: {len(all_meta)} (errors: {errors})")

    # Group by (title, artist)
    songs: dict[tuple, list] = defaultdict(list)
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

    print(f"  Unique songs: {len(songs)}")
    print(f"  Songs with 2+ mappers: {len(multi_mapper)}")
    total_sets = sum(len(s["beatmapsets"]) for s in multi_mapper)
    print(f"  Total beatmapsets in multi-mapper groups: {total_sets}")

    if len(multi_mapper) < MIN_MULTI_MAPPER_SONGS:
        print(f"\n  WARNING: Only {len(multi_mapper)} multi-mapper song(s) found.")
        print(f"  Need at least {MIN_MULTI_MAPPER_SONGS} for meaningful analysis.")
        if len(multi_mapper) == 0:
            print("  Cannot proceed. Provide a dataset with overlapping mapper coverage.")
            sys.exit(1)
        print("  Continuing with reduced dataset (results may not be statistically robust).")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index_path = DATA_DIR / "multi_mapper_index.json"
    with open(index_path, "w") as f:
        json.dump(multi_mapper, f, indent=2, ensure_ascii=False)
    print(f"  Saved index -> {index_path}  [{_elapsed(t0)}]")
    return multi_mapper


# ---------------------------------------------------------------------------
# Step 1b: Extract .osz archives for multi-mapper subset
# ---------------------------------------------------------------------------

def step1b_extract_osz(dataset_dir: Path, multi_mapper: list) -> None:
    """Extract .osz archives needed for multi-mapper analysis."""
    t0 = time.time()
    print(f"\n  Extracting .osz archives...")

    needed = {}
    for song in multi_mapper:
        for bs in song["beatmapsets"]:
            needed[bs["beatmapset_id"]] = bs

    extract_dir = DATA_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted = skipped = errors = 0
    for bsid, bs_info in sorted(needed.items()):
        out_dir = extract_dir / str(bsid)
        if out_dir.exists() and any(out_dir.glob("*.osu")):
            skipped += 1
            continue

        osz_path = dataset_dir / bs_info["filename"]
        if not osz_path.exists():
            # Try numeric stem
            osz_path = dataset_dir / f"{bsid}.osz"
        if not osz_path.exists():
            errors += 1
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(osz_path, "r") as z:
                for name in z.namelist():
                    if name.endswith(".osu") or name.endswith(".mp3") or name.endswith(".ogg"):
                        z.extract(name, out_dir)
            extracted += 1
        except Exception:
            errors += 1

    print(f"  Extracted: {extracted}, skipped: {skipped}, errors: {errors}  [{_elapsed(t0)}]")


# ---------------------------------------------------------------------------
# Step 2: Build beatmap registry
# ---------------------------------------------------------------------------

def step2_build_registry(multi_mapper: list) -> list:
    _banner(2, "Building beatmap registry")
    t0 = time.time()

    extract_dir = DATA_DIR / "extracted"
    index_path = DATA_DIR / "multi_mapper_index.json"

    # Build song group lookup
    song_lookup = {}
    for i, song in enumerate(multi_mapper):
        for bs in song["beatmapsets"]:
            song_lookup[bs["beatmapset_id"]] = {
                "song_group_idx": i,
                "song_title": song["title"],
                "song_artist": song["artist"],
                "num_mappers": song["num_mappers"],
            }

    registry = []
    mode_filtered = 0

    extract_dirs = sorted(d for d in extract_dir.iterdir() if d.is_dir())
    print(f"  Processing {len(extract_dirs)} extracted directories...")

    for d in extract_dirs:
        try:
            bsid = int(d.name)
        except ValueError:
            continue
        song_info = song_lookup.get(bsid, {})

        for osu_file in sorted(d.glob("*.osu")):
            meta = _parse_osu_metadata(osu_file)
            if meta is None:
                mode_filtered += 1
                continue
            meta["song_group_idx"] = song_info.get("song_group_idx")
            meta["num_mappers_in_group"] = song_info.get("num_mappers")
            registry.append(meta)

    print(f"  Registry entries (Mode 0): {len(registry)}")
    print(f"  Filtered (non-std mode): {mode_filtered}")

    # Select representatives: one per mapper per song (highest OD)
    groups: dict[tuple, list] = defaultdict(list)
    for r in registry:
        key = (r["song_group_idx"], r["creator"])
        groups[key].append(r)

    representatives = []
    for (song_idx, creator), beatmaps in groups.items():
        if song_idx is None:
            continue
        best = max(beatmaps, key=lambda b: b["od"] or 0)
        best["is_representative"] = True
        representatives.append(best)

    rep_paths = set(r["osu_path"] for r in representatives)
    for r in registry:
        r["is_representative"] = r["osu_path"] in rep_paths

    print(f"  Representatives selected: {len(representatives)}")

    # Save
    registry_path = DATA_DIR / "beatmap_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"  Saved registry -> {registry_path}  [{_elapsed(t0)}]")
    return registry


def _parse_osu_metadata(osu_path: Path) -> dict | None:
    try:
        content = osu_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    def get_field(name):
        m = re.search(rf"^{name}:\s*(.+)$", content, re.MULTILINE)
        return m.group(1).strip() if m else None

    def get_float(name):
        m = re.search(rf"^{name}:\s*([\d.]+)", content, re.MULTILINE)
        return float(m.group(1)) if m else None

    mode = get_field("Mode")
    if mode is None or int(mode) != 0:
        return None

    audio_filename = get_field("AudioFilename")
    audio_path = osu_path.parent / audio_filename if audio_filename else None

    return {
        "osu_path": str(osu_path),
        "beatmapset_id": int(osu_path.parent.name) if osu_path.parent.name.isdigit() else 0,
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


# ---------------------------------------------------------------------------
# Step 3: Encode to 9-dim signals
# ---------------------------------------------------------------------------

def step3_encode_9dim(registry: list) -> list:
    _banner(3, "Encoding representatives to 9-dim signals")
    t0 = time.time()

    from osu_dreamer.osu.beatmap import Beatmap
    from osu_dreamer.data.load_audio import load_audio, get_frame_times
    from osu_dreamer.data.beatmap.encode import encode_beatmap

    reps = [r for r in registry if r.get("is_representative")]
    print(f"  Representatives to encode: {len(reps)}")

    by_song: dict[int, list] = defaultdict(list)
    for r in reps:
        by_song[r["song_group_idx"]].append(r)

    enc_dir = DATA_DIR / "encodings_9dim"
    enc_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    encoded_count = error_count = alignment_issues = 0

    for song_idx, song_reps in sorted(by_song.items()):
        audio_info = {}
        for r in song_reps:
            audio_path = r.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                continue
            try:
                spec = load_audio(Path(audio_path))
                audio_info[r["osu_path"]] = {"spec": spec, "L": spec.shape[1], "audio_path": audio_path}
            except Exception:
                error_count += 1

        if not audio_info:
            continue

        frame_counts = [info["L"] for info in audio_info.values()]
        if len(set(frame_counts)) > 1:
            alignment_issues += 1
            target_L = min(frame_counts)
        else:
            target_L = frame_counts[0]

        frame_times = get_frame_times(target_L)

        for r in song_reps:
            if r["osu_path"] not in audio_info:
                continue
            try:
                bm = Beatmap(Path(r["osu_path"]))
                encoded = encode_beatmap(bm, frame_times)
                if encoded.shape[1] > target_L:
                    encoded = encoded[:, :target_L]
                elif encoded.shape[1] < target_L:
                    pad = target_L - encoded.shape[1]
                    encoded = np.pad(encoded, ((0, 0), (0, pad)), mode="constant")

                version_safe = sanitize_version(r["version"] or "unknown")
                filename = f"{r['beatmapset_id']}_{version_safe}.npy"
                np.save(enc_dir / filename, encoded)

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
                    print(f"    Encoded {encoded_count}...")
            except Exception as e:
                error_count += 1

    manifest_path = DATA_DIR / "encoding_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  Encoded: {encoded_count}, errors: {error_count}, alignment issues: {alignment_issues}")
    print(f"  Saved manifest -> {manifest_path}  [{_elapsed(t0)}]")
    return manifest


# ---------------------------------------------------------------------------
# Step 4: Track A -- cross-mapper analysis on 9-dim
# ---------------------------------------------------------------------------

def step4_track_a(manifest: list) -> dict:
    _banner(4, "Track A: Cross-mapper analysis (9-dim)")
    t0 = time.time()

    enc_dir = DATA_DIR / "encodings_9dim"
    by_song: dict[int, list] = defaultdict(list)
    for entry in manifest:
        by_song[entry["song_group_idx"]].append(entry)

    cross_mapper_results = {dim: [] for dim in DIM_NAMES}
    cross_mapper_cosine: list[float] = []
    songs_analyzed = pairs_compared = 0

    for song_idx, entries in sorted(by_song.items()):
        creators = set(e["creator"] for e in entries)
        if len(creators) < 2:
            continue

        loaded = {}
        for e in entries:
            path = enc_dir / e["filename"]
            if path.exists():
                loaded[e["filename"]] = {"data": np.load(path), "creator": e["creator"]}
        if len(loaded) < 2:
            continue

        min_L = min(v["data"].shape[1] for v in loaded.values())
        items = list(loaded.items())
        songs_analyzed += 1

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                _, info_a = items[i]
                _, info_b = items[j]
                if info_a["creator"] == info_b["creator"]:
                    continue
                enc_a = info_a["data"][:, :min_L]
                enc_b = info_b["data"][:, :min_L]
                for d, dim_name in enumerate(DIM_NAMES):
                    cross_mapper_results[dim_name].append({
                        "pearson": pearson_corr(enc_a[d], enc_b[d]),
                        "mad": mean_abs_diff(enc_a[d], enc_b[d]),
                        "cosine": cosine_sim(enc_a[d], enc_b[d]),
                    })
                frame_sims = [cosine_sim(enc_a[:, t], enc_b[:, t]) for t in range(0, min_L, 10)]
                cross_mapper_cosine.append(float(np.mean(frame_sims)))
                pairs_compared += 1

    print(f"  Songs analyzed: {songs_analyzed}")
    print(f"  Cross-mapper pairs: {pairs_compared}")

    # Aggregate
    summary: dict = {}
    for dim_name in DIM_NAMES:
        results = cross_mapper_results[dim_name]
        if not results:
            continue
        pearsons = [r["pearson"] for r in results]
        mads = [r["mad"] for r in results]
        cosines = [r["cosine"] for r in results]
        summary[dim_name] = {
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_std": float(np.std(pearsons)),
            "mad_mean": float(np.mean(mads)),
            "mad_std": float(np.std(mads)),
            "cosine_mean": float(np.mean(cosines)),
            "cosine_std": float(np.std(cosines)),
            "n_pairs": len(results),
        }

    # Print ranked
    print(f"\n  {'Dimension':<12} {'Pearson':>10}   {'Label'}")
    print(f"  {'-' * 40}")
    ranked = sorted(summary.items(), key=lambda x: -x[1]["pearson_mean"])
    for dim_name, stats in ranked:
        label = "PERCEPTUAL" if stats["pearson_mean"] > 0.3 else ("STYLISTIC" if stats["pearson_mean"] < 0.1 else "MIXED")
        print(f"  {dim_name:<12} {stats['pearson_mean']:>6.3f} +/- {stats['pearson_std']:.3f}  [{label}]")

    if cross_mapper_cosine:
        print(f"\n  Full 9-dim cosine: {np.mean(cross_mapper_cosine):.3f} +/- {np.std(cross_mapper_cosine):.3f}")

    # Save
    track_a_dir = RESULTS_DIR / "track_a"
    track_a_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "cross_mapper_cosine_mean": float(np.mean(cross_mapper_cosine)) if cross_mapper_cosine else None,
        "cross_mapper_cosine_std": float(np.std(cross_mapper_cosine)) if cross_mapper_cosine else None,
        "songs_analyzed": songs_analyzed,
        "pairs_compared": pairs_compared,
    }
    results_path = track_a_dir / "track_a_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    # Plot
    _plot_track_a(summary, ranked, track_a_dir)
    print(f"  Saved -> {results_path}  [{_elapsed(t0)}]")
    return output


def _plot_track_a(summary, ranked, track_a_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping Track A plot")
        return

    dims_sorted = [d for d, _ in ranked]
    pearsons_sorted = [summary[d]["pearson_mean"] for d in dims_sorted]
    pearson_errs = [summary[d]["pearson_std"] for d in dims_sorted]
    color_map = {
        "ONSET": "#2196F3", "COMBO": "#2196F3",
        "SLIDE": "#4CAF50", "SUSTAIN": "#4CAF50",
        "WHISTLE": "#FF9800", "FINISH": "#FF9800", "CLAP": "#FF9800",
        "X": "#9C27B0", "Y": "#9C27B0",
    }
    colors = [color_map.get(d, "#999") for d in dims_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(dims_sorted)), pearsons_sorted, xerr=pearson_errs,
            color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(range(len(dims_sorted)))
    ax.set_yticklabels(dims_sorted)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title("Track A: 9-Dim Cross-Mapper Agreement")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(track_a_dir / "cross_mapper_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Step 5: Encode through VAE
# ---------------------------------------------------------------------------

def step5_encode_latent(manifest: list) -> list:
    _banner(5, "Encoding through pre-trained VAE")
    t0 = time.time()

    import torch as th
    from osu_dreamer.latent_model.model import Model

    if not CHECKPOINT_PATH.exists():
        print(f"  ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("  Skipping VAE encoding (Track B and Synthesis will be unavailable).")
        return []

    print(f"  Loading model from {CHECKPOINT_PATH}")
    model = Model.load_from_checkpoint(str(CHECKPOINT_PATH))
    model.eval()
    device = "mps" if th.backends.mps.is_available() else ("cuda" if th.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    enc_9dim_dir = DATA_DIR / "encodings_9dim"
    latent_dir = DATA_DIR / "encodings_latent"
    latent_dir.mkdir(parents=True, exist_ok=True)

    latent_manifest = []
    encoded_count = error_count = 0

    for entry in manifest:
        src_path = enc_9dim_dir / entry["filename"]
        if not src_path.exists():
            error_count += 1
            continue
        try:
            enc_9dim = np.load(src_path)
            chart = th.tensor(enc_9dim).float().unsqueeze(0).to(device)
            with th.no_grad():
                z = model.encode(chart)
            z_np = z.squeeze(0).cpu().numpy()
            np.save(latent_dir / entry["filename"], z_np)
            latent_manifest.append({**entry, "latent_shape": list(z_np.shape), "orig_L": enc_9dim.shape[1], "latent_l": z_np.shape[1]})
            encoded_count += 1
            if encoded_count % 50 == 0:
                print(f"    Encoded {encoded_count}/{len(manifest)}...")
        except Exception:
            error_count += 1

    latent_manifest_path = DATA_DIR / "latent_manifest.json"
    with open(latent_manifest_path, "w") as f:
        json.dump(latent_manifest, f, indent=2, ensure_ascii=False)

    print(f"  Encoded: {encoded_count}, errors: {error_count}")
    print(f"  Saved manifest -> {latent_manifest_path}  [{_elapsed(t0)}]")
    return latent_manifest


# ---------------------------------------------------------------------------
# Step 6: Track B -- cross-mapper analysis on 32-dim latent
# ---------------------------------------------------------------------------

def step6_track_b(latent_manifest: list) -> dict | None:
    _banner(6, "Track B: Cross-mapper analysis (32-dim latent)")
    t0 = time.time()

    if not latent_manifest:
        print("  Skipped (no latent encodings available).")
        return None

    latent_dir = DATA_DIR / "encodings_latent"
    by_song: dict[int, list] = defaultdict(list)
    for entry in latent_manifest:
        by_song[entry["song_group_idx"]].append(entry)

    cross_mapper_pearson = {d: [] for d in range(N_LATENT)}
    cross_mapper_cosine_per_dim = {d: [] for d in range(N_LATENT)}
    cross_mapper_full_cosine: list[float] = []
    song_between_var = {d: [] for d in range(N_LATENT)}

    songs_analyzed = pairs_compared = 0

    for song_idx, entries in sorted(by_song.items()):
        creators = set(e["creator"] for e in entries)
        if len(creators) < 2:
            continue

        loaded = {}
        for e in entries:
            path = latent_dir / e["filename"]
            if path.exists():
                loaded[e["filename"]] = {"data": np.load(path), "creator": e["creator"]}
        if len(loaded) < 2:
            continue

        min_l = min(v["data"].shape[1] for v in loaded.values())
        if min_l < 2:
            continue

        items = list(loaded.items())
        songs_analyzed += 1

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                _, info_a = items[i]
                _, info_b = items[j]
                if info_a["creator"] == info_b["creator"]:
                    continue
                z_a = info_a["data"][:, :min_l]
                z_b = info_b["data"][:, :min_l]
                for d in range(N_LATENT):
                    cross_mapper_pearson[d].append(pearson_corr(z_a[d], z_b[d]))
                    cross_mapper_cosine_per_dim[d].append(cosine_sim(z_a[d], z_b[d]))
                step = max(1, min_l // 100)
                frame_sims = [cosine_sim(z_a[:, t], z_b[:, t]) for t in range(0, min_l, step)]
                cross_mapper_full_cosine.append(float(np.mean(frame_sims)))
                pairs_compared += 1

        # Variance decomposition
        by_creator: dict[str, list] = defaultdict(list)
        for _, info in items:
            by_creator[info["creator"]].append(info["data"][:, :min_l])
        for d in range(N_LATENT):
            creator_means = [np.mean([arr[d].mean() for arr in arrs]) for arrs in by_creator.values()]
            if len(creator_means) >= 2:
                song_between_var[d].append(np.var(creator_means))

    print(f"  Songs analyzed: {songs_analyzed}")
    print(f"  Cross-mapper pairs: {pairs_compared}")

    if pairs_compared == 0:
        print("  No pairs to compare. Skipping Track B.")
        return None

    dim_summary = {}
    for d in range(N_LATENT):
        pearsons = cross_mapper_pearson[d]
        cosines = cross_mapper_cosine_per_dim[d]
        bvar = song_between_var[d]
        if not pearsons:
            continue
        dim_summary[d] = {
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_std": float(np.std(pearsons)),
            "cosine_mean": float(np.mean(cosines)),
            "cosine_std": float(np.std(cosines)),
            "between_var_mean": float(np.mean(bvar)) if bvar else 0.0,
            "n_pairs": len(pearsons),
        }

    ranked = sorted(dim_summary.items(), key=lambda x: -x[1]["pearson_mean"])

    perceptual = sum(1 for _, s in dim_summary.items() if s["pearson_mean"] > 0.3)
    mixed = sum(1 for _, s in dim_summary.items() if 0.05 <= s["pearson_mean"] <= 0.3)
    stylistic = sum(1 for _, s in dim_summary.items() if s["pearson_mean"] < 0.05)
    print(f"\n  Taxonomy: {perceptual} perceptual, {mixed} mixed, {stylistic} stylistic")

    if cross_mapper_full_cosine:
        print(f"  Full 32-dim cosine: {np.mean(cross_mapper_full_cosine):.3f} +/- {np.std(cross_mapper_full_cosine):.3f}")

    # Save
    track_b_dir = RESULTS_DIR / "track_b"
    track_b_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dim_summary": {str(k): v for k, v in dim_summary.items()},
        "dim_ranking": [{"dim": d, **s} for d, s in ranked],
        "full_cosine_mean": float(np.mean(cross_mapper_full_cosine)) if cross_mapper_full_cosine else None,
        "full_cosine_std": float(np.std(cross_mapper_full_cosine)) if cross_mapper_full_cosine else None,
        "songs_analyzed": songs_analyzed,
        "pairs_compared": pairs_compared,
        "taxonomy": {"perceptual": perceptual, "mixed": mixed, "stylistic": stylistic},
    }
    results_path = track_b_dir / "track_b_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    _plot_track_b(ranked, dim_summary, cross_mapper_full_cosine, track_b_dir)
    print(f"  Saved -> {results_path}  [{_elapsed(t0)}]")
    return output


def _plot_track_b(ranked, dim_summary, cross_mapper_full_cosine, track_b_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dims_sorted = [f"z_{d}" for d, _ in ranked]
    pearsons_sorted = [s["pearson_mean"] for _, s in ranked]
    pearson_errs = [s["pearson_std"] for _, s in ranked]
    colors = ["#2196F3" if s["pearson_mean"] > 0.3 else ("#F44336" if s["pearson_mean"] < 0.05 else "#FF9800") for _, s in ranked]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(dims_sorted)), pearsons_sorted, xerr=pearson_errs, color=colors, alpha=0.8, capsize=2)
    ax.set_yticks(range(len(dims_sorted)))
    ax.set_yticklabels(dims_sorted, fontsize=8)
    ax.set_xlabel("Pearson Correlation (cross-mapper)")
    ax.set_title("Track B: VAE Latent Dimensions Cross-Mapper Agreement")
    ax.axvline(x=0.3, color="blue", linestyle="--", alpha=0.5, label="Perceptual threshold")
    ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.5, label="Stylistic threshold")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(track_b_dir / "latent_dim_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()

    if cross_mapper_full_cosine:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(cross_mapper_full_cosine, bins=40, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(x=np.mean(cross_mapper_full_cosine), color="red", linestyle="--",
                   label=f"Mean: {np.mean(cross_mapper_full_cosine):.3f}")
        ax.set_xlabel("32-dim Cosine Similarity")
        ax.set_ylabel("Count (mapper pairs)")
        ax.set_title("Cross-Mapper Latent Similarity Distribution")
        ax.legend()
        plt.tight_layout()
        fig.savefig(track_b_dir / "latent_cosine_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Step 7: Synthesis -- 32x9 correlation matrix
# ---------------------------------------------------------------------------

def step7_synthesis(track_a_output: dict, track_b_output: dict | None, manifest_9dim: list, latent_manifest: list) -> dict | None:
    _banner(7, "Synthesis: Connecting 9-dim and latent spaces")
    t0 = time.time()

    if track_b_output is None or not latent_manifest:
        print("  Skipped (Track B results not available).")
        return None

    enc_9dim_dir = DATA_DIR / "encodings_9dim"
    enc_latent_dir = DATA_DIR / "encodings_latent"

    latent_files = {e["filename"] for e in latent_manifest}
    both = [e for e in manifest_9dim if e["filename"] in latent_files]
    print(f"  Beatmaps with both encodings: {len(both)}")

    corr_matrix = np.zeros((N_LATENT, len(DIM_NAMES)))
    corr_counts = 0

    for entry in both:
        try:
            enc_9 = np.load(enc_9dim_dir / entry["filename"])
            enc_z = np.load(enc_latent_dir / entry["filename"])
            L = enc_9.shape[1]
            z_up = np.repeat(enc_z, 18, axis=1)[:, :L]
            for zi in range(N_LATENT):
                for di in range(len(DIM_NAMES)):
                    z_sig = z_up[zi]
                    d_sig = enc_9[di]
                    if np.std(z_sig) > 0 and np.std(d_sig) > 0 and len(z_sig) >= 2:
                        r = np.corrcoef(z_sig, d_sig)[0, 1]
                        if not np.isnan(r):
                            corr_matrix[zi, di] += r
            corr_counts += 1
        except Exception:
            pass

    if corr_counts > 0:
        corr_matrix /= corr_counts
    print(f"  Correlation matrix from {corr_counts} beatmaps")

    # Taxonomy
    track_b_ranking = track_b_output["dim_ranking"]
    taxonomy = []
    for item in track_b_ranking:
        d = item["dim"]
        p = item["pearson_mean"]
        z_corrs = corr_matrix[d]
        top_idx = np.argsort(np.abs(z_corrs))[::-1]
        top_3 = [(DIM_NAMES[i], float(z_corrs[i])) for i in top_idx[:3]]
        cat = "PERCEPTUAL" if p > 0.3 else ("STYLISTIC" if p < 0.05 else "MIXED")
        taxonomy.append({
            "dim": d, "category": cat, "cross_mapper_pearson": p,
            "top_9dim_correlations": top_3,
            "strongest_9dim": top_3[0][0], "strongest_9dim_corr": top_3[0][1],
        })

    # Save
    synth_dir = RESULTS_DIR / "synthesis"
    synth_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "correlation_matrix": corr_matrix.tolist(),
        "dim_names_9": DIM_NAMES,
        "n_latent": N_LATENT,
        "n_beatmaps_used": corr_counts,
        "taxonomy": taxonomy,
        "summary": {
            "track_a_songs": track_a_output["songs_analyzed"],
            "track_a_pairs": track_a_output["pairs_compared"],
            "track_b_songs": track_b_output["songs_analyzed"],
            "track_b_pairs": track_b_output["pairs_compared"],
            "track_a_cosine": track_a_output.get("cross_mapper_cosine_mean"),
            "track_b_cosine": track_b_output.get("full_cosine_mean"),
        },
    }
    results_path = synth_dir / "synthesis_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    _plot_synthesis(corr_matrix, taxonomy, track_a_output, track_b_output, synth_dir)
    print(f"  Saved -> {results_path}  [{_elapsed(t0)}]")
    return output


def _plot_synthesis(corr_matrix, taxonomy, track_a_output, track_b_output, synth_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return

    # Correlation heatmap
    dim_order = [t["dim"] for t in taxonomy]
    matrix_sorted = corr_matrix[dim_order, :]
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(matrix_sorted, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(DIM_NAMES)))
    ax.set_xticklabels(DIM_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(N_LATENT))
    ax.set_yticklabels([f"z_{d}" for d in dim_order], fontsize=7)
    ax.set_xlabel("Interpretable Dimension (9-dim)")
    ax.set_ylabel("Latent Dimension (sorted by cross-mapper agreement)")
    ax.set_title("Latent <-> Interpretable Correlation Matrix")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.7)
    plt.tight_layout()
    fig.savefig(synth_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    track_a_summary = track_a_output["summary"]
    a_dims = sorted(track_a_summary.keys(), key=lambda d: -track_a_summary[d]["pearson_mean"])
    a_pearsons = [track_a_summary[d]["pearson_mean"] for d in a_dims]
    a_errs = [track_a_summary[d]["pearson_std"] for d in a_dims]
    a_color_map = {
        "ONSET": "#2196F3", "COMBO": "#2196F3",
        "SLIDE": "#4CAF50", "SUSTAIN": "#4CAF50",
        "WHISTLE": "#FF9800", "FINISH": "#FF9800", "CLAP": "#FF9800",
        "X": "#9C27B0", "Y": "#9C27B0",
    }
    a_colors = [a_color_map.get(d, "#999") for d in a_dims]

    ax1.barh(range(len(a_dims)), a_pearsons, xerr=a_errs, color=a_colors, alpha=0.8, capsize=3)
    ax1.set_yticks(range(len(a_dims)))
    ax1.set_yticklabels(a_dims)
    ax1.set_xlabel("Pearson Correlation")
    ax1.set_title("Track A: 9-Dim Cross-Mapper Agreement")
    ax1.axvline(x=0, color="gray", linewidth=0.5)
    ax1.invert_yaxis()

    b_top = taxonomy[:9]
    b_dims = [f"z_{t['dim']}" for t in b_top]
    b_pearsons = [t["cross_mapper_pearson"] for t in b_top]
    b_colors = ["#2196F3" if t["category"] == "PERCEPTUAL" else ("#F44336" if t["category"] == "STYLISTIC" else "#FF9800") for t in b_top]

    ax2.barh(range(len(b_dims)), b_pearsons, color=b_colors, alpha=0.8)
    ax2.set_yticks(range(len(b_dims)))
    ax2.set_yticklabels(b_dims)
    ax2.set_xlabel("Pearson Correlation")
    ax2.set_title("Track B: Top-9 Latent Dim Agreement")
    ax2.axvline(x=0, color="gray", linewidth=0.5)
    ax2.invert_yaxis()

    max_x = max(max(a_pearsons), max(b_pearsons)) * 1.2
    ax1.set_xlim(-0.05, max_x)
    ax2.set_xlim(-0.05, max_x)
    plt.tight_layout()
    fig.savefig(synth_dir / "track_a_vs_b_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Step 8: Generate all remaining plots (already done inline above)
# ---------------------------------------------------------------------------

def step8_plots():
    _banner(8, "Generating summary plots")
    # Plots are generated inline in steps 4, 6, 7.
    # This step just confirms everything was produced.
    plot_files = list(RESULTS_DIR.rglob("*.png"))
    print(f"  Total plot files generated: {len(plot_files)}")
    for p in sorted(plot_files):
        print(f"    {p.relative_to(OUTPUT_DIR)}")


# ---------------------------------------------------------------------------
# Step 9: Print summary
# ---------------------------------------------------------------------------

def step9_summary(track_a: dict, track_b: dict | None, synthesis: dict | None):
    _banner(9, "Summary")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"\n  Track A (9-dim interpretable signals):")
    print(f"    Songs: {track_a['songs_analyzed']}, Pairs: {track_a['pairs_compared']}")
    if track_a.get("summary"):
        ranked = sorted(track_a["summary"].items(), key=lambda x: -x[1]["pearson_mean"])
        for dim, stats in ranked[:3]:
            print(f"    Top: {dim} = {stats['pearson_mean']:.3f}")

    if track_b:
        print(f"\n  Track B (32-dim VAE latent):")
        print(f"    Songs: {track_b['songs_analyzed']}, Pairs: {track_b['pairs_compared']}")
        tax = track_b.get("taxonomy", {})
        print(f"    Perceptual: {tax.get('perceptual', '?')}/32, Mixed: {tax.get('mixed', '?')}/32, Stylistic: {tax.get('stylistic', '?')}/32")
    else:
        print(f"\n  Track B: Skipped (no checkpoint or latent encodings)")

    if synthesis:
        print(f"\n  Synthesis:")
        print(f"    Correlation matrix: {synthesis['n_latent']}x{len(synthesis['dim_names_9'])}")
        print(f"    Computed from {synthesis['n_beatmaps_used']} beatmaps")
    else:
        print(f"\n  Synthesis: Skipped")

    print(f"\n  All results saved to: {OUTPUT_DIR}/")
    print(f"  Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: PYTHONPATH=. python experiment/run_analysis.py /path/to/osz/folder/")
        print()
        print("Provide a directory containing .osz beatmap archives.")
        print("For meaningful results, include songs mapped by multiple different mappers.")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} is not a directory")
        sys.exit(1)

    print(f"Latent Cartography Analysis Pipeline")
    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}" + (" [found]" if CHECKPOINT_PATH.exists() else " [NOT FOUND]"))

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build multi-mapper index + extract
    multi_mapper = step1_build_index(dataset_dir)
    step1b_extract_osz(dataset_dir, multi_mapper)

    # Step 2: Build registry
    registry = step2_build_registry(multi_mapper)

    # Step 3: Encode to 9-dim
    manifest_9dim = step3_encode_9dim(registry)

    if not manifest_9dim:
        print("\nERROR: No beatmaps could be encoded. Check that audio files are present.")
        sys.exit(1)

    # Step 4: Track A
    track_a_output = step4_track_a(manifest_9dim)

    # Step 5: VAE encoding
    latent_manifest = step5_encode_latent(manifest_9dim)

    # Step 6: Track B
    track_b_output = step6_track_b(latent_manifest)

    # Step 7: Synthesis
    synthesis_output = step7_synthesis(track_a_output, track_b_output, manifest_9dim, latent_manifest)

    # Step 8: Plots summary
    step8_plots()

    # Step 9: Final summary
    step9_summary(track_a_output, track_b_output, synthesis_output)


if __name__ == "__main__":
    main()
