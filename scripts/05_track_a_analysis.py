"""
Step 5: Cross-mapper comparison on 9-dim beatmap encodings (Track A).

For each song with multiple mappers, compares the representative
beatmaps across mappers using per-dimension similarity metrics.

Usage:
    uv run python scripts/05_track_a_analysis.py

Input:  experiment/output/data/encoding_manifest.json
        experiment/output/data/encodings_9dim/
Output: experiment/output/results/track_a/track_a_results.json
        experiment/output/results/track_a/cross_mapper_agreement.png
"""

import json
import warnings
import numpy as np
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "experiment" / "output" / "data"
RESULTS_DIR = REPO_ROOT / "experiment" / "output" / "results" / "track_a"
MANIFEST_PATH = DATA_DIR / "encoding_manifest.json"
ENCODINGS_DIR = DATA_DIR / "encodings_9dim"

DIM_NAMES = ["ONSET", "COMBO", "SLIDE", "SUSTAIN", "WHISTLE", "FINISH", "CLAP", "X", "Y"]


def pearson_corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a - b)))


def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    by_song = defaultdict(list)
    for entry in manifest:
        by_song[entry["song_group_idx"]].append(entry)

    print(f"Song groups with encodings: {len(by_song)}")
    print(f"Total encoded beatmaps: {len(manifest)}")

    cross_mapper_results = {dim: [] for dim in DIM_NAMES}
    cross_mapper_cosine = []

    songs_analyzed = 0
    pairs_compared = 0

    for song_idx, entries in sorted(by_song.items()):
        creators = set(e["creator"] for e in entries)
        if len(creators) < 2:
            continue

        loaded = {}
        for e in entries:
            path = ENCODINGS_DIR / e["filename"]
            if path.exists():
                loaded[e["filename"]] = {
                    "data": np.load(path),
                    "creator": e["creator"],
                }

        if len(loaded) < 2:
            continue

        min_L = min(v["data"].shape[1] for v in loaded.values())
        items = list(loaded.items())
        songs_analyzed += 1

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                fname_a, info_a = items[i]
                fname_b, info_b = items[j]

                if info_a["creator"] == info_b["creator"]:
                    continue

                enc_a = info_a["data"][:, :min_L]
                enc_b = info_b["data"][:, :min_L]

                for d, dim_name in enumerate(DIM_NAMES):
                    sig_a = enc_a[d]
                    sig_b = enc_b[d]
                    cross_mapper_results[dim_name].append({
                        "pearson": pearson_corr(sig_a, sig_b),
                        "mad": mean_abs_diff(sig_a, sig_b),
                        "cosine": cosine_sim(sig_a, sig_b),
                    })

                frame_sims = []
                for t in range(0, min_L, 10):
                    frame_sims.append(cosine_sim(enc_a[:, t], enc_b[:, t]))
                cross_mapper_cosine.append(np.mean(frame_sims))

                pairs_compared += 1

    print(f"\nSongs analyzed: {songs_analyzed}")
    print(f"Cross-mapper pairs compared: {pairs_compared}")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"CROSS-MAPPER AGREEMENT BY DIMENSION")
    print(f"{'='*70}")
    print(f"{'Dimension':<12} {'Pearson (mean±std)':<24} {'MAD (mean±std)':<24} {'Cosine (mean±std)':<24}")
    print(f"{'-'*70}")

    summary = {}
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

        print(f"{dim_name:<12} "
              f"{np.mean(pearsons):>6.3f} ± {np.std(pearsons):.3f}       "
              f"{np.mean(mads):>6.3f} ± {np.std(mads):.3f}       "
              f"{np.mean(cosines):>6.3f} ± {np.std(cosines):.3f}")

    # Ranked
    print(f"\n{'='*70}")
    print(f"DIMENSIONS RANKED BY CROSS-MAPPER AGREEMENT (Pearson)")
    print(f"{'='*70}")
    ranked = sorted(summary.items(), key=lambda x: -x[1]["pearson_mean"])
    for rank, (dim_name, stats) in enumerate(ranked, 1):
        bar = "█" * int(max(0, stats["pearson_mean"]) * 40)
        label = "PERCEPTUAL" if stats["pearson_mean"] > 0.3 else "STYLISTIC" if stats["pearson_mean"] < 0.1 else "MIXED"
        print(f"  {rank}. {dim_name:<10} {stats['pearson_mean']:>6.3f}  {bar}  [{label}]")

    if cross_mapper_cosine:
        print(f"\nFull 9-dim cosine similarity: {np.mean(cross_mapper_cosine):.3f} ± {np.std(cross_mapper_cosine):.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "cross_mapper_cosine_mean": float(np.mean(cross_mapper_cosine)) if cross_mapper_cosine else None,
        "cross_mapper_cosine_std": float(np.std(cross_mapper_cosine)) if cross_mapper_cosine else None,
        "songs_analyzed": songs_analyzed,
        "pairs_compared": pairs_compared,
    }

    with open(RESULTS_DIR / "track_a_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'track_a_results.json'}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        dims = [d for d in DIM_NAMES if d in summary]
        pearsons = [summary[d]["pearson_mean"] for d in dims]
        pearson_errs = [summary[d]["pearson_std"] for d in dims]
        mads = [summary[d]["mad_mean"] for d in dims]
        mad_errs = [summary[d]["mad_std"] for d in dims]
        cosines = [summary[d]["cosine_mean"] for d in dims]
        cosine_errs = [summary[d]["cosine_std"] for d in dims]

        colors = []
        for d in dims:
            if d in ("ONSET", "COMBO"):
                colors.append("#2196F3")
            elif d in ("SLIDE", "SUSTAIN"):
                colors.append("#4CAF50")
            elif d in ("WHISTLE", "FINISH", "CLAP"):
                colors.append("#FF9800")
            else:
                colors.append("#9C27B0")

        order = sorted(range(len(dims)), key=lambda i: -pearsons[i])
        dims_sorted = [dims[i] for i in order]
        pearsons_sorted = [pearsons[i] for i in order]
        pearson_errs_sorted = [pearson_errs[i] for i in order]
        colors_sorted = [colors[i] for i in order]

        axes[0].barh(range(len(dims_sorted)), pearsons_sorted, xerr=pearson_errs_sorted,
                     color=colors_sorted, alpha=0.8, capsize=3)
        axes[0].set_yticks(range(len(dims_sorted)))
        axes[0].set_yticklabels(dims_sorted)
        axes[0].set_xlabel("Pearson Correlation")
        axes[0].set_title("Cross-Mapper Agreement by Dimension")
        axes[0].axvline(x=0, color="gray", linewidth=0.5)

        mads_sorted = [mads[order[i]] for i in range(len(dims))]
        mad_errs_sorted = [mad_errs[order[i]] for i in range(len(dims))]
        axes[1].barh(range(len(dims_sorted)), mads_sorted, xerr=mad_errs_sorted,
                     color=colors_sorted, alpha=0.8, capsize=3)
        axes[1].set_yticks(range(len(dims_sorted)))
        axes[1].set_yticklabels(dims_sorted)
        axes[1].set_xlabel("Mean Absolute Difference")
        axes[1].set_title("Cross-Mapper Divergence by Dimension")

        cosines_sorted = [cosines[order[i]] for i in range(len(dims))]
        cosine_errs_sorted = [cosine_errs[order[i]] for i in range(len(dims))]
        axes[2].barh(range(len(dims_sorted)), cosines_sorted, xerr=cosine_errs_sorted,
                     color=colors_sorted, alpha=0.8, capsize=3)
        axes[2].set_yticks(range(len(dims_sorted)))
        axes[2].set_yticklabels(dims_sorted)
        axes[2].set_xlabel("Cosine Similarity")
        axes[2].set_title("Cross-Mapper Cosine Similarity by Dimension")

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "cross_mapper_agreement.png", dpi=150, bbox_inches="tight")
        print(f"Plot saved to {RESULTS_DIR / 'cross_mapper_agreement.png'}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plots")


if __name__ == "__main__":
    if not MANIFEST_PATH.exists():
        print(f"Error: Run 04_encode_9dim.py first — missing {MANIFEST_PATH}")
        exit(1)

    analyze()
