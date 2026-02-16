"""
Phase 2.2: Cross-mapper comparison on 9-dim beatmap encodings.

For each song with multiple mappers, compares the representative
beatmaps across mappers using per-dimension similarity metrics.

Also computes within-mapper baseline (different difficulties by
same mapper) for comparison.

Outputs analysis results and figures to results/track_a/.

Run with osu-dreamer venv:
  PYTHONPATH=.../osu-dreamer .../osu-dreamer/.venv/bin/python 05_track_a_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "track_a"
MANIFEST_PATH = DATA_DIR / "encoding_manifest.json"
INDEX_PATH = DATA_DIR / "multi_mapper_index.json"
ENCODINGS_DIR = DATA_DIR / "encodings_9dim"

DIM_NAMES = ["ONSET", "COMBO", "SLIDE", "SUSTAIN", "WHISTLE", "FINISH", "CLAP", "X", "Y"]


def pearson_corr(a, b):
    """Pearson correlation between two 1D arrays."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def mean_abs_diff(a, b):
    """Mean absolute difference between two 1D arrays."""
    return float(np.mean(np.abs(a - b)))


def cosine_sim(a, b):
    """Cosine similarity between two 1D arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(INDEX_PATH) as f:
        index = json.load(f)

    # Group manifest entries by song_group_idx
    by_song = defaultdict(list)
    for entry in manifest:
        by_song[entry["song_group_idx"]].append(entry)

    print(f"Song groups with encodings: {len(by_song)}")
    print(f"Total encoded beatmaps: {len(manifest)}")

    # Cross-mapper comparison: for each song, compare all pairs of different mappers
    cross_mapper_results = {dim: [] for dim in DIM_NAMES}
    cross_mapper_cosine = []  # Full 9-dim cosine similarity

    songs_analyzed = 0
    pairs_compared = 0

    for song_idx, entries in sorted(by_song.items()):
        # Only compare if 2+ different mappers
        creators = set(e["creator"] for e in entries)
        if len(creators) < 2:
            continue

        # Load encodings
        loaded = {}
        for e in entries:
            path = ENCODINGS_DIR / e["filename"]
            if path.exists():
                loaded[e["filename"]] = {
                    "data": np.load(path),
                    "creator": e["creator"],
                    "version": e["version"],
                    "beatmapset_id": e["beatmapset_id"],
                }

        if len(loaded) < 2:
            continue

        # Find minimum length for alignment
        min_L = min(v["data"].shape[1] for v in loaded.values())

        # Compare all pairs of different mappers
        items = list(loaded.items())
        songs_analyzed += 1

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                fname_a, info_a = items[i]
                fname_b, info_b = items[j]

                if info_a["creator"] == info_b["creator"]:
                    continue  # Skip same-mapper pairs

                enc_a = info_a["data"][:, :min_L]
                enc_b = info_b["data"][:, :min_L]

                # Per-dimension metrics
                for d, dim_name in enumerate(DIM_NAMES):
                    sig_a = enc_a[d]
                    sig_b = enc_b[d]
                    cross_mapper_results[dim_name].append({
                        "pearson": pearson_corr(sig_a, sig_b),
                        "mad": mean_abs_diff(sig_a, sig_b),
                        "cosine": cosine_sim(sig_a, sig_b),
                    })

                # Full 9-dim cosine similarity (averaged over time)
                frame_sims = []
                for t in range(0, min_L, 10):  # Sample every 10th frame for speed
                    frame_sims.append(cosine_sim(enc_a[:, t], enc_b[:, t]))
                cross_mapper_cosine.append(np.mean(frame_sims))

                pairs_compared += 1

    print(f"\nSongs analyzed: {songs_analyzed}")
    print(f"Cross-mapper pairs compared: {pairs_compared}")

    # Aggregate results
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

    # Rank dimensions by agreement
    print(f"\n{'='*70}")
    print(f"DIMENSIONS RANKED BY CROSS-MAPPER AGREEMENT (Pearson)")
    print(f"{'='*70}")
    ranked = sorted(summary.items(), key=lambda x: -x[1]["pearson_mean"])
    for rank, (dim_name, stats) in enumerate(ranked, 1):
        bar = "█" * int(max(0, stats["pearson_mean"]) * 40)
        label = "PERCEPTUAL" if stats["pearson_mean"] > 0.3 else "STYLISTIC" if stats["pearson_mean"] < 0.1 else "MIXED"
        print(f"  {rank}. {dim_name:<10} {stats['pearson_mean']:>6.3f}  {bar}  [{label}]")

    # Full 9-dim cosine
    if cross_mapper_cosine:
        print(f"\nFull 9-dim cosine similarity (cross-mapper): {np.mean(cross_mapper_cosine):.3f} ± {np.std(cross_mapper_cosine):.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "cross_mapper_cosine_mean": float(np.mean(cross_mapper_cosine)) if cross_mapper_cosine else None,
        "cross_mapper_cosine_std": float(np.std(cross_mapper_cosine)) if cross_mapper_cosine else None,
        "songs_analyzed": songs_analyzed,
        "pairs_compared": pairs_compared,
    }

    with open(RESULTS_DIR / "track_a_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'track_a_results.json'}")

    # Generate plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Bar chart: Pearson correlation by dimension
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        dims = [d for d in DIM_NAMES if d in summary]
        pearsons = [summary[d]["pearson_mean"] for d in dims]
        pearson_errs = [summary[d]["pearson_std"] for d in dims]
        mads = [summary[d]["mad_mean"] for d in dims]
        mad_errs = [summary[d]["mad_std"] for d in dims]
        cosines = [summary[d]["cosine_mean"] for d in dims]
        cosine_errs = [summary[d]["cosine_std"] for d in dims]

        # Color by category
        colors = []
        for d in dims:
            if d in ("ONSET", "COMBO"):
                colors.append("#2196F3")  # Blue = timing
            elif d in ("SLIDE", "SUSTAIN"):
                colors.append("#4CAF50")  # Green = structure
            elif d in ("WHISTLE", "FINISH", "CLAP"):
                colors.append("#FF9800")  # Orange = hitsounds
            else:
                colors.append("#9C27B0")  # Purple = spatial

        # Sort by pearson for the plot
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
        axes[0].axvline(x=0, color='gray', linewidth=0.5)

        # MAD plot
        mads_sorted = [mads[order[i]] for i in range(len(dims))]
        mad_errs_sorted = [mad_errs[order[i]] for i in range(len(dims))]
        axes[1].barh(range(len(dims_sorted)), mads_sorted, xerr=mad_errs_sorted,
                     color=colors_sorted, alpha=0.8, capsize=3)
        axes[1].set_yticks(range(len(dims_sorted)))
        axes[1].set_yticklabels(dims_sorted)
        axes[1].set_xlabel("Mean Absolute Difference")
        axes[1].set_title("Cross-Mapper Divergence by Dimension")

        # Cosine plot
        cosines_sorted = [cosines[order[i]] for i in range(len(dims))]
        cosine_errs_sorted = [cosine_errs[order[i]] for i in range(len(dims))]
        axes[2].barh(range(len(dims_sorted)), cosines_sorted, xerr=cosine_errs_sorted,
                     color=colors_sorted, alpha=0.8, capsize=3)
        axes[2].set_yticks(range(len(dims_sorted)))
        axes[2].set_yticklabels(dims_sorted)
        axes[2].set_xlabel("Cosine Similarity")
        axes[2].set_title("Cross-Mapper Cosine Similarity by Dimension")

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "cross_mapper_agreement.png", dpi=150, bbox_inches='tight')
        print(f"Plot saved to {RESULTS_DIR / 'cross_mapper_agreement.png'}")
        plt.close()

        # Legend explanation
        fig2, ax2 = plt.subplots(figsize=(8, 2))
        ax2.axis('off')
        legend_items = [
            ("Timing (ONSET, COMBO)", "#2196F3"),
            ("Structure (SLIDE, SUSTAIN)", "#4CAF50"),
            ("Hitsounds (WHISTLE, FINISH, CLAP)", "#FF9800"),
            ("Spatial (X, Y)", "#9C27B0"),
        ]
        for i, (label, color) in enumerate(legend_items):
            ax2.barh([i], [1], color=color, alpha=0.8)
            ax2.text(1.1, i, label, va='center', fontsize=10)
        ax2.set_xlim(0, 5)
        ax2.set_title("Dimension Categories")
        fig2.savefig(RESULTS_DIR / "legend.png", dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plots")

    return summary


if __name__ == "__main__":
    analyze()
