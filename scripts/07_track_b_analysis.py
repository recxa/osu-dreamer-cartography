"""
Step 7: Cross-mapper comparison in VAE latent space (Track B).

For each song with multiple mappers, compares the 32-dim latent
trajectories to identify which latent dimensions show high cross-mapper
agreement (perceptual) vs. high variance (stylistic).

Usage:
    uv run python scripts/07_track_b_analysis.py

Input:  experiment/output/data/latent_manifest.json
        experiment/output/data/encodings_latent/
Output: experiment/output/results/track_b/track_b_results.json
        experiment/output/results/track_b/latent_dim_agreement.png
"""

import json
import warnings
import numpy as np
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "experiment" / "output" / "data"
RESULTS_DIR = REPO_ROOT / "experiment" / "output" / "results" / "track_b"
LATENT_MANIFEST_PATH = DATA_DIR / "latent_manifest.json"
ENCODINGS_DIR = DATA_DIR / "encodings_latent"

N_LATENT = 32


def pearson_corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze():
    with open(LATENT_MANIFEST_PATH) as f:
        manifest = json.load(f)

    by_song = defaultdict(list)
    for entry in manifest:
        by_song[entry["song_group_idx"]].append(entry)

    print(f"Song groups with latent encodings: {len(by_song)}")
    print(f"Total latent-encoded beatmaps: {len(manifest)}")

    cross_mapper_pearson = {d: [] for d in range(N_LATENT)}
    cross_mapper_cosine_per_dim = {d: [] for d in range(N_LATENT)}
    cross_mapper_full_cosine = []
    song_between_var = {d: [] for d in range(N_LATENT)}

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

        min_l = min(v["data"].shape[1] for v in loaded.values())
        if min_l < 2:
            continue

        items = list(loaded.items())
        songs_analyzed += 1

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                fname_a, info_a = items[i]
                fname_b, info_b = items[j]

                if info_a["creator"] == info_b["creator"]:
                    continue

                z_a = info_a["data"][:, :min_l]
                z_b = info_b["data"][:, :min_l]

                for d in range(N_LATENT):
                    cross_mapper_pearson[d].append(pearson_corr(z_a[d], z_b[d]))
                    cross_mapper_cosine_per_dim[d].append(cosine_sim(z_a[d], z_b[d]))

                step = max(1, min_l // 100)
                frame_sims = []
                for t in range(0, min_l, step):
                    frame_sims.append(cosine_sim(z_a[:, t], z_b[:, t]))
                cross_mapper_full_cosine.append(np.mean(frame_sims))

                pairs_compared += 1

        # Variance decomposition
        by_creator = defaultdict(list)
        for fname, info in items:
            by_creator[info["creator"]].append(info["data"][:, :min_l])

        for d in range(N_LATENT):
            creator_means = []
            for creator, arrs in by_creator.items():
                creator_mean = np.mean([arr[d].mean() for arr in arrs])
                creator_means.append(creator_mean)
            if len(creator_means) >= 2:
                song_between_var[d].append(np.var(creator_means))

    print(f"\nSongs analyzed: {songs_analyzed}")
    print(f"Cross-mapper pairs compared: {pairs_compared}")

    # Aggregate
    print(f"\n{'='*80}")
    print(f"LATENT DIMENSIONS RANKED BY CROSS-MAPPER AGREEMENT (Pearson)")
    print(f"{'='*80}")

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
    for rank, (d, stats) in enumerate(ranked, 1):
        bar = "█" * int(max(0, stats["pearson_mean"]) * 40)
        if stats["pearson_mean"] > 0.3:
            label = "PERCEPTUAL"
        elif stats["pearson_mean"] < 0.05:
            label = "STYLISTIC"
        else:
            label = "MIXED"
        print(f"z_{d:<4} {stats['pearson_mean']:>6.3f} ± {stats['pearson_std']:.3f}  {bar}  [{label}]")

    if cross_mapper_full_cosine:
        print(f"\nFull 32-dim cosine similarity: "
              f"{np.mean(cross_mapper_full_cosine):.3f} ± {np.std(cross_mapper_full_cosine):.3f}")

    perceptual = sum(1 for d, s in dim_summary.items() if s["pearson_mean"] > 0.3)
    mixed = sum(1 for d, s in dim_summary.items() if 0.05 <= s["pearson_mean"] <= 0.3)
    stylistic = sum(1 for d, s in dim_summary.items() if s["pearson_mean"] < 0.05)
    print(f"\nDimension taxonomy: {perceptual} perceptual, {mixed} mixed, {stylistic} stylistic")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "dim_summary": {str(k): v for k, v in dim_summary.items()},
        "dim_ranking": [{"dim": d, **s} for d, s in ranked],
        "full_cosine_mean": float(np.mean(cross_mapper_full_cosine)) if cross_mapper_full_cosine else None,
        "full_cosine_std": float(np.std(cross_mapper_full_cosine)) if cross_mapper_full_cosine else None,
        "songs_analyzed": songs_analyzed,
        "pairs_compared": pairs_compared,
        "taxonomy": {"perceptual": perceptual, "mixed": mixed, "stylistic": stylistic},
    }

    with open(RESULTS_DIR / "track_b_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'track_b_results.json'}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        dims_sorted = [f"z_{d}" for d, _ in ranked]
        pearsons_sorted = [s["pearson_mean"] for _, s in ranked]
        pearson_errs_sorted = [s["pearson_std"] for _, s in ranked]

        colors = []
        for _, s in ranked:
            if s["pearson_mean"] > 0.3:
                colors.append("#2196F3")
            elif s["pearson_mean"] < 0.05:
                colors.append("#F44336")
            else:
                colors.append("#FF9800")

        ax.barh(range(len(dims_sorted)), pearsons_sorted, xerr=pearson_errs_sorted,
                color=colors, alpha=0.8, capsize=2)
        ax.set_yticks(range(len(dims_sorted)))
        ax.set_yticklabels(dims_sorted, fontsize=8)
        ax.set_xlabel("Pearson Correlation (cross-mapper)")
        ax.set_title("VAE Latent Dimensions: Cross-Mapper Agreement")
        ax.axvline(x=0.3, color="blue", linestyle="--", alpha=0.5, label="Perceptual threshold")
        ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.5, label="Stylistic threshold")
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "latent_dim_agreement.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {RESULTS_DIR}/")

    except ImportError:
        print("matplotlib not available — skipping plots")


if __name__ == "__main__":
    if not LATENT_MANIFEST_PATH.exists():
        print(f"Error: Run 06_encode_latent.py first — missing {LATENT_MANIFEST_PATH}")
        exit(1)

    analyze()
