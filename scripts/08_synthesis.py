"""
Step 8: Synthesis — Connect Track A (9-dim) to Track B (32-dim).

Computes the 32x9 correlation matrix between latent dimensions and
interpretable dimensions, revealing what each latent dim encodes.

Usage:
    uv run python scripts/08_synthesis.py

Input:  experiment/output/results/track_a/track_a_results.json
        experiment/output/results/track_b/track_b_results.json
        experiment/output/data/encodings_9dim/
        experiment/output/data/encodings_latent/
Output: experiment/output/results/synthesis/synthesis_results.json
        experiment/output/results/synthesis/correlation_heatmap.png
"""

import json
import warnings
import numpy as np
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "experiment" / "output" / "data"
RESULTS_DIR = REPO_ROOT / "experiment" / "output" / "results" / "synthesis"
TRACK_A_RESULTS = REPO_ROOT / "experiment" / "output" / "results" / "track_a" / "track_a_results.json"
TRACK_B_RESULTS = REPO_ROOT / "experiment" / "output" / "results" / "track_b" / "track_b_results.json"
ENCODINGS_9DIM_DIR = DATA_DIR / "encodings_9dim"
ENCODINGS_LATENT_DIR = DATA_DIR / "encodings_latent"
MANIFEST_9DIM_PATH = DATA_DIR / "encoding_manifest.json"
LATENT_MANIFEST_PATH = DATA_DIR / "latent_manifest.json"

DIM_NAMES = ["ONSET", "COMBO", "SLIDE", "SUSTAIN", "WHISTLE", "FINISH", "CLAP", "X", "Y"]
N_LATENT = 32


def synthesize():
    with open(MANIFEST_9DIM_PATH) as f:
        manifest_9dim = json.load(f)
    with open(LATENT_MANIFEST_PATH) as f:
        manifest_latent = json.load(f)

    latent_files = {e["filename"] for e in manifest_latent}
    both = [e for e in manifest_9dim if e["filename"] in latent_files]
    print(f"Beatmaps with both encodings: {len(both)}")

    # 32x9 correlation matrix
    corr_matrix = np.zeros((N_LATENT, len(DIM_NAMES)))
    corr_counts = 0

    for entry in both:
        try:
            enc_9 = np.load(ENCODINGS_9DIM_DIR / entry["filename"])   # [9, L]
            enc_z = np.load(ENCODINGS_LATENT_DIR / entry["filename"])  # [32, l]

            L = enc_9.shape[1]
            z_up = np.repeat(enc_z, 18, axis=1)[:, :L]  # [32, L]

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

    print(f"Correlation matrix computed from {corr_counts} beatmaps")

    with open(TRACK_A_RESULTS) as f:
        track_a = json.load(f)
    with open(TRACK_B_RESULTS) as f:
        track_b = json.load(f)

    # Dimension taxonomy
    taxonomy = []
    for item in track_b["dim_ranking"]:
        d = item["dim"]
        pearson = item["pearson_mean"]

        z_corrs = corr_matrix[d]
        top_idx = np.argsort(np.abs(z_corrs))[::-1]
        top_3 = [(DIM_NAMES[i], float(z_corrs[i])) for i in top_idx[:3]]

        if pearson > 0.3:
            category = "PERCEPTUAL"
        elif pearson < 0.05:
            category = "STYLISTIC"
        else:
            category = "MIXED"

        taxonomy.append({
            "dim": d,
            "category": category,
            "cross_mapper_pearson": pearson,
            "top_9dim_correlations": top_3,
            "strongest_9dim": top_3[0][0],
            "strongest_9dim_corr": top_3[0][1],
        })

    print(f"\n{'='*90}")
    print(f"DIMENSION TAXONOMY: LATENT → INTERPRETABLE MAPPING")
    print(f"{'='*90}")
    for t in taxonomy:
        corr_str = ", ".join(f"{name}={r:+.3f}" for name, r in t["top_9dim_correlations"])
        print(f"z_{t['dim']:<6} [{t['category']:<12}] {t['cross_mapper_pearson']:>6.3f}     {corr_str}")

    print(f"\n{'='*90}")
    print(f"SYNTHESIS SUMMARY")
    print(f"{'='*90}")
    print(f"Track A (9-dim):  {track_a['songs_analyzed']} songs, {track_a['pairs_compared']} pairs")
    print(f"Track B (32-dim): {track_b['songs_analyzed']} songs, {track_b['pairs_compared']} pairs")
    print(f"Track A full cosine:  {track_a.get('cross_mapper_cosine_mean', 'N/A')}")
    print(f"Track B full cosine:  {track_b.get('full_cosine_mean', 'N/A')}")
    print(f"\nTaxonomy:")
    print(f"  Perceptual (r>0.3):  {sum(1 for t in taxonomy if t['category']=='PERCEPTUAL')}/32")
    print(f"  Mixed (0.05-0.3):    {sum(1 for t in taxonomy if t['category']=='MIXED')}/32")
    print(f"  Stylistic (r<0.05):  {sum(1 for t in taxonomy if t['category']=='STYLISTIC')}/32")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "correlation_matrix": corr_matrix.tolist(),
        "dim_names_9": DIM_NAMES,
        "n_latent": N_LATENT,
        "n_beatmaps_used": corr_counts,
        "taxonomy": taxonomy,
        "summary": {
            "track_a_songs": track_a["songs_analyzed"],
            "track_a_pairs": track_a["pairs_compared"],
            "track_b_songs": track_b["songs_analyzed"],
            "track_b_pairs": track_b["pairs_compared"],
            "track_a_cosine": track_a.get("cross_mapper_cosine_mean"),
            "track_b_cosine": track_b.get("full_cosine_mean"),
        },
    }

    with open(RESULTS_DIR / "synthesis_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'synthesis_results.json'}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, ax = plt.subplots(figsize=(10, 12))
        dim_order = [t["dim"] for t in taxonomy]
        matrix_sorted = corr_matrix[dim_order, :]

        norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
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
        fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to {RESULTS_DIR}/")

    except ImportError:
        print("matplotlib not available — skipping plots")


if __name__ == "__main__":
    if not TRACK_A_RESULTS.exists():
        print(f"Error: Run 05_track_a_analysis.py first — missing {TRACK_A_RESULTS}")
        exit(1)
    if not TRACK_B_RESULTS.exists():
        print(f"Error: Run 07_track_b_analysis.py first — missing {TRACK_B_RESULTS}")
        exit(1)

    synthesize()
