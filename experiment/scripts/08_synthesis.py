"""
Phase 5: Synthesis — Connect Track A (9-dim) to Track B (32-dim).

Computes the 32x9 correlation matrix between latent dimensions and
interpretable dimensions, revealing what each latent dim encodes.

Also produces combined visualizations and a dimension taxonomy.

Run with osu-dreamer venv:
  PYTHONPATH=.../osu-dreamer .../osu-dreamer/.venv/bin/python 08_synthesis.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "synthesis"
TRACK_A_RESULTS = Path(__file__).parent.parent / "results" / "track_a" / "track_a_results.json"
TRACK_B_RESULTS = Path(__file__).parent.parent / "results" / "track_b" / "track_b_results.json"
ENCODINGS_9DIM_DIR = DATA_DIR / "encodings_9dim"
ENCODINGS_LATENT_DIR = DATA_DIR / "encodings_latent"
MANIFEST_9DIM_PATH = DATA_DIR / "encoding_manifest.json"
LATENT_MANIFEST_PATH = DATA_DIR / "latent_manifest.json"

DIM_NAMES = ["ONSET", "COMBO", "SLIDE", "SUSTAIN", "WHISTLE", "FINISH", "CLAP", "X", "Y"]
N_LATENT = 32


def synthesize():
    # Load manifests
    with open(MANIFEST_9DIM_PATH) as f:
        manifest_9dim = json.load(f)
    with open(LATENT_MANIFEST_PATH) as f:
        manifest_latent = json.load(f)

    # Build lookup by filename
    latent_files = {e["filename"] for e in manifest_latent}

    # Find beatmaps that have both 9-dim and latent encodings
    both = [e for e in manifest_9dim if e["filename"] in latent_files]
    print(f"Beatmaps with both encodings: {len(both)}")

    # === Compute 32x9 correlation matrix ===
    # For each beatmap, compute per-frame correlation between each latent dim
    # and each interpretable dim, then average across beatmaps.
    # Since latent has 18x downsampling, we upsample latent via nearest-neighbor.

    corr_matrix = np.zeros((N_LATENT, len(DIM_NAMES)))
    corr_counts = 0

    for entry in both:
        try:
            enc_9 = np.load(ENCODINGS_9DIM_DIR / entry["filename"])   # [9, L]
            enc_z = np.load(ENCODINGS_LATENT_DIR / entry["filename"])  # [32, l]

            L = enc_9.shape[1]
            l = enc_z.shape[1]

            # Upsample latent to match 9-dim length (nearest-neighbor)
            z_up = np.repeat(enc_z, 18, axis=1)[:, :L]  # [32, L]

            # Compute correlation between each latent dim and each interpretable dim
            for zi in range(N_LATENT):
                for di, dim_name in enumerate(DIM_NAMES):
                    z_sig = z_up[zi]
                    d_sig = enc_9[di]
                    if np.std(z_sig) > 0 and np.std(d_sig) > 0 and len(z_sig) >= 2:
                        r = np.corrcoef(z_sig, d_sig)[0, 1]
                        if not np.isnan(r):
                            corr_matrix[zi, di] += r

            corr_counts += 1

        except Exception as e:
            pass

    if corr_counts > 0:
        corr_matrix /= corr_counts

    print(f"Correlation matrix computed from {corr_counts} beatmaps")

    # === Load Track A and B results for combined analysis ===
    with open(TRACK_A_RESULTS) as f:
        track_a = json.load(f)
    with open(TRACK_B_RESULTS) as f:
        track_b = json.load(f)

    # === Dimension taxonomy ===
    taxonomy = []
    track_b_ranking = track_b["dim_ranking"]

    for item in track_b_ranking:
        d = item["dim"]
        pearson = item["pearson_mean"]

        # Find strongest 9-dim correlations
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

    # Print taxonomy
    print(f"\n{'='*90}")
    print(f"DIMENSION TAXONOMY: LATENT → INTERPRETABLE MAPPING")
    print(f"{'='*90}")
    print(f"{'Dim':<8} {'Category':<14} {'Pearson':<10} {'Strongest 9-dim correlations'}")
    print(f"{'-'*90}")

    for t in taxonomy:
        corr_str = ", ".join(f"{name}={r:+.3f}" for name, r in t["top_9dim_correlations"])
        print(f"z_{t['dim']:<6} [{t['category']:<12}] {t['cross_mapper_pearson']:>6.3f}     {corr_str}")

    # === Summary statistics ===
    print(f"\n{'='*90}")
    print(f"SYNTHESIS SUMMARY")
    print(f"{'='*90}")
    print(f"Track A (9-dim):  {track_a['songs_analyzed']} songs, {track_a['pairs_compared']} pairs")
    print(f"Track B (32-dim): {track_b['songs_analyzed']} songs, {track_b['pairs_compared']} pairs")
    print(f"")
    print(f"Track A full cosine:  {track_a.get('cross_mapper_cosine_mean', 'N/A')}")
    print(f"Track B full cosine:  {track_b.get('full_cosine_mean', 'N/A')}")
    print(f"")
    print(f"Taxonomy breakdown:")
    print(f"  Perceptual (r>0.3):  {sum(1 for t in taxonomy if t['category']=='PERCEPTUAL')}/32")
    print(f"  Mixed (0.05-0.3):    {sum(1 for t in taxonomy if t['category']=='MIXED')}/32")
    print(f"  Stylistic (r<0.05):  {sum(1 for t in taxonomy if t['category']=='STYLISTIC')}/32")

    # What do perceptual latent dims encode?
    perceptual_dims = [t for t in taxonomy if t["category"] == "PERCEPTUAL"]
    if perceptual_dims:
        print(f"\nPerceptual latent dims primarily encode:")
        strongest_counts = defaultdict(int)
        for t in perceptual_dims:
            strongest_counts[t["strongest_9dim"]] += 1
        for name, count in sorted(strongest_counts.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count} dims")

    # === Save results ===
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

    with open(RESULTS_DIR / "synthesis_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'synthesis_results.json'}")

    # === Plots ===
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        # 1. Correlation heatmap: 32 latent dims x 9 interpretable dims
        fig, ax = plt.subplots(figsize=(10, 12))
        # Sort latent dims by their cross-mapper Pearson (most perceptual on top)
        dim_order = [t["dim"] for t in taxonomy]
        matrix_sorted = corr_matrix[dim_order, :]

        norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
        im = ax.imshow(matrix_sorted, cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_xticks(range(len(DIM_NAMES)))
        ax.set_xticklabels(DIM_NAMES, rotation=45, ha='right')
        ax.set_yticks(range(N_LATENT))
        ax.set_yticklabels([f"z_{d}" for d in dim_order], fontsize=7)
        ax.set_xlabel("Interpretable Dimension (9-dim)")
        ax.set_ylabel("Latent Dimension (sorted by cross-mapper agreement)")
        ax.set_title("Latent ↔ Interpretable Correlation Matrix")
        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.7)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
        fig.savefig(RESULTS_DIR / "correlation_heatmap.pdf", bbox_inches='tight')
        plt.close()

        # 2. Side-by-side: Track A vs Track B agreement spectra
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Track A
        track_a_summary = track_a["summary"]
        a_dims = sorted(track_a_summary.keys(), key=lambda d: -track_a_summary[d]["pearson_mean"])
        a_pearsons = [track_a_summary[d]["pearson_mean"] for d in a_dims]
        a_errs = [track_a_summary[d]["pearson_std"] for d in a_dims]
        a_colors = []
        for d in a_dims:
            if d in ("ONSET", "COMBO"):
                a_colors.append("#2196F3")
            elif d in ("SLIDE", "SUSTAIN"):
                a_colors.append("#4CAF50")
            elif d in ("WHISTLE", "FINISH", "CLAP"):
                a_colors.append("#FF9800")
            else:
                a_colors.append("#9C27B0")

        ax1.barh(range(len(a_dims)), a_pearsons, xerr=a_errs, color=a_colors, alpha=0.8, capsize=3)
        ax1.set_yticks(range(len(a_dims)))
        ax1.set_yticklabels(a_dims)
        ax1.set_xlabel("Pearson Correlation")
        ax1.set_title("Track A: 9-Dim Cross-Mapper Agreement")
        ax1.axvline(x=0, color='gray', linewidth=0.5)
        ax1.invert_yaxis()

        # Track B (top 9 for comparable scale)
        b_top = taxonomy[:9]
        b_dims = [f"z_{t['dim']}" for t in b_top]
        b_pearsons = [t["cross_mapper_pearson"] for t in b_top]
        b_colors = []
        for t in b_top:
            if t["category"] == "PERCEPTUAL":
                b_colors.append("#2196F3")
            elif t["category"] == "STYLISTIC":
                b_colors.append("#F44336")
            else:
                b_colors.append("#FF9800")

        ax2.barh(range(len(b_dims)), b_pearsons, color=b_colors, alpha=0.8)
        ax2.set_yticks(range(len(b_dims)))
        ax2.set_yticklabels(b_dims)
        ax2.set_xlabel("Pearson Correlation")
        ax2.set_title("Track B: Top-9 Latent Dim Cross-Mapper Agreement")
        ax2.axvline(x=0, color='gray', linewidth=0.5)
        ax2.invert_yaxis()

        # Match x-axis scales
        max_x = max(max(a_pearsons), max(b_pearsons)) * 1.2
        ax1.set_xlim(-0.05, max_x)
        ax2.set_xlim(-0.05, max_x)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "track_a_vs_b_comparison.png", dpi=150, bbox_inches='tight')
        fig.savefig(RESULTS_DIR / "track_a_vs_b_comparison.pdf", bbox_inches='tight')
        plt.close()

        print(f"Plots saved to {RESULTS_DIR}/")

    except ImportError:
        print("matplotlib not available — skipping plots")

    return output


if __name__ == "__main__":
    synthesize()
