# Latent Cartography Experiment

This experiment investigates whether a VAE trained on osu! beatmap data learns a meaningful separation between **perceptual** (music-driven) and **stylistic** (mapper-driven) aspects of rhythm annotation. The core idea: if multiple mappers independently annotate the same song, the dimensions where they agree must reflect something about the music itself, while dimensions where they diverge capture personal mapping style.

We test this at two levels of representation. **Track A** operates on osu-dreamer's 9-dimensional interpretable encoding (onset, combo, slide, sustain, hitsounds, cursor position). **Track B** passes those signals through a pre-trained VAE encoder to get 32-dimensional latent trajectories, then repeats the cross-mapper comparison. A synthesis step connects the two by computing the 32x9 correlation matrix, revealing what each latent dimension actually encodes.

## Setup

```bash
git clone https://github.com/recxa/osu-dreamer-cartography.git
cd osu-dreamer-cartography
uv sync
```

That's it. All dependencies (including PyTorch) are managed by `uv` via `pyproject.toml`.

## Quick Start

```bash
uv run python experiment/run_analysis.py /path/to/your/osz/files/
```

Point it at a directory containing `.osz` beatmap archives. For meaningful results, your dataset should include songs that have been mapped by multiple different people (the script automatically identifies these by matching Title/Artist metadata across .osz files).

## What the Output Means

Results are written to `experiment/output/`:

```
experiment/output/
  data/                          # Intermediate data (index, registry, encodings)
  results/
    track_a/
      track_a_results.json       # Per-dimension Pearson correlations (9-dim)
      cross_mapper_agreement.png # Bar chart of agreement by dimension
    track_b/
      track_b_results.json       # Per-dimension Pearson correlations (32-dim latent)
      latent_dim_agreement.png   # Ranked latent dimensions
      latent_cosine_distribution.png
    synthesis/
      synthesis_results.json     # 32x9 correlation matrix + taxonomy
      correlation_heatmap.png    # Latent-to-interpretable mapping
      track_a_vs_b_comparison.png
```

**Track A results** show which of the 9 interpretable dimensions are consistent across mappers (high Pearson = perceptual, low = stylistic). **Track B results** do the same for the 32 latent dimensions. The **synthesis** reveals what interpretable features each latent dimension primarily encodes.

## Pre-trained Checkpoint

The bundled checkpoint (`experiment/checkpoints/epoch=3-step=58000.ckpt`) was trained with these parameters:

- **Architecture:** osu-dreamer's VAE (latent model) -- WaveNet encoder/decoder, 32-dim latent
- **Training data:** 850 mapsets (single timing point subset), all difficulties
- **Training:** 58,000 steps (~4 epochs), batch size 8, lr 0.002
- **Hardware:** Apple M1 Pro (MPS backend)
- **Config:** See `experiment/config/latent_model.yml`

**Important:** This checkpoint was trained on the full dataset without a train/test split. A cross-validated version with proper held-out evaluation is in progress.

## Preliminary Findings (Track A)

From our initial run on 292 multi-mapper songs (460 cross-mapper pairs):

| Dimension | Pearson r | Category |
|-----------|----------|----------|
| ONSET     | 0.582    | Perceptual |
| COMBO     | 0.444    | Perceptual |
| CLAP      | 0.346    | Perceptual |
| SUSTAIN   | 0.294    | Mixed |
| FINISH    | 0.256    | Mixed |
| SLIDE     | 0.249    | Mixed |
| WHISTLE   | 0.215    | Mixed |
| X         | 0.008    | Stylistic |
| Y         | 0.004    | Stylistic |

The gradient from timing (highly perceptual) through hitsounds (mixed) to spatial position (purely stylistic) matches musical intuition: different mappers place notes at roughly the same times but move them to very different positions on screen.

## Individual Scripts

The pipeline script combines the logic from these standalone scripts in `experiment/scripts/`:

| Script | Phase | Description |
|--------|-------|-------------|
| `01_build_index.py` | 1.1 | Scan .osz files, build multi-mapper index |
| `02_extract_osz.py` | 1.2 | Extract .osz archives for multi-mapper subset |
| `03_build_registry.py` | 1.3 | Parse .osu files, select representative diffs |
| `04_encode_9dim.py` | 2.1 | Encode beatmaps to 9-dim temporal signals |
| `05_track_a_analysis.py` | 2.2 | Cross-mapper comparison on 9-dim |
| `06_encode_latent.py` | 4.1 | Encode through trained VAE |
| `07_track_b_analysis.py` | 4.2 | Cross-mapper comparison on 32-dim latent |
| `08_synthesis.py` | 5 | Connect Track A and Track B (32x9 matrix) |

These scripts have hardcoded paths from our original development environment. Use `run_analysis.py` for a portable, self-contained pipeline.

## Reference

This experiment is part of the OSUxMIR research project exploring perceptual rhythm representations learned from osu! beatmap data, targeting ISMIR 2026.

osu-dreamer is by [@jaswon](https://github.com/jaswon/osu-dreamer).
