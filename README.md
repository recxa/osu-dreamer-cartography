# Latent Space Cartography

**Cross-mapper beatmap analysis for ISMIR 2026**

[Live results](https://recxa.github.io/osu-dreamer-cartography/)

This experiment investigates whether a VAE trained on osu! beatmap data learns a meaningful separation between **perceptual** (music-driven) and **stylistic** (mapper-driven) aspects of rhythm annotation. The core idea: if multiple mappers independently annotate the same song, dimensions where they agree reflect the music itself, while dimensions where they diverge capture personal style.

We test at two representation levels:
- **Track A**: osu-dreamer's 9-dim interpretable encoding (onset, combo, slide, sustain, hitsounds, cursor XY)
- **Track B**: 32-dim VAE latent trajectories from a pre-trained encoder

A synthesis step computes the 32x9 correlation matrix, revealing what each latent dimension encodes.

## Key Findings

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

8 of 32 latent dimensions classified as perceptual (r > 0.3), primarily encoding timing (ONSET + COMBO). The gradient from timing through hitsounds to spatial position matches musical intuition.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/recxa/osu-dreamer-cartography.git
cd osu-dreamer-cartography
uv sync
```

If you cloned without Git LFS, fetch the checkpoint:
```bash
git lfs pull
```

## Quick Start (One Command)

```bash
uv run python experiment/run_analysis.py /path/to/your/osz/files/
```

This runs the full 8-step pipeline. Point it at a directory of `.osz` beatmap archives — the script automatically identifies songs mapped by multiple people.

## Step-by-Step Reproduction

For more control, run individual scripts from the repo root:

```bash
# Step 1: Build multi-mapper index from .osz files
uv run python scripts/01_build_index.py /path/to/osz/files/

# Step 2: Extract .osz archives for multi-mapper songs
uv run python scripts/02_extract_osz.py /path/to/osz/files/

# Step 3: Parse .osu files, build beatmap registry
uv run python scripts/03_build_registry.py

# Step 4: Encode beatmaps to 9-dim temporal signals
uv run python scripts/04_encode_9dim.py

# Step 5: Cross-mapper analysis on 9-dim (Track A)
uv run python scripts/05_track_a_analysis.py

# Step 6: Encode through pre-trained VAE to 32-dim latent
uv run python scripts/06_encode_latent.py

# Step 7: Cross-mapper analysis on 32-dim latent (Track B)
uv run python scripts/07_track_b_analysis.py

# Step 8: Synthesis — 32x9 correlation matrix
uv run python scripts/08_synthesis.py
```

Each script validates its prerequisites and prints clear progress. Steps 1-2 require a dataset path; steps 3-8 read from previous outputs.

## Output

All generated data goes to `experiment/output/` (gitignored):

```
experiment/output/
  data/
    multi_mapper_index.json      # Song groups with 2+ mappers
    extracted/                   # .osu + audio files per beatmapset
    beatmap_registry.json        # All Mode 0 beatmaps with metadata
    encodings_9dim/              # [9, L] arrays per beatmap
    encoding_manifest.json
    encodings_latent/            # [32, l] arrays per beatmap
    latent_manifest.json
  results/
    track_a/track_a_results.json
    track_b/track_b_results.json
    synthesis/synthesis_results.json
```

## Reference Results

`reference_results/` contains the JSON outputs from our run on 292 multi-mapper songs (460 cross-mapper pairs, 850 training mapsets). Compare your outputs against these to validate reproduction.

## Pre-trained Checkpoint

The bundled checkpoint (`experiment/checkpoints/epoch=3-step=58000.ckpt`, 148MB, Git LFS) was trained with:

- **Architecture:** osu-dreamer VAE (WaveNet encoder/decoder, 32-dim latent, 18x temporal compression)
- **Data:** 850 mapsets (single timing point subset), all difficulties
- **Training:** 58K steps (~4 epochs), batch 8, lr 0.002
- **Hardware:** Apple M1 Pro (MPS)
- **Config:** `experiment/config/latent_model.yml`

To retrain from scratch:
```bash
uv run python -m osu_dreamer fit-latent --config experiment/config/latent_model.yml
```

## Repository Structure

```
osu-dreamer-cartography/
├── scripts/                    # 8 numbered analysis scripts
├── experiment/
│   ├── run_analysis.py         # All-in-one pipeline
│   ├── checkpoints/            # Pre-trained VAE (Git LFS)
│   ├── config/                 # Training configs
│   ├── results/                # Pre-computed result plots
│   └── output/                 # Generated data (gitignored)
├── osu_dreamer/                # Minimal osu-dreamer bundle
│   ├── osu/                    # .osu file parser
│   ├── data/                   # Audio loading, 9-dim encoding
│   ├── latent_model/           # VAE model (encode/decode)
│   └── modules/                # Neural network building blocks
├── reference_results/          # Expected outputs for validation
├── docs/                       # GitHub Pages interactive site
└── pyproject.toml              # Dependencies (managed by uv)
```

## Dataset

Our analysis used the single-timing-point subset of ranked osu! maps (~3,586 .osz files). Any collection of `.osz` files with songs mapped by multiple creators will work.

## Troubleshooting

**"Checkpoint not found"**: Run `git lfs pull` to download the 148MB checkpoint.

**MPS errors on Apple Silicon**: Some PyTorch operations may not be supported on MPS. The scripts fall back to CPU automatically. If you encounter issues, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Audio alignment warnings**: Different .osz archives for the same song may have slightly different audio lengths. The scripts handle this by truncating to the shortest common length.

**Low multi-mapper count**: Your dataset needs songs mapped by 2+ different creators. The more overlap, the more statistical power.

## Credits

- osu-dreamer by [@jaswon](https://github.com/jaswon/osu-dreamer)
- Part of the OSUxMIR research project (ISMIR 2026)
