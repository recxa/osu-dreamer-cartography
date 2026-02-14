# Latent Space Cartography – Interactive Results

Interactive visualization of cross-mapper beatmap analysis results for ISMIR 2026.

## What's Here

- **Part A**: Cross-mapper agreement on 9-dim interpretable encoding
- **Part B**: Cross-mapper agreement in 32-dim VAE latent space
- **Part C**: Interactive 32×9 correlation heatmap (synthesis)
- **Part D**: Cross-validation results confirming generalization

## Local Development

Serve locally:
```bash
python3 -m http.server 8000 --directory docs
```

Then open http://localhost:8000

## Deployment

This site is designed for GitHub Pages. Just push the `docs/` folder and enable Pages in repository settings.

## Structure

```
docs/
├── index.html           # Landing page
├── track_a.html         # Part A details
├── track_b.html         # Part B details
├── synthesis.html       # Interactive heatmap
├── cross_validation.html # CV comparison
├── css/
│   ├── style.css       # Global styles
│   └── heatmap.css     # Heatmap-specific styles
├── js/
│   ├── main.js         # Landing page logic
│   └── heatmap.js      # Interactive heatmap
└── data/
    ├── track_a.json
    ├── track_b.json
    ├── synthesis.json
    ├── cv_comparison.json
    ├── track_b_cv.json
    └── raw_baseline.json
```

## Data Sources

All JSON files are generated from the analysis scripts in `scripts/01-09_*.py`.
