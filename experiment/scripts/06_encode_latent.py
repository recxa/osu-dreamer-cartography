"""
Phase 4.1: Encode representative beatmaps through the trained VAE.

Takes the 9-dim encodings from Phase 2.1 and passes them through
the VAE encoder to get 32-dim latent representations.

Input:  data/encodings_9dim/{beatmapset_id}_{version}.npy  — shape [9, L]
Output: data/encodings_latent/{beatmapset_id}_{version}.npy — shape [32, l]
        data/latent_manifest.json

Run with osu-dreamer venv:
  PYTHONPATH=.../osu-dreamer .../osu-dreamer/.venv/bin/python 06_encode_latent.py
"""

import json
import numpy as np
import torch as th
from pathlib import Path

DATA_DIR = Path("/Users/red/Downloads/rcx26/OSUxMIR/experiments/latent-cartography/data")
ENCODINGS_9DIM_DIR = DATA_DIR / "encodings_9dim"
OUTPUT_DIR = DATA_DIR / "encodings_latent"
MANIFEST_9DIM_PATH = DATA_DIR / "encoding_manifest.json"
LATENT_MANIFEST_PATH = DATA_DIR / "latent_manifest.json"

CHECKPOINT_PATH = Path(
    "/Users/red/Downloads/rcx26/OSUxMIR/experiments/latent-cartography"
    "/runs/latent/version_2/checkpoints/epoch=3-step=58000.ckpt"
)


def encode_latent():
    from osu_dreamer.latent_model.model import Model

    # Load trained VAE
    print(f"Loading model from {CHECKPOINT_PATH}")
    model = Model.load_from_checkpoint(str(CHECKPOINT_PATH))
    model.eval()

    # Use MPS if available, else CPU
    device = "mps" if th.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Model on device: {device}")

    # Load 9-dim manifest
    with open(MANIFEST_9DIM_PATH) as f:
        manifest_9dim = json.load(f)

    print(f"Beatmaps to encode: {len(manifest_9dim)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latent_manifest = []
    encoded_count = 0
    error_count = 0

    for entry in manifest_9dim:
        src_path = ENCODINGS_9DIM_DIR / entry["filename"]
        if not src_path.exists():
            print(f"  Missing: {entry['filename']}")
            error_count += 1
            continue

        try:
            # Load 9-dim encoding
            enc_9dim = np.load(src_path)  # [9, L]

            # Convert to tensor and add batch dim
            chart = th.tensor(enc_9dim).float().unsqueeze(0).to(device)  # [1, 9, L]

            # Encode through VAE
            with th.no_grad():
                z = model.encode(chart)  # [1, 32, l]

            # Save latent encoding
            z_np = z.squeeze(0).cpu().numpy()  # [32, l]
            np.save(OUTPUT_DIR / entry["filename"], z_np)

            latent_manifest.append({
                **entry,
                "latent_shape": list(z_np.shape),
                "orig_L": enc_9dim.shape[1],
                "latent_l": z_np.shape[1],
            })

            encoded_count += 1
            if encoded_count % 50 == 0:
                print(f"  Encoded {encoded_count}/{len(manifest_9dim)}...")

        except Exception as e:
            print(f"  Error ({entry['filename']}): {e}")
            error_count += 1

    # Save manifest
    with open(LATENT_MANIFEST_PATH, 'w') as f:
        json.dump(latent_manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Encoded: {encoded_count}")
    print(f"  Errors: {error_count}")
    print(f"  Latent manifest: {LATENT_MANIFEST_PATH}")
    print(f"  Latent encodings: {OUTPUT_DIR}/")


if __name__ == "__main__":
    encode_latent()
