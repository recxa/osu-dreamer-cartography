"""
Step 6: Encode representative beatmaps through the trained VAE.

Takes the 9-dim encodings from Step 4 and passes them through
the VAE encoder to get 32-dim latent representations.

Usage:
    uv run python scripts/06_encode_latent.py

Input:  experiment/output/data/encodings_9dim/*.npy
        experiment/output/data/encoding_manifest.json
        experiment/checkpoints/epoch=3-step=58000.ckpt
Output: experiment/output/data/encodings_latent/*.npy
        experiment/output/data/latent_manifest.json
"""

import json
import numpy as np
import torch as th
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "experiment" / "output" / "data"
ENCODINGS_9DIM_DIR = DATA_DIR / "encodings_9dim"
OUTPUT_DIR = DATA_DIR / "encodings_latent"
MANIFEST_9DIM_PATH = DATA_DIR / "encoding_manifest.json"
LATENT_MANIFEST_PATH = DATA_DIR / "latent_manifest.json"
CHECKPOINT_PATH = REPO_ROOT / "experiment" / "checkpoints" / "epoch=3-step=58000.ckpt"


def encode_latent():
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"If you cloned without Git LFS, run: git lfs pull")
        exit(1)

    from osu_dreamer.latent_model.model import Model

    print(f"Loading model from {CHECKPOINT_PATH}")
    model = Model.load_from_checkpoint(str(CHECKPOINT_PATH))
    model.eval()

    if th.cuda.is_available():
        device = "cuda"
    elif th.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    print(f"Model on device: {device}")

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
            enc_9dim = np.load(src_path)  # [9, L]
            chart = th.tensor(enc_9dim).float().unsqueeze(0).to(device)  # [1, 9, L]

            with th.no_grad():
                z = model.encode(chart)  # [1, 32, l]

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

    with open(LATENT_MANIFEST_PATH, "w") as f:
        json.dump(latent_manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Encoded: {encoded_count}")
    print(f"  Errors: {error_count}")
    print(f"  Latent manifest: {LATENT_MANIFEST_PATH}")


if __name__ == "__main__":
    if not MANIFEST_9DIM_PATH.exists():
        print(f"Error: Run 04_encode_9dim.py first — missing {MANIFEST_9DIM_PATH}")
        exit(1)

    encode_latent()
