"""
Pipeline runner: wraps run_analysis.py step functions with progress callbacks.

Captures stdout from each step and emits structured events for SSE streaming.
Detects pre-computed data and skips already-completed steps.
"""

import io
import json
import os
import sys
import tarfile
import threading
import traceback
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Add repo root to path so run_analysis imports work
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.run_analysis import (
    step1_build_index,
    step1b_extract_osz,
    step2_build_registry,
    step3_encode_9dim,
    step4_track_a,
    step5_encode_latent,
    step6_track_b,
    step7_synthesis,
    step8_plots,
    step9_summary,
    OUTPUT_DIR,
    RESULTS_DIR,
    DATA_DIR,
    CHECKPOINT_PATH,
)

# URL for pre-computed sample data (GitHub Release asset)
SAMPLE_DATA_URL = "https://github.com/recxa/osu-dreamer-cartography/releases/download/v0.1.0/cartography-sample-data.tar.gz"
SAMPLE_DATA_LOCAL = REPO_ROOT / "experiment" / "output" / "data" / ".sample-data-downloaded"


STEPS = [
    {"id": 1, "name": "Build multi-mapper index", "fn": "step1"},
    {"id": 2, "name": "Extract .osz archives", "fn": "step1b"},
    {"id": 3, "name": "Build beatmap registry", "fn": "step2"},
    {"id": 4, "name": "Encode to 9-dim signals", "fn": "step3"},
    {"id": 5, "name": "Track A: 9-dim analysis", "fn": "step4"},
    {"id": 6, "name": "Encode through VAE", "fn": "step5"},
    {"id": 7, "name": "Track B: latent analysis", "fn": "step6"},
    {"id": 8, "name": "Synthesis: 32x9 correlation", "fn": "step7"},
    {"id": 9, "name": "Summary", "fn": "step9"},
]


def detect_precomputed() -> dict:
    """Check what pre-computed data exists in the output directory."""
    info = {
        "has_index": (DATA_DIR / "multi_mapper_index.json").exists(),
        "has_registry": (DATA_DIR / "beatmap_registry.json").exists(),
        "has_9dim_manifest": (DATA_DIR / "encoding_manifest.json").exists(),
        "has_latent_manifest": (DATA_DIR / "latent_manifest.json").exists(),
        "n_9dim": 0,
        "n_latent": 0,
        "has_track_a": (RESULTS_DIR / "track_a" / "track_a_results.json").exists(),
        "has_track_b": (RESULTS_DIR / "track_b" / "track_b_results.json").exists(),
        "has_synthesis": (RESULTS_DIR / "synthesis" / "synthesis_results.json").exists(),
        "sample_data_downloaded": SAMPLE_DATA_LOCAL.exists(),
    }

    enc_9dim_dir = DATA_DIR / "encodings_9dim"
    enc_latent_dir = DATA_DIR / "encodings_latent"
    if enc_9dim_dir.exists():
        info["n_9dim"] = len(list(enc_9dim_dir.glob("*.npy")))
    if enc_latent_dir.exists():
        info["n_latent"] = len(list(enc_latent_dir.glob("*.npy")))

    # Can skip steps 1-4 if we have manifests + encodings
    info["can_skip_encoding"] = (
        info["has_9dim_manifest"] and info["n_9dim"] > 0
    )
    # Can skip VAE encoding if we have latent data
    info["can_skip_vae"] = (
        info["has_latent_manifest"] and info["n_latent"] > 0
    )

    return info


def download_sample_data(progress_callback=None) -> bool:
    """Download and extract the pre-computed sample data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "cartography-sample-data.tar.gz"

    try:
        if progress_callback:
            progress_callback("downloading", "Downloading sample data (~900MB)...")

        # Download with progress
        def _reporthook(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                progress_callback("downloading", f"Downloading: {mb:.0f}/{total_mb:.0f} MB ({pct}%)")

        urllib.request.urlretrieve(SAMPLE_DATA_URL, str(tar_path), _reporthook)

        if progress_callback:
            progress_callback("extracting", "Extracting data...")

        # Extract to DATA_DIR
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=str(DATA_DIR))

        # Clean up tar
        tar_path.unlink(missing_ok=True)

        # Mark as downloaded
        SAMPLE_DATA_LOCAL.touch()

        if progress_callback:
            progress_callback("done", "Sample data ready!")

        return True

    except Exception as e:
        if progress_callback:
            progress_callback("error", f"Download failed: {e}")
        tar_path.unlink(missing_ok=True)
        return False


class PipelineRunner:
    """Runs the analysis pipeline with progress callbacks."""

    def __init__(self, dataset_dir: str, checkpoint_path: str = None,
                 use_precomputed: bool = False):
        self.dataset_dir = Path(dataset_dir).resolve() if dataset_dir and dataset_dir != "." else None
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else CHECKPOINT_PATH
        self.use_precomputed = use_precomputed
        self.listeners = []
        self.running = False
        self.cancel_requested = False
        self.thread = None

        # Pipeline state (passed between steps)
        self._multi_mapper = None
        self._registry = None
        self._manifest_9dim = None
        self._track_a = None
        self._latent_manifest = None
        self._track_b = None
        self._synthesis = None

    def add_listener(self, callback):
        """Add a callback(event_type, data) for progress events."""
        self.listeners.append(callback)

    def remove_listener(self, callback):
        self.listeners.remove(callback)

    def _emit(self, event_type: str, data: dict):
        for cb in self.listeners:
            try:
                cb(event_type, data)
            except Exception:
                pass

    def _skip_step(self, step_id: int, step_name: str, reason: str):
        """Mark a step as skipped (pre-computed data found)."""
        self._emit("step_start", {"step": step_id, "name": step_name, "total": len(STEPS)})
        self._emit("step_log", {"step": step_id, "log": f"  Skipped: {reason}\n"})
        self._emit("step_done", {"step": step_id, "name": step_name})

    def _capture_step(self, step_id: int, step_name: str, fn, *args, **kwargs):
        """Run a step function, capturing stdout and emitting progress."""
        if self.cancel_requested:
            return None

        self._emit("step_start", {"step": step_id, "name": step_name, "total": len(STEPS)})

        # Capture stdout
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        result = None
        error = None
        try:
            result = fn(*args, **kwargs)
        except SystemExit:
            error = "Step called sys.exit (non-fatal, continuing)"
        except Exception as e:
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

        log_text = captured.getvalue()
        if log_text:
            self._emit("step_log", {"step": step_id, "log": log_text})

        if error:
            self._emit("step_error", {"step": step_id, "name": step_name, "error": error})
        else:
            self._emit("step_done", {"step": step_id, "name": step_name})

        return result

    def _load_precomputed(self):
        """Load manifests from pre-computed data."""
        manifest_path = DATA_DIR / "encoding_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self._manifest_9dim = json.load(f)

        latent_manifest_path = DATA_DIR / "latent_manifest.json"
        if latent_manifest_path.exists():
            with open(latent_manifest_path) as f:
                self._latent_manifest = json.load(f)

        index_path = DATA_DIR / "multi_mapper_index.json"
        if index_path.exists():
            with open(index_path) as f:
                self._multi_mapper = json.load(f)

    def run(self):
        """Run the full pipeline (blocking). Call from a thread for async."""
        self.running = True
        self.cancel_requested = False

        import experiment.run_analysis as ra
        original_ckpt = ra.CHECKPOINT_PATH
        if self.checkpoint_path != CHECKPOINT_PATH:
            ra.CHECKPOINT_PATH = self.checkpoint_path

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        precomputed = detect_precomputed()
        skip_encoding = self.use_precomputed and precomputed["can_skip_encoding"]
        skip_vae = self.use_precomputed and precomputed["can_skip_vae"]

        try:
            active_steps = []
            for s in STEPS:
                step = dict(s)
                if skip_encoding and s["id"] in (1, 2, 3, 4):
                    step["skip"] = True
                if skip_vae and s["id"] == 6:
                    step["skip"] = True
                active_steps.append(step)

            self._emit("pipeline_start", {
                "dataset_dir": str(self.dataset_dir) if self.dataset_dir else "(using pre-computed data)",
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_found": self.checkpoint_path.exists(),
                "using_precomputed": skip_encoding,
                "steps": [{"id": s["id"], "name": s["name"], "skip": s.get("skip", False)} for s in active_steps],
            })

            if skip_encoding:
                # Load pre-computed manifests
                self._load_precomputed()
                n9 = len(self._manifest_9dim) if self._manifest_9dim else 0
                nz = len(self._latent_manifest) if self._latent_manifest else 0

                self._skip_step(1, "Build multi-mapper index",
                    f"using pre-computed data ({len(self._multi_mapper or [])} song groups)")
                self._skip_step(2, "Extract .osz archives", "using pre-computed data")
                self._skip_step(3, "Build beatmap registry", "using pre-computed data")
                self._skip_step(4, "Encode to 9-dim signals",
                    f"using pre-computed encodings ({n9} files)")

                if not self._manifest_9dim:
                    self._emit("pipeline_done", {"success": False, "error": "Pre-computed manifest not found"})
                    return
            else:
                # Full pipeline from .osz files
                if not self.dataset_dir or not self.dataset_dir.is_dir():
                    self._emit("pipeline_done", {"success": False, "error": "No dataset directory specified"})
                    return

                self._multi_mapper = self._capture_step(
                    1, "Build multi-mapper index",
                    step1_build_index, self.dataset_dir
                )
                if not self._multi_mapper:
                    self._emit("pipeline_done", {"success": False, "error": "No multi-mapper songs found"})
                    return

                self._capture_step(
                    2, "Extract .osz archives",
                    step1b_extract_osz, self.dataset_dir, self._multi_mapper
                )

                self._registry = self._capture_step(
                    3, "Build beatmap registry",
                    step2_build_registry, self._multi_mapper
                )

                self._manifest_9dim = self._capture_step(
                    4, "Encode to 9-dim signals",
                    step3_encode_9dim, self._registry
                )
                if not self._manifest_9dim:
                    self._emit("pipeline_done", {"success": False, "error": "No beatmaps could be encoded"})
                    return

            # Step 5: Track A (always runs — fast)
            self._track_a = self._capture_step(
                5, "Track A: 9-dim analysis",
                step4_track_a, self._manifest_9dim
            )

            # Step 6: VAE encode
            if skip_vae:
                nz = len(self._latent_manifest) if self._latent_manifest else 0
                self._skip_step(6, "Encode through VAE",
                    f"using pre-computed latent encodings ({nz} files)")
            else:
                self._latent_manifest = self._capture_step(
                    6, "Encode through VAE",
                    step5_encode_latent, self._manifest_9dim
                )

            # Step 7: Track B
            self._track_b = self._capture_step(
                7, "Track B: latent analysis",
                step6_track_b, self._latent_manifest or []
            )

            # Step 8: Synthesis
            self._synthesis = self._capture_step(
                8, "Synthesis: 32x9 correlation",
                step7_synthesis, self._track_a, self._track_b,
                self._manifest_9dim, self._latent_manifest or []
            )

            # Step 9: Summary
            self._capture_step(
                9, "Summary",
                step9_summary, self._track_a, self._track_b, self._synthesis
            )

            self._emit("pipeline_done", {"success": True})

        except Exception as e:
            self._emit("pipeline_done", {
                "success": False,
                "error": f"{type(e).__name__}: {e}"
            })
        finally:
            ra.CHECKPOINT_PATH = original_ckpt
            self.running = False

    def run_async(self):
        """Run the pipeline in a background thread."""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self.thread

    def cancel(self):
        """Request cancellation (checked between steps)."""
        self.cancel_requested = True

    def get_results(self) -> dict:
        """Load results from disk (for the results viewer)."""
        results = {}

        track_a_path = RESULTS_DIR / "track_a" / "track_a_results.json"
        if track_a_path.exists():
            with open(track_a_path) as f:
                results["track_a"] = json.load(f)

        track_b_path = RESULTS_DIR / "track_b" / "track_b_results.json"
        if track_b_path.exists():
            with open(track_b_path) as f:
                results["track_b"] = json.load(f)

        synth_path = RESULTS_DIR / "synthesis" / "synthesis_results.json"
        if synth_path.exists():
            with open(synth_path) as f:
                results["synthesis"] = json.load(f)

        results["has_track_a"] = "track_a" in results
        results["has_track_b"] = "track_b" in results
        results["has_synthesis"] = "synthesis" in results
        results["output_dir"] = str(OUTPUT_DIR)

        return results
