"""
Pipeline runner: wraps run_analysis.py step functions with progress callbacks.

Captures stdout from each step and emits structured events for SSE streaming.
"""

import io
import json
import sys
import threading
import traceback
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


class PipelineRunner:
    """Runs the analysis pipeline with progress callbacks."""

    def __init__(self, dataset_dir: str, checkpoint_path: str = None):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else CHECKPOINT_PATH
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

    def _capture_step(self, step_id: str, step_name: str, fn, *args, **kwargs):
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

    def run(self):
        """Run the full pipeline (blocking). Call from a thread for async."""
        self.running = True
        self.cancel_requested = False

        import experiment.run_analysis as ra
        # Override checkpoint path if custom
        original_ckpt = ra.CHECKPOINT_PATH
        if self.checkpoint_path != CHECKPOINT_PATH:
            ra.CHECKPOINT_PATH = self.checkpoint_path

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        try:
            self._emit("pipeline_start", {
                "dataset_dir": str(self.dataset_dir),
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_found": self.checkpoint_path.exists(),
                "steps": [{"id": s["id"], "name": s["name"]} for s in STEPS],
            })

            # Step 1: Build index
            self._multi_mapper = self._capture_step(
                1, "Build multi-mapper index",
                step1_build_index, self.dataset_dir
            )
            if not self._multi_mapper:
                self._emit("pipeline_done", {"success": False, "error": "No multi-mapper songs found"})
                return

            # Step 2: Extract
            self._capture_step(
                2, "Extract .osz archives",
                step1b_extract_osz, self.dataset_dir, self._multi_mapper
            )

            # Step 3: Build registry
            self._registry = self._capture_step(
                3, "Build beatmap registry",
                step2_build_registry, self._multi_mapper
            )

            # Step 4: Encode 9-dim
            self._manifest_9dim = self._capture_step(
                4, "Encode to 9-dim signals",
                step3_encode_9dim, self._registry
            )
            if not self._manifest_9dim:
                self._emit("pipeline_done", {"success": False, "error": "No beatmaps could be encoded"})
                return

            # Step 5: Track A
            self._track_a = self._capture_step(
                5, "Track A: 9-dim analysis",
                step4_track_a, self._manifest_9dim
            )

            # Step 6: VAE encode
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

        # Track A
        track_a_path = RESULTS_DIR / "track_a" / "track_a_results.json"
        if track_a_path.exists():
            with open(track_a_path) as f:
                results["track_a"] = json.load(f)

        # Track B
        track_b_path = RESULTS_DIR / "track_b" / "track_b_results.json"
        if track_b_path.exists():
            with open(track_b_path) as f:
                results["track_b"] = json.load(f)

        # Synthesis
        synth_path = RESULTS_DIR / "synthesis" / "synthesis_results.json"
        if synth_path.exists():
            with open(synth_path) as f:
                results["synthesis"] = json.load(f)

        results["has_track_a"] = "track_a" in results
        results["has_track_b"] = "track_b" in results
        results["has_synthesis"] = "synthesis" in results
        results["output_dir"] = str(OUTPUT_DIR)

        return results
