"""
HTTP server with API endpoints and SSE streaming for the GUI.

Endpoints:
  GET /              -> static/index.html
  GET /static/...    -> static files (CSS, JS)
  POST /api/pick-folder  -> open native folder picker (Tkinter subprocess)
  POST /api/pick-file    -> open native file picker
  POST /api/run          -> start pipeline (returns immediately, stream via SSE)
  POST /api/cancel       -> cancel running pipeline
  GET  /api/sse          -> SSE stream for pipeline progress
  GET  /api/results      -> load results JSON from disk
  GET  /api/status       -> check if pipeline is running
"""

import json
import os
import queue
import subprocess
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from gui.pipeline_runner import PipelineRunner

STATIC_DIR = Path(__file__).parent / "static"


class GUIHandler(SimpleHTTPRequestHandler):
    """Handler for the GUI web server."""

    runner: PipelineRunner = None
    sse_queues: list = []
    _lock = threading.Lock()

    def log_message(self, format, *args):
        # Suppress default request logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._serve_file(STATIC_DIR / "index.html", "text/html")
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            file_path = STATIC_DIR / rel
            if file_path.exists() and file_path.is_file():
                content_type = self._guess_type(file_path)
                self._serve_file(file_path, content_type)
            else:
                self.send_error(404)
        elif path == "/api/sse":
            self._handle_sse()
        elif path == "/api/results":
            self._handle_results()
        elif path == "/api/status":
            self._handle_status()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        body = b""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)

        if path == "/api/pick-folder":
            self._handle_pick_folder()
        elif path == "/api/pick-file":
            self._handle_pick_file(body)
        elif path == "/api/run":
            self._handle_run(body)
        elif path == "/api/cancel":
            self._handle_cancel()
        else:
            self.send_error(404)

    def _serve_file(self, file_path: Path, content_type: str):
        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(500)

    def _json_response(self, data: dict, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _guess_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")

    # ─── File Pickers (Tkinter via subprocess) ───

    def _handle_pick_folder(self):
        path = _pick_dialog("folder")
        self._json_response({"path": path})

    def _handle_pick_file(self, body: bytes):
        params = json.loads(body) if body else {}
        filetypes = params.get("filetypes", [("Checkpoint", "*.ckpt")])
        path = _pick_dialog("file", filetypes=filetypes)
        self._json_response({"path": path})

    # ─── Pipeline Control ───

    def _handle_run(self, body: bytes):
        params = json.loads(body) if body else {}
        dataset_dir = params.get("dataset_dir", "")
        checkpoint = params.get("checkpoint", "")

        if not dataset_dir or not Path(dataset_dir).is_dir():
            self._json_response({"error": "Invalid dataset directory"}, 400)
            return

        cls = type(self)
        if cls.runner and cls.runner.running:
            self._json_response({"error": "Pipeline already running"}, 409)
            return

        cls.runner = PipelineRunner(dataset_dir, checkpoint or None)

        def on_event(event_type, data):
            with cls._lock:
                for q in cls.sse_queues:
                    q.put((event_type, data))

        cls.runner.add_listener(on_event)
        cls.runner.run_async()

        self._json_response({"status": "started"})

    def _handle_cancel(self):
        cls = type(self)
        if cls.runner and cls.runner.running:
            cls.runner.cancel()
            self._json_response({"status": "cancelling"})
        else:
            self._json_response({"status": "not_running"})

    def _handle_status(self):
        cls = type(self)
        running = cls.runner is not None and cls.runner.running
        self._json_response({"running": running})

    # ─── SSE Stream ───

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = queue.Queue()
        cls = type(self)
        with cls._lock:
            cls.sse_queues.append(q)

        try:
            while True:
                try:
                    event_type, data = q.get(timeout=30)
                    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                    if event_type == "pipeline_done":
                        break
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with cls._lock:
                if q in cls.sse_queues:
                    cls.sse_queues.remove(q)

    # ─── Results ───

    def _handle_results(self):
        cls = type(self)
        if cls.runner:
            results = cls.runner.get_results()
        else:
            # Try loading from disk anyway
            runner = PipelineRunner(".", None)
            results = runner.get_results()
        self._json_response(results)


def _pick_dialog(kind: str, filetypes=None) -> str:
    """Open a native file/folder dialog via Tkinter subprocess.

    Uses subprocess to avoid macOS thread-safety issues with Tkinter.
    """
    if kind == "folder":
        script = (
            "import tkinter as tk; "
            "root = tk.Tk(); root.withdraw(); "
            "from tkinter import filedialog; "
            "p = filedialog.askdirectory(title='Select Beatmap Folder'); "
            "print(p if p else '')"
        )
    else:
        ft_str = repr(filetypes) if filetypes else "[('All files', '*.*')]"
        script = (
            "import tkinter as tk; "
            "root = tk.Tk(); root.withdraw(); "
            "from tkinter import filedialog; "
            f"p = filedialog.askopenfilename(title='Select File', filetypes={ft_str}); "
            "print(p if p else '')"
        )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()
    except Exception:
        return ""


def create_server(port: int = 8765) -> HTTPServer:
    """Create and return the GUI HTTP server."""
    server = HTTPServer(("127.0.0.1", port), GUIHandler)
    return server
