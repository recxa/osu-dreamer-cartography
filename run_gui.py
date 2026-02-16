#!/usr/bin/env python3
"""
One-click GUI launcher for Latent Space Cartography.

Starts a local web server and opens the browser.

Usage:
    python run_gui.py
    python run_gui.py --port 8888
"""

import argparse
import sys
import webbrowser
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from gui.server import create_server


def main():
    parser = argparse.ArgumentParser(description="Latent Space Cartography GUI")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    server = create_server(args.port)
    url = f"http://127.0.0.1:{args.port}"

    print(f"Latent Space Cartography GUI")
    print(f"  Server: {url}")
    print(f"  Press Ctrl+C to stop\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
