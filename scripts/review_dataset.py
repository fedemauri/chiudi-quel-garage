#!/usr/bin/env python3
"""Local web page to review and relabel the imported Telegram dataset.

Serves the photos of a dataset built by import_telegram_export.py and lets you
set a final label per photo (open / closed / exclude). The task is binary:
a door open even a few cm is open; "partial" review labels are proposed as open. Decisions are
saved immediately to the final_label / final_note columns of manifest.csv.
The server listens on 127.0.0.1 only: photos never leave this machine.

Usage:
    python scripts/review_dataset.py                  # data/telegram, port 8765
    python scripts/review_dataset.py data/telegram --port 8800 --no-browser
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data/telegram"
UI_PATH = Path(__file__).with_name("review_ui.html")
FINAL_LABELS = {"open", "closed", "exclude", ""}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PHOTO_FIELDS = (
    "file", "label", "kind", "date", "confidence", "flags",
    "review_label", "review_note", "final_label", "final_note",
)
MAX_BODY = 2 * 1024 * 1024


class Dataset:
    """manifest.csv held in memory, written back atomically on every change."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "manifest.csv"
        self._lock = threading.Lock()
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.fields = list(reader.fieldnames or [])
            self.rows = list(reader)
        for col in ("final_label", "final_note"):
            if col not in self.fields:
                self.fields.append(col)
                for row in self.rows:
                    row[col] = ""
        # Same numbering as the review contact sheets: chronological file name.
        kept = sorted((r for r in self.rows if r["file"]), key=lambda r: Path(r["file"]).name)
        self.by_file = {r["file"]: r for r in kept}
        self.numbers = {r["file"]: n for n, r in enumerate(kept, start=1)}

        backup = self.path.with_name("manifest.backup.csv")
        if not backup.exists():
            shutil.copy2(self.path, backup)

    def photos(self) -> list[dict]:
        return [
            {"n": self.numbers[f], **{k: r.get(k, "") for k in PHOTO_FIELDS}}
            for f, r in self.by_file.items()
        ]

    def update(self, items: list[dict]) -> int:
        with self._lock:
            for item in items:
                row = self.by_file.get(item.get("file", ""))
                label = item.get("final_label", "")
                if row is None or label not in FINAL_LABELS:
                    raise ValueError(f"invalid item: {item!r}")
                row["final_label"] = label
                if "final_note" in item:
                    row["final_note"] = str(item["final_note"])[:500]
            self._save()
        return len(items)

    def _save(self) -> None:
        tmp = self.path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.path)

    def image_path(self, rel: str) -> Path | None:
        path = (self.root / rel).resolve()
        if path.is_relative_to(self.root) and path.suffix.lower() in IMAGE_EXTS and path.is_file():
            return path
        return None


def make_handler(dataset: Dataset):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _local_host(self) -> bool:
            # Reject requests addressed to another host name (DNS rebinding).
            host = self.headers.get("Host", "").rsplit(":", 1)[0]
            return host in ("127.0.0.1", "localhost")

        def do_GET(self) -> None:
            if not self._local_host():
                return self._json(403, {"error": "forbidden"})
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/photos":
                return self._json(200, dataset.photos())
            if path.startswith("/img/"):
                img = dataset.image_path(unquote(path[len("/img/"):]))
                if img is not None:
                    ctype = mimetypes.guess_type(img.name)[0] or "application/octet-stream"
                    return self._send(200, img.read_bytes(), ctype, cache=True)
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._local_host() or urlparse(self.path).path != "/api/labels":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY:
                return self._json(413, {"error": "bad body size"})
            try:
                items = json.loads(self.rfile.read(length))["items"]
                saved = dataset.update(items)
            except (ValueError, KeyError, TypeError) as exc:
                return self._json(400, {"error": str(exc)})
            self._json(200, {"saved": saved})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DIR, help=f"Dataset folder (default: {DEFAULT_DIR})")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(dataset))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Reviewing {len(dataset.by_file)} photos from {dataset.root}")
    print(f"Open {url}  (Ctrl+C to stop; decisions are saved to manifest.csv as you go)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
