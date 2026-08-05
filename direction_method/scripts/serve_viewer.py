"""Local HTTP server for browsing run_full_sweep.py's logs -- the "webpage to visually check
failures later" the sweep's rich per-run JSON records were built for.

Zero external dependencies (stdlib http.server only) so it runs in any environment that can run
the rest of this repo. Binds to 127.0.0.1 by default -- this is a local dev tool serving files
from arbitrary absolute paths on disk (subject to the allow-list below), not something to expose
on a shared network interface. If you're on a remote box, reach it via an SSH tunnel:

    ssh -L 8765:localhost:8765 <host>

then open http://localhost:8765/ locally.

Run: `python scripts/serve_viewer.py` (defaults: log root = the sweep's own default output dir,
port 8765). Override with `VIEWER_LOG_ROOT=... VIEWER_PORT=... python scripts/serve_viewer.py`.
"""

import json
import mimetypes
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = DIRECTION_ROOT / "viewer"

LOG_ROOT = Path(os.environ.get("VIEWER_LOG_ROOT", "/data/gyeom/coin_challenge/direction_method_logs/full_sweep_real"))
EPISODES_DIR = LOG_ROOT / "episodes"
PORT = int(os.environ.get("VIEWER_PORT", "8765"))

# Only paths under these roots can be served via /api/image -- the images are already-persistent
# distractor/target files under IMAGE_ROOT, or synthesized boxed images under the log dir itself.
IMAGE_ROOT = Path("/data/gyeom/coin_challenge/images").resolve()
ALLOWED_IMAGE_ROOTS = (IMAGE_ROOT, LOG_ROOT.resolve())


def _load_all_records():
    records = {}
    for p in sorted(EPISODES_DIR.glob("*.json")):
        with open(p) as f:
            records[p.stem] = json.load(f)
    return records


def _record_summary(key: str, record: dict) -> dict:
    return {
        "key": key,
        "episode_idx": record["episode_idx"],
        "episode_id": record["episode_id"],
        "task_type": record["task_type"],
        "target_description": record["target_description"],
        "category": record["category"],
        "outcome": record["outcome"],
        "full_success": record["full_success"],
        "n_successes": record["n_successes"],
        "n_distractors": record["n_distractors"],
        "n_questions": record["n_questions"],
        "candidates_seen": record["candidates_seen"],
        "wall_clock_s": record["wall_clock_s"],
        "n_candidate_logs": len(record["candidates"]),
    }


def _is_allowed_image_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in ALLOWED_IMAGE_ROOTS)


class Handler(BaseHTTPRequestHandler):
    records = {}  # populated once at startup by main(); key -> full record dict

    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None):
        if not path.is_file():
            self._send_json({"error": f"not found: {path}"}, status=404)
            return
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/" or parsed.path == "/index.html":
            self._send_file(VIEWER_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path.startswith("/static/"):
            rel = parsed.path[len("/static/"):]
            target = (VIEWER_DIR / rel).resolve()
            if VIEWER_DIR.resolve() not in target.parents and target != VIEWER_DIR.resolve():
                self._send_json({"error": "forbidden"}, status=403)
                return
            self._send_file(target)
        elif parsed.path == "/api/records":
            summaries = [_record_summary(k, r) for k, r in self.records.items()]
            summaries.sort(key=lambda s: (s["episode_idx"], s["task_type"]))
            self._send_json(summaries)
        elif parsed.path == "/api/episode":
            key = qs.get("key", [None])[0]
            record = self.records.get(key)
            if record is None:
                self._send_json({"error": f"unknown key: {key}"}, status=404)
                return
            self._send_json(record)
        elif parsed.path == "/api/image":
            raw = qs.get("path", [None])[0]
            if not raw:
                self._send_json({"error": "missing path"}, status=400)
                return
            path = Path(raw)
            if not _is_allowed_image_path(path):
                self._send_json({"error": "path not allowed"}, status=403)
                return
            self._send_file(path)
        else:
            self._send_json({"error": "not found"}, status=404)


def main() -> None:
    if not EPISODES_DIR.is_dir():
        print(f"No episodes dir at {EPISODES_DIR} -- run scripts/run_full_sweep.py first.")
        sys.exit(1)

    print(f"loading records from {EPISODES_DIR} ...")
    Handler.records = _load_all_records()
    print(f"loaded {len(Handler.records)} records")

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"serving on http://127.0.0.1:{PORT}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
