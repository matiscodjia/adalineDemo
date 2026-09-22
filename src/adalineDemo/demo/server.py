"""Local web demo: `adaline-demo` (or `python -m adalineDemo.demo`).

Standard library only. The server binds to localhost by default (`--lan` opens it to the local
network, for a phone on the same Wi-Fi) and, unless `--lan`, refuses any Host header that isn't
localhost, so a page on the open internet cannot redirect a browser into this server.
"""

import argparse
import json
import re
import socket
import ssl
import subprocess
import tempfile
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from adalineDemo import _native, chip
from adalineDemo.demo import runner

STATIC = Path(__file__).parent / "static"
ASSET = re.compile(r"^[a-z0-9_.-]+\.(css|js|json|svg)$")
ASSET_TYPES = {"css": "text/css", "js": "text/javascript", "json": "application/json", "svg": "image/svg+xml"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
MEDIA_NAME = re.compile(r"^[0-9a-f]{32}\.wav$")
MAX_BODY = 4_096
MAX_JOBS = 6


class Jobs:
    """Runs one job at a time (timings must not compete), polled by the page."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._running = threading.Lock()

    def start(self, work) -> str | None:
        if not self._running.acquire(blocking=False):
            return None
        job_id = uuid.uuid4().hex
        job = {"state": "running", "stage": "Starting"}
        with self._lock:
            self._jobs[job_id] = job
            for old in [k for k, v in self._jobs.items() if v["state"] != "running"][:-MAX_JOBS]:
                del self._jobs[old]

        def target():
            try:
                job["result"] = work(lambda stage: job.update(stage=stage))
                job["state"] = "done"
            except Exception as exc:  # decoding, math edge cases, ...: report, keep serving
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["state"] = "error"
            finally:
                self._running.release()

        threading.Thread(target=target, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


class Handler(BaseHTTPRequestHandler):
    media_dir: Path
    jobs = Jobs()
    lan = False

    server_version = "adalineDemo"

    def log_message(self, fmt, *args):
        pass

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in LOCAL_HOSTS or (self.lan and bool(IPV4.match(host)))

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
        if self.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path.startswith("/static/"):
            return self._asset(self.path[len("/static/") :])
        if self.path == "/api/config":
            return self._json(200, self._config())
        if self.path.startswith("/api/job/"):
            job = self.jobs.get(self.path[len("/api/job/") :])
            return self._json(200, job) if job else self._json(HTTPStatus.NOT_FOUND, {"error": "unknown job"})
        if self.path.startswith("/media/"):
            return self._media(self.path[len("/media/") :])
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    do_HEAD = do_GET

    def _asset(self, name: str):
        file = STATIC / name
        if not ASSET.match(name) or not file.is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        self._send(200, file.read_bytes(), ASSET_TYPES[name.rsplit(".", 1)[1]] + "; charset=utf-8")

    def do_POST(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
        if self.path != "/api/run":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                raise ValueError("bad request body")
            req = json.loads(self.rfile.read(length))
            taps, fs = int(req["taps"]), int(req["fs"])
            lr, duration_s = float(req["lr"]), float(req["duration_s"])
            runner.validate(taps, fs, lr, duration_s)  # fail fast, before starting a job
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})
        job_id = self.jobs.start(lambda progress: runner.run(taps, fs, lr, duration_s, str(self.media_dir), progress))
        if job_id is None:
            return self._json(HTTPStatus.CONFLICT, {"error": "another run is in progress, wait for it to finish"})
        self._json(HTTPStatus.ACCEPTED, {"job": job_id})

    def _config(self) -> dict:
        return {
            "taps_ladder": chip.TAPS_LADDER,
            "sample_rates": chip.SAMPLE_RATES,
            "durations_s": chip.DURATIONS_S,
            "default_duration_s": chip.DEFAULT_DURATION_S,
            "default_lr": runner.DEFAULT_LR,
            "lr_min": runner.LR_MIN,
            "lr_max": runner.LR_MAX,
            "chip": {
                "name": chip.CHIP_NAME,
                "clock_hz": chip.CHIP_CLOCK_HZ,
                "verified_taps": chip.VERIFIED_TAPS,
                "verified_cycles": chip.VERIFIED_CYCLES,
            },
            "signal": dict(zip(("f_signal", "a_signal", "f_noise", "a_noise"), _native.signal_params())),
        }

    def _media(self, name: str):
        if not MEDIA_NAME.match(name) or not (self.media_dir / name).is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        data = (self.media_dir / name).read_bytes()
        headers = {"Accept-Ranges": "bytes"}
        match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if match and (match[1] or match[2]):
            if match[1]:
                start = int(match[1])
                end = int(match[2]) if match[2] else len(data) - 1
            else:
                start, end = max(0, len(data) - int(match[2])), len(data) - 1
            end = min(end, len(data) - 1)
            if start > end:
                return self._send(416, b"", "audio/wav", {"Content-Range": f"bytes */{len(data)}"})
            headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            return self._send(206, data[start : end + 1], "audio/wav", headers)
        self._send(200, data, "audio/wav", headers)


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))  # no packet sent: only picks the outgoing interface
        return probe.getsockname()[0]


def wrap_tls(server: ThreadingHTTPServer, ip: str, directory: str) -> None:
    """Self-signed HTTPS, for browsers (Safari with "HTTPS-Only") that refuse plain http:// links."""
    cert, key = f"{directory}/cert.pem", f"{directory}/key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "7", "-subj", "/CN=adalineDemo",
         "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost", "-keyout", key, "-out", cert],
        check=True, capture_output=True,
    )  # fmt: skip
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hear frugal_ml's ADALINE (LMS) cancel noise, live.")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--lan", action="store_true", help="also listen on the local network, to open the demo from a phone")
    parser.add_argument("--https", action="store_true", help="serve HTTPS with a self-signed certificate")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="adaline-demo-") as media_dir, tempfile.TemporaryDirectory(prefix="adaline-tls-") as tls_dir:
        Handler.media_dir = Path(media_dir)
        Handler.lan = args.lan
        server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), Handler)
        ip = lan_ip() if args.lan else "127.0.0.1"
        if args.https:
            wrap_tls(server, ip, tls_dir)
        scheme = "https" if args.https else "http"
        url = f"{scheme}://localhost:{args.port}"
        print(f"adalineDemo on {url}  (Ctrl+C to stop)", flush=True)
        if args.lan:
            print(f"On your phone (same Wi-Fi): {scheme}://{ip}:{args.port}", flush=True)
            if args.https:
                print("The browser will warn about the certificate (self-signed): choose to visit the site anyway.", flush=True)
            print("Anyone on this network can use the demo while it runs.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
