"""Tiny HTTP server for probes, metrics and a status page.

The standard library is enough: four read-only endpoints, scraped every 30
seconds and probed every few seconds. A web framework would add dependencies
and attack surface for nothing.

- ``/healthz``: the process is alive (liveness).
- ``/readyz``: the model is loaded and warmed up (readiness).
- ``/metrics``: Prometheus text format.
- ``/status``: JSON summary for people (exposed through the HTTPRoute).

Stages can add their own read-only routes (the ranking service serves its
JSON API and the leaderboard page this way): ``routes`` maps a path to a
function taking the query string and returning (status, body, content type).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

Route = Callable[[dict[str, list[str]]], tuple[int, bytes, str]]


class StatusServer:
    def __init__(
        self,
        port: int,
        metrics,
        status: Callable[[], dict[str, Any]],
        host: str = "0.0.0.0",
        routes: dict[str, Route] | None = None,
    ):
        self.metrics = metrics
        self.status = status
        self.routes = dict(routes or {})
        self.ready = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parts = urlsplit(self.path)
                path = parts.path
                if path in server.routes:
                    try:
                        code, body, content_type = server.routes[path](parse_qs(parts.query))
                    except Exception as err:  # a route bug must not kill the server
                        code, body, content_type = 500, f"error: {err}\n".encode(), "text/plain"
                    self._send(code, body, content_type)
                elif path == "/healthz":
                    self._send(200, b"ok\n", "text/plain")
                elif path == "/readyz":
                    if server.ready.is_set():
                        self._send(200, b"ready\n", "text/plain")
                    else:
                        self._send(503, b"loading\n", "text/plain")
                elif path == "/metrics":
                    self._send(200, server.metrics.render(), "text/plain; version=0.0.4")
                elif path in ("/", "/status") and path not in server.routes:
                    body = json.dumps(server.status(), indent=2, default=str).encode() + b"\n"
                    self._send(200, body, "application/json")
                else:
                    self._send(404, b"not found\n", "text/plain")

            def _send(self, code: int, body: bytes, content_type: str):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                # The leaderboard is public read-only data; these keep a
                # browser from sniffing types or framing it elsewhere.
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                # Probes every few seconds would drown the useful logs.
                pass

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="status-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
