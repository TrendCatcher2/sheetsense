"""Run the Lambda handler locally: python -m app.local_server  (http://127.0.0.1:8080)."""

from __future__ import annotations

import base64
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.handler import lambda_handler


class H(BaseHTTPRequestHandler):
    def _go(self, method: str):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        ev = {"rawPath": self.path.split("?")[0], "body": body, "isBase64Encoded": False, "requestContext": {"http": {"method": method}}}
        r = lambda_handler(ev)
        data = base64.b64decode(r["body"]) if r.get("isBase64Encoded") else r["body"].encode()
        self.send_response(r["statusCode"])
        for k, v in r["headers"].items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._go("GET")

    def do_POST(self):
        self._go("POST")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"SheetSense on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
