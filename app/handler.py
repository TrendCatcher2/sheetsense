"""AWS Lambda entry point (Function URL, payload format 2.0).

GET  /                 -> web app
GET  /static/<file>    -> printable sheet, sample photos
POST /api/grade        -> run the grading agent on one photo
GET  /api/health       -> version info

Photos are decoded in memory and never written to disk, S3 or logs. Each request
emits one structured log line in CloudWatch Embedded Metric Format, so latency,
outcome and review load become CloudWatch metrics without extra API calls.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path

import cv2

from sheetsense import __version__
from sheetsense import template as T
from sheetsense import vision as V
from sheetsense.agent import GradingAgent

STATIC = Path(__file__).parent / "static"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
AGENT = GradingAgent()

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
}
CSP = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"


def _resp(status: int, body, ctype: str = "application/json", extra: dict | None = None, binary: bool = False):
    headers = {"content-type": ctype, **SECURITY_HEADERS, **(extra or {})}
    if binary:
        return {"statusCode": status, "headers": headers, "body": base64.b64encode(body).decode(), "isBase64Encoded": True}
    if not isinstance(body, str):
        body = json.dumps(body)
    return {"statusCode": status, "headers": headers, "body": body, "isBase64Encoded": False}


def _err(status: int, msg: str):
    return _resp(status, {"error": msg})


def _log(request_id: str, route: str, status: str, latency_ms: float, extra: dict):
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "SheetSense", "Dimensions": [["Route", "Outcome"]],
            "Metrics": [{"Name": "LatencyMs", "Unit": "Milliseconds"}, {"Name": "ReviewQuestions", "Unit": "Count"}, {"Name": "ToolCalls", "Unit": "Count"}]}]},
        "Route": route, "Outcome": status, "LatencyMs": round(latency_ms, 1), "request_id": request_id, **extra,
    }))


def grade(body: dict, request_id: str):
    t0 = time.perf_counter()
    b64 = body.get("image_b64", "")
    if not isinstance(b64, str) or not b64:
        return _err(400, "image_b64 is required")
    if "," in b64[:100]:  # tolerate data: URLs
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return _err(400, "image_b64 is not valid base64")
    if len(raw) > MAX_IMAGE_BYTES:
        return _err(413, "image too large; send at most 4 MB (the web app resizes photos automatically)")
    try:
        n_q = int(body.get("n_questions", T.N_QUESTIONS))
    except (TypeError, ValueError):
        return _err(400, "n_questions must be an integer")
    if not 1 <= n_q <= T.N_QUESTIONS:
        return _err(400, f"n_questions must be between 1 and {T.N_QUESTIONS}")
    img = V.decode_image(raw)
    if img is None:
        return _err(400, "could not decode image; send a JPEG or PNG photo")
    res = AGENT.run(img, n_q)
    out = res.to_dict()
    out["request_id"] = request_id
    out["version"] = __version__
    out["opencv"] = cv2.__version__
    _log(request_id, "grade", res.status, (time.perf_counter() - t0) * 1000, {
        "ReviewQuestions": len(res.review), "ToolCalls": len(res.trace), "image_px": [int(img.shape[1]), int(img.shape[0])],
        "trace": [{"tool": s.tool, "decision": s.decision, "ms": s.ms} for s in res.trace], "retake_reasons": res.retake_reasons,
    })
    return _resp(200, out)


def lambda_handler(event, context=None):
    request_id = getattr(context, "aws_request_id", None) or str(uuid.uuid4())
    http = event.get("requestContext", {}).get("http", {})
    method = http.get("method", "GET").upper()
    path = event.get("rawPath", "/")
    try:
        if method == "GET" and path in ("/", "/index.html"):
            return _resp(200, (STATIC / "index.html").read_text(), "text/html; charset=utf-8", {"content-security-policy": CSP})
        if method == "GET" and path == "/api/health":
            return _resp(200, {"ok": True, "version": __version__, "opencv": cv2.__version__, "arch": os.uname().machine})
        if method == "GET" and path.startswith("/static/"):
            rel = path[len("/static/"):]
            f = (STATIC / rel).resolve()
            if STATIC.resolve() not in f.parents or not f.is_file():
                return _err(404, "not found")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            return _resp(200, f.read_bytes(), ctype, {"cache-control": "public, max-age=3600"}, binary=True)
        if method == "POST" and path == "/api/grade":
            raw = event.get("body") or ""
            if event.get("isBase64Encoded"):
                raw = base64.b64decode(raw).decode("utf-8", "replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return _err(400, "body must be JSON")
            if not isinstance(body, dict):
                return _err(400, "body must be a JSON object")
            return grade(body, request_id)
        return _err(404, "not found")
    except Exception as e:  # never leak a stack trace to the client
        _log(request_id, path, "ERROR", 0.0, {"error": type(e).__name__, "detail": str(e)[:300]})
        return _err(500, f"internal error (request {request_id})")
