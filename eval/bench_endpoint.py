"""Measure end-to-end latency of the deployed endpoint (network + Lambda + OpenCV).

    python -m eval.bench_endpoint https://xxxx.lambda-url.ap-south-1.on.aws/ --n 20
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"


def call(url: str, payload: bytes) -> tuple[float, dict]:
    req = urllib.request.Request(url.rstrip("/") + "/api/grade", data=payload, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/endpoint_latency.json")
    a = ap.parse_args()
    out = {}
    for name in ("sample-clean.jpg", "sample-review.jpg", "sample-blurry.jpg"):
        payload = json.dumps({"image_b64": base64.b64encode((STATIC / name).read_bytes()).decode(), "n_questions": 40}).encode()
        e2e, server, status = [], [], None
        for _ in range(a.n):
            ms, body = call(a.url, payload)
            e2e.append(ms); server.append(body["ms_total"]); status = body["status"]
        out[name] = {"status": status, "n": a.n, "e2e_ms_p50": float(np.percentile(e2e, 50)), "e2e_ms_p95": float(np.percentile(e2e, 95)),
                     "agent_ms_p50": float(np.percentile(server, 50)), "agent_ms_p95": float(np.percentile(server, 95)), "first_call_ms": e2e[0]}
        print(name, json.dumps(out[name]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
