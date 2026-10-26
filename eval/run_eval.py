"""Score the agent (and baselines) against synthetic ground truth.

    python -m eval.run_eval --data data/synth --out results/test
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from sheetsense.agent import GradingAgent

VARIANTS = {
    "agent": dict(),
    "no_zoom": dict(use_zoom=False),
    "no_review": dict(allow_review=False),
    "baseline_single_pass": dict(use_zoom=False, use_flatten=False, allow_review=False, escalate=False, quality_gate=False),
}


def evaluate(data: Path, variant: str, limit: int | None = None, policy: dict | None = None, keep: bool = False):
    gt = json.loads((data / "gt.json").read_text())["sheets"]
    if limit:
        gt = gt[:limit]
    agent = GradingAgent(policy=policy, render=False, **VARIANTS[variant])
    rows = []
    for s in gt:
        img = cv2.imread(str(data / s["file"]))
        t0 = time.perf_counter()
        r = agent.run(img)
        ms = (time.perf_counter() - t0) * 1000
        row = {"file": s["file"], "split": s["split"], "expect": s["expect"], "retake_kind": s["retake_kind"], "status": r.status, "ms": ms,
               "steps": len(r.trace), "tools": [st.tool for st in r.trace], "reasons": r.retake_reasons, "q": []}
        if r.status != "RETAKE":
            for i, (lab, kind) in enumerate(zip(s["labels"], s["kinds"])):
                pred = r.answers[i]
                row["q"].append({"i": i + 1, "gt": lab, "kind": kind, "pred": pred, "auto": pred != "?", "correct": pred == lab})
        rows.append(row)
    return rows


def summarise(rows):
    out = {}
    gradeable = [r for r in rows if r["expect"] == "GRADE"]
    retake = [r for r in rows if r["expect"] == "RETAKE"]
    qs = [q for r in gradeable for q in r["q"]]
    auto = [q for q in qs if q["auto"]]
    n_total_q = 40 * len(gradeable)
    out["sheets_gradeable"] = len(gradeable)
    out["false_retake_rate"] = sum(r["status"] == "RETAKE" for r in gradeable) / max(1, len(gradeable))
    out["questions_total"] = n_total_q
    out["auto_coverage"] = len(auto) / max(1, n_total_q)  # share of all questions decided without a human
    out["auto_accuracy"] = sum(q["correct"] for q in auto) / max(1, len(auto))  # precision of automatic decisions
    out["silent_errors"] = sum(not q["correct"] for q in auto)
    out["silent_errors_per_1000q"] = 1000 * out["silent_errors"] / max(1, n_total_q)
    out["review_rate"] = sum(not q["auto"] for q in qs) / max(1, n_total_q)
    out["sheets_fully_auto_and_correct"] = sum(r["status"] == "READY" and all(q["correct"] for q in r["q"]) for r in gradeable) / max(1, len(gradeable))
    out["sheets_needing_review"] = sum(r["status"] == "NEEDS_REVIEW" for r in gradeable) / max(1, len(gradeable))
    out["mean_review_q_per_sheet"] = sum(not q["auto"] for q in qs) / max(1, len(gradeable))
    by_split = defaultdict(list)
    for r in gradeable:
        by_split[r["split"]].append(r)
    out["by_split"] = {}
    for sp, rs in by_split.items():
        qq = [q for r in rs for q in r["q"]]
        aa = [q for q in qq if q["auto"]]
        out["by_split"][sp] = {"sheets": len(rs), "false_retake": sum(r["status"] == "RETAKE" for r in rs),
                               "auto_accuracy": sum(q["correct"] for q in aa) / max(1, len(aa)),
                               "silent_errors": sum(not q["correct"] for q in aa),
                               "review_rate": sum(not q["auto"] for q in qq) / max(1, 40 * len(rs))}
    by_kind = defaultdict(list)
    for q in qs:
        by_kind[q["kind"]].append(q)
    out["by_kind"] = {k: {"n": len(v), "auto_accuracy": sum(q["correct"] for q in v if q["auto"]) / max(1, sum(q["auto"] for q in v)),
                          "silent_errors": sum(q["auto"] and not q["correct"] for q in v),
                          "review_rate": sum(not q["auto"] for q in v) / max(1, len(v))} for k, v in sorted(by_kind.items())}
    out["retake_sheets"] = len(retake)
    out["retake_requested"] = sum(r["status"] == "RETAKE" for r in retake) / max(1, len(retake))
    rq = [q for r in retake for q in r["q"] if q["auto"]]
    out["retake_set_silent_errors"] = sum(not q["correct"] for q in rq)
    out["retake_by_kind"] = {k: {"n": len(v), "retake": sum(r["status"] == "RETAKE" for r in v),
                                 "silent_errors": sum(1 for r in v for q in r["q"] if q["auto"] and not q["correct"])}
                             for k, v in sorted(_group(retake, "retake_kind").items())}
    ms = np.array([r["ms"] for r in rows])
    out["latency_ms_p50"] = float(np.percentile(ms, 50))
    out["latency_ms_p95"] = float(np.percentile(ms, 95))
    out["mean_tool_calls"] = float(np.mean([r["steps"] for r in rows]))
    return out


def _group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="results/test")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allsum = {}
    for v in a.variants.split(","):
        rows = evaluate(Path(a.data), v, a.limit)
        (out / f"rows_{v}.json").write_text(json.dumps(rows))
        allsum[v] = summarise(rows)
        s = allsum[v]
        print(f"{v:22s} auto_acc={s['auto_accuracy']:.4f} coverage={s['auto_coverage']:.3f} silent/1000q={s['silent_errors_per_1000q']:.2f} "
              f"review={s['review_rate']:.3f} false_retake={s['false_retake_rate']:.3f} retake_ok={s['retake_requested']:.2f} p50={s['latency_ms_p50']:.0f}ms")
    (out / "summary.json").write_text(json.dumps(allsum, indent=1))


if __name__ == "__main__":
    main()
