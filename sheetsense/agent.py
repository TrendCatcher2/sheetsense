"""The grading agent: a perception -> decision -> action loop over OpenCV 5 tools.

The agent never just runs a fixed pipeline. What it sees decides what it does next:

  * markers not found       -> try enhancement variants, then a page-outline fallback,
                               then ask the user for a specific retake
  * photo too blurry / far  -> ask for a retake with a concrete instruction
  * glare over some rows    -> re-sample those rows; still unreadable -> teacher review
  * uncertain bubbles       -> zoom into that question at high resolution with a second
                               measurement; still uncertain -> teacher review with a crop
  * everything confident    -> propose a grade that a teacher approves before release

Every step is appended to a trace (tool, inputs, observation, decision, time) that the
UI shows and the logs keep, so a teacher can see why the agent acted as it did.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import template as T
from . import vision as V

# Policy thresholds. Calibrated on the synthetic dev split (eval/calibrate.py) and
# then frozen before running the test split.
POLICY = {
    "min_src_px_per_mm": 2.4,  # below this the bubbles are only a few pixels wide
    "min_sharpness": 3.0,
    "min_global_sharpness": 400.0,  # Laplacian variance of the whole photo, used only when no sheet is found  # V.measure_quality sharpness score
    "max_glare_questions": 10,  # more rows than this under glare -> retake instead of review
    "fill_on": 0.62,  # normalised score >= this counts as a mark
    "fill_off": 0.12,  # normalised score < this counts as empty
    "zoom_cover_on": 0.50,  # high-res ink coverage >= this backs up a mark
    "erasure_cover_max": 0.15,  # a mid-grey smudge with almost no solid ink = erasure residue
}

MAX_RETAKE_HINTS = 3


@dataclass
class Step:
    tool: str
    args: dict
    observation: dict
    decision: str
    ms: float


@dataclass
class Result:
    status: str  # READY | NEEDS_REVIEW | RETAKE
    answers: list[str] = field(default_factory=list)  # per question: A-D, BLANK, MULTI, or ? (review)
    confidence: list[float] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)  # {q, reason, candidates, crop_jpg_b64}
    retake_reasons: list[str] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    annotated_jpg_b64: str | None = None
    ms_total: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["trace"] = [s.__dict__ for s in self.trace]
        return d


def _b64jpg(img: np.ndarray, q: int = 82) -> str:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(enc.tobytes()).decode()


class GradingAgent:
    def __init__(self, policy: dict | None = None, use_zoom: bool = True, use_flatten: bool = True, allow_review: bool = True,
                 escalate: bool = True, quality_gate: bool = True, render: bool = True):
        self.p = {**POLICY, **(policy or {})}
        self.escalate = escalate
        self.quality_gate = quality_gate
        self.use_zoom = use_zoom
        self.use_flatten = use_flatten
        self.allow_review = allow_review
        self.render = render

    # ------------------------------------------------------------ helpers
    def _do(self, res: Result, tool: str, args: dict, fn):
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000
        step = Step(tool, args, {}, "", round(ms, 1))
        res.trace.append(step)
        return out, step

    def _classify(self, scores: np.ndarray) -> tuple[str, float, str]:
        """Map normalised per-option scores for one question to (label, confidence, reason)."""
        on, off = self.p["fill_on"], self.p["fill_off"]
        order = np.argsort(-scores)
        s1, s2 = float(scores[order[0]]), float(scores[order[1]])
        n_on = int((scores >= on).sum())
        n_mid = int(((scores >= off) & (scores < on)).sum())
        if n_on == 0 and n_mid == 0:
            return "BLANK", min(1.0, (off - s1) / off), ""
        if n_on == 1 and n_mid == 0:
            return T.OPTIONS[order[0]], min(1.0, (s1 - on) / (1 - on) * 0.5 + (off - s2) / off * 0.5 + 0.5), ""
        if n_on >= 2 and n_mid == 0:
            return "MULTI", min(1.0, (s2 - on) / (1 - on) + 0.5), ""
        if n_on == 0:
            return "?", 0.0, "faint mark"
        return "?", 0.0, "possible erasure or second mark" if n_on == 1 else "unclear multiple marks"

    @staticmethod
    def calibrate(dark: np.ndarray) -> tuple[float, float]:
        """Per-sheet calibration: empty-bubble level and filled-bubble level from the sheet itself."""
        empty = float(np.median(dark))
        filled = float(np.percentile(dark.max(axis=1), 75))
        if filled - empty < 0.15:  # almost nothing filled; fall back to absolute scale
            filled = empty + 0.45
        return empty, filled

    @classmethod
    def _normalise(cls, dark: np.ndarray, cal: tuple[float, float] | None = None) -> np.ndarray:
        empty, filled = cal or cls.calibrate(dark)
        return np.clip((dark - empty) / (filled - empty), 0, 1.5)

    # ------------------------------------------------------------ main loop
    def run(self, img: np.ndarray, n_questions: int = T.N_QUESTIONS) -> Result:
        t_start = time.perf_counter()
        res = Result(status="RETAKE")
        gray = V.to_gray(img)

        # 1. PERCEIVE: find the sheet. Escalate through detector variants only as needed.
        loc = None
        attempts = [("aruco", "none", 1.0), ("aruco", "clahe", 1.0), ("aruco", "flatten", 1.0), ("aruco", "none", 0.5), ("contour", "", 1.0)]
        if not self.escalate:
            attempts = attempts[:1]
        for i, (method, enh, ds) in enumerate(attempts):
            if method == "aruco":
                cand, step = self._do(res, "locate_sheet", {"method": "aruco", "enhance": enh, "downscale": ds}, lambda: V.locate_aruco(gray, enh, ds))
            else:
                cand, step = self._do(res, "locate_sheet", {"method": "page_outline+template_match"}, lambda: V.locate_contour(gray))
            step.observation = {"ok": cand.ok, "markers": cand.markers, "note": cand.note, "reproj_mm": None if cand.reproj_mm is None else round(cand.reproj_mm, 3)}
            if cand.ok:
                step.decision = "sheet located; continue to quality check"
                loc = cand
                break
            nxt = attempts[i + 1] if i + 1 < len(attempts) else None
            step.decision = f"not located; try {nxt[0]}{'/' + nxt[1] if nxt and nxt[1] else ''}" if nxt else "give up; request retake"
        if loc is None:
            last = res.trace[0].observation.get("markers", [])
            missing = [i for i in range(4) if i not in last]
            names = {0: "top-left", 1: "top-right", 2: "bottom-right", 3: "bottom-left"}
            small = cv2.resize(gray, None, fx=800 / max(gray.shape), fy=800 / max(gray.shape), interpolation=cv2.INTER_AREA)
            global_sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
            res.quality = {"global_sharpness": round(global_sharp, 1)}
            if global_sharp < self.p["min_global_sharpness"]:
                res.retake_reasons.append("The photo is too blurry to find the sheet. Hold the phone steady and tap to focus, then retake.")
            elif 0 < len(last) < 4:
                res.retake_reasons.append(f"Corner square(s) not visible: {', '.join(names[m] for m in missing)}. Keep all four black corner squares in the frame and uncovered.")
            else:
                res.retake_reasons.append("Could not find the answer sheet. Fill the frame with the whole page and keep all four corner squares visible.")
            res.ms_total = round((time.perf_counter() - t_start) * 1000, 1)
            return res

        # 2. PERCEIVE: is the photo good enough to read?
        q, step = self._do(res, "measure_quality", {}, lambda: V.measure_quality(gray, loc, n_questions))
        res.quality = {"sharpness": round(q.sharpness, 2), "contrast": round(q.contrast, 1), "glare_frac": round(q.glare_frac, 4), "glare_questions": [x + 1 for x in q.glare_questions], "src_px_per_mm": round(q.src_px_per_mm, 2), "locate_method": loc.method}
        step.observation = dict(res.quality)
        if q.src_px_per_mm < self.p["min_src_px_per_mm"]:
            res.retake_reasons.append("The sheet is too small in the photo. Move closer so the page fills most of the frame.")
        if q.sharpness < self.p["min_sharpness"]:
            res.retake_reasons.append("The photo is blurry. Hold the phone steady, tap to focus on the bubbles, and retake.")
        if len(q.glare_questions) > self.p["max_glare_questions"]:
            res.retake_reasons.append("Strong glare is washing out the bubbles. Tilt the phone or move away from the light and retake.")
        if not self.quality_gate:
            res.retake_reasons = []
        if res.retake_reasons:
            step.decision = "photo not readable; request retake"
            res.ms_total = round((time.perf_counter() - t_start) * 1000, 1)
            return res
        step.decision = "readable" + (f"; glare on {len(q.glare_questions)} rows -> re-check those rows" if q.glare_questions else "")

        # 3. PERCEIVE: read every bubble once on the rectified sheet.
        mode = "flatten" if self.use_flatten else "raw"
        dark, step = self._do(res, "read_bubbles", {"mode": mode, "px_per_mm": V.READ_PX_PER_MM}, lambda: V.bubble_darkness(gray, loc, n_questions, mode))
        cal = self.calibrate(dark)
        scores = self._normalise(dark, cal)
        labels, confs, reasons = [], [], []
        for qi in range(n_questions):
            lab, c, why = self._classify(scores[qi])
            if qi in q.glare_questions and lab != "?":
                lab, c, why = "?", 0.0, "glare over this row"
            labels.append(lab); confs.append(c); reasons.append(why)
        unsure = [i for i, l in enumerate(labels) if l == "?"]
        step.observation = {"confident": n_questions - len(unsure), "uncertain": [i + 1 for i in unsure]}
        step.decision = f"zoom into {len(unsure)} uncertain question(s)" if unsure and self.use_zoom else ("all confident" if not unsure else "send uncertain questions to review")

        # 4. DECIDE + ACT: active perception on uncertain questions only.
        if unsure and self.use_zoom:
            for qi in unsure:
                (zd, zc), step = self._do(res, "zoom_question", {"q": qi + 1, "px_per_mm": V.ZOOM_PX_PER_MM}, lambda: V.zoom_question(gray, loc, qi))
                z = self._normalise(zd, cal)
                on, off = self.p["fill_on"], self.p["fill_off"]
                marked = [o for o in range(len(T.OPTIONS)) if z[o] >= on and zc[o] >= self.p["zoom_cover_on"]]
                unclear = [o for o in range(len(T.OPTIONS)) if z[o] >= off and o not in marked]
                erasures = [o for o in unclear if zc[o] <= self.p["erasure_cover_max"] and z[o] < on]
                step.observation = {"score": {T.OPTIONS[o]: round(float(z[o]), 2) for o in range(len(T.OPTIONS))},
                                    "ink_coverage": {T.OPTIONS[o]: round(float(zc[o]), 2) for o in range(len(T.OPTIONS))},
                                    "reason_before": reasons[qi]}
                glare_row = qi in q.glare_questions
                if glare_row:
                    step.decision = "row under glare; ask teacher" if self.allow_review else "forced best guess"
                elif not unclear:
                    lab = "BLANK" if not marked else (T.OPTIONS[marked[0]] if len(marked) == 1 else "MULTI")
                    labels[qi], confs[qi], reasons[qi] = lab, 0.6, ""
                    step.decision = f"resolved as {lab}"
                elif len(marked) == 1 and set(unclear) == set(erasures):
                    labels[qi], confs[qi], reasons[qi] = T.OPTIONS[marked[0]], 0.5, ""
                    step.decision = f"resolved as {T.OPTIONS[marked[0]]}; {', '.join(T.OPTIONS[o] for o in erasures)} looks erased (smudge, no solid ink)"
                else:
                    step.decision = "still uncertain; ask teacher" if self.allow_review else "forced best guess"
                if labels[qi] == "?" and not self.allow_review:
                    labels[qi] = T.OPTIONS[int(np.argmax(z))] if z.max() >= off else "BLANK"
                    confs[qi] = 0.3
        elif unsure and not self.allow_review:
            for qi in unsure:
                labels[qi] = T.OPTIONS[int(np.argmax(scores[qi]))] if scores[qi].max() >= self.p["fill_off"] else "BLANK"
                confs[qi] = 0.3

        # 5. ACT: package a proposed grade; anything still unsure goes to the teacher.
        review = [i for i, l in enumerate(labels) if l == "?"]
        for qi in review:
            cand = [T.OPTIONS[o] for o in np.argsort(-scores[qi])[:2] if scores[qi][o] >= self.p["fill_off"] * 0.7]
            item = {"q": qi + 1, "reason": reasons[qi] or "uncertain", "candidates": cand}
            if self.render:
                item["crop_jpg_b64"] = _b64jpg(V.question_crop(img, loc, qi))
            res.review.append(item)
        res.answers, res.confidence = labels, [round(float(c), 3) for c in confs]
        res.status = "NEEDS_REVIEW" if review else "READY"
        _, step = self._do(res, "propose_grade", {"n_questions": n_questions}, lambda: None)
        step.observation = {"auto": n_questions - len(review), "review": [r["q"] for r in res.review]}
        step.decision = f"ask teacher to confirm {len(review)} question(s), then approve" if review else "all answers read; waiting for teacher approval"
        if self.render:
            res.annotated_jpg_b64 = _b64jpg(annotate(img, loc, labels, n_questions))
        res.ms_total = round((time.perf_counter() - t_start) * 1000, 1)
        return res


def annotate(img: np.ndarray, loc: V.Location, labels: list[str], n_questions: int, px_per_mm: float = 4.0) -> np.ndarray:
    """Rectified sheet with the agent's reading drawn on top (blue = read mark, amber = needs review)."""
    w = V.warp(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), loc.H_mm, px_per_mm)
    r = int(T.BUBBLE_R_MM * px_per_mm) + 2
    for qi in range(n_questions):
        lab = labels[qi]
        y = (T.GRID_TOP_MM + (qi % T.ROWS_PER_COL) * T.ROW_PITCH_MM) * px_per_mm
        if lab == "?":
            x0, y0, x1, y1 = (int(v * px_per_mm) for v in T.question_box_mm(qi))
            cv2.rectangle(w, (x0, y0), (x1, y1), (0, 170, 255), 2)
            continue
        for o, letter in enumerate(T.OPTIONS):
            if lab == letter or (lab == "MULTI"):
                x = (T.COL_X_MM[qi // T.ROWS_PER_COL] + o * T.OPT_PITCH_MM) * px_per_mm
                if lab == "MULTI":
                    continue
                cv2.circle(w, (int(x), int(y)), r, (230, 120, 0), 2, cv2.LINE_AA)
        if lab in ("MULTI", "BLANK"):
            x = (T.COL_X_MM[qi // T.ROWS_PER_COL] + 3 * T.OPT_PITCH_MM + 6) * px_per_mm
            cv2.putText(w, "multi" if lab == "MULTI" else "blank", (int(x), int(y + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 200), 1, cv2.LINE_AA)
    return w
