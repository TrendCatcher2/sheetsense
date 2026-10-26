"""Generate phone-style photos of filled answer sheets with exact ground truth.

    python -m eval.synth --out data/synth --seed 2026

Every per-question label and the expected sheet outcome are written to gt.json,
so evaluation numbers are fully reproducible from the seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from sheetsense import template as T

PX = 8.0  # render scale for the "printed" sheet

# question-level behaviour mix
P_BLANK, P_MULTI, P_ERASED, P_FAINT = 0.04, 0.03, 0.05, 0.02

SPLITS = {"clean": 50, "typical": 80, "hard": 60, "retake": 40}


def _fill(img, rng, cx, cy, r, ink, cover):
    """Draw a hand-filled bubble: a solid-ish blob made of overlapping strokes (patch-local)."""
    pad = int(r * 2.2)
    x0, y0 = int(cx) - pad, int(cy) - pad
    size = 2 * pad + 1
    lcx, lcy = cx - x0, cy - y0
    layer = np.zeros((size, size), np.float32)
    if rng.random() < 0.6:  # solid fill
        rr = r * rng.uniform(0.82, 1.05)
        ox, oy = rng.normal(0, 0.06 * r, 2)
        cv2.ellipse(layer, (int(lcx + ox), int(lcy + oy)), (int(rr), int(rr * rng.uniform(0.85, 1.05))), rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
    else:  # scribble strokes
        n = int(10 + 25 * cover)
        th = max(2, int(r * 0.28))
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(0, r * 0.8)
            px, py = lcx + d * math.cos(a), lcy + d * math.sin(a)
            b = a + math.pi + rng.normal(0, 0.6)
            ln = rng.uniform(0.6, 1.6) * r
            cv2.line(layer, (int(px), int(py)), (int(px + ln * math.cos(b)), int(py + ln * math.sin(b))), 1.0, th, cv2.LINE_AA)
        circ = np.zeros_like(layer)
        cv2.circle(circ, (int(lcx), int(lcy)), int(r * 1.05), 1.0, -1)
        layer *= circ
    tex = cv2.GaussianBlur(rng.uniform(0.75, 1.0, layer.shape).astype(np.float32), (3, 3), 0)
    alpha = np.clip(layer * tex * cover, 0, 1)
    sub = img[y0:y0 + size, x0:x0 + size]
    img[y0:y0 + size, x0:x0 + size] = sub * (1 - alpha) + ink * alpha
    return img


def make_sheet(rng: np.random.Generator, n_q: int = T.N_QUESTIONS):
    img = T.render(PX, n_q).astype(np.float32)
    r = T.BUBBLE_R_MM * PX
    pencil = rng.random() < 0.5
    labels, kinds = [], []
    for q in range(n_q):
        u = rng.random()
        opts = list(range(len(T.OPTIONS)))
        ink = rng.uniform(55, 115) if pencil else rng.uniform(15, 70)

        def at(o):
            return (T.COL_X_MM[q // T.ROWS_PER_COL] + o * T.OPT_PITCH_MM) * PX, (T.GRID_TOP_MM + (q % T.ROWS_PER_COL) * T.ROW_PITCH_MM) * PX

        if u < P_BLANK:
            labels.append("BLANK"); kinds.append("blank")
            continue
        if u < P_BLANK + P_MULTI:
            a, b = rng.choice(opts, 2, replace=False)
            for o in (a, b):
                img = _fill(img, rng, *at(o), r, ink, rng.uniform(0.8, 1.0))
            labels.append("MULTI"); kinds.append("multi")
            continue
        ans = int(rng.integers(len(opts)))
        if u < P_BLANK + P_MULTI + P_ERASED:
            old = int(rng.choice([o for o in opts if o != ans]))
            residue = rng.uniform(0.12, 0.35)
            img = _fill(img, rng, *at(old), r, ink, residue)
            img = _fill(img, rng, *at(ans), r, ink, rng.uniform(0.85, 1.0))
            kinds.append("erased")
        elif u < P_BLANK + P_MULTI + P_ERASED + P_FAINT:
            img = _fill(img, rng, *at(ans), r, ink, rng.uniform(0.45, 0.65))
            kinds.append("faint")
        else:
            img = _fill(img, rng, *at(ans), r, ink, rng.uniform(0.85, 1.0))
            kinds.append("normal")
        labels.append(T.OPTIONS[ans])
    # handwriting-ish scrawl on the name line and a few stray marks outside bubbles
    for _ in range(int(rng.integers(6, 14))):
        x = rng.uniform(52, 150) * PX; y = rng.uniform(36, 39.5) * PX
        cv2.line(img, (int(x), int(y)), (int(x + rng.uniform(5, 30)), int(y + rng.uniform(-12, 12))), 40.0, 3, cv2.LINE_AA)
    for _ in range(int(rng.integers(0, 4))):
        x = rng.uniform(25, 190) * PX; y = rng.uniform(80, 255) * PX
        cv2.circle(img, (int(x), int(y)), int(rng.uniform(2, 5)), 60.0, -1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8), labels, kinds


PRESETS = {
    #            persp  fill_lo fill_hi grad_lo shadow blur_hi noise jpeg_lo glare
    "clean":   (0.02, 0.80, 0.92, 0.90, 0.0, 0.5, 2.0, 88, 0.0),
    "typical": (0.07, 0.70, 0.90, 0.70, 0.3, 1.0, 4.0, 70, 0.1),
    "hard":    (0.12, 0.60, 0.85, 0.50, 0.6, 1.6, 6.0, 55, 0.3),
}


def _background(rng, h, w):
    base = rng.uniform(60, 170, 3)
    bg = np.ones((h, w, 3), np.float32) * base
    stripes = cv2.resize(rng.normal(0, 18, (1, 64)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    bg += stripes[..., None]
    bg += rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    return bg


def photograph(sheet: np.ndarray, rng: np.random.Generator, preset: str, retake_kind: str | None = None):
    """Simulate a phone photo of the printed sheet. Returns a BGR uint8 image."""
    if preset == "retake":
        persp, flo, fhi, glo, psh, bhi, noise, jlo, pglare = PRESETS["typical"]
    else:
        persp, flo, fhi, glo, psh, bhi, noise, jlo, pglare = PRESETS[preset]
    portrait = rng.random() < 0.8
    W, H = (1200, 1600) if portrait else (1600, 1200)
    sh, sw = sheet.shape
    fill = rng.uniform(flo, fhi)
    if retake_kind == "far":
        fill = rng.uniform(0.16, 0.22)
    # destination quad: scaled sheet, rotated, jittered, centred
    scale = fill * min(H / sh, W / sw) if portrait else fill * min(H / sw, W / sh)
    hw, hh = sw * scale / 2, sh * scale / 2
    quad = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], np.float32)
    rot = rng.normal(0, 6) + (0 if portrait else 90) + (180 if rng.random() < 0.1 else 0)
    a = math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    quad = quad @ R.T
    quad += rng.uniform(-persp, persp, quad.shape).astype(np.float32) * max(hw, hh)
    quad += np.array([W / 2, H / 2], np.float32) + rng.normal(0, 0.03, 2).astype(np.float32) * [W, H]
    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad)
    paper = np.dstack([sheet] * 3).astype(np.float32) * rng.uniform(0.93, 1.0, 3).astype(np.float32)
    warped = cv2.warpPerspective(paper, M, (W, H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.ones((sh, sw), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    img = _background(rng, H, W) * (1 - mask) + warped * mask

    # lighting: linear gradient + optional soft shadow
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = rng.uniform(glo, 1.0)
    ang = rng.uniform(0, 2 * math.pi)
    t = (xx * math.cos(ang) + yy * math.sin(ang)); t = (t - t.min()) / (t.max() - t.min())
    light = g + (1 - g) * t
    if rng.random() < psh:
        pts = (rng.uniform(0, 1, (5, 2)) * [W, H]).astype(np.int32)
        sm = np.zeros((H, W), np.float32)
        cv2.fillPoly(sm, [cv2.convexHull(pts)], 1.0)
        sm = cv2.GaussianBlur(sm, (0, 0), rng.uniform(15, 50))
        light *= 1 - sm * rng.uniform(0.25, 0.45)
    img *= light[..., None] * rng.uniform(0.9, 1.15)

    glare = rng.random() < pglare or retake_kind == "glare"
    if glare:
        cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.35, 0.7) * H
        sig = rng.uniform(25, 60) if retake_kind != "glare" else rng.uniform(170, 260)
        amp = rng.uniform(80, 180) if retake_kind != "glare" else 400
        img += (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)))[..., None]

    if retake_kind == "occluded":  # a hand / another sheet covering two corner markers
        side = rng.integers(4)
        q = quad.copy()
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0)][side]
        for i in pairs:
            cx, cy = q[i]
            cv2.circle(img, (int(cx), int(cy)), int(0.16 * max(hw, hh)), (int(rng.uniform(90, 160)),) * 3, -1)

    blur = rng.uniform(0, bhi)
    if retake_kind == "blur":
        blur = rng.uniform(4.5, 7.0)
    if blur > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if retake_kind == "motion":
        k = int(rng.uniform(28, 45)); ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
        Mr = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1.0)
        ker = cv2.warpAffine(ker, Mr, (k, k)); ker /= ker.sum()
        img = cv2.filter2D(img, -1, ker)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(jlo, 92))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


RETAKE_KINDS = ["blur", "motion", "occluded", "glare", "far"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply split sizes (e.g. 0.1 for a smoke test)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    gt = []
    for split, n in SPLITS.items():
        for i in range(max(1, int(n * a.scale))):
            sheet, labels, kinds = make_sheet(rng)
            rk = RETAKE_KINDS[i % len(RETAKE_KINDS)] if split == "retake" else None
            photo = photograph(sheet, rng, split, rk)
            name = f"{split}_{i:03d}.jpg"
            cv2.imwrite(str(out / name), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gt.append({"file": name, "split": split, "retake_kind": rk, "expect": "RETAKE" if rk else "GRADE", "labels": labels, "kinds": kinds})
    (out / "gt.json").write_text(json.dumps({"seed": a.seed, "sheets": gt}, indent=1))
    print(f"wrote {len(gt)} sheets to {out}")


if __name__ == "__main__":
    main()
