"""OpenCV 5 perception tools used by the grading agent.

Each public function is a "tool": it takes images plus parameters and returns
plain observations the agent can reason over. None of them decide what to do
next; that is agent.py's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import template as T

READ_PX_PER_MM = 6.0  # canonical frame used for reading
ZOOM_PX_PER_MM = 14.0  # re-perception frame for single questions


# ---------------------------------------------------------------- utilities

def decode_image(data: bytes, max_side: int = 4000) -> np.ndarray | None:
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        f = max_side / max(h, w)
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canonical_size(px_per_mm: float) -> tuple[int, int]:
    return int(round(T.PAGE_W_MM * px_per_mm)), int(round(T.PAGE_H_MM * px_per_mm))


def scale_h(H: np.ndarray, px_per_mm: float, offset_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Turn a photo->mm homography into photo->pixels at px_per_mm (optionally cropped at offset)."""
    S = np.array([[px_per_mm, 0, -offset_mm[0] * px_per_mm], [0, px_per_mm, -offset_mm[1] * px_per_mm], [0, 0, 1]], np.float64)
    return S @ H


# ---------------------------------------------------------------- locate

@dataclass
class Location:
    ok: bool
    method: str
    H_mm: np.ndarray | None = None  # photo pixels -> sheet millimetres
    markers: list[int] = field(default_factory=list)
    reproj_mm: float | None = None
    src_px_per_mm: float | None = None  # how many photo pixels cover one mm of paper
    note: str = ""


def _src_resolution(H_mm: np.ndarray) -> float:
    """Approximate photo pixels per sheet mm at the page centre."""
    Hinv = np.linalg.inv(H_mm)
    c = np.array([[[T.PAGE_W_MM / 2, T.PAGE_H_MM / 2]], [[T.PAGE_W_MM / 2 + 10, T.PAGE_H_MM / 2]], [[T.PAGE_W_MM / 2, T.PAGE_H_MM / 2 + 10]]], np.float64)
    p = cv2.perspectiveTransform(c, Hinv).reshape(-1, 2)
    return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 20.0)


def preprocess(gray: np.ndarray, method: str) -> np.ndarray:
    """Image enhancement variants the agent can request before re-detecting."""
    if method == "none":
        return gray
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    if method == "flatten":  # divide out uneven lighting / shadows
        k = max(31, (min(gray.shape) // 12) | 1)
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
        return cv2.normalize(cv2.divide(gray, bg, scale=255), None, 0, 255, cv2.NORM_MINMAX)
    if method == "sharpen":
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        return cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    raise ValueError(method)


def locate_aruco(gray: np.ndarray, enhance: str = "none", downscale: float = 1.0) -> Location:
    g = preprocess(gray, enhance)
    if downscale != 1.0:
        g = cv2.resize(g, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMax = 53
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(T.ARUCO_DICT), params)
    corners, ids, _ = detector.detectMarkers(g)
    ref = T.marker_corners_mm()
    src, dst, found = [], [], []
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i in ref and i not in found:
                found.append(i)
                src.append(c.reshape(4, 2) / downscale)
                dst.append(ref[i])
    found.sort()
    if len(found) < 3:
        return Location(False, f"aruco[{enhance}]", markers=found, note=f"{len(found)}/4 corner markers found")
    src_a = np.concatenate(src).astype(np.float32)
    dst_a = np.concatenate(dst).astype(np.float32)
    H, _ = cv2.findHomography(src_a, dst_a, cv2.RANSAC, 1.5)
    if H is None:
        return Location(False, f"aruco[{enhance}]", markers=found, note="homography failed")
    proj = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.sqrt(np.mean(np.sum((proj - dst_a) ** 2, axis=1))))
    if err > 2.0:
        return Location(False, f"aruco[{enhance}]", markers=found, reproj_mm=err, note="markers inconsistent with sheet geometry")
    return Location(True, f"aruco[{enhance}]", H, found, err, _src_resolution(H))


_TEMPLATE_SMALL: np.ndarray | None = None


def _template_small() -> np.ndarray:
    global _TEMPLATE_SMALL
    if _TEMPLATE_SMALL is None:
        _TEMPLATE_SMALL = cv2.GaussianBlur(T.render(2.0), (5, 5), 0).astype(np.float32)
    return _TEMPLATE_SMALL


def locate_contour(gray: np.ndarray) -> Location:
    """Fallback: find the paper outline, then pick the rotation that best matches the template."""
    h, w = gray.shape
    f = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(c)
        if area < 0.15 * small.shape[0] * small.shape[1]:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx.reshape(4, 2).astype(np.float32) / f
            break
    if best is None:
        return Location(False, "contour", note="no page outline found")
    # order corners clockwise starting top-left in photo space
    s = best.sum(1)
    d = np.diff(best, axis=1).ravel()
    quad = np.array([best[np.argmin(s)], best[np.argmin(d)], best[np.argmax(s)], best[np.argmax(d)]], np.float32)
    page = np.array([[0, 0], [T.PAGE_W_MM, 0], [T.PAGE_W_MM, T.PAGE_H_MM], [0, T.PAGE_H_MM]], np.float32)
    tpl = _template_small()
    scores = []
    for rot in range(4):
        H = cv2.getPerspectiveTransform(np.roll(quad, -rot, axis=0), page)
        warped = cv2.warpPerspective(gray, scale_h(H, 2.0), (tpl.shape[1], tpl.shape[0]))
        warped = cv2.GaussianBlur(warped, (5, 5), 0).astype(np.float32)
        score = float(cv2.matchTemplate(warped, tpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        scores.append((score, rot, H))
    score, rot, H = max(scores, key=lambda t: t[0])
    if score < 0.35:
        return Location(False, "contour", note=f"page found but layout match too weak ({score:.2f})")
    return Location(True, "contour", H, [], None, _src_resolution(H), note=f"template match {score:.2f}, rotation {rot * 90} deg")


def warp(img: np.ndarray, H_mm: np.ndarray, px_per_mm: float = READ_PX_PER_MM) -> np.ndarray:
    return cv2.warpPerspective(img, scale_h(H_mm, px_per_mm), canonical_size(px_per_mm), flags=cv2.INTER_LINEAR, borderValue=255 if img.ndim == 2 else (255, 255, 255))


def warp_region(img: np.ndarray, H_mm: np.ndarray, box_mm: tuple[float, float, float, float], px_per_mm: float = ZOOM_PX_PER_MM) -> np.ndarray:
    x0, y0, x1, y1 = box_mm
    size = (int(round((x1 - x0) * px_per_mm)), int(round((y1 - y0) * px_per_mm)))
    return cv2.warpPerspective(img, scale_h(H_mm, px_per_mm, (x0, y0)), size, flags=cv2.INTER_CUBIC, borderValue=255 if img.ndim == 2 else (255, 255, 255))


# ---------------------------------------------------------------- quality

GRID_BOX_MM = (T.COL_X_MM[0] - 6, T.GRID_TOP_MM - 6, T.COL_X_MM[1] + 3 * T.OPT_PITCH_MM + 6, T.GRID_TOP_MM + (T.ROWS_PER_COL - 1) * T.ROW_PITCH_MM + 6)


@dataclass
class Quality:
    sharpness: float  # mean gradient energy on the printed bubble rings (higher = sharper)
    contrast: float  # p95 - p5 of the grid area
    glare_frac: float  # fraction of question rows whose printed rings are washed out
    glare_questions: list[int]
    src_px_per_mm: float


def measure_quality(gray: np.ndarray, loc: Location, n_questions: int) -> Quality:
    wg = warp(gray, loc.H_mm)
    s = READ_PX_PER_MM
    x0, y0, x1, y1 = (int(v * s) for v in GRID_BOX_MM)
    grid = wg[y0:y1, x0:x1]
    # Sharpness: Laplacian energy normalised by local contrast so dim photos are not called blurry.
    lap = cv2.Laplacian(cv2.GaussianBlur(grid, (3, 3), 0), cv2.CV_32F)
    p5, p95 = np.percentile(grid, [5, 95])
    contrast = float(p95 - p5)
    sharp = float(np.sqrt(np.mean(lap ** 2)) / max(contrast, 1.0) * 100.0)
    # Legibility per row: the printed bubble rings are a known calibration target. Glare or
    # over-exposure washes them out, so a row whose rings have lost their contrast cannot be
    # trusted, whatever the bubble interiors look like.
    norm = flatten_warped(wg, s)
    r = int(round(T.BUBBLE_R_MM * s))
    ring = np.zeros((2 * r + 9, 2 * r + 9), np.uint8)
    c = r + 4
    cv2.circle(ring, (c, c), int(r * 1.12), 1, -1)
    cv2.circle(ring, (c, c), int(r * 0.86), 0, -1)
    ring = ring.astype(bool)
    ring_dark = np.zeros((n_questions, len(T.OPTIONS)), np.float32)
    for b in T.bubbles(n_questions):
        cx, cy = int(round(b.x_mm * s)), int(round(b.y_mm * s))
        patch = norm[cy - c:cy + c + 1, cx - c:cx + c + 1]
        if patch.shape == ring.shape:
            ring_dark[b.q, b.opt] = 1.0 - float(patch[ring].mean())
    ref = float(np.median(ring_dark))
    # a single washed-out ring is enough: glare that hides one ring also hides the mark inside it
    glare_q = [q for q in range(n_questions) if ring_dark[q].min() < max(0.05, 0.5 * ref)]
    glare_frac = len(glare_q) / max(1, n_questions)
    return Quality(sharp, contrast, glare_frac, glare_q, loc.src_px_per_mm or 0.0)


# ---------------------------------------------------------------- read bubbles

def _disk_mask(r: int, frac: float) -> np.ndarray:
    m = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(m, (r, r), max(1, int(r * frac)), 1, -1)
    return m.astype(bool)


def flatten_warped(wg: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Remove lighting gradients and shadows: paper becomes ~1.0, ink ~0."""
    k = int(T.BUBBLE_R_MM * 2.6 * px_per_mm) | 1
    bg = cv2.morphologyEx(wg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return np.clip(wg.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0), 0, 1)


def bubble_darkness(gray: np.ndarray, loc: Location, n_questions: int, mode: str = "flatten") -> np.ndarray:
    """Return an (n_questions, n_options) array of ink darkness inside each bubble (0 white .. 1 black).

    mode="flatten": illumination-normalised (agent default)
    mode="raw":     plain grey level, no normalisation (baseline)
    """
    s = READ_PX_PER_MM
    wg = warp(gray, loc.H_mm, s)
    norm = flatten_warped(wg, s) if mode == "flatten" else wg.astype(np.float32) / 255.0
    r = int(round(T.BUBBLE_R_MM * s))
    mask = _disk_mask(r, 0.62)
    out = np.zeros((n_questions, len(T.OPTIONS)), np.float32)
    for b in T.bubbles(n_questions):
        cx, cy = int(round(b.x_mm * s)), int(round(b.y_mm * s))
        patch = norm[cy - r:cy + r + 1, cx - r:cx + r + 1]
        if patch.shape != mask.shape:
            out[b.q, b.opt] = 0.0
            continue
        out[b.q, b.opt] = 1.0 - float(patch[mask].mean())
    return out


def zoom_question(gray: np.ndarray, loc: Location, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Active perception: re-sample one question row at high resolution straight from the photo.

    Returns (darkness per option, ink-coverage per option). Coverage counts pixels that are
    clearly ink (below a local Otsu threshold), which separates a solid fill from an erasure
    smudge or a faint stray mark better than mean darkness does.
    """
    s = ZOOM_PX_PER_MM
    box = T.question_box_mm(q)
    reg = warp_region(gray, loc.H_mm, box, s)
    norm = flatten_warped(reg, s)
    u8 = (norm * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(thr, 150)  # never call mid-grey paper "ink"
    r = int(round(T.BUBBLE_R_MM * s))
    mask = _disk_mask(r, 0.62)
    dark = np.zeros(len(T.OPTIONS), np.float32)
    cover = np.zeros(len(T.OPTIONS), np.float32)
    for o in range(len(T.OPTIONS)):
        cx = int(round((T.COL_X_MM[q // T.ROWS_PER_COL] + o * T.OPT_PITCH_MM - box[0]) * s))
        cy = int(round((T.GRID_TOP_MM + (q % T.ROWS_PER_COL) * T.ROW_PITCH_MM - box[1]) * s))
        # small search window absorbs residual registration error at this zoom
        best_d, best_c = 0.0, 0.0
        for dy in (-3, 0, 3):
            for dx in (-3, 0, 3):
                p = norm[cy + dy - r:cy + dy + r + 1, cx + dx - r:cx + dx + r + 1]
                pu = u8[cy + dy - r:cy + dy + r + 1, cx + dx - r:cx + dx + r + 1]
                if p.shape != mask.shape:
                    continue
                d = 1.0 - float(p[mask].mean())
                if d > best_d:
                    best_d, best_c = d, float((pu[mask] < thr).mean())
        dark[o], cover[o] = best_d, best_c
    return dark, cover


def question_crop(img: np.ndarray, loc: Location, q: int) -> np.ndarray:
    return warp_region(img, loc.H_mm, T.question_box_mm(q), 10.0)
