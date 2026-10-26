"""Printable answer-sheet layout.

Everything is defined in millimetres on an A4 page so the same layout can be
rendered for printing (high px/mm) and used as the canonical frame the reader
warps photos into (lower px/mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAGE_W_MM = 210.0
PAGE_H_MM = 297.0

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_MM = 18.0
MARKER_MARGIN_MM = 10.0
# Marker id -> top-left corner of the marker square (mm). Ids encode orientation,
# so a sheet photographed upside down is still rectified correctly.
MARKER_POS_MM = {
    0: (MARKER_MARGIN_MM, MARKER_MARGIN_MM),
    1: (PAGE_W_MM - MARKER_MARGIN_MM - MARKER_MM, MARKER_MARGIN_MM),
    2: (PAGE_W_MM - MARKER_MARGIN_MM - MARKER_MM, PAGE_H_MM - MARKER_MARGIN_MM - MARKER_MM),
    3: (MARKER_MARGIN_MM, PAGE_H_MM - MARKER_MARGIN_MM - MARKER_MM),
}

OPTIONS = "ABCD"
N_QUESTIONS = 40
ROWS_PER_COL = 20
BUBBLE_R_MM = 3.4
OPT_PITCH_MM = 11.0
ROW_PITCH_MM = 8.8
GRID_TOP_MM = 84.0
COL_X_MM = (40.0, 125.0)  # x of option A in each column


@dataclass(frozen=True)
class Bubble:
    q: int  # 0-based question index
    opt: int  # 0-based option index
    x_mm: float
    y_mm: float


def bubbles(n_questions: int = N_QUESTIONS) -> list[Bubble]:
    out = []
    for q in range(n_questions):
        col, row = divmod(q, ROWS_PER_COL)
        for o in range(len(OPTIONS)):
            out.append(Bubble(q, o, COL_X_MM[col] + o * OPT_PITCH_MM, GRID_TOP_MM + row * ROW_PITCH_MM))
    return out


def marker_corners_mm() -> dict[int, np.ndarray]:
    """Marker id -> 4x2 corners (TL, TR, BR, BL) in mm, matching ArUco corner order."""
    out = {}
    for mid, (x, y) in MARKER_POS_MM.items():
        out[mid] = np.array([[x, y], [x + MARKER_MM, y], [x + MARKER_MM, y + MARKER_MM], [x, y + MARKER_MM]], np.float32)
    return out


def question_box_mm(q: int) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of one question row, used for review crops."""
    col, row = divmod(q, ROWS_PER_COL)
    y = GRID_TOP_MM + row * ROW_PITCH_MM
    x0 = COL_X_MM[col] - 16.0
    x1 = COL_X_MM[col] + (len(OPTIONS) - 1) * OPT_PITCH_MM + 7.0
    return x0, y - 4.4, x1, y + 4.4


def _put(img, text, x_mm, y_mm, s, scale_mm=3.0, thick=1, color=0, center=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = scale_mm * s / 22.0
    th = max(1, int(round(thick * s / 4)))
    (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
    x = int(x_mm * s - (tw / 2 if center else 0))
    y = int(y_mm * s + (tht / 2 if center else 0))
    cv2.putText(img, text, (x, y), font, fs, color, th, cv2.LINE_AA)


def render(px_per_mm: float = 8.0, n_questions: int = N_QUESTIONS, title: str = "SheetSense MCQ Answer Sheet") -> np.ndarray:
    """Render the blank sheet as a grayscale image (white paper, black ink)."""
    s = px_per_mm
    w, h = int(round(PAGE_W_MM * s)), int(round(PAGE_H_MM * s))
    img = np.full((h, w), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    msize = int(round(MARKER_MM * s))
    for mid, (x, y) in MARKER_POS_MM.items():
        m = cv2.aruco.generateImageMarker(dictionary, mid, msize)
        x0, y0 = int(round(x * s)), int(round(y * s))
        img[y0:y0 + msize, x0:x0 + msize] = m

    _put(img, title, PAGE_W_MM / 2, 22, s, 4.2, 2, center=True)
    _put(img, "Name: ______________________________", 36, 40, s, 3.2)
    _put(img, "Class: ________   Roll no: __________", 36, 50, s, 3.2)
    _put(img, "Fill ONE bubble per question completely with a dark pen or pencil.", 36, 60, s, 2.4)
    _put(img, "To change an answer, erase cleanly. Keep all four corner squares visible.", 36, 65, s, 2.4)

    r = int(round(BUBBLE_R_MM * s))
    ring = max(1, int(round(0.35 * s)))
    for col in range(2):
        for o, letter in enumerate(OPTIONS):
            _put(img, letter, COL_X_MM[col] + o * OPT_PITCH_MM, GRID_TOP_MM - 7.0, s, 2.6, 1, center=True)
    for b in bubbles(n_questions):
        cx, cy = int(round(b.x_mm * s)), int(round(b.y_mm * s))
        cv2.circle(img, (cx, cy), r, 0, ring, cv2.LINE_AA)
        if b.opt == 0:
            _put(img, f"{b.q + 1:>2}", b.x_mm - 13.0, b.y_mm, s, 2.8, 1, center=False)
    _put(img, "sheetsense v1 - 40 questions - A4", PAGE_W_MM / 2, PAGE_H_MM - 16, s, 2.2, 1, 120, center=True)
    return img
