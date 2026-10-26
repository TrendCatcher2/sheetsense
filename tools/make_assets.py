"""Build static assets: printable sheet (PDF + PNG) and three demo photos.

    python -m tools.make_assets

The demo photos are synthetic (eval/synth.py) and answer the web app's demo key,
with a few deliberate wrong answers so the score is not trivially 40/40.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval import synth as S
from sheetsense import template as T

OUT = Path(__file__).resolve().parent.parent / "app" / "static"
DEMO_KEY = "BDACCBADDABCABDCBADCACBDDBCAABDCDACBBDCA"


def sheet_for(answers: dict[int, str], erased: dict[int, str], faint: set[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = T.render(S.PX).astype(np.float32)
    r = T.BUBBLE_R_MM * S.PX

    def at(q, o):
        return (T.COL_X_MM[q // T.ROWS_PER_COL] + o * T.OPT_PITCH_MM) * S.PX, (T.GRID_TOP_MM + (q % T.ROWS_PER_COL) * T.ROW_PITCH_MM) * S.PX

    for q, a in answers.items():
        if a == "BLANK":
            continue
        ink = 45.0
        if q in erased:
            img = S._fill(img, rng, *at(q, T.OPTIONS.index(erased[q])), r, ink, 0.25)
        img = S._fill(img, rng, *at(q, T.OPTIONS.index(a)), r, ink, 0.5 if q in faint else 0.95)
    out = np.clip(img, 0, 255).astype(np.uint8)
    cv2.putText(out, "Ayesha K.", (int(60 * S.PX), int(39 * S.PX)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.2, 40, 3, cv2.LINE_AA)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hi = T.render(300 / 25.4)  # 300 dpi
    cv2.imwrite(str(OUT / "sheetsense-answer-sheet.png"), hi)
    Image.fromarray(hi).convert("L").save(OUT / "sheetsense-answer-sheet.pdf", "PDF", resolution=300.0)

    key = {i: c for i, c in enumerate(DEMO_KEY)}
    wrong = {3: "A", 11: "D", 17: "C", 26: "B", 33: "A"}  # five wrong answers
    student = {**key, **wrong, 21: "BLANK"}
    rng = np.random.default_rng(11)
    clean = sheet_for(student, {8: "A"}, set(), 1)
    cv2.imwrite(str(OUT / "sample-clean.jpg"), S.photograph(clean, rng, "clean"), [cv2.IMWRITE_JPEG_QUALITY, 88])
    review = sheet_for(student, {8: "A", 14: "C"}, {5, 30}, 2)
    cv2.imwrite(str(OUT / "sample-review.jpg"), S.photograph(review, np.random.default_rng(5), "typical"), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(OUT / "sample-blurry.jpg"), S.photograph(clean, np.random.default_rng(9), "retake", "motion"), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("assets written to", OUT)


if __name__ == "__main__":
    main()
