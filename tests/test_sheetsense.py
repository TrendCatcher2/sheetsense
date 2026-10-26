import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.handler import lambda_handler
from sheetsense import template as T
from sheetsense import vision as V
from sheetsense.agent import GradingAgent

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
DEMO_KEY = "BDACCBADDABCABDCBADCACBDDBCAABDCDACBBDCA"


def _ev(method, path, body=None):
    return {"rawPath": path, "requestContext": {"http": {"method": method}}, "body": body, "isBase64Encoded": False}


def test_layout_fits_between_markers():
    ys = [b.y_mm for b in T.bubbles()]
    assert min(ys) - T.BUBBLE_R_MM > T.MARKER_MARGIN_MM + T.MARKER_MM
    assert max(ys) + T.BUBBLE_R_MM < T.PAGE_H_MM - T.MARKER_MARGIN_MM - T.MARKER_MM


def test_blank_render_reads_as_all_blank():
    img = cv2.cvtColor(T.render(6.0), cv2.COLOR_GRAY2BGR)
    r = GradingAgent(render=False).run(img)
    assert r.status == "READY"
    assert set(r.answers) == {"BLANK"}


def test_upside_down_photo_is_rectified():
    img = cv2.imread(str(STATIC / "sample-clean.jpg"))
    a = GradingAgent(render=False).run(img).answers
    b = GradingAgent(render=False).run(cv2.rotate(img, cv2.ROTATE_180)).answers
    assert a == b


def test_clean_sample_matches_expected_answers():
    r = GradingAgent(render=False).run(cv2.imread(str(STATIC / "sample-clean.jpg")))
    assert r.status == "READY"
    expected = list(DEMO_KEY)
    for q, a in {3: "A", 11: "D", 17: "C", 26: "B", 33: "A", 21: "BLANK"}.items():
        expected[q] = a
    assert r.answers == expected


def test_review_sample_flags_faint_marks():
    r = GradingAgent(render=False).run(cv2.imread(str(STATIC / "sample-review.jpg")))
    assert r.status == "NEEDS_REVIEW"
    assert sorted(x["q"] for x in r.review) == [6, 31]


def test_blurry_sample_requests_retake():
    r = GradingAgent(render=False).run(cv2.imread(str(STATIC / "sample-blurry.jpg")))
    assert r.status == "RETAKE" and r.retake_reasons
    assert [s.tool for s in r.trace].count("locate_sheet") > 1  # agent escalated before giving up


def test_handler_routes_and_validation():
    assert lambda_handler(_ev("GET", "/"))["statusCode"] == 200
    assert lambda_handler(_ev("GET", "/static/../handler.py"))["statusCode"] == 404
    assert lambda_handler(_ev("POST", "/api/grade", "not json"))["statusCode"] == 400
    assert lambda_handler(_ev("POST", "/api/grade", json.dumps({"image_b64": "!!"})))["statusCode"] == 400
    assert lambda_handler(_ev("POST", "/api/grade", json.dumps({"image_b64": base64.b64encode(b"x").decode()})))["statusCode"] == 400
    big = base64.b64encode(b"\0" * (5 * 1024 * 1024)).decode()
    assert lambda_handler(_ev("POST", "/api/grade", json.dumps({"image_b64": big})))["statusCode"] == 413


def test_handler_grades_sample():
    b64 = base64.b64encode((STATIC / "sample-review.jpg").read_bytes()).decode()
    r = lambda_handler(_ev("POST", "/api/grade", json.dumps({"image_b64": b64, "n_questions": 40})))
    body = json.loads(r["body"])
    assert r["statusCode"] == 200 and body["status"] == "NEEDS_REVIEW"
    assert body["opencv"].startswith("5.")
    assert all("crop_jpg_b64" in x for x in body["review"])


@pytest.mark.parametrize("n", [1, 20, 40])
def test_question_count(n):
    r = GradingAgent(render=False).run(cv2.imread(str(STATIC / "sample-clean.jpg")), n)
    assert len(r.answers) == n
