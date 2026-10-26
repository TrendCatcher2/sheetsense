"""Render docs/*.svg to PNG with headless Chromium (Playwright): python tools/svg2png.py docs/a.svg out.png"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

src, out = Path(sys.argv[1]).resolve(), sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    pg = b.new_page(device_scale_factor=2)
    pg.goto(src.as_uri())
    pg.locator("svg").screenshot(path=out)
    b.close()
