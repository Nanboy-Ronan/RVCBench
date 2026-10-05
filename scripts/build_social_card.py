#!/usr/bin/env python3
"""Render the homepage/docs share image (requires Playwright and Chromium)."""
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1200, 'height': 630}, device_scale_factor=1)
        page.goto((root / 'docs/site-src/social-card.html').as_uri(), wait_until='networkidle')
        page.screenshot(path=str(root / 'docs/assets/og-image.png'))
        browser.close()


if __name__ == '__main__':
    main()
