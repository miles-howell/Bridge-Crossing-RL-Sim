"""Build assets/readme/hero-illustrated.png from the illustrated hero layout.

Renders source/hero-illustrated-layout.svg (written by build_readme_svgs.py)
in headless Chromium at 2x, with transparent rounded corners. Publishing a
PNG rather than the SVG avoids relying on GitHub to resolve the SVG's
relative image link. Needs Playwright (with a Chromium); run from the repo
root after build_readme_svgs.py:

    python assets/readme/source/build_hero_png.py
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

SOURCE = Path(__file__).resolve().parent
LAYOUT = SOURCE / "hero-illustrated-layout.svg"
OUT = SOURCE.parent / "hero-illustrated.png"


async def main():
    async with async_playwright() as p:
        chromium = Path("/opt/pw-browsers/chromium")
        browser = await p.chromium.launch(executable_path=str(chromium) if chromium.is_file() else None)
        page = await browser.new_page(viewport={"width": 1200, "height": 420}, device_scale_factor=2)
        await page.goto(LAYOUT.as_uri())
        await page.wait_for_load_state("load")
        await page.locator("svg").screenshot(path=str(OUT), omit_background=True)
        await browser.close()
    print(f"wrote {OUT}: {OUT.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    asyncio.run(main())
