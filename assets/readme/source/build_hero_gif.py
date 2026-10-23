"""Build assets/readme/hero.gif: Nigel walking the route in the README hero.

The run is scripted, not simulated: Nigel walks the same route the static
hero draws (spawn -> log -> river -> far bank -> house) one tile at a time,
with his real walk cycle, while the manager's current sub-goal updates on
the left. Each frame is the hero SVG from build_readme_svgs.py, rendered in
headless Chromium and encoded with the beautify-github-readme skill's GIF
encoder (palette, transparent rounded corners).

Needs Pillow, Playwright (with a Chromium) and ffmpeg; none are app
dependencies. Run from the repo root:

    python assets/readme/source/build_hero_gif.py
"""
import asyncio
import functools
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_readme_svgs as svgs  # noqa: E402

ROOT = svgs.ROOT
OUT = ROOT / "assets" / "readme" / "hero.gif"
ENCODER = ROOT / ".claude" / "skills" / "beautify-github-readme" / "scripts" / "render_motion_gif.py"

FPS = 15
FRAMES_PER_TILE = 4      # walking speed: ~3.75 tiles a second
WALK_FRAME_EVERY = 2     # advance the walk cycle every N frames
SPEC = {"fps": FPS, "colors": 128, "dither": "none", "transparent_color": "#ff00ff"}

# Each leg ends on its sub-goal's milestone: (manager's sub-goal, waypoints,
# what the milestone changes).
LEGS = [
    ("GOTO_LOG", [(1, 9), (3, 9)], {"log": "carried"}),
    ("GOTO_RIVER", [(3, 6), (8, 6)], {"log": "placed"}),
    ("GOTO_FAR_BANK", [(12, 6)], {}),
    ("GOTO_HOUSE", [(15, 6), (15, 3)], {}),
]


def facing(a, b):
    dc, dr = b[0] - a[0], b[1] - a[1]
    if dc:
        return "RIGHT" if dc > 0 else "LEFT"
    return "DOWN" if dr > 0 else "UP"


def timeline():
    """Yield one hero state per frame."""
    pos, trail = svgs.SPAWN, [svgs.SPAWN]
    state = {"log": "ground", "reached": 0, "facing": "DOWN", "step": 0}

    def frame(**kw):
        return {**state, "pos": pos, "trail": list(trail), **kw}

    def hold(seconds, **kw):
        for _ in range(round(seconds * FPS)):
            yield frame(step=0, **kw)

    yield from hold(0.8, status=LEGS[0][0])
    n = 0
    for subgoal, waypoints, milestone in LEGS:
        for target in waypoints:
            while pos != target:
                state["facing"] = facing(pos, target)
                dc = (target[0] > pos[0]) - (target[0] < pos[0])
                dr = (target[1] > pos[1]) - (target[1] < pos[1])
                nxt = (pos[0] + dc, pos[1] + dr)
                for f in range(1, FRAMES_PER_TILE + 1):
                    t = f / FRAMES_PER_TILE
                    here = (pos[0] + dc * t, pos[1] + dr * t)
                    yield {**state, "pos": here, "trail": trail + [here], "status": subgoal,
                           "step": (n // WALK_FRAME_EVERY) % 4}
                    n += 1
                pos = nxt
                trail.append(pos)
        state.update(milestone)
        state["reached"] += 1
        state["facing"] = "DOWN" if subgoal != "GOTO_HOUSE" else "UP"
        yield from hold(0.5, status=subgoal)
    # Into the house, then rest on the finished map before looping.
    for i in range(1, 9):
        yield frame(step=0, status="home ✓", nigel_opacity=1 - i / 8)
    yield from hold(1.6, status="home ✓", nigel_opacity=0)


def load_encoder():
    spec = importlib.util.spec_from_file_location("render_motion_gif", ENCODER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def render(frames_dir, enc):
    key = enc.parse_hex_color(SPEC["transparent_color"])
    async with async_playwright() as p:
        chromium = Path("/opt/pw-browsers/chromium")
        browser = await p.chromium.launch(executable_path=str(chromium) if chromium.is_file() else None)
        page = await browser.new_page(viewport={"width": 1200, "height": 420})
        count = 0
        for i, state in enumerate(timeline()):
            await page.set_content(f'<body style="margin:0">{svgs.hero(state)}</body>')
            png = frames_dir / f"raw-{i:04d}.png"
            await page.locator("svg").screenshot(path=str(png), omit_background=True)
            from PIL import Image
            enc.flatten_to_chroma(Image.open(png).convert("RGBA"), key, 128).save(frames_dir / f"frame-{i:04d}.png")
            png.unlink()
            count += 1
        await browser.close()
    return count


def main():
    enc = load_encoder()
    ffmpeg = shutil.which("ffmpeg") or sys.exit("ffmpeg is required")
    with tempfile.TemporaryDirectory(prefix="hero-gif-") as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir()
        count = asyncio.run(render(frames_dir, enc))
        enc.encode_gif(frames_dir, OUT, SPEC, ffmpeg, count, True)
    print(f"wrote {OUT}: {count} frames, {count / FPS:.1f}s, {OUT.stat().st_size / 1024 / 1024:.2f} MB")


svgs.sprite_frame = functools.lru_cache(maxsize=None)(svgs.sprite_frame)

if __name__ == "__main__":
    main()
