"""Build the README hero and how-it-works SVGs.

The hero map mirrors the real world layout in simulation/engine.py
(20x12 grid, river at cols 8-12, bridge row 6, log at (3, 9), house at
cols 15-16 rows 1-3, agent spawn (1, 1)). In the static hero Nigel is
drawn from static/images/favicon.svg; build_hero_gif.py animates the
same hero with his real walk cycle. Run from the repo root:

    python assets/readme/source/build_readme_svgs.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "assets" / "readme"

# Palette: the app's dark "Mission Control" theme (index.html).
BG = "#0e1116"
SURFACE = "#151a21"
BORDER = "#262d38"
TEXT = "#e6eaf0"
MUTED = "#8b95a5"
ACCENT = "#f5a524"
POS = "#4ade80"
NEG = "#f87171"
GRASS = "#16261b"
GRASS_LINE = "#1d3324"
BANK = "#5b4a2e"
WATER = "#1c4a6b"
WATER_HI = "#38bdf8"
WOOD = "#8f5230"
WOOD_DARK = "#6b3a1f"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# World geometry, from engine.py.
COLS, ROWS = 20, 12
RIVER_START, RIVER_W = 8, 5
BRIDGE_ROW = 6
LOG = (3, 9)
HOUSE = (15, 1, 2, 3)
SPAWN = (1, 1)
GOALS = [("GOTO_LOG", LOG), ("GOTO_RIVER", (8, 6)),
         ("GOTO_FAR_BANK", (12, 6)), ("GOTO_HOUSE", (15, 3))]


def nigel(x, y, size):
    """Nigel's sprite from the favicon, minus its amber background."""
    src = (ROOT / "static" / "images" / "favicon.svg").read_text()
    rects = re.findall(r'<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)" fill="(#\w+)"/>', src)
    s = size / 16
    body = "".join(
        f'<rect x="{int(rx)}" y="{int(ry)}" width="{int(w)}" height="{int(h)}" fill="{fill}"/>'
        for rx, ry, w, h, fill in rects if fill != ACCENT
    )
    return (f'<g id="nigel" transform="translate({x} {y}) scale({s:g})" '
            f'shape-rendering="crispEdges">{body}</g>')


# The route the hero draws, as grid steps from spawn through the four sub-goals.
ROUTE = [SPAWN, (1, 9), (3, 9), (3, 6), (8, 6), (12, 6), (15, 6), (15, 3)]


def sprite_frame(direction, step):
    """One 16x32 frame of Nigel's real walk cycle (static/images/character.png),
    as a data URI. Rows match CHAR_SPRITE_MAP in index.html."""
    import base64, io
    from PIL import Image
    row = {"DOWN": 0, "RIGHT": 2, "UP": 4, "LEFT": 6}[direction]
    sheet = Image.open(ROOT / "static" / "images" / "character.png")
    frame = sheet.crop((step * 16, row * 16, step * 16 + 16, row * 16 + 32))
    buf = io.BytesIO()
    frame.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def hero(state=None):
    """The static hero, or with `state` one frame of the animated hero.

    state keys: pos (col, row floats), trail (list of points walked), facing,
    step (walk-cycle frame), log ("ground" | "carried" | "placed"), reached
    (number of sub-goal markers lit), status (manager line on the left),
    nigel_opacity.
    """
    W, H = 1200, 420
    T = 26  # tile size
    mx, my = 638, 54  # map origin
    tx = lambda c: mx + c * T
    ty = lambda r: my + r * T
    cx = lambda c: tx(c) + T / 2
    cy = lambda r: ty(r) + T / 2
    log_state = state["log"] if state else "ground"

    m = []
    # Grass and grid.
    m.append(f'<rect x="{mx}" y="{my}" width="{COLS*T}" height="{ROWS*T}" rx="6" fill="{GRASS}"/>')
    for c in range(1, COLS):
        m.append(f'<line x1="{tx(c)}" y1="{my}" x2="{tx(c)}" y2="{ty(ROWS)}" stroke="{GRASS_LINE}" stroke-width="1"/>')
    for r in range(1, ROWS):
        m.append(f'<line x1="{mx}" y1="{ty(r)}" x2="{tx(COLS)}" y2="{ty(r)}" stroke="{GRASS_LINE}" stroke-width="1"/>')
    # River: one bank tile, three water tiles, one bank tile.
    m.append(f'<rect x="{tx(RIVER_START)}" y="{my}" width="{T}" height="{ROWS*T}" fill="{BANK}"/>')
    m.append(f'<rect x="{tx(RIVER_START+4)}" y="{my}" width="{T}" height="{ROWS*T}" fill="{BANK}"/>')
    m.append(f'<rect x="{tx(RIVER_START+1)}" y="{my}" width="{3*T}" height="{ROWS*T}" fill="{WATER}"/>')
    for r in range(0, ROWS, 2):
        y = cy(r) + (6 if r % 4 else -4)
        m.append(f'<path d="M{tx(9)+8} {y} q8 -5 16 0 t16 0" fill="none" stroke="{WATER_HI}" stroke-opacity="0.45" stroke-width="2"/>')
    if log_state == "placed":
        # The log laid across the three water tiles.
        m.append(f'<rect x="{tx(9)-4}" y="{ty(BRIDGE_ROW)+3}" width="{3*T+8}" height="{T-6}" rx="5" fill="{WOOD}"/>')
        m.append(f'<rect x="{tx(9)-4}" y="{ty(BRIDGE_ROW)+3}" width="{3*T+8}" height="4" rx="2" fill="#b06a3c"/>')
    else:
        # Bridge slot (where the log goes), drawn as a dashed outline.
        m.append(f'<rect x="{tx(9)+2}" y="{ty(BRIDGE_ROW)+3}" width="{3*T-4}" height="{T-6}" rx="3" '
                 f'fill="none" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 4"/>')
    if log_state == "ground":
        # Log: three tiles centred on GOTO_LOG.
        lx = tx(LOG[0] - 1)
        m.append(f'<rect x="{lx+3}" y="{ty(LOG[1])+6}" width="{3*T-6}" height="{T-12}" rx="7" fill="{WOOD}"/>')
        m.append(f'<rect x="{lx+3}" y="{ty(LOG[1])+6}" width="{3*T-6}" height="4" rx="2" fill="#b06a3c"/>')
        m.append(f'<circle cx="{tx(LOG[0]+2)-8}" cy="{cy(LOG[1])}" r="5" fill="{WOOD_DARK}"/>')
    # House: 2x3 tiles.
    hc, hr, hw, hh = HOUSE
    hx, hy = tx(hc), ty(hr)
    m.append(f'<path d="M{hx-3} {hy+T} L{hx+hw*T/2} {hy+4} L{hx+hw*T+3} {hy+T} Z" fill="#c8323a"/>')
    m.append(f'<rect x="{hx+3}" y="{hy+T}" width="{hw*T-6}" height="{(hh-1)*T-2}" fill="#d9c7a3"/>')
    m.append(f'<rect x="{hx+hw*T/2-7}" y="{hy+hh*T-20}" width="14" height="18" fill="{WOOD_DARK}"/>')

    # The route through the four sub-goals: the whole plan when static,
    # only the tiles walked so far when animated.
    pts = state["trail"] if state else ROUTE
    if len(pts) > 1:
        d = "M" + " L".join(f"{cx(c):g} {cy(r):g}" for c, r in pts)
        m.append(f'<path d="{d}" fill="none" stroke="{ACCENT}" stroke-width="3" '
                 f'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="2 7"/>')
    reached = state["reached"] if state else len(GOALS)
    for i, (_, (c, r)) in enumerate(GOALS, 1):
        lit = i <= reached
        m.append(f'<circle cx="{cx(c)}" cy="{cy(r)}" r="12" fill="{ACCENT if lit else BG}" '
                 f'stroke="{BG if lit else ACCENT}" stroke-width="{3 if lit else 2}"/>')
        m.append(f'<text x="{cx(c)}" y="{cy(r)+6}" text-anchor="middle" font-family="{MONO}" '
                 f'font-size="16" font-weight="700" fill="{BG if lit else ACCENT}">{i}</text>')
    if state is None:
        m.append(nigel(tx(SPAWN[0]) - 4, ty(SPAWN[1]) - 6, 34))
    else:
        # Nigel's sprite covers his tile and the one above, like the app draws him.
        c, r = state["pos"]
        g = [f'<g id="nigel" opacity="{state.get("nigel_opacity", 1):g}">']
        g.append(f'<image x="{tx(c):g}" y="{ty(r - 1):g}" width="{T}" height="{2*T}" '
                 f'style="image-rendering:pixelated" href="{sprite_frame(state["facing"], state["step"])}"/>')
        if log_state == "carried":
            # Resting on his head, like drawCarriedLog() in index.html.
            g.append(f'<rect x="{tx(c) - 2:g}" y="{ty(r - 1) + 5:g}" width="{T + 4}" height="9" rx="4" fill="{WOOD}"/>')
            g.append(f'<rect x="{tx(c) - 2:g}" y="{ty(r - 1) + 5:g}" width="{T + 4}" height="3" rx="1.5" fill="#b06a3c"/>')
        g.append('</g>')
        m.append("".join(g))
    m.append(f'<text x="{mx}" y="{ty(ROWS)+28}" font-family="{MONO}" font-size="16" fill="{MUTED}">'
             f'20 × 12 grid · spawn → log → river → far bank → house</text>')

    if state is None:
        footer = f'<text x="0" y="0">Django · PyTorch · trains live in your browser</text>'
    else:
        footer = (f'<text x="0" y="0">manager → <tspan fill="{ACCENT}" font-weight="700">'
                  f'{state["status"]}</tspan></text>')

    left = f'''
  <g id="title-block" transform="translate(64 0)">
    <text x="0" y="86" font-family="{MONO}" font-size="18" letter-spacing="2" fill="{ACCENT}">HIERARCHICAL RL · DQN + HER</text>
    <text x="0" y="158" font-family="{SANS}" font-size="64" font-weight="800" fill="{TEXT}">Nigel’s Quest</text>
    <text x="0" y="198" font-family="{MONO}" font-size="22" fill="{MUTED}">bridge-crossing-rl-sim</text>
    <text x="0" y="256" font-family="{SANS}" font-size="24" fill="{TEXT}">A manager picks the next sub-goal.</text>
    <text x="0" y="290" font-family="{SANS}" font-size="24" fill="{TEXT}">A worker learns to walk there.</text>
    <text x="0" y="324" font-family="{SANS}" font-size="24" fill="{MUTED}">Watch them learn to build a bridge home.</text>
    <g transform="translate(0 358)" font-family="{MONO}" font-size="16" fill="{MUTED}">
      {footer}
    </g>
  </g>'''

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">Nigel’s Quest: Bridge-Crossing-RL-Sim</title>
  <desc id="desc">A hierarchical reinforcement learning agent, Nigel, on the simulation's real 20 by 12 grid. A numbered route runs from the log, to the river, across the bridge slot to the far bank, and into the house.</desc>
  <rect width="{W}" height="{H}" rx="20" fill="{BG}"/>
  {left}
  <g id="world-map">
    {"".join(m)}
  </g>
</svg>
'''


def box(x, y, w, h, title, tag, lines, color):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{SURFACE}" stroke="{BORDER}" stroke-width="2"/>',
           f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{color}"/>',
           f'<text x="{x+28}" y="{y+44}" font-family="{SANS}" font-size="28" font-weight="700" fill="{TEXT}">{title}</text>',
           f'<text x="{x+28}" y="{y+74}" font-family="{MONO}" font-size="18" fill="{color}">{tag}</text>']
    yy = y + 118
    for label, values in lines:
        out.append(f'<text x="{x+28}" y="{yy}" font-family="{MONO}" font-size="16" letter-spacing="1" fill="{MUTED}">{label}</text>')
        for value in values:
            yy += 26
            out.append(f'<text x="{x+28}" y="{yy}" font-family="{SANS}" font-size="20" fill="{TEXT}">{value}</text>')
        yy += 40
    return "".join(out)


def reward(x, y, w, who, items):
    out = [f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="18" letter-spacing="1" fill="{MUTED}">{who} IS PAID FOR</text>']
    for i, (amount, text, color) in enumerate(items):
        yy = y + 40 + i * 36
        out.append(f'<text x="{x}" y="{yy}" font-family="{MONO}" font-size="20" font-weight="700" fill="{color}">{amount}</text>')
        out.append(f'<text x="{x+92}" y="{yy}" font-family="{SANS}" font-size="20" fill="{TEXT}">{text}</text>')
    return "".join(out)


def how_it_works():
    W, H = 1200, 600
    arrow = lambda x1, x2, y, label: (
        f'<line x1="{x1}" y1="{y}" x2="{x2-10}" y2="{y}" stroke="{ACCENT}" stroke-width="3"/>'
        f'<path d="M{x2-12} {y-8} L{x2} {y} L{x2-12} {y+8} Z" fill="{ACCENT}"/>'
        f'<text x="{(x1+x2)/2}" y="{y-14}" text-anchor="middle" font-family="{MONO}" font-size="18" fill="{ACCENT}">{label}</text>')
    body = [
        f'<rect width="{W}" height="{H}" rx="20" fill="{BG}"/>',
        f'<text x="56" y="62" font-family="{SANS}" font-size="40" font-weight="800" fill="{TEXT}">How Nigel learns</text>',
        f'<text x="56" y="96" font-family="{SANS}" font-size="20" fill="{MUTED}">Two DQNs, two jobs, and no shared rewards.</text>',
        box(56, 128, 316, 254, "Manager", "DQN · plans", [
            ("SEES", ["has log · bridge placed ·", "crossed · last goal failed"]),
            ("PICKS", ["one of 4 sub-goals"])], ACCENT),
        arrow(372, 442, 230, "goal"),
        box(442, 128, 316, 254, "Worker", "goal-conditioned DQN + HER", [
            ("SEES", ["position · flags · goal (x, y)"]),
            ("PICKS", ["UP · DOWN · LEFT · RIGHT"])], WATER_HI),
        arrow(758, 828, 230, "move"),
        box(828, 128, 316, 254, "World", "20 × 12 grid", [
            ("MILESTONES", ["log picked up · bridge placed", "· crossed · reached home"]),
            ("ENDS", ["at home, or drowned"])], POS),
        f'<line x1="56" y1="420" x2="1144" y2="420" stroke="{BORDER}" stroke-width="2"/>',
        reward(56, 462, 500, "MANAGER", [
            ("+100", "its sub-goal produced the milestone", POS),
            ("−100", "anything else, then it re-plans", NEG)]),
        reward(628, 462, 500, "WORKER", [
            ("+100", "reaching the tile it was given", POS),
            ("−step", "each move, plus a small novelty bonus", MUTED),
            ("−200", "walking into the river", NEG)]),
    ]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
  <title id="title">How Nigel learns</title>
  <desc id="desc">The manager sees four progress flags and picks a sub-goal coordinate. The worker sees its position, the flags and that goal, and picks a move. The manager earns plus or minus 100 for whether its sub-goal produced a milestone; the worker earns 100 for reaching its tile, pays a step penalty, and loses 200 for drowning.</desc>
  {"".join(body)}
</svg>
'''


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hero.svg").write_text(hero())
    (OUT / "how-it-works.svg").write_text(how_it_works())
    print("wrote", OUT / "hero.svg", OUT / "how-it-works.svg")
