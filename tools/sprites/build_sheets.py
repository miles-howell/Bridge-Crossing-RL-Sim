"""Build agent character spritesheets from text pixel grids.

Every character shares one body template (Nigel's layout): a 112x128 sheet of
16x32 frames in four rows.

    row 0  facing down   7 frames
    row 1  facing right  4 frames
    row 2  facing up     7 frames
    row 3  facing left   4 frames (row 1 mirrored)

Frame timing for the 7-frame rows is
    0 idle, 1 stride A (bob), 2 idle, 3 stride B (bob), 4 idle (bob), 5 idle (bob), 6 wave
and for the 4-frame side rows
    0 stride A (bob), 1 idle, 2 stride B (bob), 3 idle.

A "bob" frame draws the upper body one pixel lower; the legs never move with it.

The template owns the legs and the frame timing. A character file in
characters/<name>.py owns its palette and its upper body:

    NAME        display name
    PALETTE     {letter: (r, g, b)}. Must define the leg letters W (leg/boot
                outline), X (trouser), Y (trouser shade/inner seam), Z (boot).
    FRONT       17 lines x 16 chars, upper body facing down
    FRONT_WAVE  FRONT with the character's right arm raised (viewer's right)
    SIDE        17 lines x 16 chars, upper body facing right
    BACK        17 lines x 16 chars, upper body facing up
    BACK_WAVE   BACK with the same arm raised (viewer's left from behind)
    FRONT_LEGS / SIDE_LEGS   optional overrides of the template legs

'.' is transparent. Upper line 0 lands on cell row 5, so the head may use
lines 0-11, shoulders sit on line 12, hands on lines 14-15 and the torso ends
on line 16. Legs start on line 17 and idle boots end on cell row 25.

Usage (needs Pillow):
    python tools/sprites/build_sheets.py [name ...] [--preview-dir DIR]
With no names, every file in characters/ is built. Sheets are written to
static/images/characters/<name>.png. With --preview-dir, an 8x preview of each
sheet and a lineup.png of all built characters are written there too.
"""
import argparse
import runpy
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CHAR_DIR = HERE / "characters"
OUT_DIR = REPO / "static" / "images" / "characters"
GRASS_SHEET = REPO / "static" / "images" / "Overworld.png"

CELL_W, CELL_H = 16, 32
UPPER_TOP = 5
UPPER_LINES = 17
LEG_LETTERS = "WXYZ"

FRONT_LEGS = {
    "idle": ["...WYXXYYXXYW...", "...WZWWWWWWZW...", "...WZZZWWZZZW...", "....WWW..WWW...."],
    "walkA": ["...WYXXYYXXYW...", "...WZWWWYXXYW...", "...WZZZWWZWWW...", "....WWW.WZZZW...", ".........WWW...."],
    "walkB": ["...WYXXYYXXYW...", "...WYXXYWWWZW...", "...WWWZWWZZZW...", "...WZZZW.WWW....", "....WWW........."],
}
SIDE_LEGS = {
    "idle": [".....WYXXYW.....", ".....WZZZZW.....", ".....WZZZZZW....", "......WWWWW....."],
    "walkA": [".....WYXXYW.....", "....WXXWWXXW....", "...WYYW..WXXW...", "...WZZZW.WZZZW..", "....WWW...WWW..."],
    "walkB": [".....WYXXYW.....", "....WYYWWXXW....", "...WXXW..WYYW...", "...WZZZW.WZZZW..", "....WWW...WWW..."],
}

SEVEN = [("up", "idle", 0), ("up", "walkA", 1), ("up", "idle", 0), ("up", "walkB", 1),
         ("up", "idle", 1), ("up", "idle", 1), ("wave", "idle", 0)]
FOUR = [("up", "walkA", 1), ("up", "idle", 0), ("up", "walkB", 1), ("up", "idle", 0)]


def load(name):
    data = runpy.run_path(str(CHAR_DIR / f"{name}.py"))
    data.setdefault("FRONT_LEGS", FRONT_LEGS)
    data.setdefault("SIDE_LEGS", SIDE_LEGS)
    return data


def validate(name, c):
    errors = []
    pal = c.get("PALETTE", {})
    for letter in LEG_LETTERS:
        if letter not in pal:
            errors.append(f"PALETTE is missing leg letter {letter}")
    grids = {k: c.get(k) for k in ("FRONT", "FRONT_WAVE", "SIDE", "BACK", "BACK_WAVE")}
    for key in ("FRONT_LEGS", "SIDE_LEGS"):
        for pose, lines in c[key].items():
            grids[f"{key}[{pose}]"] = lines
    for key, lines in grids.items():
        if lines is None:
            errors.append(f"{key} is missing")
            continue
        if key in ("FRONT", "FRONT_WAVE", "SIDE", "BACK", "BACK_WAVE") and len(lines) != UPPER_LINES:
            errors.append(f"{key} has {len(lines)} lines, expected {UPPER_LINES}")
        for i, row in enumerate(lines):
            if len(row) != CELL_W:
                errors.append(f"{key} line {i} is {len(row)} wide, expected {CELL_W}")
            for ch in set(row) - {"."}:
                if ch not in pal:
                    errors.append(f"{key} line {i} uses {ch!r}, which is not in PALETTE")
    return [f"{name}: {e}" for e in errors]


def render_cell(c, upper, legs, bob, mirror=False):
    cell = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))

    def put(lines, top):
        for y, row in enumerate(lines):
            for x, ch in enumerate(row):
                if ch != ".":
                    cell.putpixel((x, top + y), tuple(c["PALETTE"][ch]) + (255,))

    put(legs, UPPER_TOP + UPPER_LINES)
    put(upper, UPPER_TOP + bob)
    return cell.transpose(Image.FLIP_LEFT_RIGHT) if mirror else cell


def build_sheet(c):
    sheet = Image.new("RGBA", (CELL_W * 7, CELL_H * 4), (0, 0, 0, 0))
    rows = [
        (SEVEN, {"up": c["FRONT"], "wave": c["FRONT_WAVE"]}, c["FRONT_LEGS"], False),
        (FOUR, {"up": c["SIDE"]}, c["SIDE_LEGS"], False),
        (SEVEN, {"up": c["BACK"], "wave": c["BACK_WAVE"]}, c["FRONT_LEGS"], False),
        (FOUR, {"up": c["SIDE"]}, c["SIDE_LEGS"], True),
    ]
    for r, (timing, uppers, legs, mirror) in enumerate(rows):
        for col, (upper, pose, bob) in enumerate(timing):
            cell = render_cell(c, uppers[upper], legs[pose], bob, mirror)
            sheet.paste(cell, (col * CELL_W, r * CELL_H))
    return sheet


def grass_tile():
    try:
        return Image.open(GRASS_SHEET).convert("RGBA").crop((0, 0, 16, 16))
    except OSError:
        return None


def build_lineup(sheets):
    """Idle front, side and back of each character, on grass and on a dark panel."""
    scale = 4
    grass = grass_tile()
    poses = [(0, 1), (1, 1), (2, 0), (3, 1)]  # (row, col) of an idle frame
    width = CELL_W * len(poses) * len(sheets)
    lineup = Image.new("RGBA", (width, CELL_H * 2), (32, 32, 40, 255))
    if grass:
        for x in range(0, width, 16):
            for y in range(0, CELL_H, 16):
                lineup.paste(grass, (x, y))
    for i, sheet in enumerate(sheets):
        for j, (row, col) in enumerate(poses):
            frame = sheet.crop((col * CELL_W, row * CELL_H, (col + 1) * CELL_W, (row + 1) * CELL_H))
            x = (i * len(poses) + j) * CELL_W
            lineup.alpha_composite(frame, (x, 0))
            lineup.alpha_composite(frame, (x, CELL_H))
    return lineup.resize((lineup.width * scale, lineup.height * scale), Image.NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    names = args.names or sorted(p.stem for p in CHAR_DIR.glob("*.py"))
    chars = {n: load(n) for n in names}
    errors = [e for n, c in chars.items() for e in validate(n, c)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheets = []
    for name, c in chars.items():
        sheet = build_sheet(c)
        sheet.save(OUT_DIR / f"{name}.png")
        sheets.append(sheet)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            sheet.resize((sheet.width * 8, sheet.height * 8), Image.NEAREST).save(args.preview_dir / f"{name}_preview.png")
        print(f"built {OUT_DIR.relative_to(REPO) / (name + '.png')}")
    if args.preview_dir:
        build_lineup(sheets).save(args.preview_dir / "lineup.png")


if __name__ == "__main__":
    main()
