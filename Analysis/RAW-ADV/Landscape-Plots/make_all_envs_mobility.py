#!/usr/bin/env python3
"""4×4 all-environment top-10% mobility figure for NeurIPS.

Layout
------
  Rows    : AntMaze-M, AntMaze-L, Cube, Scene
  Columns : CRL, GCIQL, GCIVL, QRL

Row labels are rotated 90° in a left strip.
Column labels sit in a header strip above the grid.
Thin 1-px separator lines divide rows and columns.
Target width ≈ 7 inches at 300 DPI (NeurIPS textwidth).

Output
------
  Landscape-Plots/combined/mobility_all_envs_top10.png   300 DPI PNG
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

BASE = Path(__file__).parent

AGENTS = ["crl", "gciql", "gcivl", "qrl"]
AGENT_LABELS = ["CRL", "GCIQL", "GCIVL", "QRL"]

ENVS_ORDERED = ["antmaze-medium", "antmaze-large", "cube", "scene"]
ENV_LABELS = {
    "antmaze-medium": "AntMaze-M",
    "antmaze-large": "AntMaze-L",
    "cube": "Cube",
    "scene": "Scene",
}
ENV_DATA = {
    "antmaze-medium": (
        "2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip",
        "antmaze-medium-navigate-v0",
    ),
    "antmaze-large": (
        "2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip",
        "antmaze-large-navigate-v0",
    ),
    "cube": (
        "2026-04-27-logs-advantage-cube-single-fixed-batch.zip",
        "cube-single-play-v0",
    ),
    "scene": (
        "2026-05-03-logs-advantage-scene-single-fixed-batch.zip",
        "scene-play-v0",
    ),
}
THRESHOLD = "90"

# ── Layout constants (tune here) ─────────────────────────────────────────────
CELL_W = 480  # px per cell width  → total ≈ 2070 px = 6.9 in @ 300 DPI
ROW_LBL_W = 96  # px for left row-label strip
COL_HDR_H = 72  # px for column-header strip
GAP_C = 2  # px gap between columns (thin separator line shows here)
GAP_R = 2  # px gap between rows
SEP_COLOR = (200, 200, 200)  # separator line color
BG = (255, 255, 255)  # background white
FG = (20, 20, 20)  # label text color
DPI = 300

OUT_DIR = BASE / "combined"
OUT_DIR.mkdir(exist_ok=True)


def get_font(size: int):
    candidates = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 1),
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("/Library/Fonts/Arial Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    ]
    for path, idx in candidates:
        try:
            return ImageFont.truetype(path, size, index=idx)
        except Exception:
            continue
    return ImageFont.load_default()


def rotated_label(
    text: str, font, strip_w: int, strip_h: int, fg=FG, bg=BG
) -> Image.Image:
    """Render `text` rotated 90° CCW, centred in a (strip_w × strip_h) patch."""
    # Draw on a temporary tall-and-narrow canvas, then rotate
    tmp = Image.new("RGBA", (strip_h, strip_w), bg + (255,))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (strip_h - tw) // 2
    y = (strip_w - th) // 2
    draw.text((x, y), text, fill=fg + (255,), font=font)
    rotated = tmp.rotate(90, expand=True)  # CCW: text reads bottom→top
    return rotated.convert("RGB")


def main() -> None:
    # ── 1. Load and scale all 16 source images ───────────────────────────────
    imgs: dict[tuple, Image.Image | None] = {}
    ref_aspect = None

    for env_key in ENVS_ORDERED:
        dir_name, env_tag = ENV_DATA[env_key]
        mob_dir = BASE / dir_name / "mobility" / THRESHOLD
        for agent in AGENTS:
            p = mob_dir / f"mobility-{agent}-{env_tag}.png"
            if not p.exists():
                print(f"  [missing] {p}")
                imgs[(env_key, agent)] = None
                continue
            src = Image.open(p).convert("RGB")
            if ref_aspect is None:
                ref_aspect = src.height / src.width
            cell_h = int(CELL_W * (src.height / src.width))
            imgs[(env_key, agent)] = src.resize((CELL_W, cell_h), Image.LANCZOS)

    cell_h = int(CELL_W * ref_aspect) if ref_aspect else CELL_W

    n_rows = len(ENVS_ORDERED)
    n_cols = len(AGENTS)

    total_w = ROW_LBL_W + n_cols * CELL_W + (n_cols - 1) * GAP_C
    total_h = COL_HDR_H + n_rows * cell_h + (n_rows - 1) * GAP_R

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    font_col = get_font(44)  # algorithm column headers
    font_row = get_font(40)  # environment row labels (rendered into rotated patch)

    # ── 2. Column headers ────────────────────────────────────────────────────
    for c_idx, label in enumerate(AGENT_LABELS):
        x_left = ROW_LBL_W + c_idx * (CELL_W + GAP_C)
        cx = x_left + CELL_W // 2
        bbox = draw.textbbox((0, 0), label, font=font_col)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, (COL_HDR_H - th) // 2), label, fill=FG, font=font_col)

    # ── 3. Row labels + cell images ──────────────────────────────────────────
    for r_idx, env_key in enumerate(ENVS_ORDERED):
        y_top = COL_HDR_H + r_idx * (cell_h + GAP_R)

        # Rotated row label
        label_img = rotated_label(ENV_LABELS[env_key], font_row, ROW_LBL_W, cell_h)
        canvas.paste(label_img, (0, y_top))

        for c_idx, agent in enumerate(AGENTS):
            img = imgs.get((env_key, agent))
            x_left = ROW_LBL_W + c_idx * (CELL_W + GAP_C)
            if img is not None:
                canvas.paste(img, (x_left, y_top))
            else:
                # Grey placeholder for missing image
                draw.rectangle(
                    [x_left, y_top, x_left + CELL_W - 1, y_top + cell_h - 1],
                    fill=(230, 230, 230),
                )
                msg = "missing"
                bbox = draw.textbbox((0, 0), msg, font=get_font(28))
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (x_left + (CELL_W - tw) // 2, y_top + (cell_h - th) // 2),
                    msg,
                    fill=(160, 160, 160),
                )

    # ── 4. Separator lines ────────────────────────────────────────────────────
    # Horizontal separators between rows
    for r_idx in range(1, n_rows):
        y = COL_HDR_H + r_idx * (cell_h + GAP_R) - 1
        draw.line([(ROW_LBL_W, y), (total_w - 1, y)], fill=SEP_COLOR, width=1)

    # Vertical separators between columns
    for c_idx in range(1, n_cols):
        x = ROW_LBL_W + c_idx * (CELL_W + GAP_C) - 1
        draw.line([(x, COL_HDR_H), (x, total_h - 1)], fill=SEP_COLOR, width=1)

    # Outer border
    draw.rectangle(
        [ROW_LBL_W, COL_HDR_H, total_w - 1, total_h - 1],
        outline=(180, 180, 180),
        width=1,
    )

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out = OUT_DIR / "mobility_all_envs_top10.png"
    canvas.save(str(out), dpi=(DPI, DPI))
    print(f"Saved → {out}")
    print(
        f"  Size : {total_w} × {total_h} px  ({total_w / DPI:.2f} × {total_h / DPI:.2f} in @ {DPI} dpi)"
    )


if __name__ == "__main__":
    main()
