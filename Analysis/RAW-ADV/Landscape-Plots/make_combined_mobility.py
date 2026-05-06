#!/usr/bin/env python3
"""Combine per-agent mobility PNGs into one 4-across panel per (env, threshold).

Layout matches the existing make_combined.py style:
  - TITLE_H  strip: centred figure title  ("AntMaze Large: Top 10% Configurations")
  - HEADER_H strip: centred column labels (algorithm names)
  - Cell row: images scaled to CELL_W

Output: Landscape-Plots/combined/mobility_{env}_{threshold}.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

BASE = Path(__file__).parent

AGENTS = ["crl", "gciql", "gcivl", "qrl"]
AGENT_LABELS = ["CRL", "GCIQL", "GCIVL", "QRL"]
THRESHOLDS = ["90", "95"]

ENVS = {
    "antmaze-medium": (
        "2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip",
        "antmaze-medium-navigate-v0",
        "AntMaze Medium",
    ),
    "antmaze-large": (
        "2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip",
        "antmaze-large-navigate-v0",
        "AntMaze Large",
    ),
    "cube": (
        "2026-04-27-logs-advantage-cube-single-fixed-batch.zip",
        "cube-single-play-v0",
        "Cube",
    ),
    "scene": (
        "2026-05-03-logs-advantage-scene-single-fixed-batch.zip",
        "scene-play-v0",
        "Scene",
    ),
}

# folder "90" = top 10% (≥90% of max), "95" = top 5%
THRESHOLD_LABELS = {"90": "Top 10% Configurations", "95": "Top 5% Configurations"}

CELL_W = 1200
HEADER_H = 90
TITLE_H = 100
PADDING = 10

OUT_DIR = BASE / "combined"
OUT_DIR.mkdir(exist_ok=True)


def get_font(size: int):
    for path in [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]:
        for idx in [1, 0]:
            try:
                return ImageFont.truetype(path, size, index=idx)
            except Exception:
                continue
    return ImageFont.load_default()


def make_panel(env_key: str, threshold: str) -> None:
    dir_name, env_tag, env_label = ENVS[env_key]
    mob_dir = BASE / dir_name / "mobility" / threshold

    # Load and scale all images
    imgs = {}
    for agent in AGENTS:
        p = mob_dir / f"mobility-{agent}-{env_tag}.png"
        if not p.exists():
            print(f"  [missing] {p}")
            imgs[agent] = None
            continue
        img = Image.open(p).convert("RGBA")
        scale = CELL_W / img.width
        imgs[agent] = img.resize((CELL_W, int(img.height * scale)), Image.LANCZOS)

    ref = next((v for v in imgs.values() if v is not None), None)
    if ref is None:
        print(f"  [skip] no images for {env_key}/{threshold}")
        return

    cell_w, cell_h = ref.size
    n_cols = len(AGENTS)
    total_w = n_cols * cell_w + (n_cols - 1) * PADDING
    total_h = TITLE_H + HEADER_H + cell_h

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_title = get_font(56)
    font_header = get_font(48)

    # ── Title strip ───────────────────────────────────────────────────────────
    title = f"{env_label}: {THRESHOLD_LABELS[threshold]}"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((total_w - tw) // 2, (TITLE_H - th) // 2),
        title,
        fill=(0, 0, 0, 255),
        font=font_title,
    )

    # ── Column headers (algorithm names) ──────────────────────────────────────
    for c_idx, label in enumerate(AGENT_LABELS):
        cx = c_idx * (cell_w + PADDING) + cell_w // 2
        cy = TITLE_H + HEADER_H // 2
        bbox = draw.textbbox((0, 0), label, font=font_header)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2, cy - th // 2), label, fill=(0, 0, 0, 255), font=font_header
        )

    # ── Paste images ──────────────────────────────────────────────────────────
    for c_idx, agent in enumerate(AGENTS):
        img = imgs.get(agent)
        if img is not None:
            canvas.paste(img, (c_idx * (cell_w + PADDING), TITLE_H + HEADER_H), img)

    out = OUT_DIR / f"mobility_{env_key}_{threshold}.png"
    canvas.convert("RGB").save(out, dpi=(300, 300))
    print(f"  Saved → {out.relative_to(BASE)}")


def main() -> None:
    for env_key in ENVS:
        print(f"\n{env_key}")
        for threshold in THRESHOLDS:
            make_panel(env_key, threshold)


if __name__ == "__main__":
    main()
