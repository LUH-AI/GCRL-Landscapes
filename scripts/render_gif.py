"""Render one landscape GIF per task and layout: one frame per phase.

Reads the GP grids written by fit_grids.py and uses the KDE settings of
gcrl_landscapes.evaluation.landscapes.mobility_plot, so a frame is one phase of
the corresponding paper figure. Frames cut between phases: interpolating would
draw landscapes that were never measured.

Usage: python render_gif.py <grids.pkl> <output_dir> <task_slug>
"""
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from PIL import Image

warnings.filterwarnings("ignore")

grids_path, out_dir, task = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = pickle.load(open(grids_path, "rb"))
AGENTS = ["GCIQL", "GCIVL", "CRL", "QRL"]
PHASES = [1, 2, 3, 4]
THRESHOLD = 0.8
MIN_PEAK = 0.05
PALETTE = sns.color_palette("viridis", n_colors=4)
GHOST = "#8a8a94"
INK = "#1d1d1f"
KDE = dict(thresh=0.05, bw_adjust=0.7)

sns.set_theme(style="white", context="paper")


def region(agent, phase):
    d = data[(agent.lower(), phase)]
    # Normalizing by a peak near zero would draw a region out of GP noise.
    if d["Z"].max() < MIN_PEAK:
        return pd.DataFrame({"x": [], "y": []}), d["bounds"]
    z = d["Z"] / d["Z"].max()
    keep = z > THRESHOLD
    return pd.DataFrame({"x": d["px"][keep], "y": d["py"][keep]}), d["bounds"]


regions = {(a, p): region(a, p) for a in AGENTS for p in PHASES}
LAYOUTS = {
    "wide": dict(rows=1, cols=4, size=(12.8, 3.5), dpi=130, top=0.92, title=14, label=11),
    "narrow": dict(rows=2, cols=2, size=(7.0, 6.6), dpi=115, top=0.94, title=13, label=10.5),
}


def kde(ax, df, bx, by, **kw):
    # A region of a few grid points has no density to draw.
    if len(df) < 5:
        return
    sns.kdeplot(data=df, x="x", y="y", log_scale=(bx[2], by[2]), ax=ax, **KDE, **kw)


def render(name, spec):
    frames = []
    for phase in PHASES:
        fig, axes = plt.subplots(spec["rows"], spec["cols"], figsize=spec["size"],
                                 sharex=True, sharey=True)
        for i, (ax, agent) in enumerate(zip(axes.ravel(), AGENTS)):
            df, (bx, by) = regions[(agent, phase)]
            for prev in PHASES[: phase - 1]:
                kde(ax, regions[(agent, prev)][0], bx, by, fill=False, levels=[0.7], color=GHOST,
                    linewidths=0.9, alpha=0.55, linestyles="--")
            color = PALETTE[phase - 1]
            kde(ax, df, bx, by, fill=True, levels=4, alpha=0.28, color=color, linewidth=0)
            kde(ax, df, bx, by, fill=False, levels=[0.7], color=color, linewidths=2.8)
            trail = [regions[(agent, p)][0] for p in PHASES[:phase]]
            trail = [t for t in trail if len(t)]
            cx = [t["x"].median() for t in trail]
            cy = [t["y"].median() for t in trail]
            if len(df):
                ax.plot(cx, cy, color=INK, lw=1.1, alpha=0.7, zorder=50)
                ax.scatter(cx[:-1], cy[:-1], s=22, c=INK, alpha=0.7, zorder=51)
                ax.scatter(cx[-1], cy[-1], s=170, color=color, edgecolors="white", lw=1.2, zorder=52)
                ax.annotate(str(phase), (cx[-1], cy[-1]), ha="center", va="center", color="white",
                            fontsize=10, fontweight="bold", zorder=53)
            else:
                ax.text(0.5, 0.5, "goal-distance return near 0:\nno region to draw",
                        transform=ax.transAxes, ha="center", va="center", fontsize=10,
                        color="#5b5b66")
            ax.set_title(agent, fontsize=spec["title"], fontweight="bold", color=INK)
            ax.set_xlim(bx[0], bx[1])
            ax.set_ylim(by[0], by[1])
            ax.set_xscale("log")
            bottom_row = i >= (spec["rows"] - 1) * spec["cols"]
            ax.set_xlabel("Learning rate" if bottom_row else "", fontsize=spec["label"])
            ax.set_ylabel("AWR temperature" if i % spec["cols"] == 0 else "", fontsize=spec["label"])
            ax.grid(True, alpha=0.15)
            sns.despine(ax=ax)
        step = 0.075 if spec["cols"] == 4 else 0.135
        for i, p in enumerate(PHASES):
            fig.text(0.36 + i * step if spec["cols"] == 4 else 0.135 + i * step, 0.965,
                     f"Phase {p}", ha="center", va="center", fontsize=11,
                     fontweight="bold" if p == phase else "normal",
                     color=INK if p == phase else "#b4b4bc")
        handles = [Line2D([], [], color=PALETTE[phase - 1], lw=2.8, label="this phase"),
                   Line2D([], [], color=GHOST, lw=0.9, ls="--", label="earlier phases")]
        fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2, fontsize=9,
                   bbox_to_anchor=(0.995, 1.005))
        fig.tight_layout(rect=(0, 0, 1, spec["top"]))
        png = out_dir / f"{task}_{name}_p{phase}.png"
        fig.savefig(png, dpi=spec["dpi"], facecolor="white")
        plt.close(fig)
        frames.append(Image.open(png).convert("RGB"))

    # One palette across frames so phase colours do not flicker.
    sheet = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for i, f in enumerate(frames):
        sheet.paste(f, (0, i * frames[0].height))
    # Median cut averages the background into an off-white; pin it back to the page colour.
    colours = sheet.quantize(colors=128, method=Image.MEDIANCUT).getpalette()
    for i in range(0, len(colours), 3):
        if min(colours[i : i + 3]) >= 240:
            colours[i : i + 3] = [255, 255, 255]
    palette = Image.new("P", (1, 1))
    palette.putpalette(colours)
    pframes = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    gif = out_dir / f"{task}_{name}.gif"
    pframes[0].save(gif, save_all=True, append_images=pframes[1:],
                    duration=[1500, 1500, 1500, 2600], loop=0)
    # Reduced-motion readers get the final frame; the other PNGs are intermediates.
    for phase in PHASES[:-1]:
        (out_dir / f"{task}_{name}_p{phase}.png").unlink()
    (out_dir / f"{task}_{name}_p4.png").rename(out_dir / f"{task}_{name}.png")
    print(gif, gif.stat().st_size // 1024, "KB", frames[0].size, flush=True)


out_dir.mkdir(parents=True, exist_ok=True)
for layout, spec in LAYOUTS.items():
    render(layout, spec)
