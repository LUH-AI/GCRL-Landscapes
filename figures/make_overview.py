"""Generate paper teaser figure (Fig. 1) — hourglass layout.

Run: python figures/make_overview.py
Output: figures/approach-overview.svg (and .pdf)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── Matplotlib / LaTeX setup ───────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
""",
    "figure.dpi": 150,
})

# ── Palette ────────────────────────────────────────────────────────────────
COLORS = {
    "GCIQL": "#4C72B0",
    "GCIVL": "#DD8452",
    "QRL":   "#55A868",
    "CRL":   "#C44E52",
}
GRAY      = "#444444"
LIGHTGRAY = "#CCCCCC"
BG        = "white"

# ── Canvas ─────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 16, 9
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(BG)

# ── Layout constants ───────────────────────────────────────────────────────
MARGIN  = 0.35
FULL_W  = FIG_W - 2 * MARGIN
FULL_X  = MARGIN
ROW_GAP = 0.32

R1_H = 1.25
R2_H = 2.15
R3_H = 1.05
R4_H = 2.85

R4_Y = 0.30
R3_Y = R4_Y + R4_H + ROW_GAP
R2_Y = R3_Y + R3_H + ROW_GAP
R1_Y = R2_Y + R2_H + ROW_GAP

# Alg column layout
N_ALG   = 4
ALG_GAP = 0.28
ALG_W   = (FULL_W - (N_ALG - 1) * ALG_GAP) / N_ALG
ALG_XS  = [FULL_X + i * (ALG_W + ALG_GAP) for i in range(N_ALG)]

# Row 4 equal split
R4_INNER_GAP = 0.30
R4a_W = (FULL_W - R4_INNER_GAP) / 2
R4b_W = R4a_W
R4a_X = FULL_X
R4b_X = FULL_X + R4a_W + R4_INNER_GAP

# ── Helpers ────────────────────────────────────────────────────────────────
def rounded_rect(ax_, x, y, w, h, color, lw=1.8, fc="white", zorder=2, radius=0.18):
    box = FancyBboxPatch(
        (x + radius, y + radius),
        w - 2 * radius, h - 2 * radius,
        boxstyle=f"round,pad={radius}",
        linewidth=lw, edgecolor=color, facecolor=fc,
        zorder=zorder, clip_on=False,
    )
    ax_.add_patch(box)
    return box


def header_strip(ax_, x, y_top, w, h=0.40, color="#DDDDDD"):
    s = FancyBboxPatch(
        (x + 0.16, y_top - h),
        w - 0.32, h - 0.04,
        boxstyle="round,pad=0.04", linewidth=0, facecolor=color, zorder=3,
    )
    ax_.add_patch(s)


# ══════════════════════════════════════════════════════════════════════════
# ROW 1 — Offline Dataset
# ══════════════════════════════════════════════════════════════════════════
rounded_rect(ax, FULL_X, R1_Y, FULL_W, R1_H, color=GRAY, fc="#F5F5F5")

thumb_h = 0.75
thumb_w = 1.60
thumb_y = R1_Y + (R1_H - thumb_h) / 2
ant_x   = FULL_X + 0.38

for tx, tlabel in [(ant_x, r"\textit{antmaze}"),
                   (ant_x + thumb_w + 0.20, r"\textit{cube / scene}")]:
    rounded_rect(ax, tx, thumb_y, thumb_w, thumb_h, color=LIGHTGRAY, fc="#E8E8E8", lw=1.0)
    ax.text(tx + thumb_w / 2, thumb_y + thumb_h / 2, tlabel,
            ha="center", va="center", fontsize=9, color="#777777")

label_cx = (ant_x + 2 * thumb_w + 0.20 + 0.5 + FULL_X + FULL_W) / 2
ax.text(label_cx, R1_Y + R1_H * 0.67, r"\textbf{Offline Dataset}",
        ha="center", va="center", fontsize=13.5, color=GRAY)
ax.text(label_cx, R1_Y + R1_H * 0.28,
        r"\footnotesize antmaze-medium $\cdot$ antmaze-large $\cdot$ cube-play $\cdot$ scene-play",
        ha="center", va="center", fontsize=9, color="#555555")

# ══════════════════════════════════════════════════════════════════════════
# Fan-out: Row 1 bottom-center -> 4 alg box top-centers
# ══════════════════════════════════════════════════════════════════════════
src_x = FULL_X + FULL_W / 2
for i, alg in enumerate(["GCIQL", "GCIVL", "QRL", "CRL"]):
    ax.annotate("",
        xy=(ALG_XS[i] + ALG_W / 2, R2_Y + R2_H), xytext=(src_x, R1_Y),
        arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.11",
                        color=COLORS[alg], lw=1.3), zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# ROW 2 — Algorithm boxes
# ══════════════════════════════════════════════════════════════════════════
ALGS = [
    {"name": "GCIQL", "formula": r"$Q(s,a,g) - V(s,g)$",                "semantic": r"\textit{Bootstrapped Q--V margin}"},
    {"name": "GCIVL", "formula": r"$V(s',g) - V(s,g)$",                 "semantic": r"\textit{Local value progress}"},
    {"name": "QRL",   "formula": r"$d_\phi(s,g) - d_\phi(s',g)$",       "semantic": r"\textit{Quasimetric distance reduction}"},
    {"name": "CRL",   "formula": r"$f_\theta(s,a,g) - f_\theta^V(s,g)$","semantic": r"\textit{Approx.\ log-reachability ratio}"},
]
for i, alg in enumerate(ALGS):
    name = alg["name"]; c = COLORS[name]
    x, y, w, h = ALG_XS[i], R2_Y, ALG_W, R2_H
    hdr_h = 0.44
    rounded_rect(ax, x, y, w, h, color=c, fc="white", lw=1.8)
    ax.add_patch(FancyBboxPatch((x + 0.16, y + h - hdr_h), w - 0.32, hdr_h - 0.05,
        boxstyle="round,pad=0.05", linewidth=0, facecolor=c, zorder=3))
    ax.text(x + w / 2, y + h - hdr_h / 2, r"\textbf{" + name + r"}",
            ha="center", va="center", fontsize=12, color="white", zorder=4)
    ax.text(x + w / 2, y + h * 0.47, alg["formula"],
            ha="center", va="center", fontsize=10.5, color="#222222", zorder=3)
    ax.text(x + w / 2, y + h * 0.15, alg["semantic"],
            ha="center", va="center", fontsize=8.0, color="#555555", zorder=3)

# ══════════════════════════════════════════════════════════════════════════
# Fan-in: Row 2 bottom-centers -> Row 3 top-center
# ══════════════════════════════════════════════════════════════════════════
dst_x = FULL_X + FULL_W / 2
for i, alg in enumerate(["GCIQL", "GCIVL", "QRL", "CRL"]):
    ax.annotate("",
        xy=(dst_x, R3_Y + R3_H), xytext=(ALG_XS[i] + ALG_W / 2, R2_Y),
        arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.11",
                        color=COLORS[alg], lw=1.3), zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# ROW 3 — Shared AWR Policy Extractor
# ══════════════════════════════════════════════════════════════════════════
rounded_rect(ax, FULL_X, R3_Y, FULL_W, R3_H, color=GRAY, fc="#F5F5F5", lw=1.8)
cx3 = FULL_X + FULL_W / 2
ax.text(cx3, R3_Y + R3_H * 0.82, r"\textbf{Shared AWR Policy Extractor}",
        ha="center", va="center", fontsize=12.5, color=GRAY)
ax.text(cx3, R3_Y + R3_H * 0.53,
        r"$\mathcal{L}_{\mathrm{AWR}}(\pi) = -\mathbb{E}\bigl[w_\alpha(s,a,g)\log\pi(a\mid s,g)\bigr]$",
        ha="center", va="center", fontsize=10.5, color="#222222")
ax.text(cx3, R3_Y + R3_H * 0.20,
        r"$w_\alpha = \min\!\left\{\exp\!\left(\alpha\, A_{\mathrm{alg}}\right),\, w_{\max}\right\}$",
        ha="center", va="center", fontsize=10, color="#444444")

# Diverging arrows Row 3 -> Row 4a/4b top-centers
for rx in [R4a_X + R4a_W / 2, R4b_X + R4b_W / 2]:
    ax.annotate("",
        xy=(rx, R4_Y + R4_H), xytext=(cx3, R3_Y),
        arrowprops=dict(arrowstyle="->,head_width=0.17,head_length=0.12",
                        color=GRAY, lw=1.4), zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# ROW 4a — Advantage Quality (left half)
# ══════════════════════════════════════════════════════════════════════════
rounded_rect(ax, R4a_X, R4_Y, R4a_W, R4_H, color=GRAY, fc="#F8F8F8", lw=1.6)
header_strip(ax, R4a_X, R4_Y + R4_H, R4a_W)
ax.text(R4a_X + R4a_W / 2, R4_Y + R4_H - 0.20,
        r"\textbf{Advantage Quality}",
        ha="center", va="center", fontsize=11, color=GRAY, zorder=4)

# Subtitle inside box, below header strip
ax.text(R4a_X + R4a_W / 2, R4_Y + R4_H - 0.56,
        r"\textit{future-random AUC across training phases}",
        ha="center", va="center", fontsize=8, color="#666666")

# Sparklines ─────────────────────────────────────────────────────────────
phases = np.arange(1, 11)
np.random.seed(42)

def schematic_frac_auc(pattern):
    if   pattern == "CRL":   base = 0.55 + 0.32*(1-np.exp(-phases*0.55)); sig = 0.012
    elif pattern == "QRL":   base = 0.53 + 0.26*(1-np.exp(-phases*0.45)); sig = 0.015
    elif pattern == "GCIQL": base = 0.50 + 0.20*(1-np.exp(-phases*0.28)); sig = 0.025
    else:                    base = 0.51 + 0.16*(1-np.exp(-phases*0.33)); sig = 0.018
    return np.clip(base + np.random.normal(0, sig, len(phases)), 0.45, 1.0)

# Layout: generous left margin for alg names, tight right margin
spark_pad_l = 0.80   # room for alg name labels
spark_pad_r = 0.22
spark_pad_t = 0.75   # below header+subtitle
spark_pad_b = 0.45   # space for x-axis label
spark_gap   = 0.08
n_sparks    = 4

spark_x0   = R4a_X + spark_pad_l
spark_y0   = R4_Y  + spark_pad_b
spark_w    = R4a_W - spark_pad_l - spark_pad_r
spark_h_total = R4_H - spark_pad_t - spark_pad_b
spark_h    = (spark_h_total - spark_gap*(n_sparks-1)) / n_sparks

alg_order = ["CRL", "QRL", "GCIQL", "GCIVL"]

for k, alg in enumerate(alg_order):
    sk_y = spark_y0 + (n_sparks - 1 - k) * (spark_h + spark_gap)
    ax_sp = ax.inset_axes([spark_x0, sk_y, spark_w, spark_h], transform=ax.transData)
    vals  = schematic_frac_auc(alg)
    ax_sp.plot(phases, vals, color=COLORS[alg], lw=1.8, solid_capstyle="round")
    ax_sp.fill_between(phases, 0.45, vals, color=COLORS[alg], alpha=0.15)
    ax_sp.set_xlim(1, 10)
    ax_sp.set_ylim(0.44, 1.01)
    # No y-tick labels — just a thin left spine as scale reference
    ax_sp.set_yticks([0.5, 0.75, 1.0])
    ax_sp.set_yticklabels([])
    ax_sp.tick_params(axis="y", length=2, width=0.5)
    if k == n_sparks - 1:
        ax_sp.set_xticks([1, 5, 10])
        ax_sp.tick_params(axis="x", labelsize=6.5, pad=1, length=2)
        ax_sp.set_xlabel(r"training phase", fontsize=8, labelpad=2)
    else:
        ax_sp.set_xticks([])
    ax_sp.spines["top"].set_visible(False)
    ax_sp.spines["right"].set_visible(False)
    ax_sp.spines["left"].set_linewidth(0.6)
    ax_sp.spines["bottom"].set_linewidth(0.6)

    # Alg name: colored bold label to the LEFT of the sparkline
    ax.text(spark_x0 - 0.10, sk_y + spark_h / 2,
            r"\textbf{" + alg + r"}",
            ha="right", va="center", fontsize=9, color=COLORS[alg],
            transform=ax.transData)

# Y-axis label: vertical text, centered on sparkline stack
ax.text(R4a_X + 0.20, spark_y0 + spark_h_total / 2,
        r"FR-AUC",
        ha="center", va="center", fontsize=8, color="#444444",
        rotation=90, transform=ax.transData)

# ══════════════════════════════════════════════════════════════════════════
# ROW 4b — Trainability Landscapes (right half)
# ══════════════════════════════════════════════════════════════════════════
rounded_rect(ax, R4b_X, R4_Y, R4b_W, R4_H, color=GRAY, fc="#F8F8F8", lw=1.6)
header_strip(ax, R4b_X, R4_Y + R4_H, R4b_W)
ax.text(R4b_X + R4b_W / 2, R4_Y + R4_H - 0.20,
        r"\textbf{Trainability Landscapes}",
        ha="center", va="center", fontsize=11, color=GRAY, zorder=4)
ax.text(R4b_X + R4b_W / 2, R4_Y + R4_H - 0.56,
        r"\textit{$\eta \times \alpha$ performance heatmaps}",
        ha="center", va="center", fontsize=8, color="#666666")

# 2x2 grid of mini heatmaps ───────────────────────────────────────────────
# Compute cell geometry so we know the aspect ratio for shape correction
grid_pad_l = 0.50
grid_pad_r = 0.25
grid_pad_t = 0.78   # below header+subtitle+cell-label
grid_pad_b = 0.52   # space for shared x-axis label
grid_gap_x = 0.30
grid_gap_y = 0.40   # includes space for per-cell alg labels above

grid_x0  = R4b_X + grid_pad_l
grid_y0  = R4_Y  + grid_pad_b
grid_w   = R4b_W - grid_pad_l - grid_pad_r
grid_h   = R4_H  - grid_pad_t - grid_pad_b

cell_w = (grid_w - grid_gap_x) / 2
cell_h = (grid_h - grid_gap_y) / 2

# Aspect ratio of each cell in display-inches (data units == inches here)
# To make contours appear circular we stretch x-axis contribution
cell_aspect = cell_w / cell_h   # ≈ 4–5  (wide cells)

nx, ny = 60, 60
xx = np.linspace(0, 1, nx)
yy = np.linspace(0, 1, ny)
XX, YY = np.meshgrid(xx, yy)

def schematic_landscape(alg, ar):
    """ar = cell_w/cell_h; stretch XX to compensate so contours look circular."""
    Xs = (XX - 0.5) * ar   # scale x to match y in display space
    if alg == "GCIQL":
        # narrow peak (fragile)
        return 0.85 * np.exp(-(Xs**2 + (YY - 0.5)**2) / 0.06)
    elif alg == "GCIVL":
        # medium width
        return 0.75 * np.exp(-(Xs**2 + (YY - 0.5)**2) / 0.18)
    elif alg == "QRL":
        # broad plateau (robust) — wide flat top
        return (0.70 * np.exp(-(Xs**2 + (YY - 0.5)**2) / 0.55)
              + 0.15 * np.exp(-((XX - 0.3)**2 * ar**2 + (YY - 0.72)**2) / 0.10))
    else:  # CRL — asymmetric, off-center ridge
        return (0.72 * np.exp(-(((XX-0.6)*ar)**2) / 0.12 - ((YY-0.45)**2) / 0.06)
              + 0.28 * np.exp(-(((XX-0.35)*ar)**2) / 0.08 - ((YY-0.62)**2) / 0.05))

# row 0 = bottom row, row 1 = top row
alg_grid = [["QRL", "CRL"], ["GCIQL", "GCIVL"]]

for row in range(2):
    for col in range(2):
        alg    = alg_grid[row][col]
        cell_x = grid_x0 + col * (cell_w + grid_gap_x)
        cell_y = grid_y0 + row * (cell_h + grid_gap_y)

        ax_mini = ax.inset_axes([cell_x, cell_y, cell_w, cell_h],
                                transform=ax.transData)
        Z = schematic_landscape(alg, cell_aspect)
        ax_mini.contourf(XX, YY, Z, levels=12, cmap="RdYlGn")
        ax_mini.contour(XX, YY, Z, levels=6, colors="white",
                        linewidths=0.4, alpha=0.5)
        ax_mini.set_xticks([]); ax_mini.set_yticks([])
        for spine in ax_mini.spines.values():
            spine.set_linewidth(2.0)
            spine.set_edgecolor(COLORS[alg])

        # Alg name above cell (in data coords, with enough gap_y to fit)
        ax.text(cell_x + cell_w / 2, cell_y + cell_h + 0.06,
                r"\textbf{" + alg + r"}",
                ha="center", va="bottom", fontsize=9, color=COLORS[alg],
                transform=ax.transData)

# Shared axis labels outside the 2x2 grid
mid_x = grid_x0 + grid_w / 2
ax.text(mid_x, grid_y0 - 0.28, r"learning rate $\eta$",
        ha="center", va="top", fontsize=8.5, color="#444444",
        transform=ax.transData)
ax.text(R4b_X + 0.17, grid_y0 + grid_h / 2, r"temperature $\alpha$",
        ha="center", va="center", fontsize=8.5, color="#444444",
        rotation=90, transform=ax.transData)

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════
out_dir  = os.path.dirname(os.path.abspath(__file__))
svg_path = os.path.join(out_dir, "approach-overview.svg")
pdf_path = os.path.join(out_dir, "approach-overview.pdf")

fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=BG)
fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Saved: {svg_path}")
print(f"Saved: {pdf_path}")
