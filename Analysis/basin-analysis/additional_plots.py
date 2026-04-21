import ast
import json
import re
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# legend for markers (use phase 1 scatter as proxy)
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans

BASE = Path("/Users/adityamohan/git/GCRL/plots/new-plots/raw-data/logs-antmaze")
OUT = Path("/Users/adityamohan/git/GCRL/plots/new-plots/advantage-dist-phases")
OUT.mkdir(exist_ok=True)

CLIP = 100.0
LOG100 = np.log(CLIP)  # ≈ 4.605
PCTS = list(range(0, 101))

MEDIUM_ENVS = {
    "antmaze-medium-navigate-v0",
    "antmaze-medium-explore-v0",
    "antmaze-medium-explore40navigate-v0",
    "antmaze-medium-explore80navigate-v0",
    "antmaze-medium-explore-v0 antmaze-medium-explore80navigate-v0 antmaze-medium-explore40navigate-v0 antmaze-medium-navigate-v0",
}

ALGOS = ["CRL", "GCIQL", "GCIVL", "QRL"]
PHASES = [1, 2, 3, 4]


def tail1(path: Path) -> str:
    return subprocess.run(
        ["tail", "-1", str(path)], capture_output=True, text=True
    ).stdout.strip()


def head1(path: Path) -> str:
    return subprocess.run(
        ["head", "-1", str(path)], capture_output=True, text=True
    ).stdout.strip()


# ── Data collection ──────────────────────────────────────────────────────────

records = []
z_quant_records = []

algo_env_dirs = sorted(
    d
    for d in BASE.iterdir()
    if d.is_dir() and re.match(r"^(CRL|GCIQL|GCIVL|QRL)_.*_64c_lr-alpha$", d.name)
)

for algo_env_dir in algo_env_dirs:
    m = re.match(r"^(CRL|GCIQL|GCIVL|QRL)_(.*?)_64c_lr-alpha$", algo_env_dir.name)
    algo, env = m.group(1), m.group(2)

    if env not in MEDIUM_ENVS:
        print(f"  skip {algo}/{env}")
        continue

    print(f"Processing {algo} / {env} ...")

    config_info: dict[int, dict] = {}
    for cjson in (algo_env_dir / "configurations").glob("configuration_*.json"):
        idx = int(re.search(r"(\d+)", cjson.name).group(1))
        cfg = json.loads(cjson.read_text())
        config_info[idx] = {
            "alpha": cfg.get("alpha"),
            "lr": cfg.get("lr"),
        }

    run_logs = algo_env_dir / "run_logs"

    all_phase_steps: set[int] = set()
    for config_dir in run_logs.iterdir():
        if not config_dir.is_dir():
            continue
        for phase_dir in config_dir.iterdir():
            if phase_dir.is_dir():
                all_phase_steps.add(int(phase_dir.name.split("_")[1]))
    phase_rank = {s: i + 1 for i, s in enumerate(sorted(all_phase_steps))}

    sample_csv = next(run_logs.rglob("train_log.csv"), None)
    if sample_csv is None:
        continue
    header = head1(sample_csv).split(";")
    adv_col_idx = (
        header.index("advantage/actor") if "advantage/actor" in header else None
    )
    if adv_col_idx is None:
        print("  no advantage/actor column, skipping")
        continue

    for config_dir in sorted(run_logs.iterdir()):
        if not config_dir.is_dir():
            continue
        config_idx = int(config_dir.name.split("_")[1])
        info = config_info.get(config_idx, {})
        alpha = info.get("alpha")
        lr = info.get("lr")
        if alpha is None or alpha <= 0:
            continue

        for phase_dir in sorted(config_dir.iterdir()):
            if not phase_dir.is_dir():
                continue
            phase_step = int(phase_dir.name.split("_")[1])
            phase_num = phase_rank[phase_step]

            for seed_dir in sorted(phase_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                seed = int(seed_dir.name.split("_")[1])
                csv_file = seed_dir / "train_log.csv"
                eval_file = seed_dir / "eval_log.csv"
                if not csv_file.exists():
                    continue

                try:
                    last_line = tail1(csv_file)
                    fields = last_line.split(";")
                    adv_str = fields[adv_col_idx]
                    if not adv_str.startswith("["):
                        continue
                    A = np.array(ast.literal_eval(adv_str), dtype=np.float32)
                    z = (alpha * A).clip(-500, 500)  # z = α·A

                    n = len(z)
                    w = np.minimum(np.exp(z), CLIP)  # AWR weights

                    # ── Sample fractions (z-space) ──
                    frac_neg = float(np.mean(z < 0))
                    frac_mid = float(np.mean((z >= 0) & (z <= LOG100)))
                    frac_sat = float(np.mean(z > LOG100))

                    # ── Normalized weight mass ──
                    w_sum = w.sum()
                    if w_sum > 0:
                        wt = w / w_sum
                        W_neg = float(wt[z < 0].sum())
                        W_mid = float(wt[(z >= 0) & (z <= LOG100)].sum())
                        W_sat = float(wt[z > LOG100].sum())
                    else:
                        W_neg = W_mid = W_sat = 0.0

                    # ── ESS ──
                    w2_sum = (w**2).sum()
                    ess = float((w_sum**2) / (n * w2_sum)) if w2_sum > 0 else 0.0

                    # ── Regime map coordinates ──
                    p_plus = float(np.mean(z > 0))
                    pos_mask = z > 0
                    saturation = (
                        float(np.mean(z[pos_mask] > LOG100)) if pos_mask.any() else 0.0
                    )

                    # ── Success rate ──
                    success = float("nan")
                    if eval_file.exists():
                        try:
                            row = tail1(eval_file)
                            val = row.split(",")[0]
                            success = float(val)
                        except Exception:
                            pass

                    records.append(
                        {
                            "algo": algo,
                            "env": env,
                            "config": config_idx,
                            "seed": seed,
                            "phase": phase_num,
                            "alpha": alpha,
                            "lr": lr,
                            "frac_neg": frac_neg,
                            "frac_mid": frac_mid,
                            "frac_sat": frac_sat,
                            "W_neg": W_neg,
                            "W_mid": W_mid,
                            "W_sat": W_sat,
                            "ess": ess,
                            "p_plus": p_plus,
                            "saturation": saturation,
                            "success": success,
                        }
                    )

                    # z-quantiles (wide format)
                    q_row = {
                        "algo": algo,
                        "env": env,
                        "config": config_idx,
                        "seed": seed,
                        "phase": phase_num,
                        "alpha": alpha,
                    }
                    for p, v in zip(PCTS, np.percentile(z, PCTS)):
                        q_row[f"q{p:03d}"] = float(v)
                    z_quant_records.append(q_row)

                except Exception as exc:
                    print(f"    error {csv_file}: {exc}")

df = pd.DataFrame(records)
df_q = pd.DataFrame(z_quant_records)

df.to_csv(OUT / "additional_stats_raw.csv", index=False)
df_q.to_csv(OUT / "z_quantiles_raw.csv", index=False)
print(f"\nRaw records: {len(df)}")

# ── Shared aesthetics ─────────────────────────────────────────────────────────

sns.set_theme(context="paper", style="whitegrid")
PHASE_COLORS = [cm.viridis(x) for x in np.linspace(0.15, 0.85, 4)]
ALGO_COLORS = {
    "CRL": "#4878CF",
    "GCIQL": "#6ACC65",
    "GCIVL": "#D65F5F",
    "QRL": "#B47CC7",
}
REG_COLORS = ["#4878CF", "#6ACC65", "#D65F5F"]  # neg / mid / sat
Q_COLS = [f"q{p:03d}" for p in PCTS]

# =============================================================================
# Plot 1 — z = αA distribution (CDF) per algorithm, coloured by phase
# =============================================================================

z_agg = df_q.groupby(["algo", "phase"])[Q_COLS].median().reset_index()
z_agg.to_csv(OUT / "z_distribution_aggregated.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

for ax, algo in zip(axes, ALGOS):
    sub = z_agg[z_agg["algo"] == algo].sort_values("phase")
    for _, row in sub.iterrows():
        ph = int(row["phase"])
        zq = row[Q_COLS].values.astype(float)
        ax.plot(
            zq,
            np.array(PCTS) / 100,
            color=PHASE_COLORS[ph - 1],
            lw=1.8,
            label=f"Phase {ph}",
        )

    ax.axvline(0, color="gray", ls="--", lw=1.0)
    ax.axvline(LOG100, color="red", ls="--", lw=1.0)
    ax.set_title(algo, fontweight="bold", fontsize=13)
    ax.set_xlabel("z = αA")
    ax.set_xlim(-15, 8)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("CDF")
# legend on last panel
handles = [plt.Line2D([0], [0], color=PHASE_COLORS[i], lw=2) for i in range(4)]
labels = [f"Phase {i + 1}" for i in range(4)]
handles += [
    plt.Line2D([0], [0], color="gray", ls="--", lw=1),
    plt.Line2D([0], [0], color="red", ls="--", lw=1),
]
labels += ["z = 0 (weight boundary)", f"z = log(100) ≈ {LOG100:.2f} (clip)"]
axes[-1].legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=8,
    title="Legend",
)

fig.suptitle(
    "z = αA CDF Across Training Phases\n"
    "(AntMaze Medium, median across configs, seeds & environments)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT / "z_distribution_cdf.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "z_distribution_cdf.pdf", bbox_inches="tight")
plt.close()
print("Plot 1 saved: z_distribution_cdf")

# =============================================================================
# Plot 2 — GCIQL basins: k-means on (log10 lr, log10 α), then fractions + ESS
# =============================================================================

gciql = df[df["algo"] == "GCIQL"].copy()
gciql = gciql.dropna(subset=["lr", "alpha"])
gciql = gciql[(gciql["lr"] > 0) & (gciql["alpha"] > 0)]

# Config-level summary (mean across seeds, median across envs, last phase)
cfg_summary = (
    gciql[gciql["phase"] == gciql["phase"].max()]
    .groupby(["config", "alpha", "lr"])["success"]
    .mean()
    .reset_index()
    .dropna(subset=["success"])
)

# k-means on log-scale hyperparameters
X = np.column_stack(
    [
        np.log10(cfg_summary["lr"].values.astype(float)),
        np.log10(cfg_summary["alpha"].values.astype(float)),
    ]
)
km = KMeans(n_clusters=2, random_state=0, n_init=20)
cfg_summary["basin"] = km.fit_predict(X)

# Label basins by mean alpha: higher mean alpha → basin A ("upper")
basin_alpha = cfg_summary.groupby("basin")["alpha"].mean()
upper_basin = int(basin_alpha.idxmax())
cfg_summary["basin_label"] = cfg_summary["basin"].map(
    {upper_basin: "Upper (high α)", 1 - upper_basin: "Lower (low α)"}
)
cfg_summary.to_csv(OUT / "gciql_basin_assignments.csv", index=False)

# Merge basin labels back to full gciql records
basin_map = cfg_summary.set_index("config")["basin_label"].to_dict()
gciql["basin"] = gciql["config"].map(basin_map)
gciql_basined = gciql.dropna(subset=["basin"])

# Aggregate: median per basin × phase
basin_agg = (
    gciql_basined.groupby(["basin", "phase"])[
        ["frac_neg", "frac_mid", "frac_sat", "W_neg", "W_mid", "W_sat", "ess"]
    ]
    .median()
    .reset_index()
)
basin_agg.to_csv(OUT / "gciql_basin_fractions.csv", index=False)

# ── subplot: stacked bars (fractions) + ESS line ──
basins = ["Upper (high α)", "Lower (low α)"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)

for row_i, basin in enumerate(basins):
    sub = basin_agg[basin_agg["basin"] == basin].set_index("phase").reindex(PHASES)

    # stacked fractions
    ax_f = axes[row_i][0]
    bottoms = np.zeros(len(PCTS[:4]))
    region_labels = [
        "z < 0\n(zero weight)",
        "0 ≤ z ≤ log(100)\n(active)",
        "z > log(100)\n(clipped)",
    ]
    for col, color, lbl in zip(
        ["frac_neg", "frac_mid", "frac_sat"], REG_COLORS, region_labels
    ):
        vals = sub[col].fillna(0).values
        ax_f.bar(PHASES, vals, bottom=bottoms, color=color, label=lbl, width=0.6)
        bottoms += vals
    ax_f.set_ylim(0, 1)
    ax_f.set_xticks(PHASES)
    ax_f.set_xlabel("Phase")
    ax_f.set_title(f"{basin} — sample fractions", fontsize=10, fontweight="bold")
    if row_i == 0:
        ax_f.legend(fontsize=7, loc="lower right")

    # ESS
    ax_e = axes[row_i][1]
    ess_vals = sub["ess"].fillna(0).values
    ax_e.bar(PHASES, ess_vals, color="#888888", width=0.6)
    ax_e.set_ylim(0, max(ess_vals.max() * 1.2, 0.01))
    ax_e.set_xticks(PHASES)
    ax_e.set_xlabel("Phase")
    ax_e.set_ylabel("ESS (normalised)")
    ax_e.set_title(f"{basin} — ESS", fontsize=10, fontweight="bold")

fig.suptitle(
    "GCIQL Hyperparameter Basins — z-Space Fractions & ESS per Phase\n"
    "(k-means k=2 on log₁₀(lr) × log₁₀(α); median across configs, seeds & envs)",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT / "gciql_basins.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "gciql_basins.pdf", bbox_inches="tight")
plt.close()
print("Plot 2 saved: gciql_basins")

# ── bonus scatter: basin map ──
fig, ax = plt.subplots(figsize=(5, 4))
for basin, grp in cfg_summary.groupby("basin_label"):
    sc = ax.scatter(
        np.log10(grp["lr"]),
        np.log10(grp["alpha"]),
        c=grp["success"],
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        s=60,
        label=basin,
        edgecolors="k",
        linewidths=0.4,
        marker=("o" if "Upper" in basin else "s"),
    )
plt.colorbar(sc, ax=ax, label="Success rate (last phase)")
ax.set_xlabel("log₁₀(lr)")
ax.set_ylabel("log₁₀(α)")
ax.legend(fontsize=8)
ax.set_title("GCIQL basin map\n(k-means k=2)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "gciql_basin_map.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "gciql_basin_map.pdf", bbox_inches="tight")
plt.close()
print("Plot 2b saved: gciql_basin_map")

# =============================================================================
# Plot 3 — Weight mass by region (W_neg, W_mid, W_sat), not sample fractions
# =============================================================================

# Aggregate like original plot: median across configs & seeds, mean across envs
w_agg = (
    df.groupby(["algo", "env", "phase"])[["W_neg", "W_mid", "W_sat"]]
    .median()
    .reset_index()
)
w_agg_mean = (
    w_agg.groupby(["algo", "phase"])[["W_neg", "W_mid", "W_sat"]].mean().reset_index()
)
w_agg.to_csv(OUT / "weight_mass_per_env.csv", index=False)
w_agg_mean.to_csv(OUT / "weight_mass_aggregated.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
w_labels = [
    "z < 0\n(zero weight)",
    "0 ≤ z ≤ log(100)\n(active)",
    "z > log(100)\n(clipped at 100)",
]

for ax, algo in zip(axes, ALGOS):
    sub = w_agg_mean[w_agg_mean["algo"] == algo].set_index("phase").reindex(PHASES)
    bottoms = np.zeros(len(PHASES))
    for col, color, lbl in zip(["W_neg", "W_mid", "W_sat"], REG_COLORS, w_labels):
        vals = sub[col].fillna(0).values
        ax.bar(PHASES, vals, bottom=bottoms, color=color, label=lbl, width=0.6)
        bottoms += vals
    ax.set_title(algo, fontweight="bold", fontsize=13)
    ax.set_xlabel("Phase")
    ax.set_xticks(PHASES)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Normalized weight mass")
axes[-1].legend(
    title="Region",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=8,
    title_fontsize=8,
)
fig.suptitle(
    "AWR Weight Mass by Region Across Training Phases\n"
    "(AntMaze Medium, median across configs & seeds)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT / "weight_mass_dist.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "weight_mass_dist.pdf", bbox_inches="tight")
plt.close()
print("Plot 3 saved: weight_mass_dist")

# =============================================================================
# Plot 4 — 2-D regime map: p+ vs saturation, coloured by success rate
# =============================================================================

# Per config, aggregate over seeds & envs: median p+, saturation; mean success
regime = (
    df.groupby(["algo", "config", "phase"])
    .agg(
        p_plus=("p_plus", "median"),
        saturation=("saturation", "median"),
        success=("success", "mean"),
        alpha=("alpha", "median"),
    )
    .reset_index()
)
regime.to_csv(OUT / "regime_map_data.csv", index=False)

# One panel per phase
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True, sharex=True)

for ax, phase in zip(axes, PHASES):
    sub = regime[regime["phase"] == phase]
    for algo in ALGOS:
        g = sub[sub["algo"] == algo].dropna(subset=["success"])
        if g.empty:
            continue
        sc = ax.scatter(
            g["p_plus"],
            g["saturation"],
            c=g["success"],
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            s=30,
            alpha=0.7,
            marker={"CRL": "o", "GCIQL": "s", "GCIVL": "^", "QRL": "D"}[algo],
            label=algo,
        )

    ax.set_title(f"Phase {phase}", fontweight="bold")
    ax.set_xlabel("p₊ = Pr(z > 0)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

axes[0].set_ylabel("Saturation = Pr(z > log(100) | z > 0)")

# shared colorbar
plt.colorbar(sc, ax=axes.tolist(), label="Success rate", shrink=0.8)


legend_elems = [
    Line2D([0], [0], marker=m, color="gray", ls="none", ms=7, label=a)
    for a, m in [("CRL", "o"), ("GCIQL", "s"), ("GCIVL", "^"), ("QRL", "D")]
]
axes[0].legend(handles=legend_elems, fontsize=8, loc="upper right")

fig.suptitle(
    "AWR 2-D Regime Map: winner fraction vs saturation, coloured by success rate\n"
    "(AntMaze Medium — one dot per config, median across seeds & environments)",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT / "regime_map.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "regime_map.pdf", bbox_inches="tight")
plt.close()
print("Plot 4 saved: regime_map")

print("\nDone. All plots and CSVs written to:", OUT)
