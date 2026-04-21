"""
Three analyses from test.md:

  1. Distribution Stability   — Wasserstein-1 between consecutive-phase z-distributions,
                                per (algo, config, seed).  Reveals whether advantages
                                converge or keep drifting as training proceeds.

  2. α-Sensitivity            — How frac_sat and frac_mid shift when the temperature α
                                is scaled by k ∈ {0.1 … 10}. Computed from z-quantiles
                                so no raw-data re-read needed.  Reveals how brittle
                                policy extraction is to the temperature choice.

  3. Advantage→Return Corr.  — Cross-config Spearman correlation between advantage
                                geometry stats (p_plus, W_mid, W_sat, ESS, median_z)
                                and empirical success rate.  Tells us whether the critic
                                is well-calibrated.

All inputs come from the pre-computed CSVs in OUT/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from scipy import stats as sci_stats

OUT = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/advantage-dist-phases/basin-analysis"
)
OUTD = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/advantage-dist-phases/advantage-trustworthiness"
)
OUTD.mkdir(exist_ok=True)

LOG100 = np.log(100.0)  # ≈ 4.605
ALGOS = ["CRL", "GCIQL", "GCIVL", "QRL"]
PHASES = [1, 2, 3, 4]

df = pd.read_csv(OUT / "additional_stats_raw.csv")
df_q = pd.read_csv(OUT / "z_quantiles_raw.csv")

Q_COLS = [f"q{p:03d}" for p in range(101)]
PCTS = np.arange(101) / 100.0

sns.set_theme(context="paper", style="whitegrid")
PHASE_COLORS = [cm.viridis(x) for x in np.linspace(0.15, 0.85, 4)]
ALGO_COLORS = {
    "CRL": "#4878CF",
    "GCIQL": "#6ACC65",
    "GCIVL": "#D65F5F",
    "QRL": "#B47CC7",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DISTRIBUTION STABILITY — Wasserstein-1 between consecutive phases
# ─────────────────────────────────────────────────────────────────────────────
# W₁ from quantile functions: W₁ = ∫₀¹ |F⁻¹₁(p) - F⁻¹₂(p)| dp ≈ mean|q1_p - q2_p|

stab_records = []
for (algo, env, config, seed), grp in df_q.groupby(["algo", "env", "config", "seed"]):
    grp = grp.sort_values("phase").reset_index(drop=True)
    q_mat = grp[Q_COLS].values.astype(float)  # (n_phases, 101)
    phases = grp["phase"].tolist()
    for i in range(len(phases) - 1):
        w1 = float(np.mean(np.abs(q_mat[i] - q_mat[i + 1])))
        stab_records.append(
            {
                "algo": algo,
                "env": env,
                "config": config,
                "seed": seed,
                "phase_from": phases[i],
                "phase_to": phases[i + 1],
                "transition": f"{phases[i]}->{phases[i + 1]}",
                "w1": w1,
            }
        )

df_stab = pd.DataFrame(stab_records)
df_stab.to_csv(OUTD / "stability_w1_raw.csv", index=False)

# aggregate: median across configs & seeds (already mixed environments inside config id)
stab_agg = (
    df_stab.groupby(["algo", "transition"])["w1"]
    .agg(
        median=("median"),
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    .reset_index()
)
stab_agg.to_csv(OUTD / "stability_w1_aggregated.csv", index=False)

# Plot
transitions = ["1->2", "2->3", "3->4"]
x = np.arange(len(transitions))
width = 0.18

fig, ax = plt.subplots(figsize=(8, 4))
for i, algo in enumerate(ALGOS):
    sub = (
        stab_agg[stab_agg["algo"] == algo].set_index("transition").reindex(transitions)
    )
    medians = sub["median"].fillna(0).values
    q25 = sub["q25"].fillna(0).values
    q75 = sub["q75"].fillna(0).values
    offset = (i - 1.5) * width
    bars = ax.bar(
        x + offset, medians, width=width, color=ALGO_COLORS[algo], label=algo, alpha=0.9
    )
    ax.errorbar(
        x + offset,
        medians,
        yerr=[medians - q25, q75 - medians],
        fmt="none",
        color="black",
        capsize=3,
        lw=1,
    )

ax.set_xticks(x)
ax.set_xticklabels(transitions)
ax.set_xlabel("Phase transition")
ax.set_ylabel("Wasserstein-1 distance in z-space")
ax.legend(fontsize=9)
ax.set_title(
    "Phase-to-Phase Distribution Stability of z = alpha*A\n"
    "(median W1 across configs & seeds; error bars = IQR)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTD / "stability_w1.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTD / "stability_w1.pdf", bbox_inches="tight")
plt.close()
print("Stability plot saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. α-SENSITIVITY — vary temperature multiplier k, track region shifts
# ─────────────────────────────────────────────────────────────────────────────
# For α' = k·α, z' = k·z.
#   frac_neg stays the same (k > 0 doesn't flip sign)
#   frac_sat(k) = Pr(z > log100 / k) — interpolated from stored quantiles
#   frac_mid(k) = 1 − frac_neg − frac_sat(k)

K_VALS = [0.1, 0.2, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def frac_above(q_sorted: np.ndarray, threshold: float) -> float:
    """Approximate Pr(Z > threshold) from 101 sorted quantile values."""
    idx = np.searchsorted(q_sorted, threshold, side="right")  # 0..101
    return float(np.clip(1.0 - idx / 100.0, 0.0, 1.0))


sens_records = []
for _, row in df_q.iterrows():
    z_q = row[Q_COLS].values.astype(float)
    frac_neg = float(frac_above(-z_q[::-1], 0.0))  # Pr(z < 0) = Pr(-z > 0)
    frac_neg = float(np.searchsorted(z_q, 0.0, side="left") / 100.0)
    frac_neg = float(np.clip(frac_neg, 0, 1))

    for k in K_VALS:
        threshold = LOG100 / k  # z above this → clipped under α'=kα
        frac_sat_k = frac_above(z_q, threshold)
        frac_mid_k = float(np.clip(1.0 - frac_neg - frac_sat_k, 0.0, 1.0))
        sens_records.append(
            {
                "algo": row["algo"],
                "config": row["config"],
                "seed": row["seed"],
                "phase": row["phase"],
                "k": k,
                "frac_neg": frac_neg,
                "frac_mid": frac_mid_k,
                "frac_sat": frac_sat_k,
            }
        )

df_sens = pd.DataFrame(sens_records)
df_sens.to_csv(OUTD / "alpha_sensitivity_raw.csv", index=False)

sens_agg = (
    df_sens.groupby(["algo", "phase", "k"])[["frac_neg", "frac_mid", "frac_sat"]]
    .median()
    .reset_index()
)
sens_agg.to_csv(OUTD / "alpha_sensitivity_aggregated.csv", index=False)

# Plot: one panel per algo, x=k, lines=phase, y=frac_sat (and frac_mid)
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)

for col_i, algo in enumerate(ALGOS):
    sub = sens_agg[sens_agg["algo"] == algo]

    for row_i, (metric, ylabel, title_sfx) in enumerate(
        [
            ("frac_sat", "frac_sat(k) = Pr(z > log100/k)", "Saturated (clipped)"),
            ("frac_mid", "frac_mid(k)", "Active (useful gradient)"),
        ]
    ):
        ax = axes[row_i][col_i]
        for ph in PHASES:
            grp = sub[sub["phase"] == ph].sort_values("k")
            ax.plot(
                grp["k"],
                grp[metric],
                color=PHASE_COLORS[ph - 1],
                lw=2,
                marker="o",
                ms=4,
                label=f"Phase {ph}",
            )
        ax.axvline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("α multiplier k")
        ax.set_ylabel(ylabel if col_i == 0 else "")
        if row_i == 0:
            ax.set_title(algo, fontweight="bold", fontsize=12)
        if col_i == 3 and row_i == 0:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))

fig.suptitle(
    "α-Sensitivity: how region fractions shift as temperature is scaled\n"
    "(top = clipped fraction; bottom = active fraction; dashed = current α)",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTD / "alpha_sensitivity.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTD / "alpha_sensitivity.pdf", bbox_inches="tight")
plt.close()
print("α-sensitivity plot saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ADVANTAGE→RETURN CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
# We use success rate as the empirical return proxy.
# Cross-config Spearman(advantage_stat, success) tells us if the critic's
# learned advantages track which configurations actually work.
#
# Per-sample ground-truth returns are not available in the logs, so we test
# the coarser but meaningful question: do configs whose advantage geometry
# looks "healthy" also achieve higher success?

# Merge in median z (q050 from z_quantiles)
median_z = df_q[["algo", "env", "config", "seed", "phase", "q050"]].copy()
median_z.rename(columns={"q050": "median_z"}, inplace=True)
df_full = df.merge(median_z, on=["algo", "env", "config", "seed", "phase"], how="left")

# Aggregate per (algo, config, phase): mean success across seeds & envs, median stats
corr_base = (
    df_full.dropna(subset=["success"])
    .groupby(["algo", "config", "phase"])
    .agg(
        success=("success", "mean"),
        p_plus=("p_plus", "median"),
        W_mid=("W_mid", "median"),
        W_sat=("W_sat", "median"),
        W_neg=("W_neg", "median"),
        ess=("ess", "median"),
        median_z=("median_z", "median"),
        frac_sat=("frac_sat", "median"),
    )
    .reset_index()
)
corr_base.to_csv(OUTD / "advantage_return_corr_data.csv", index=False)

# Compute Spearman r per (algo, phase) for each stat
stats_to_test = {
    "p+ = Pr(z>0)": "p_plus",
    "W_mid (active)": "W_mid",
    "W_sat (clipped)": "W_sat",
    "ESS": "ess",
    "Median z": "median_z",
}

corr_records = []
for algo in ALGOS:
    for phase in PHASES:
        sub = corr_base[(corr_base["algo"] == algo) & (corr_base["phase"] == phase)]
        if len(sub) < 5:
            continue
        y = sub["success"].values
        for label, col in stats_to_test.items():
            x = sub[col].values
            if np.std(x) < 1e-10:
                continue
            r, p = sci_stats.spearmanr(x, y)
            corr_records.append(
                {
                    "algo": algo,
                    "phase": phase,
                    "stat": label,
                    "col": col,
                    "spearman_r": r,
                    "p_value": p,
                }
            )

df_corr = pd.DataFrame(corr_records)
df_corr.to_csv(OUTD / "spearman_correlations.csv", index=False)

# ── Heatmap of Spearman r per algo×stat (averaged across phases) ──
corr_pivot = (
    df_corr.groupby(["algo", "stat"])["spearman_r"]
    .mean()
    .unstack("algo")
    .reindex(columns=ALGOS)
    .reindex(index=list(stats_to_test.keys()))
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(
    corr_pivot,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 10},
)
ax.set_title(
    "Spearman r: advantage geometry stat vs. success rate\n"
    "(mean across phases; each cell = r over all configs for that algo)",
    fontweight="bold",
)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUTD / "advantage_return_corr_heatmap.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTD / "advantage_return_corr_heatmap.pdf", bbox_inches="tight")
plt.close()
print("Correlation heatmap saved.")

# ── Scatter: p_plus vs success per algo (phase=last) ──
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

for ax, algo in zip(axes, ALGOS):
    sub_all = corr_base[corr_base["algo"] == algo]
    for ph in PHASES:
        sub = sub_all[sub_all["phase"] == ph]
        ax.scatter(
            sub["p_plus"],
            sub["success"],
            color=PHASE_COLORS[ph - 1],
            alpha=0.5,
            s=12,
            label=f"Phase {ph}",
        )
    # overall regression across all phases
    x = sub_all["p_plus"].values
    y = sub_all["success"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 5:
        slope, intercept, r, p, _ = sci_stats.linregress(x[mask], y[mask])
        xr = np.array([x[mask].min(), x[mask].max()])
        ax.plot(xr, slope * xr + intercept, "k--", lw=1.2)
        r_sp, _ = sci_stats.spearmanr(x[mask], y[mask])
        ax.text(0.05, 0.92, f"ρ = {r_sp:.2f}", transform=ax.transAxes, fontsize=9)
    ax.set_title(algo, fontweight="bold")
    ax.set_xlabel("p+ = Pr(z > 0)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

axes[0].set_ylabel("Success rate")
handles = [
    plt.Line2D(
        [0], [0], marker="o", ls="none", color=PHASE_COLORS[i], label=f"Phase {i + 1}"
    )
    for i in range(4)
]
axes[-1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.suptitle(
    "Does p+ = Pr(z > 0) predict success rate?\n"
    "(one dot per config; ρ = Spearman; dashed = OLS; coloured by phase)",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTD / "pplus_vs_success.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTD / "pplus_vs_success.pdf", bbox_inches="tight")
plt.close()
print("p_plus vs success scatter saved.")

# ── Scatter: W_mid vs success (the 'useful gradient mass' metric) ──
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

for ax, algo in zip(axes, ALGOS):
    sub_all = corr_base[corr_base["algo"] == algo]
    for ph in PHASES:
        sub = sub_all[sub_all["phase"] == ph]
        ax.scatter(
            sub["W_mid"],
            sub["success"],
            color=PHASE_COLORS[ph - 1],
            alpha=0.5,
            s=12,
            label=f"Phase {ph}",
        )
    x = sub_all["W_mid"].values
    y = sub_all["success"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 5:
        slope, intercept, *_ = sci_stats.linregress(x[mask], y[mask])
        xr = np.array([x[mask].min(), x[mask].max()])
        ax.plot(xr, slope * xr + intercept, "k--", lw=1.2)
        r_sp, _ = sci_stats.spearmanr(x[mask], y[mask])
        ax.text(0.05, 0.92, f"ρ = {r_sp:.2f}", transform=ax.transAxes, fontsize=9)
    ax.set_title(algo, fontweight="bold")
    ax.set_xlabel("W_mid (active weight mass)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

axes[0].set_ylabel("Success rate")
axes[-1].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.suptitle(
    "Does W_mid (unsaturated weight mass) predict success rate?\n"
    "(one dot per config; ρ = Spearman; dashed = OLS; coloured by phase)",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTD / "wmid_vs_success.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTD / "wmid_vs_success.pdf", bbox_inches="tight")
plt.close()
print("W_mid vs success scatter saved.")

print(f"\nDone. All outputs in: {OUTD}")
