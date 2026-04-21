import ast
import json
import re
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path("/Users/adityamohan/git/GCRL/plots/new-plots/raw-data/logs-antmaze")
OUT = Path("/Users/adityamohan/git/GCRL/plots/new-plots/advantage-dist-phases")
OUT.mkdir(exist_ok=True)

CLIP = 100.0

MEDIUM_ENVS = {
    "antmaze-medium-navigate-v0",
    "antmaze-medium-explore-v0",
    "antmaze-medium-explore40navigate-v0",
    "antmaze-medium-explore80navigate-v0",
    "antmaze-medium-explore-v0 antmaze-medium-explore80navigate-v0 antmaze-medium-explore40navigate-v0 antmaze-medium-navigate-v0",
}


def tail1(path: Path) -> str:
    return subprocess.run(
        ["tail", "-1", str(path)], capture_output=True, text=True
    ).stdout.strip()


def head1(path: Path) -> str:
    return subprocess.run(
        ["head", "-1", str(path)], capture_output=True, text=True
    ).stdout.strip()


# collect records
records = []

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

    # alpha per config
    config_alpha: dict[int, float] = {}
    for cjson in (algo_env_dir / "configurations").glob("configuration_*.json"):
        idx = int(re.search(r"(\d+)", cjson.name).group(1))
        config_alpha[idx] = json.loads(cjson.read_text())["alpha"]

    run_logs = algo_env_dir / "run_logs"

    # phase step -> rank (1-4)
    all_phase_steps: set[int] = set()
    for config_dir in run_logs.iterdir():
        if not config_dir.is_dir():
            continue
        for phase_dir in config_dir.iterdir():
            if phase_dir.is_dir():
                all_phase_steps.add(int(phase_dir.name.split("_")[1]))
    phase_rank = {s: i + 1 for i, s in enumerate(sorted(all_phase_steps))}

    # read header once (same across all files in this algo-env dir)
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
        alpha = config_alpha.get(config_idx)
        if alpha is None or alpha <= 0:
            continue
        A_clip = np.log(CLIP) / alpha

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
                if not csv_file.exists():
                    continue

                try:
                    last_line = tail1(csv_file)
                    fields = last_line.split(";")
                    adv_str = fields[adv_col_idx]
                    if not adv_str.startswith("["):
                        continue
                    adv = np.array(ast.literal_eval(adv_str), dtype=np.float32)

                    frac_neg = float(np.mean(adv < 0))
                    frac_sub = float(np.mean((adv >= 0) & (adv <= A_clip)))
                    frac_clip = float(np.mean(adv > A_clip))

                    records.append(
                        {
                            "algo": algo,
                            "env": env,
                            "config": config_idx,
                            "seed": seed,
                            "phase": phase_num,
                            "alpha": alpha,
                            "A_clip": A_clip,
                            "frac_neg": frac_neg,
                            "frac_sub": frac_sub,
                            "frac_clip": frac_clip,
                        }
                    )
                except Exception as exc:
                    print(f"    error {csv_file}: {exc}")
                    continue

df = pd.DataFrame(records)
df.to_csv(OUT / "advantage_fractions_raw.csv", index=False)
print(f"\nRaw records: {len(df)}")

# aggregate: median across configs and seeds, mean across envs
agg = (
    df.groupby(["algo", "env", "phase"])[["frac_neg", "frac_sub", "frac_clip"]]
    .median()
    .reset_index()
)
agg_mean = (
    agg.groupby(["algo", "phase"])[["frac_neg", "frac_sub", "frac_clip"]]
    .mean()
    .reset_index()
)
agg.to_csv(OUT / "advantage_fractions_per_env.csv", index=False)
agg_mean.to_csv(OUT / "advantage_fractions_aggregated.csv", index=False)
print("CSVs saved.")

# plot
sns.set_theme(context="paper", style="whitegrid")

algos = ["CRL", "GCIQL", "GCIVL", "QRL"]
phases = [1, 2, 3, 4]
region_labels = [
    "A < 0\n(zero weight)",
    "0 ≤ A ≤ A_clip\n(active)",
    "A > A_clip\n(clipped at 100)",
]
region_cols = ["frac_neg", "frac_sub", "frac_clip"]
region_colors = ["#4878CF", "#6ACC65", "#D65F5F"]

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

for ax, algo in zip(axes, algos):
    sub = agg_mean[agg_mean["algo"] == algo].set_index("phase").reindex(phases)
    bottoms = np.zeros(len(phases))
    for col, color, label in zip(region_cols, region_colors, region_labels):
        vals = sub[col].fillna(0).values
        ax.bar(phases, vals, bottom=bottoms, color=color, label=label, width=0.6)
        bottoms += vals

    ax.set_title(algo, fontsize=13, fontweight="bold")
    ax.set_xlabel("Phase")
    ax.set_xticks(phases)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Fraction of advantage samples")
axes[-1].legend(
    title="Region  (α_clip = log(100)/α ≈ 0.31 at α=15)",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=8,
    title_fontsize=8,
)

fig.suptitle(
    "AWR Weight Region Distribution Across Training Phases\n(AntMaze Medium, median across configs & seeds)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT / "advantage_dist_phases.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "advantage_dist_phases.pdf", bbox_inches="tight")
plt.close()
print("Plot saved.")

# also plot per-env facets
env_labels = {
    "antmaze-medium-navigate-v0": "Navigate",
    "antmaze-medium-explore-v0": "Explore",
    "antmaze-medium-explore40navigate-v0": "Explore 40%",
    "antmaze-medium-explore80navigate-v0": "Explore 80%",
    "antmaze-medium-explore-v0 antmaze-medium-explore80navigate-v0 antmaze-medium-explore40navigate-v0 antmaze-medium-navigate-v0": "Scheduled",
}
envs = list(env_labels.keys())

fig, axes = plt.subplots(
    len(algos), len(envs), figsize=(18, 12), sharey=True, sharex=True
)

for r, algo in enumerate(algos):
    for c, env in enumerate(envs):
        ax = axes[r][c]
        sub = (
            agg[(agg["algo"] == algo) & (agg["env"] == env)]
            .set_index("phase")
            .reindex(phases)
        )
        bottoms = np.zeros(len(phases))
        for col, color, label in zip(region_cols, region_colors, region_labels):
            vals = sub[col].fillna(0).values
            ax.bar(phases, vals, bottom=bottoms, color=color, label=label, width=0.6)
            bottoms += vals
        ax.set_ylim(0, 1)
        ax.set_xticks(phases)
        if r == 0:
            ax.set_title(env_labels[env], fontsize=9)
        if c == 0:
            ax.set_ylabel(algo, fontsize=10, fontweight="bold")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in region_colors]
fig.legend(
    handles,
    region_labels,
    loc="lower center",
    ncol=3,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
)
fig.suptitle(
    "AWR Weight Region Distribution Across Phases — per Algorithm × Environment\n(median across configs & seeds)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUT / "advantage_dist_phases_per_env.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "advantage_dist_phases_per_env.pdf", bbox_inches="tight")
plt.close()
print("Per-env plot saved.")
