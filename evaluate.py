# ruff: noqa
# %%
# %% tags=["parameters"]
from pathlib import Path
import argparse

_p = argparse.ArgumentParser()
_p.add_argument("--zipfiles", nargs="+", type=Path)
_args, _ = _p.parse_known_args()
zipfiles = _args.zipfiles or [
    Path(
        "/home/mtoepperwien/Documents/gcrl/log_zips/2026-04-07-logs_antmaze-medium-all-algs.zip"
    )
]  # <- edit this
plot_dir = Path("plots") / zipfiles[0].name
tables_dir = Path("tables") / zipfiles[0].name

# %%
import os

os.makedirs(plot_dir / "cdf_swapped", exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# %%
import types
import gcrl_landscapes.evaluation.tabular as _tabular

_tabular.args = types.SimpleNamespace(no_multiprocessing=True)

from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)

# Memory optimization
to_category_columns = ["dataset", "hps", "agent"] + merged_training_df.columns[
    merged_training_df.columns.str.startswith("hp.")
].tolist()
for col in to_category_columns:
    merged_training_df[col] = merged_training_df[col].astype("category")
float_cols = merged_training_df.select_dtypes(include="float64").columns
for col in float_cols:
    merged_training_df[col] = merged_training_df[col].astype("float16")
merged_training_df = merged_training_df.drop(
    columns=[c for c in merged_training_df.columns if c.startswith("advantage/")],
    errors="ignore",
)

# %%
import pandas as pd
import numpy as np
from scipy.stats import trim_mean


def marginalize_seeds(df: pd.DataFrame):
    def trim_mean_with_assert(values, proportiontocut=0.25):
        assert values.nunique() == 5
        return trim_mean(values, proportiontocut=proportiontocut)

    def agg_constant_col(column_values):
        if column_values.nunique() > 1:
            raise ValueError(f"Not all values are the same:\n{column_values}")
        return column_values.iloc[0]

    return (
        df.groupby(
            by=[
                "agent",
                "dataset",
                "constant_dataset",
                "hps",
            ]
            + [
                "phase_num",
                "config_index",
            ]
        )
        .agg(
            {
                "success": lambda column_values: trim_mean(
                    column_values, proportiontocut=0.25
                ),
                "mean_normalized_goal_distance_return": lambda column_values: trim_mean(
                    column_values, proportiontocut=0.25
                ),
                "mean_normalized_goal_distance_return_normalized_regret": lambda column_values: trim_mean(
                    column_values, proportiontocut=0.25
                ),
                "seed": trim_mean_with_assert,
                **{
                    col: agg_constant_col
                    for col in df.columns[df.columns.str.startswith("hp.")]
                },
            }
        )
        .reset_index()
    )


def eps_optimality(df: pd.DataFrame, col: str) -> pd.Series:
    grouping = [
        "agent",
        "dataset",
        "constant_dataset",
        "hps",
        "phase_num",
    ]

    return df[col] / df.groupby(grouping)[col].transform("max")


def regret(df: pd.DataFrame, col: str) -> pd.Series:
    grouping = [
        "agent",
        "dataset",
        "constant_dataset",
        "phase_num",
    ]
    return df.groupby(grouping)[col].transform("max") - df[col]


# %%
merged_training_df: pd.DataFrame = merged_training_df
merged_results_df: pd.DataFrame = merged_results_df
merged_results_df["mean_normalized_goal_distance_return_eps_optimality"] = (
    eps_optimality(merged_results_df, "mean_normalized_goal_distance_return")
)
merged_results_df["mean_normalized_goal_distance_return_regret"] = regret(
    merged_results_df, "mean_normalized_goal_distance_return"
)
merged_results_df["mean_normalized_goal_distance_return_normalized_regret"] = regret(
    merged_results_df, "mean_normalized_goal_distance_return_eps_optimality"
)
merged_marginalized_results_df: pd.DataFrame = marginalize_seeds(merged_results_df)
merged_marginalized_results_df[
    "mean_normalized_goal_distance_return_eps_optimality"
] = eps_optimality(
    merged_marginalized_results_df, "mean_normalized_goal_distance_return"
)
merged_marginalized_results_df[
    "mean_normalized_goal_distance_return_normalized_regret"
] = regret(merged_marginalized_results_df, "mean_normalized_goal_distance_return")
merged_training_df.columns.tolist()

# %% [markdown]
# Let's convert `eval_step` to a percentage of training and bin that, so that we can compare different stages of training.
# Also extract exploration schedule from dataset names

# %%
import re


def datasets_to_exploration_schedule(dataset_str: str) -> str:
    def dataset_to_exploration_percentage(dataset: str) -> int:
        try:
            return int(re.search(r"explore(\d+)\w+", dataset).group(1))
        except:
            if re.match(r".*explore-.*", dataset):
                return 100
            elif re.match(r".*navigate-.*", dataset):
                return 0
            else:
                return 0  # non-antmaze datasets (e.g. cube) treated as fully expert

    datasets = dataset_str.split(",")
    return ",".join(
        [str(dataset_to_exploration_percentage(dataset)) for dataset in datasets]
    )


print(
    datasets_to_exploration_schedule(
        "antmaze-medium-explore-v0,antmaze-medium-explore90navigate-v0"
    )
)

# %%
# TODO: generalize this to dfs containing multiple hyperparameter combinations. Does this even have an influence?
merged_training_df["eval_percent"] = round(
    merged_training_df["eval_step"]
    / merged_training_df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform(
        "max"
    )
    * 100
).astype("category")
merged_training_df["eval_bins10"] = (
    merged_training_df.groupby(["hp.agent_name", "dataset"])["eval_percent"]
    .transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    .astype("category")
)
merged_training_df["eval_bins5"] = (
    merged_training_df.groupby(["hp.agent_name", "dataset"])["eval_percent"]
    .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    .astype("category")
)
merged_training_df["exploration_schedule"] = (
    merged_training_df["dataset"]
    .apply(datasets_to_exploration_schedule)
    .astype("category")
)

# %% [markdown]
# Get dataframe with only best performing model per experiment.
# We will have to merge training results so that we know the performance (iqm).

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print(merged_results_df.columns)
print(merged_training_df.columns.tolist())


# %%
iqm_df = merged_results_df[
    merged_results_df["eval_step"]
    == merged_results_df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform(
        "max"
    )
]
iqm_df = (
    iqm_df.groupby(["hp.agent_name", "dataset", "config_index"])[
        "mean_normalized_goal_distance_return"
    ]
    .apply(lambda x: trim_mean(x, proportiontocut=0.25))
    .reset_index(name="iqm")
)
# save iqm in merged_training_df for later
merged_training_with_iqm_df = merged_training_df.merge(
    iqm_df[["hp.agent_name", "dataset", "config_index", "iqm"]],
    how="inner",
    on=["hp.agent_name", "dataset", "config_index"],
)
# add mean_normalized_goal_distance_return and regret
# merged_training_with_iqm_df["phase"].describe()
merged_training_with_iqm_df = merged_training_with_iqm_df.merge(
    merged_results_df[
        [
            "seed",
            "hp.agent_name",
            "dataset",
            "phase",
            "config_index",
            "mean_normalized_goal_distance_return",
            "mean_normalized_goal_distance_return_regret",
            "mean_normalized_goal_distance_return_normalized_regret",
        ]
    ],
    how="left",
    on=["hp.agent_name", "dataset", "config_index", "phase", "seed"],
)
iqm_df = iqm_df[
    iqm_df["iqm"]
    == iqm_df.groupby(["hp.agent_name", "dataset"])["iqm"].transform("max")
]
best_config_df = merged_training_with_iqm_df.merge(
    iqm_df[["hp.agent_name", "dataset", "config_index"]],
    how="inner",
    on=["hp.agent_name", "dataset", "config_index"],
)

# %% [markdown]
# Let's evaluate the rank of the model:

# %%
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")
import matplotlib.pyplot as plt

merged_training_with_iqm_df["name_dataset_combination"] = (
    merged_training_with_iqm_df["hp.agent_name"].astype("str")
    + " - "
    + merged_training_with_iqm_df["dataset"].astype("str")
).astype("category")
end_of_training_df = merged_training_with_iqm_df[
    merged_training_with_iqm_df["eval_step"]
    == merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset"])[
        "eval_step"
    ].transform("max")
]
if "feature/embedding_rank" in merged_training_with_iqm_df.columns:
    print(
        merged_training_with_iqm_df.groupby("hp.agent_name")[
            "feature/embedding_rank"
        ].describe()
    )
    print(
        merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset", "eval_bins5"])[
            "feature/embedding_rank"
        ].describe()
    )

    # %% [markdown]
    # Capture this in a boxplot:

    # %%
    fig, ax = plt.subplots()
    sns.boxplot(data=end_of_training_df, x="hp.agent_name", y="feature/embedding_rank")
    plt.savefig(plot_dir / "rank_boxplot_agent.png")
    plt.close()
    for dataset in merged_training_with_iqm_df["dataset"].unique():
        fig, ax = plt.subplots()
        sns.boxplot(
            data=end_of_training_df[end_of_training_df["dataset"] == dataset],
            x="hp.agent_name",
            y="feature/embedding_rank",
        )
        print(dataset)
        print(
            merged_training_with_iqm_df[
                merged_training_with_iqm_df["dataset"] == dataset
            ]
            .groupby("hp.agent_name")["iqm"]
            .max()
        )
        plt.savefig(plot_dir / f"rank_boxplot_agent_{dataset}.png")
        plt.close()

    # %% [markdown]
    # Now for only best config per experiment:

    # %%
    print(best_config_df.groupby("hp.agent_name")["feature/embedding_rank"].describe())
    print(
        best_config_df.groupby(["hp.agent_name", "exploration_schedule", "eval_bins5"])[
            "feature/embedding_rank"
        ].describe()
    )


# %% [markdown]
# # Target drift


# %%
def literal_lists_to_numpy(s):
    import ast  # I do not know why this can't be imported up front

    try:
        return np.array(ast.literal_eval(s), dtype=np.float16)
    except:
        return None


try:
    merged_training_with_iqm_df["target/held_out_val_batch_values_np"] = (
        merged_training_with_iqm_df["target/held_out_val_batch_values"].apply(
            literal_lists_to_numpy
        )
    )
    merged_training_with_iqm_df = merged_training_with_iqm_df.drop(
        ["target/held_out_val_batch_values"], axis=1
    )
except KeyError:
    pass


# %%
def compute_drift_to(group: pd.DataFrame, target: str = "end"):
    def target_drift(a, b):
        if a is None or b is None:
            return None
        if not a.shape == b.shape:
            raise ValueError("Arrays must have the same shape")
        return np.linalg.norm(a - b)

    start_value = group.loc[
        group["eval_step"] == group["eval_step"].min(),
        "target/held_out_val_batch_values_np",
    ].values[0]
    end_value = group.loc[
        group["eval_step"] == group["eval_step"].max(),
        "target/held_out_val_batch_values_np",
    ].values[0]
    drift_start_end = target_drift(start_value, end_value)
    if target == "neighbor":
        values = group["target/held_out_val_batch_values_np"]
        compare_values = group["target/held_out_val_batch_values_np"].shift(1)
        return (
            pd.Series(
                [target_drift(a, b) for a, b in zip(values, compare_values)],
                index=group.index,
            )
            / drift_start_end
        )
    elif target == "end":
        compare_value = end_value
    elif target == "start":
        compare_value = start_value
    else:
        raise ValueError(f"Unknown target: {target}")
    return group["target/held_out_val_batch_values_np"].apply(
        lambda x: target_drift(x, compare_value) / drift_start_end
        if drift_start_end
        else None
    )


if "target/held_out_val_batch_values_np" in merged_training_with_iqm_df.columns:
    merged_training_with_iqm_df["target_drift_end"] = (
        merged_training_with_iqm_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "seed"], group_keys=False
        ).apply(compute_drift_to, target="end")
    )
    merged_training_with_iqm_df["target_drift_start"] = (
        merged_training_with_iqm_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "seed"], group_keys=False
        ).apply(compute_drift_to, target="start")
    )
    merged_training_with_iqm_df["target_drift_neighbor"] = (
        merged_training_with_iqm_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "seed"], group_keys=False
        ).apply(compute_drift_to, target="neighbor")
    )
    print(
        merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])[
            "target_drift_end"
        ].agg(["mean", "std"])
    )
    print(
        merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])[
            "target_drift_start"
        ].agg(["mean", "std"])
    )
    print(
        merged_training_with_iqm_df.groupby(
            ["hp.agent_name", "exploration_schedule", "eval_bins5"]
        )["target_drift_neighbor"].agg(["mean", "std"])
    )
# %% [markdown]
#

# %% [markdown]
# # Gradient Interference
#
# ## Gradients
#
# Let's look at gradient interference data. We start with **cosine similarity**:

# %%
good_configs_df = merged_training_with_iqm_df[merged_training_with_iqm_df["iqm"] > 0.5]
if "grad/value_cosine_similarity_mean" in merged_training_with_iqm_df.columns:
    print(
        merged_training_with_iqm_df.groupby(["hp.agent_name"])[
            ["grad/value_cosine_similarity_mean", "grad/actor_cosine_similarity_mean"]
        ].mean()
    )
    print("Only best configuration per landscape for next print")
    print(
        best_config_df.groupby(["exploration_schedule", "hp.agent_name"])[
            ["grad/value_cosine_similarity_mean", "grad/actor_cosine_similarity_mean"]
        ].mean()
    )

    print(
        merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])[
            "grad/value_cosine_similarity_mean"
        ].mean()
    )
    print("Only best configuration per landscape for next print")
    print(
        best_config_df.groupby(["hp.agent_name", "eval_bins5"])[
            "grad/value_cosine_similarity_mean"
        ].mean()
    )

    # %% [markdown]
    # Keep only configurations that reach a minimal performance and do analysis

    # %%
    print(
        good_configs_df.groupby(["dataset", "hp.agent_name"])[
            ["grad/value_cosine_similarity_mean", "grad/actor_cosine_similarity_mean"]
        ].describe()
    )
    print(
        good_configs_df.groupby(["hp.agent_name"])[
            ["grad/value_cosine_similarity_std", "grad/actor_cosine_similarity_std"]
        ].describe()
    )

    print("-------------------")

    def cvar(x, alpha=0.05):
        var = np.quantile(x, alpha)
        return x[x <= var].mean()

    print(
        good_configs_df.groupby(["eval_bins5", "dataset", "hp.agent_name"])[
            ["grad/value_cosine_similarity_mean"]
        ].describe()
    )
    print(
        good_configs_df.groupby(["eval_bins5", "hp.agent_name"])[
            ["grad/value_cosine_similarity_std"]
        ].describe()
    )

    print(
        merged_training_with_iqm_df.groupby(["eval_bins5", "hp.agent_name"])[
            ["grad/value_cosine_similarity_mean"]
        ].apply(
            lambda x: (
                x[x <= np.quantile(x, 0.25)].mean(),
                x[x >= np.quantile(x, 0.75)].mean(),
            )
        )
    )
    # %% [markdown]
    # Look at metrics inside of batch

    # %%
    bad_configs_df = merged_training_with_iqm_df[
        (merged_training_with_iqm_df["iqm"] < 0.1)
    ]
    bad_configs_df.groupby(["hp.agent_name"])[
        "grad/value_cosine_similarity_cvar0.25"
    ].mean()
    # (merged_training_with_iqm_df.groupby(["hp.agent_name"])[merged_training_with_iqm_df.columns[merged_training_with_iqm_df.columns.str.contains(r"grad/.*quant\d+")]].mean())

    # %%
    fig, ax = plt.subplots()
    sns.displot(
        data=merged_training_with_iqm_df,
        x="grad/value_cosine_similarity_mean",
        hue="hp.agent_name",
    )
    plt.savefig(plot_dir / "grad_cosine_similarity_distributions.png")
    plt.close()

    # %%
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        "grad/value_cosine_similarity_quant0.05"
    ].describe()

# %% [markdown]
# ### Intra-Batch Goal Gradient Alignment

# %%
merged_training_with_iqm_df["id"] = merged_training_with_iqm_df.index
# %%
# df = merged_training_with_iqm_merged_with_iqm_df.copy()
try:
    merged_training_with_iqm_df = merged_training_with_iqm_df.drop(
        "target/held_out_val_batch_values", axis=1
    )
except:
    pass
quant_cols = [
    col
    for col in merged_training_with_iqm_df.columns
    if "grad/value_cosine_similarity_quant" in col
]
id_cols = [
    col
    for col in merged_training_with_iqm_df.columns
    if "quant" not in col and "target/held_out_val_batch_values_np" not in col
]


def keep_constants(g):
    out = {}
    for col in g.columns:
        if g[col].nunique() == 1:
            out[col] = g[col].iloc[0]
        else:
            print(col)
            out[col] = g[col].mean()
    return pd.Series(out)


marginalized_training_df = (
    merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset", "config_index"])[
        quant_cols
    ]
    .mean()
    .reset_index()
)
df_long = marginalized_training_df.melt(
    id_vars=["hp.agent_name", "dataset", "config_index"],
    value_vars=quant_cols,
    var_name="column",
    value_name="value",
)
# df_long["dataset"] = df_long["dataset"].astype("category")

df_long["quantile"] = df_long["column"].str.extract(r"quant([\d.]+)")[0].astype(float)

# Extract the base column name (everything before 'quant')
df_long["variable"] = (
    df_long["column"].str.replace(r"_?quant[\d.]+", "", regex=True).astype("category")
)

# Clean up
df_long = df_long.drop("column", axis=1).rename(columns={"hp.agent_name": "Algorithm"})
df_long["Algorithm"] = df_long["Algorithm"].str.upper().astype("category")

mem_usage = df_long.memory_usage(deep=True)
print(mem_usage.sort_values(ascending=False).sum() / 1024 / 1024 / 1024)
print(mem_usage.sort_values(ascending=False) / 1024 / 1024 / 1024)


# %%
def plot_cdf(df: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # print(df_long[["hp.agent_name", "quantile", "value"]].groupby(["hp.agent_name", "quantile"]).describe())
    ax = sns.lineplot(
        data=df, x="value", y="quantile", hue="Algorithm", errorbar=("pi", 95)
    )

    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.title("Inter-Goal Gradient Alignment")
    plt.xlabel("Gradient Cosine Similarity")
    plt.ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.savefig(plot_dir / f"gradient-alignment-cdf-{name}.png", dpi=1200)

    plt.close()


plot_cdf(
    df_long[df_long["variable"] == "grad/value_cosine_similarity"]
    .groupby(["Algorithm", "quantile"])["value"]
    .mean()
    .reset_index(),
    "antmaze-medium-all",
)
for name, group in df_long[
    df_long["variable"] == "grad/value_cosine_similarity"
].groupby(["dataset"]):
    plot_cdf(
        group.groupby(["Algorithm", "quantile"])["value"].mean().reset_index(), name
    )


# %%
def plot_swapped_cdf(df: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # print(df_long[["hp.agent_name", "quantile", "value"]].groupby(["hp.agent_name", "quantile"]).describe())
    ax = sns.lineplot(
        data=df, x="quantile", y="value", hue="Algorithm", errorbar=("pi", 95)
    )

    for line in ax.lines:
        # get data from first line of the plot
        newx = line.get_ydata()
        newy = line.get_xdata()

        # set new x- and y- data for the line
        line.set_xdata(newx)
        line.set_ydata(newy)
    from matplotlib.collections import PolyCollection

    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            verts = coll.get_paths()[0].vertices  # shape: (npoints, 2)
            verts[:, [0, 1]] = verts[:, [1, 0]]  # swap columns x<->y
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.title("Inter-Goal Gradient Alignment")
    plt.xlabel("Gradient Cosine Similarity")
    plt.ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "cdf_swapped" / f"gradient-alignment-cdf-{name}.png", dpi=1200
    )

    plt.close()


sns.set_theme(context="paper", style="whitegrid")
plot_swapped_cdf(df_long[df_long["variable"] == "grad/value_cosine_similarity"], "all")
for name, group in df_long[
    df_long["variable"] == "grad/value_cosine_similarity"
].groupby(["dataset"]):
    datasets = name[0].split(",")
    dataset = (
        datasets[0]
        if len(set(datasets)) == 1
        else f"{re.match(r'(.*)-(explore|navigate).*', datasets[0]).group(1)}-scheduled.png"
    )
    plot_swapped_cdf(group, dataset)

# %% [markdown]
# **Look at tail statistics**
#
# Simulate data and test

# %%

# %% [markdown]
# We can not properly do this due to aggregation. Maybe use CVar?

# %%
merged_training_with_iqm_df.groupby(["hp.agent_name"])[
    "grad/value_cosine_similarity_cvar0.25"
].std()

# %%
quantile_mean_df = (
    df_long[df_long["variable"] == "grad/value_cosine_similarity"]
    .groupby(["Algorithm", "quantile"])["value"]
    .mean()
    .reset_index()
)

# quantile_mean_df.groupby(["Algorithm"]).apply(lambda group: group[group["value"] < -0.2][["quantile", "value"]])


# %% [markdown]
# ### Magnitude Similarity
#
# Let's take a look at gradient magnitude similarity:

# %%
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        "grad/value_magnitude_similarity_mean"
    ].mean()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["hp.agent_name"])[
        "grad/value_magnitude_similarity_mean"
    ].mean()
)

print(
    merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])[
        "grad/value_magnitude_similarity_mean"
    ].mean()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["hp.agent_name", "eval_bins5"])[
        "grad/value_magnitude_similarity_mean"
    ].mean()
)


# %% [markdown]
# To validate that nothing strange is going on, take a look at the size of gradients in general:

# %%
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        "grad/value_scale_mean"
    ].mean()
)
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        "grad/actor_scale_mean"
    ].mean()
)

print("Only best configuration per landscape for next prints")
print(best_config_df.groupby(["hp.agent_name"])["grad/value_scale_mean"].mean())
print(best_config_df.groupby(["hp.agent_name"])["grad/actor_scale_mean"].mean())

print("Compare the loss sizes")
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        merged_training_with_iqm_df.columns[
            merged_training_with_iqm_df.columns.str.contains("loss")
            | merged_training_with_iqm_df.columns.str.contains("lam")
        ]
    ].mean(numeric_only=True)
)

# %% [markdown]
# ### Correlation
#
# Is there correlation between learning rate (and maybe discount factor) and cosine similarity?

# %%
print(
    good_configs_df.groupby(["eval_bins5", "hp.agent_name"])[
        ["grad/value_cosine_similarity_mean", "hp.lr"]
    ].corr()
)

# %%
col = "grad/value_cosine_similarity_std"
corr = (
    merged_training_with_iqm_df[
        merged_training_with_iqm_df.columns[
            ~merged_training_with_iqm_df.columns.str.contains("grad/|update/")
        ].tolist()
        + [col]
    ]
    .groupby(["hp.agent_name"])
    .apply(lambda g: g.corr(numeric_only=True)[col])
)
corr_sorted = corr.stack().rename("corr").reset_index()
corr_sorted = corr_sorted.reindex(
    corr_sorted["corr"].abs().sort_values(ascending=False).index
)
# %%
corr_sorted

# %% [markdown]
# Difference between the **two**agents (this will not work when adding CRL)

# %%
if "hiql" in corr.index and "qrl" in corr.index:
    corr_diff = corr.loc["hiql"] - corr.loc["qrl"]
    corr_diff_sorted = corr_diff.reindex(corr_diff.abs().sort_values(ascending=False).index)
    # only keep entries without "grad/" and "update/"
    display(corr_diff_sorted[~corr_diff_sorted.index.str.contains("grad/|update/")])
else:
    print(f"Skipping corr_diff: available agents in corr = {corr.index.tolist()}")

# %% [markdown]
# **We have found correlation for HIQL between regret/performance and gradient alignment.**
# Let us take a closer look.
# There was also a bit of correlation for QRL between eval_step and gradient alignment.

# %%
from scipy.stats import bootstrap, pearsonr

col1, col2 = (
    "grad/value_cosine_similarity_std",
    "mean_normalized_goal_distance_return_normalized_regret",
)
# col1, col2 = "grad/value_cosine_similarity_std", "eval_step"
for name, group in merged_training_with_iqm_df.groupby(["hp.agent_name"]):
    # filtered_df = group[[col1, col2]].reset_index()
    #
    # def corr_bootstrap(indices):
    #   return pearsonr(filtered_df.loc[indices, col1], filtered_df.loc[indices, col2]).statistic
    #
    # bootstrap((filtered_df.index, ), corr_bootstrap)
    numpy_col1 = group[col1].to_numpy()
    print(f"{name[0]}: {pearsonr(group[col1], group[col2])}")
    print(len(group))


# %% [markdown]
# Now lets do this sorted by training progress

# %%
col = "grad/value_cosine_similarity_std"
corr_progress = (
    merged_training_with_iqm_df[
        merged_training_with_iqm_df.columns[
            ~merged_training_with_iqm_df.columns.str.contains("grad/|update/")
        ].tolist()
        + [col]
    ]
    .groupby(["hp.agent_name", "eval_bins5"])
    .apply(lambda g: g.corr(numeric_only=True)[col])
)
corr_progress_sorted = corr_progress.stack().rename("corr").reset_index()
corr_progress_sorted = corr_progress_sorted.reindex(
    corr_progress_sorted["corr"].abs().sort_values(ascending=False).index
)
corr_progress_sorted[corr_progress_sorted["eval_bins5"] == 0]

# %% [markdown]
# ## Parameter Updates
#
# We use Adam for optimization and therefore parameter updates might look a bit different directly compared to the gradients. Let's analyze the parameter updates.

# %%
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        ["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]
    ].describe()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["hp.agent_name"])[
        ["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]
    ].mean()
)

print("---------------------------------------------")

print(
    merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])[
        "update/value_cosine_similarity_mean"
    ].mean()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["hp.agent_name", "eval_bins5"])[
        "update/value_cosine_similarity_mean"
    ].mean()
)
# %% [markdown]
# Differentiate between experiments. **For now only exploration schedule as this is antmaze only**.

# %%
print(
    merged_training_with_iqm_df.groupby(["exploration_schedule", "hp.agent_name"])[
        ["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]
    ].describe()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["exploration_schedule", "hp.agent_name"])[
        ["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]
    ].mean()
)

print(
    merged_training_with_iqm_df.groupby(
        ["exploration_schedule", "eval_bins5", "hp.agent_name"]
    )["update/value_cosine_similarity_mean"].mean()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["exploration_schedule", "eval_bins5", "hp.agent_name"])[
        "update/value_cosine_similarity_mean"
    ].mean()
)


# %% [markdown]
# Do this analysis for standard deviation:
# %%
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        ["update/value_cosine_similarity_std", "update/actor_cosine_similarity_std"]
    ].describe()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["exploration_schedule", "hp.agent_name"])[
        ["update/value_cosine_similarity_std", "update/actor_cosine_similarity_std"]
    ].mean()
)

print(
    merged_training_with_iqm_df.groupby(
        ["exploration_schedule", "eval_bins5", "hp.agent_name"]
    )["update/value_cosine_similarity_std"].mean()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["exploration_schedule", "eval_bins5", "hp.agent_name"])[
        "update/value_cosine_similarity_std"
    ].mean()
)


# %% [markdown]
# Let's take a look at gradient magnitude similarity:

# %%
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        "update/value_magnitude_similarity_mean"
    ].mean()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["hp.agent_name"])[
        "update/value_magnitude_similarity_mean"
    ].mean()
)

print(
    merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])[
        "update/value_magnitude_similarity_mean"
    ].mean()
)
print("Only best configuration per landscape for next print")
print(
    best_config_df.groupby(["hp.agent_name", "eval_bins5"])[
        "update/value_magnitude_similarity_mean"
    ].mean()
)


# %% [markdown]
# To validate that nothing strange is going on, take a look at the size of gradients in general:

# %%
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        "update/value_scale_mean"
    ].mean()
)
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        "update/actor_scale_mean"
    ].mean()
)

print("Only best configuration per landscape for next prints")
print(best_config_df.groupby(["hp.agent_name"])["update/value_scale_mean"].mean())
print(best_config_df.groupby(["hp.agent_name"])["update/actor_scale_mean"].mean())

# %% [markdown]
# Is there correlation between learning rate (and maybe discount factor) and cosine similarity?

# %%
print(
    merged_training_with_iqm_df.groupby(["hp.agent_name"])[
        ["update/value_cosine_similarity_std", "hp.lr"]
    ].corr()
)

# %%
corr = merged_training_with_iqm_df.groupby(["hp.agent_name"]).apply(
    lambda g: g.corr(numeric_only=True)["update/value_cosine_similarity_mean"]
)
corr_sorted = corr.stack().rename("corr").reset_index()
corr_sorted = corr_sorted.reindex(
    corr_sorted["corr"].abs().sort_values(ascending=False).index
)
corr_sorted

# %% [markdown]
# Difference between the **two**agents (this will not work when adding CRL)

# %%
if "hiql" in corr.index and "qrl" in corr.index:
    corr_diff = corr.loc["hiql"] - corr.loc["qrl"]
    corr_diff_sorted = corr_diff.reindex(corr_diff.abs().sort_values(ascending=False).index)
    display(corr_diff_sorted)
else:
    print(f"Skipping corr_diff: available agents in corr = {corr.index.tolist()}")

# %% [markdown]
# Now lets do this sorted by training progress

# %%
corr_progress = merged_training_with_iqm_df.groupby(
    ["hp.agent_name", "eval_bins5"]
).apply(lambda g: g.corr(numeric_only=True)["grad/value_cosine_similarity_mean"])
corr_progress_sorted = corr_progress.stack().rename("corr").reset_index()
corr_progress_sorted = corr_progress_sorted.reindex(
    corr_progress_sorted["corr"].abs().sort_values(ascending=False).index
)
corr_progress_sorted[corr_progress_sorted["eval_bins5"] == 0]


# %% [markdown]
# # Combine Landscape Plots
#
# ## Seaborn KDE-like plot

# %%
merged_results_df.groupby(["hp.agent_name", "dataset"])

# %%
from adjustText import adjust_text
from src.gcrl_landscapes.util.eval import fit_model
from src.gcrl_landscapes.configurations import get_bounds, sobol_codomain_to_hp
from gcrl_landscapes.plots.triple_gp import create_contour_plot
from gcrl_landscapes.evaluation.common import map_labels

grid_length = 100


def mobility_plot(
    df, agent_name, title, by_col: str = "phase_num", performance_threshold=0.95
):
    clipped_merged_results_df = df.copy()
    clipped_merged_results_df["mean_normalized_goal_distance_return"] = (
        clipped_merged_results_df["mean_normalized_goal_distance_return"].clip(0, 1)
    )
    data_temp = clipped_merged_results_df
    # for phase_num ...
    point_dfs = []
    for name, group in data_temp.groupby([by_col]):
        group_copy = group.copy().reset_index()
        model = fit_model(
            group_copy, "mean_normalized_goal_distance_return", ["hp.lr", "hp.alpha"]
        )
        model.fit()
        create_contour_plot(
            model,
            x_dim=0,
            y_dim=1,
            z_dim="mean_normalized_goal_distance_return",
            bounds=[0, 1],
            filename="test_contour.png",
            dim_label_mapping=map_labels,
            agent_name=agent_name,
            z_transform=lambda x, _: x,
            discrete_levels=None,
            last_phase_best_config=None,
        )
        x_lower, x_upper, x_log = get_bounds(
            model.hp_names[0].removeprefix("hp."), agent_name
        )
        y_lower, y_upper, y_log = get_bounds(
            model.hp_names[1].removeprefix("hp."), agent_name
        )
        x, y = np.linspace(0, 1, grid_length), np.linspace(0, 1, grid_length)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel()]).transpose()
        Z = model.get_middle(points).clip(0, 1)
        Z_normalized = Z / Z.max()
        Z_selected = (Z_normalized > 0.9).squeeze()
        points_x, points_y = (
            sobol_codomain_to_hp(points[:, 0], x_lower, x_upper, x_log),
            sobol_codomain_to_hp(points[:, 1], y_lower, y_upper, y_log),
        )
        print(np.vstack([points_x, points_y]))

        # # TODO: decide
        # group_copy["hp.lr"] = hp_to_sobol_codomain(group_copy["hp.lr"], x_lower, x_upper, x_log)
        # group_copy["hp.discount"] = hp_to_sobol_codomain(group_copy["hp.discount"], y_lower, y_upper, y_log)
        # marginalized_group_copy = group_copy.groupby(["hp.lr", "hp.discount"])["mean_normalized_goal_distance_return"].apply(lambda values: trim_mean(values, proportiontocut=0.25)).reset_index()
        # marginalized_group_copy["goal_distance_return_eps"] = marginalized_group_copy["mean_normalized_goal_distance_return"] / marginalized_group_copy["mean_normalized_goal_distance_return"].max()

        points_prediction_df = pd.DataFrame(
            {
                "hp.lr": points_x,
                "hp.alpha": points_y,
                "mean_normalized_goal_distance_return": Z_normalized.squeeze(),
            }
        )
        # points_prediction_df = marginalized_group_copy[marginalized_group_copy["goal_distance_return_eps"] > 0.8]
        points_prediction_df[by_col] = name[0]
        point_dfs.append(points_prediction_df)
    point_df = pd.concat(point_dfs)
    print(point_df.describe())

    print(len(point_df))
    # sns.scatterplot(data=point_df[point_df["mean_normalized_goal_distance_return"] > 0.9], x="hp.lr", y="hp.discount", hue="phase_num")
    x_lower, x_upper, x_log = get_bounds("lr", agent_name)
    y_lower, y_upper, y_log = get_bounds("alpha", agent_name)
    # ax = sns.kdeplot(data=point_df[point_df["mean_normalized_goal_distance_return"] > 0.95], x="hp.lr", y="hp.discount", hue="phase_num", log_scale=(x_log, y_log), levels=10, bw_adjust=1, fill=True, alpha=0.4, palette="rocket")
    df = point_df[
        point_df["mean_normalized_goal_distance_return"] > performance_threshold
    ]

    # ax = sns.kdeplot(
    #     data=df,
    #     x="hp.lr", y="hp.discount",
    #     hue="phase_num",
    #     log_scale=(x_log, y_log),
    #     fill=True,
    #     levels=2,
    #     thresh=0.15,
    #     bw_adjust=1.0,
    #     alpha=0.12,
    #     palette="magma",
    #     linewidth=0,
    # )
    #
    # sns.kdeplot(
    #     data=df,
    #     x="hp.lr", y="hp.discount",
    #     hue="phase_num",
    #     log_scale=(x_log, y_log),
    #     fill=False,
    #     levels=[0.5, 0.8],
    #     thresh=0.15,
    #     bw_adjust=1.0,
    #     alpha=0.9,
    #     palette="magma",
    #     linewidths=2.0,
    #     ax=ax,
    # )
    fig, ax = plt.subplots(figsize=(6, 4))

    palette = sns.color_palette("viridis", n_colors=df[by_col].nunique())

    sns.set_context(context="paper", font_scale=1.75)

    # plt.rcParams.update({
    #     "font.size": 20,          # base font size
    #     "axes.titlesize": 16,
    #     "axes.labelsize": 20,
    #     "xtick.labelsize": 20,
    #     "ytick.labelsize": 20,
    #     "legend.fontsize": 12,
    #     "figure.titlesize": 18,
    # })

    texts = []
    centroids = []
    for i, phase in enumerate(sorted(df[by_col].unique())):
        phase_df = df[df[by_col] == phase]
        color = palette[i]

        sns.kdeplot(
            data=phase_df,
            x="hp.lr",
            y="hp.alpha",
            log_scale=(x_log, y_log),
            fill=True,
            levels=4,
            thresh=0.05,
            bw_adjust=0.7,
            alpha=0.12,
            color=color,
            linewidth=0,
            ax=ax,
        )

        sns.kdeplot(
            data=phase_df,
            x="hp.lr",
            y="hp.alpha",
            log_scale=(x_log, y_log),
            fill=False,
            levels=[0.7],
            thresh=0.05,
            bw_adjust=0.7,
            alpha=0.9,
            color=color,
            linewidths=3.5,
            ax=ax,
            label=f"Phase {phase}",
        )

        centroid_x = phase_df["hp.lr"].median()
        centroid_y = phase_df["hp.alpha"].median()
        ax.scatter(
            centroid_x,
            centroid_y,
            s=750 if by_col == "phase_num" else 1500,
            c=[color],
            edgecolors="white",
            linewidths=1,
            zorder=100,
            marker="o",
            alpha=0.75,
        )
        texts.append(
            ax.annotate(
                str(phase),
                xy=(centroid_x, centroid_y),
                xytext=(centroid_x, centroid_y),
                fontsize=20,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                zorder=101,
                alpha=1,
            )
        )
        # texts.append(ax.text(centroid_x, centroid_y, str(phase), fontsize=20, fontweight='bold',
        # ha='center', va='center', color='black', zorder=101, alpha=1))
        centroids.append((centroid_x, centroid_y))
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Alpha")
    # ax.set_title("Evolution of Optimal Hyperparameter Regions Across Training Phases",
    #              fontsize=15, fontweight='bold', pad=20)

    # ax.legend(title="Training Phase", title_fontsize=12, fontsize=11,
    #           loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True,
    #           fancybox=True, shadow=True)
    # Replace your current legend section with this:

    # Create the legend
    # legend = plt.legend(
    #     handles = [1, 2, 3, 4],
    #     title="Training Phase",
    #     title_fontsize=14,
    #     fontsize=12,
    #     loc="center left",
    #     bbox_to_anchor=(1.05, 0.5),  # Position to the right of the plot
    #     frameon=True,
    #     fancybox=True,
    #     shadow=True,
    #     borderpad=1.2,  # Padding inside legend box
    #     labelspacing=1.2,  # Space between legend entries
    #     handlelength=2.5,  # Length of the legend lines
    #     handleheight=1.5   # Height of the legend lines
    # )

    # ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    # ax.set_facecolor('#fafafa')
    # for i in range(len(centroids)-1):
    #     ax.annotate('', xy=centroids[i+1], xytext=centroids[i],
    #                 arrowprops=dict(arrowstyle='->', lw=2.5, color='black', alpha=0.6,
    #                                connectionstyle="arc3,rad=0.1"))
    #

    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="phase")
    ax.grid(True, alpha=0.15)
    print(ax.collections[0].levels)
    plt.xlim(x_lower, x_upper)
    plt.ylim(y_lower, y_upper)
    if x_log:
        ax.set_xscale("log", base=10)
    if y_log:
        ax.set_yscale("log", base=10)
    x, y = zip(*centroids)
    texts, patches = adjust_text(
        texts,
        avoid_self=False,
        pull_threshold=0.000001,
        pull_force=0.1,
        force_static=0.001,
        force_explode=0.5,
        arrowprops=dict(arrowstyle="-", color="white"),
    )
    for item in texts:
        item.set_zorder(102)
    for item in patches:
        item.set_zorder(101)
    plt.tight_layout()

    from pathlib import Path

    path = plot_dir / "mobility" / f"{title}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=1200)
    plt.close()


# %%

merged_results_df["dataset_condensed"] = (
    merged_results_df["dataset"]
    .apply(lambda x: x.split(",")[0] if len(set(x.split(","))) == 1 else x)
    .astype("category")
)
for name, group in merged_results_df.groupby(["hp.agent_name", "dataset_condensed"]):
    agent = name[0]
    datasets = name[1]

    print(f"mobility-{agent}-{datasets}")
    performance_threshold = 95
    mobility_plot(
        group,
        agent,
        f"{performance_threshold}/mobility-{agent}-{datasets}",
        performance_threshold=performance_threshold / 100,
    )
    performance_threshold = 90
    mobility_plot(
        group,
        agent,
        f"{performance_threshold}/mobility-{agent}-{datasets}",
        performance_threshold=performance_threshold / 100,
    )


# %% [markdown]
# **Now do it across dataset qualities for the last phase**

# %%
merged_results_df_constant_last_phase = merged_results_df[
    (merged_results_df["constant_dataset"]) & (merged_results_df["phase_num"] == 4)
].copy()
merged_results_df_constant_last_phase["env"] = (
    merged_results_df_constant_last_phase["dataset_condensed"]
    .apply(lambda x: re.match(r"(.*)-(explore|navigate).*", x).group(1))
    .astype("category")
)
merged_results_df_constant_last_phase["Exploration Ratio"] = (
    merged_results_df_constant_last_phase["dataset"]
    .apply(lambda x: datasets_to_exploration_schedule(x).split(",")[0])
    .astype("category")
)

# %%
for name, group in merged_results_df_constant_last_phase.groupby(
    ["hp.agent_name", "env"]
):
    agent = name[0]
    envs = name[1]
    if group["Exploration Ratio"].nunique() == 1:
        print(f"Skipping: mobility-last-phase-{agent}-{envs}")
        continue

    print(f"mobility-last-phase-{agent}-{envs}")
    performance_threshold = 95
    mobility_plot(
        group,
        agent,
        f"across-dataquality/{performance_threshold}/mobility-last-phase-{agent}-{envs}",
        by_col="Exploration Ratio",
        performance_threshold=performance_threshold / 100,
    )
    performance_threshold = 90
    mobility_plot(
        group,
        agent,
        f"across-dataquality/{performance_threshold}/mobility-last-phase-{agent}-{envs}",
        by_col="Exploration Ratio",
        performance_threshold=performance_threshold / 100,
    )

# %% [markdown]
# ## Table for across phase/quality

# %%
from src.gcrl_landscapes.configurations import get_bounds, sobol_codomain_to_hp
import warnings
from typing import Callable


def optimum_share_table(
    df,
    agent_name,
    by_col: str = "phase_num",
    performance_threshold=0.95,
    sorter: Callable = lambda x: sorted(x),
    grid_length=100,
):
    clipped_merged_results_df = df.copy()
    clipped_merged_results_df["mean_normalized_goal_distance_return"] = (
        clipped_merged_results_df["mean_normalized_goal_distance_return"].clip(0, 1)
    )
    data_temp = clipped_merged_results_df

    # This is an example model to correctly set grid
    model = fit_model(df, "mean_normalized_goal_distance_return", ["hp.lr", "hp.alpha"])
    hpname_x = model.hp_names[0].removeprefix("hp.")
    hpname_y = model.hp_names[1].removeprefix("hp.")
    model.fit()
    x_lower, x_upper, x_log = get_bounds(hpname_x, agent_name)
    y_lower, y_upper, y_log = get_bounds(hpname_y, agent_name)
    x, y = np.linspace(0, 1, grid_length), np.linspace(0, 1, grid_length)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).transpose()

    def get_optimal_point_selector(group_df):
        group_copy = group_df.copy().reset_index()
        model = fit_model(
            group_copy, "mean_normalized_goal_distance_return", ["hp.lr", "hp.alpha"]
        )
        model.fit()
        assert (
            model.hp_names[0].removeprefix("hp.") == hpname_x
            and model.hp_names[1].removeprefix("hp.") == hpname_y
        )
        Z = model.get_middle(points).clip(0, 1)
        Z_normalized = Z / Z.max()
        Z_selected = (Z_normalized > performance_threshold).squeeze()
        return Z_selected

    optimal_points_per_col = data_temp.groupby(by_col).apply(get_optimal_point_selector)
    col_sorted = sorter(optimal_points_per_col.index.tolist())
    transition_point_share = {
        f"{col1}->{col2}": np.sum(
            optimal_points_per_col.loc[col1] & optimal_points_per_col.loc[col2]
        )
        / np.sum(optimal_points_per_col.loc[col1] | optimal_points_per_col.loc[col2])
        for col1, col2 in zip(col_sorted[:-1], col_sorted[1:])
    }
    return pd.Series(transition_point_share, name="transition")


# %% [markdown]
# ### Across Phases

# %%
for name, group in merged_results_df.groupby(["hp.agent_name", "dataset"]):
    print(name)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    phase_optimum_share_df = merged_results_df.groupby(
        ["hp.agent_name", "dataset"]
    ).apply(
        lambda group: optimum_share_table(
            group,
            group["hp.agent_name"].iloc[0],
            by_col="phase_num",
            performance_threshold=0.90,
            grid_length=100,
        )
    )
    if isinstance(phase_optimum_share_df, pd.Series):
        phase_optimum_share_df = phase_optimum_share_df.unstack()
# for name, group in merged_results_df.groupby(["hp.agent_name", "dataset"]):
#   agent_name = name[0]
#   datasets = name[1]
#   optimum_share_table(group, agent_name, f"across-phase-{agent_name}-{datasets}", by_col="phase_num", performance_threshold=0.90, grid_length=100)

# %%
phase_optimum_share_df.columns
phase_optimum_share_df_long = phase_optimum_share_df.reset_index().melt(
    id_vars=["hp.agent_name", "dataset"],
    var_name="transition",
    value_name="Jaccard Index of Optimum",
)
phase_optimum_share_df_long["constant_dataset"] = phase_optimum_share_df_long[
    "dataset"
].apply(lambda x: len(set(x.split(","))) == 1)
phase_optimum_share_df_long["hp.agent_name"] = (
    phase_optimum_share_df_long["hp.agent_name"].str.upper().astype("category")
)

# %%
fig, ax = plt.subplots(figsize=(3.5, 2.5))
sns.lineplot(
    data=phase_optimum_share_df_long,
    x="transition",
    y="Jaccard Index of Optimum",
    hue="hp.agent_name",
    errorbar=("ci", 95),
)
plt.ylim(0, 1)
plt.xlabel("Phase Transition")
plt.ylabel(r"Overlap of Optimum")
plt.legend(title="Algorithm")
plt.tight_layout()
plt.savefig(plot_dir / "optimum-overlap-phases.png", dpi=1200)
plt.close()
# %% [markdown]
# Now lets get tabular data


# %%
def dataset_to_name(dataset):
    datasets = dataset.split(",")
    if datasets[0].startswith("antmaze-medium") or datasets[0].startswith(
        "antmaze-large"
    ):
        exploration_shares = datasets_to_exploration_schedule(dataset).split(",")
        if len(set(exploration_shares)) == 1:
            return f"{re.match(r'(antmaze-medium|antmaze-large)', dataset).group(1)}-{exploration_shares[0]}\%explore"
        else:
            return f"{re.match(r'(antmaze-medium|antmaze-large)', dataset).group(1)}-scheduled"
    return dataset if len(set(datasets)) > 1 else datasets[0]


latex_phase_optimum_df = (
    phase_optimum_share_df.swaplevel(0, 1)
    .rename(index=lambda idx: dataset_to_name(idx), level="dataset")
    .rename(index=lambda idx: idx.upper(), level="hp.agent_name")
    .rename_axis(index={"hp.agent_name": "Algorithm", "dataset": "Setting"})
    .sort_index()
)

with open(tables_dir / "optimum-overlap-phases.tex", "w") as f:
    f.writelines(
        latex_phase_optimum_df.to_latex(
            index=True,
            caption="Optimum Overlap across Phase Transitions",
            label="tab:optimum-overlap-phases",
            float_format="%.3f",
            bold_rows=False,
            longtable=False,
        )
    )

# %% [markdown]
# ### Across Data Quality

# %%
merged_results_df_constant_last_phase["Exploration Ratio"] = (
    merged_results_df_constant_last_phase["Exploration Ratio"].astype("int")
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    quality_optimum_share_df = (
        merged_results_df_constant_last_phase[
            (merged_results_df_constant_last_phase["env"] == "antmaze-medium")
            | (merged_results_df_constant_last_phase["env"] == "antmaze-large")
        ]
        .groupby(["hp.agent_name", "env"])
        .apply(
            lambda group: optimum_share_table(
                group.reset_index(),
                group["hp.agent_name"].reset_index().iloc[0],
                by_col="Exploration Ratio",
                sorter=lambda x: list(reversed(sorted(x))),
                performance_threshold=0.90,
                grid_length=100,
            )
        )
    )
# pandas >= 2.x may return a long Series with MultiIndex instead of a wide DataFrame
if isinstance(quality_optimum_share_df, pd.Series):
    quality_optimum_share_df = quality_optimum_share_df.unstack()
# %%
quality_optimum_share_df_long = quality_optimum_share_df.reset_index().melt(
    id_vars=["hp.agent_name", "env"],
    var_name="transition",
    value_name="Jaccard Index of Optimum",
)
quality_optimum_share_df_long["hp.agent_name"] = (
    quality_optimum_share_df_long["hp.agent_name"].str.upper().astype("category")
)

# %%
fig, ax = plt.subplots(figsize=(3.5, 2.5))
sns.lineplot(
    data=quality_optimum_share_df_long,
    x="transition",
    y="Jaccard Index of Optimum",
    hue="hp.agent_name",
    errorbar=("ci", 95),
)
plt.ylim(0, 1)
plt.xlabel("Exploration Ratio")
plt.ylabel(r"Overlap of Optimum")
plt.legend(title="Algorithm")
plt.tight_layout()
plt.savefig(plot_dir / "optimum-overlap-quality.png", dpi=1200)
plt.close()

# %%
latex_quality_optimum_df = (
    quality_optimum_share_df.swaplevel(0, 1)
    .rename(index=lambda idx: idx.upper(), level="hp.agent_name")
    .rename_axis(index={"hp.agent_name": "Algorithm", "dataset": "Setting"})
    .sort_index()
)

with open(tables_dir / "optimum-overlap-quality.tex", "w") as f:
    f.writelines(
        latex_quality_optimum_df.to_latex(
            index=True,
            caption="Optimum Overlap across Exploration Ratios",
            label="tab:optimum-overlap-quality",
            float_format="%.3f",
            bold_rows=False,
            longtable=False,
        )
    )

# %% [markdown]
# ### Scheduled vs non-Scheduled

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


# %% [markdown]
# ## Optimum Movement line plot

# %%
data_temp = merged_results_df[
    (
        merged_results_df["dataset"]
        == "antmaze-medium-explore-v0,antmaze-medium-explore80navigate-v0,antmaze-medium-explore40navigate-v0,antmaze-medium-navigate-v0"
    )
    & (merged_results_df["hps"] == frozenset(set(["lr", "alpha"])))
]
optima = []
for name, group in data_temp.groupby(["hp.agent_name", "phase_num"]):
    marginalized_group = (
        group.groupby(["hp.lr", "hp.alpha"])["mean_normalized_goal_distance_return"]
        .apply(lambda values: trim_mean(values, proportiontocut=0.25))
        .reset_index()
    )
    t = marginalized_group[
        marginalized_group["mean_normalized_goal_distance_return"]
        == marginalized_group["mean_normalized_goal_distance_return"].max()
    ].iloc[0][["hp.lr", "hp.alpha"]]
    t["Algorithm"] = name[0]
    optima.append(t)
agent_name = ""
x_lower, x_upper, x_log = get_bounds("lr", agent_name)
y_lower, y_upper, y_log = get_bounds("alpha", agent_name)
fig, ax = plt.subplots()
sns.lineplot(
    data=pd.DataFrame(optima),
    x="hp.lr",
    y="hp.alpha",
    hue="Algorithm",
    errorbar=None,
    marker="o",
)
ax.set(xscale="log")
plt.xlim(x_lower, x_upper)
plt.ylim(y_lower, y_upper)
plt.savefig(plot_dir / "test.png")
plt.close()
