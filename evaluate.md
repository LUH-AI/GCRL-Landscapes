```python
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
                **{col: agg_constant_col for col in df.columns[df.columns.str.startswith("hp.")]},
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
```

```python
merged_training_df: pd.DataFrame = merged_training_df
merged_results_df: pd.DataFrame = merged_results_df
merged_marginalized_results_df: pd.DataFrame = marginalize_seeds(merged_results_df)
merged_marginalized_results_df["mean_normalized_goal_distance_return_eps_optimality"] = eps_optimality(merged_marginalized_results_df, "mean_normalized_goal_distance_return")
merged_training_df.columns.tolist()
```

Let's convert `eval_step` to a percentage of training and bin that, so that we can compare different stages of training.
Also extract exploration schedule from dataset names

```python
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
        raise ValueError(f"Unknown dataset: {dataset}")
  datasets = dataset_str.split(",")
  return ",".join([str(dataset_to_exploration_percentage(dataset)) for dataset in datasets])
print(datasets_to_exploration_schedule("antmaze-medium-explore-v0,antmaze-medium-explore90navigate-v0"))
```

```python
import re
# TODO: generalize this to dfs containing multiple hyperparameter combinations. Does this even have an influence?
merged_training_df["eval_percent"] = round(merged_training_df["eval_step"] / merged_training_df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform("max") * 100)
merged_training_df["eval_bins10"] = merged_training_df.groupby(["hp.agent_name", "dataset"])["eval_percent"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
merged_training_df["eval_bins5"] = merged_training_df.groupby(["hp.agent_name", "dataset"])["eval_percent"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
merged_training_df["exploration_schedule"] = merged_training_df["dataset"].apply(datasets_to_exploration_schedule)
```

Get dataframe with only best performing model per experiment.
We will have to merge training results so that we know the performance (iqm).

```python
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print(merged_results_df.columns)
print(merged_training_df.columns.tolist())
```


```python
iqm_df = merged_results_df[merged_results_df["eval_step"] == merged_results_df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform("max")]
iqm_df = iqm_df.groupby(["hp.agent_name", "dataset", "config_index"])["mean_normalized_goal_distance_return"].apply(lambda x: trim_mean(x, proportiontocut=0.25)).reset_index(name="iqm")
# save iqm in merged_training_df for later
merged_training_with_iqm_df = merged_training_df.merge(iqm_df[["hp.agent_name", "dataset", "config_index", "iqm"]], how="inner", on=["hp.agent_name", "dataset", "config_index"])
iqm_df = iqm_df[iqm_df["iqm"] == iqm_df.groupby(["hp.agent_name", "dataset"])["iqm"].transform("max")]
best_config_df = merged_training_with_iqm_df.merge(iqm_df[["hp.agent_name", "dataset", "config_index"]], how="inner", on=["hp.agent_name", "dataset", "config_index"])
```

Let's evaluate the rank of the model:  

```python
print(merged_training_with_iqm_df.groupby("hp.agent_name")["feature/embedding_rank"].describe())
print(merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset", "eval_bins5"])["feature/embedding_rank"].describe())
```

Capture this in a boxplot:

```python
import seaborn as sns
sns.set_theme(context="paper", style="whitegrid")
import matplotlib.pyplot as plt
merged_training_with_iqm_df["name_dataset_combination"] = merged_training_with_iqm_df["hp.agent_name"] + " - " + merged_training_with_iqm_df["dataset"]
end_of_training_df = merged_training_with_iqm_df[merged_training_with_iqm_df["eval_step"] == merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform("max")]
fig, ax = plt.subplots()
sns.boxplot(data=end_of_training_df, x="hp.agent_name", y="feature/embedding_rank")
plt.savefig("rank_boxplot_agent.png")
for dataset in merged_training_with_iqm_df["dataset"].unique():
  fig, ax = plt.subplots()
  sns.boxplot(data=end_of_training_df[end_of_training_df["dataset"] == dataset], x="hp.agent_name", y="feature/embedding_rank")
  print(dataset)
  print(merged_training_with_iqm_df[merged_training_with_iqm_df["dataset"] == dataset].groupby("hp.agent_name")["iqm"].max())
  plt.savefig(f"rank_boxplot_agent_{dataset}.png")
```

Now for only best config per experiment:

```python
print(best_config_df.groupby("hp.agent_name")["feature/embedding_rank"].describe())
print(best_config_df.groupby(["hp.agent_name", "exploration_schedule", "eval_bins5"])["feature/embedding_rank"].describe())
```

# Gradient Interference

## Gradients

Let's look at gradient interference data. We start with **cosine similarity**:

```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])[["grad/value_cosine_similarity_mean", "grad/actor_cosine_similarity_mean"]].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["exploration_schedule", "hp.agent_name"])[["grad/value_cosine_similarity_mean", "grad/actor_cosine_similarity_mean"]].mean())

print(merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])["grad/value_cosine_similarity_mean"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["hp.agent_name", "eval_bins5"])["grad/value_cosine_similarity_mean"].mean())
```

Keep only configurations that reach a minimal performance and do analysis

```python
good_configs_df = merged_training_with_iqm_df[(merged_training_with_iqm_df["iqm"] > -1.1)]
print(good_configs_df.groupby(["dataset", "hp.agent_name"])[["grad/value_cosine_similarity_mean", "grad/actor_cosine_similarity_mean"]].describe())
print(good_configs_df.groupby(["hp.agent_name"])[["grad/value_cosine_similarity_std", "grad/actor_cosine_similarity_std"]].describe())

print("-------------------")

def cvar(x, alpha=0.05):
  var = np.quantile(x, alpha)
  return x[x <= var].mean()

print(good_configs_df.groupby(["eval_bins5", "dataset", "hp.agent_name"])[["grad/value_cosine_similarity_mean"]].describe())
print(good_configs_df.groupby(["eval_bins5", "hp.agent_name"])[["grad/value_cosine_similarity_std"]].describe())

print(merged_training_with_iqm_df.groupby(["eval_bins5", "hp.agent_name"])[["grad/value_cosine_similarity_mean"]].apply(lambda x: (x[x <= np.quantile(x, 0.25)].mean(), x[x >= np.quantile(x, 0.75)].mean())))
```
Look at metrics inside of batch

```python
bad_configs_df = merged_training_with_iqm_df[(merged_training_with_iqm_df["iqm"] < 0.2)]
bad_configs_df.groupby(["hp.agent_name"])["grad/value_cosine_similarity_quant0.25"].mean()
#(merged_training_with_iqm_df.groupby(["hp.agent_name"])[merged_training_with_iqm_df.columns[merged_training_with_iqm_df.columns.str.contains(r"grad/.*quant\d+")]].mean())

```

```python
fig, ax = plt.subplots()
sns.displot(data=merged_training_with_iqm_df, x="grad/value_cosine_similarity_mean", hue="hp.agent_name")
plt.savefig("grad_cosine_similarity_distributions.png")
```


```python
merged_training_with_iqm_df.groupby(["hp.agent_name"])["grad/value_cosine_similarity_quant0.05"].describe()
```

### Intra-Batch Goal Gradient Alignment

```python
fig, ax = plt.subplots()
merged_training_with_iqm_df["id"] = merged_training_with_iqm_df.index
# df = merged_training_with_iqm_merged_with_iqm_df.copy()
float_cols = merged_training_with_iqm_df.select_dtypes(include="float64").columns
# for col in float_cols:
#   print(col)
#   merged_training_with_iqm_df[col] = merged_training_with_iqm_df[col].astype("float16")
quant_cols = [col for col in merged_training_with_iqm_df.columns if 'grad/value_cosine_similarity_quant' in col]
id_cols = [col for col in merged_training_with_iqm_df.columns if 'quant' not in col]

df_long = merged_training_with_iqm_df.melt(
    id_vars=id_cols,
    value_vars=quant_cols,
    var_name='column',
    value_name='value'
)
df_long['quantile'] = df_long['column'].str.extract(r'quant([\d.]+)')[0].astype(float)

# Extract the base column name (everything before 'quant')
df_long['variable'] = df_long['column'].str.replace(r'_?quant[\d.]+', '', regex=True)

# Clean up
df_long = df_long.drop('column', axis=1).rename(columns={"hp.agent_name": "Algorithm"})
df_long["Algorithm"] = df_long["Algorithm"].str.upper()
```

```python
def plot_cdf(df: pd.DataFrame, name: str) -> None:
  import matplotlib.pyplot as plt
  import seaborn as sns
  fig, ax = plt.subplots(figsize=(3.5, 2.5))
  # print(df_long[["hp.agent_name", "quantile", "value"]].groupby(["hp.agent_name", "quantile"]).describe())
  ax = sns.lineplot(data=df, x="value", y="quantile", hue="Algorithm", errorbar=None)

  plt.xlim(-1, 1)
  plt.ylim(0, 1)
  plt.title("Inter-Goal Gradient Alignment")
  plt.xlabel("Gradient Cosine Similarity")
  plt.ylabel("Cumulative Probability")
  plt.tight_layout()
  plt.savefig(f"plots/gradient-alignment-cdf-{name}.png", dpi=1200)

  plt.close()


plot_cdf(df_long[df_long["variable"] == "grad/value_cosine_similarity"].groupby(["Algorithm", "quantile"])["value"].mean().reset_index(), "antmaze-medium-all")
for name, group in df_long[df_long["variable"] == "grad/value_cosine_similarity"].groupby(["dataset"]):
  plot_cdf(group.groupby(["Algorithm", "quantile"])["value"].mean().reset_index(), name)

```

```python
def plot_swapped_cdf(df: pd.DataFrame, name: str) -> None:
  fig, ax = plt.subplots(figsize=(3.5, 2.5))
  # print(df_long[["hp.agent_name", "quantile", "value"]].groupby(["hp.agent_name", "quantile"]).describe())
  ax = sns.lineplot(data=df, x="quantile", y="value", hue="Algorithm", errorbar="ci")
  print(df.groupby(["eval_bins5"]).size())
  # print(df.groupby(["quantile"]).size())

  plt.ylim(-1, 1)
  plt.xlim(0, 1)
  plt.title("Inter-Goal Gradient Alignment")
  plt.ylabel("Gradient Cosine Similarity")
  plt.xlabel("Cumulative Probability")
  plt.tight_layout()
  plt.savefig(f"plots/cdf_swapped/gradient-alignment-cdf-{name}.png", dpi=1200)

  plt.close()


plot_swapped_cdf(df_long[df_long["variable"] == "grad/value_cosine_similarity"], "antmaze-medium-all")
for name, group in df_long[df_long["variable"] == "grad/value_cosine_similarity"].groupby(["dataset"]):
  plot_swapped_cdf(group, name)
```

```python
#df_long[df_long["variable"] == "grad/value_cosine_similarity"].groupby(["Algorithm", "dataset"])["value"].describe()
t = df_long[df_long["variable"] == "grad/value_cosine_similarity"]
# t.groupby(["dataset"])["value"].size()
df_teleport = t[t["dataset"].str.contains("teleport")]
df_teleport.loc[:, "value"] = df_teleport["value"].astype("float64")
# df_teleport.groupby(["quantile"])["value"].describe()  # Here we can see too many values for the bumps
error_quant = df_teleport[df_teleport["quantile"] == 0.75]
non_error_quant = df_teleport[df_teleport["quantile"] == 0.74]
print(error_quant["run_id"].describe())
print(non_error_quant["run_id"].describe())
print(error_quant.mean(numeric_only=True) - non_error_quant.mean(numeric_only=True))
print(merged_training_with_iqm_df[(merged_training_with_iqm_df["grad/value_cosine_similarity_quant0.74"].isna()) & (merged_training_with_iqm_df["dataset"].str.contains("teleport"))].iloc[0])
```

### Magnitude Similarity

Let's take a look at gradient magnitude similarity:

```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])["grad/value_magnitude_similarity_mean"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["hp.agent_name"])["grad/value_magnitude_similarity_mean"].mean())

print(merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])["grad/value_magnitude_similarity_mean"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["hp.agent_name", "eval_bins5"])["grad/value_magnitude_similarity_mean"].mean())

```

To validate that nothing strange is going on, take a look at the size of gradients in general:

```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])["grad/value_scale_mean"].mean())
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])["grad/actor_scale_mean"].mean())

print("Only best configuration per landscape for next prints")
print(best_config_df.groupby(["hp.agent_name"])["grad/value_scale_mean"].mean())
print(best_config_df.groupby(["hp.agent_name"])["grad/actor_scale_mean"].mean())

print("Compare the loss sizes")
print(
  merged_training_with_iqm_df.groupby(["hp.agent_name"])[merged_training_with_iqm_df.columns[merged_training_with_iqm_df.columns.str.contains("loss") | merged_training_with_iqm_df.columns.str.contains("lam")]].mean(numeric_only=True)
      )
```

Is there correlation between learning rate (and maybe discount factor) and cosine similarity?

```python
print(good_configs_df.groupby(["eval_bins5", "hp.agent_name"])[["grad/value_cosine_similarity_mean", "hp.lr"]].corr())
```

```python
corr = (
    merged_training_with_iqm_df.groupby(["hp.agent_name"]).apply(lambda g: g.corr(numeric_only=True)["grad/value_cosine_similarity_mean"]))
corr_sorted = corr.stack().rename("corr").reset_index()
corr_sorted = corr_sorted.reindex(corr_sorted["corr"].abs().sort_values(ascending=False).index)
corr_sorted
```

Difference between the **two**agents (this will not work when adding CRL)

```python
corr_diff = corr.loc["hiql"] - corr.loc["qrl"]
corr_diff_sorted = corr_diff.reindex(corr_diff.abs().sort_values(ascending=False).index)
corr_diff_sorted
```

Now lets do this sorted by training progress

```python
corr_progress = good_configs_df.groupby(["hp.agent_name", "eval_bins5"]).apply(lambda g: g.corr(numeric_only=True)["grad/value_cosine_similarity_mean"])
corr_progress_sorted = corr_progress.stack().rename("corr").reset_index()
corr_progress_sorted = corr_progress_sorted.reindex(corr_progress_sorted["corr"].abs().sort_values(ascending=False).index)
corr_progress_sorted[corr_progress_sorted["eval_bins5"] == 0]
```

## Parameter Updates

We use Adam for optimization and therefore parameter updates might look a bit different directly compared to the gradients. Let's analyze the parameter updates.

```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])[["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]].describe())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["hp.agent_name"])[["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]].mean())

print("---------------------------------------------")

print(merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])["update/value_cosine_similarity_mean"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["hp.agent_name", "eval_bins5"])["update/value_cosine_similarity_mean"].mean())
```
Differentiate between experiments. **For now only exploration schedule as this is antmaze only**.

```python
print(merged_training_with_iqm_df.groupby(["exploration_schedule", "hp.agent_name"])[["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]].describe())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["exploration_schedule", "hp.agent_name"])[["update/value_cosine_similarity_mean", "update/actor_cosine_similarity_mean"]].mean())

print(merged_training_with_iqm_df.groupby(["exploration_schedule", "eval_bins5", "hp.agent_name"])["update/value_cosine_similarity_mean"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["exploration_schedule", "eval_bins5", "hp.agent_name"])["update/value_cosine_similarity_mean"].mean())
```


Do this analysis for standard deviation:
```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])[["update/value_cosine_similarity_std", "update/actor_cosine_similarity_std"]].describe())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["exploration_schedule", "hp.agent_name"])[["update/value_cosine_similarity_std", "update/actor_cosine_similarity_std"]].mean())

print(merged_training_with_iqm_df.groupby(["exploration_schedule", "eval_bins5", "hp.agent_name"])["update/value_cosine_similarity_std"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["exploration_schedule", "eval_bins5", "hp.agent_name"])["update/value_cosine_similarity_std"].mean())
```


Let's take a look at gradient magnitude similarity:

```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])["update/value_magnitude_similarity_mean"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["hp.agent_name"])["update/value_magnitude_similarity_mean"].mean())

print(merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])["update/value_magnitude_similarity_mean"].mean())
print("Only best configuration per landscape for next print")
print(best_config_df.groupby(["hp.agent_name", "eval_bins5"])["update/value_magnitude_similarity_mean"].mean())

```

To validate that nothing strange is going on, take a look at the size of gradients in general:

```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])["update/value_scale_mean"].mean())
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])["update/actor_scale_mean"].mean())

print("Only best configuration per landscape for next prints")
print(best_config_df.groupby(["hp.agent_name"])["update/value_scale_mean"].mean())
print(best_config_df.groupby(["hp.agent_name"])["update/actor_scale_mean"].mean())
```

Is there correlation between learning rate (and maybe discount factor) and cosine similarity?

```python
print(merged_training_with_iqm_df.groupby(["hp.agent_name"])[["update/value_cosine_similarity_std", "hp.lr"]].corr())
```

```python
corr = (
    merged_training_with_iqm_df.groupby(["hp.agent_name"]).apply(lambda g: g.corr(numeric_only=True)["update/value_cosine_similarity_mean"]))
corr_sorted = corr.stack().rename("corr").reset_index()
corr_sorted = corr_sorted.reindex(corr_sorted["corr"].abs().sort_values(ascending=False).index)
corr_sorted
```

Difference between the **two**agents (this will not work when adding CRL)

```python
corr_diff = corr.loc["hiql"] - corr.loc["qrl"]
corr_diff_sorted = corr_diff.reindex(corr_diff.abs().sort_values(ascending=False).index)
corr_diff_sorted
```

Now lets do this sorted by training progress

```python
corr_progress = merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"]).apply(lambda g: g.corr(numeric_only=True)["grad/value_cosine_similarity_mean"])
corr_progress_sorted = corr_progress.stack().rename("corr").reset_index()
corr_progress_sorted = corr_progress_sorted.reindex(corr_progress_sorted["corr"].abs().sort_values(ascending=False).index)
corr_progress_sorted[corr_progress_sorted["eval_bins5"] == 0]
```


# Target drift

```python
def literal_lists_to_numpy(s):
  import ast  # I do not know why this can't be imported up front
  try:
    return np.array(ast.literal_eval(s))
  except:
    return None
merged_training_with_iqm_df["target/held_out_val_batch_values_np"] = merged_training_with_iqm_df["target/held_out_val_batch_values"].apply(literal_lists_to_numpy)
```
```python
def compute_drift_to(group: pd.DataFrame, target: str = "end"):
  def target_drift(a, b):
    if a is None or b is None:
      return None
    if not a.shape == b.shape:
      raise ValueError("Arrays must have the same shape")
    return np.linalg.norm(a - b)

  start_value = group.loc[group["eval_step"] == group["eval_step"].min(), "target/held_out_val_batch_values_np"].values[0]
  end_value = group.loc[group["eval_step"] == group["eval_step"].max(), "target/held_out_val_batch_values_np"].values[0]
  drift_start_end = target_drift(start_value, end_value)
  if target == "neighbor":
    values = group["target/held_out_val_batch_values_np"]
    compare_values = group["target/held_out_val_batch_values_np"].shift(1)
    return pd.Series([target_drift(a, b) for a, b in zip(values, compare_values)], index=group.index) / drift_start_end
  elif target == "end":
    compare_value = end_value
  elif target == "start":
    compare_value = start_value
  else:
    raise ValueError(f"Unknown target: {target}")
  return group["target/held_out_val_batch_values_np"].apply(lambda x: target_drift(x, compare_value) / drift_start_end if drift_start_end else None)

merged_training_with_iqm_df["target_drift_end"] = merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset", "config_index", "seed"], group_keys=False).apply(compute_drift_to, target="end")
merged_training_with_iqm_df["target_drift_start"] = merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset", "config_index", "seed"], group_keys=False).apply(compute_drift_to, target="start")
merged_training_with_iqm_df["target_drift_neighbor"] = merged_training_with_iqm_df.groupby(["hp.agent_name", "dataset", "config_index", "seed"], group_keys=False).apply(compute_drift_to, target="neighbor")
print(merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])["target_drift_end"].agg(["mean", "std"]))
print(merged_training_with_iqm_df.groupby(["hp.agent_name", "eval_bins5"])["target_drift_start"].agg(["mean", "std"]))
print(merged_training_with_iqm_df.groupby(["hp.agent_name", "exploration_schedule", "eval_bins5"])["target_drift_neighbor"].agg(["mean", "std"]))
```

# Combine Landscape Plots

## Seaborn KDE-like plot

```python
merged_results_df.groupby(["hp.agent_name", "dataset"])
```

```python
from src.gcrl_landscapes.util.eval import fit_model
from src.gcrl_landscapes.configurations import get_bounds, sobol_codomain_to_hp
from gcrl_landscapes.plots.triple_gp import create_contour_plot
from gcrl_landscapes.evaluation.common import map_labels
agent_name = "crl"
grid_length = 100

fig, ax = plt.subplots()
clipped_merged_results_df = merged_results_df.copy()
clipped_merged_results_df["mean_normalized_goal_distance_return"] = clipped_merged_results_df["mean_normalized_goal_distance_return"].clip(0, 1)
data_temp = clipped_merged_results_df[ (merged_results_df["hp.agent_name"] == agent_name)
                                & (merged_results_df["dataset"] == "antmaze-medium-explore-v0,antmaze-medium-explore80navigate-v0,antmaze-medium-explore40navigate-v0,antmaze-medium-navigate-v0")
                                & (merged_results_df["hps"] == frozenset(set(["lr", "discount"])))
                                ]
# for phase_num ...
point_dfs = []
for name, group in data_temp.groupby(["phase_num"]):
  group_copy = group.copy().reset_index()
  model = fit_model(group, "mean_normalized_goal_distance_return", ["hp.lr", "hp.discount"])
  model.fit()
  create_contour_plot(model, x_dim=0, y_dim=1, z_dim="mean_normalized_goal_distance_return", bounds=[0, 1], filename="test_contour.png", dim_label_mapping=map_labels, agent_name=agent_name, z_transform=lambda x, _: x, discrete_levels=None, last_phase_best_config=None)
  x_lower, x_upper, x_log = get_bounds(model.hp_names[0].removeprefix("hp."), agent_name)
  y_lower, y_upper, y_log = get_bounds(model.hp_names[1].removeprefix("hp."), agent_name)
  x, y = np.linspace(0, 1, grid_length), np.linspace(0, 1, grid_length)
  X, Y = np.meshgrid(x, y)
  points = np.vstack([X.ravel(), Y.ravel()]).transpose()
  Z = model.get_middle(points).clip(0, 1)
  Z_normalized = Z / Z.max()
  Z_selected = (Z_normalized > 0.9).squeeze()
  points_x, points_y = sobol_codomain_to_hp(points[:, 0], x_lower, x_upper, x_log), sobol_codomain_to_hp(points[:, 1], y_lower, y_upper, y_log)
  print(np.vstack([points_x, points_y]))

  # # TODO: decide
  # group_copy["hp.lr"] = hp_to_sobol_codomain(group_copy["hp.lr"], x_lower, x_upper, x_log)
  # group_copy["hp.discount"] = hp_to_sobol_codomain(group_copy["hp.discount"], y_lower, y_upper, y_log)
  # marginalized_group_copy = group_copy.groupby(["hp.lr", "hp.discount"])["mean_normalized_goal_distance_return"].apply(lambda values: trim_mean(values, proportiontocut=0.25)).reset_index()
  # marginalized_group_copy["goal_distance_return_eps"] = marginalized_group_copy["mean_normalized_goal_distance_return"] / marginalized_group_copy["mean_normalized_goal_distance_return"].max()


  points_prediction_df = pd.DataFrame({"hp.lr": points_x, "hp.discount": points_y, "mean_normalized_goal_distance_return": Z_normalized.squeeze()})
  # points_prediction_df = marginalized_group_copy[marginalized_group_copy["goal_distance_return_eps"] > 0.8]
  points_prediction_df["phase_num"] = name[0]
  point_dfs.append(points_prediction_df)
point_df = pd.concat(point_dfs)
print(point_df.describe())

print(len(point_df))
# sns.scatterplot(data=point_df[point_df["mean_normalized_goal_distance_return"] > 0.9], x="hp.lr", y="hp.discount", hue="phase_num")
x_lower, x_upper, x_log = get_bounds("lr", agent_name)
y_lower, y_upper, y_log = get_bounds("discount", agent_name)
# ax = sns.kdeplot(data=point_df[point_df["mean_normalized_goal_distance_return"] > 0.95], x="hp.lr", y="hp.discount", hue="phase_num", log_scale=(x_log, y_log), levels=10, bw_adjust=1, fill=True, alpha=0.4, palette="rocket")
df = point_df[point_df["mean_normalized_goal_distance_return"] > 0.90]

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

palette = sns.color_palette("viridis", n_colors=df["phase_num"].nunique())

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

for i, phase in enumerate(sorted(df["phase_num"].unique())):
    phase_df = df[df["phase_num"] == phase]
    color = palette[i]
    
    sns.kdeplot(
        data=phase_df,
        x="hp.lr", 
        y="hp.discount",
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
        y="hp.discount",
        log_scale=(x_log, y_log),
        fill=False,
        levels=[0.7],
        thresh=0.05,
        bw_adjust=0.7,
        alpha=0.9,
        color=color,
        linewidths=3.5,
        ax=ax,
        label=f"Phase {phase}"
    )
    
    centroid_x = phase_df["hp.lr"].median()
    centroid_y = phase_df["hp.discount"].median()
    ax.scatter(centroid_x, centroid_y, s=150, c=[color], edgecolors='white', 
               linewidths=2, zorder=100, marker='o')
    ax.text(centroid_x, centroid_y, str(phase), fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white', zorder=101)

# ax.set_xlabel("")
# ax.set_ylabel("")
ax.set_xlabel("Learning Rate", fontsize=14, fontweight='bold')
ax.set_ylabel("Discount Factor", fontsize=14, fontweight='bold')
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

#sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="phase")
ax.grid(True, alpha=0.15)
print(ax.collections[0].levels)
plt.xlim(x_lower, x_upper)
plt.ylim(y_lower, y_upper)
if x_log:
  ax.set_xscale("log", base=10)
if y_log:
  ax.set_yscale("log", base=10)
plt.tight_layout()
plt.savefig("test.png", dpi=600)
plt.close()



```

## Optimum Movement line plot

```python
data_temp = merged_results_df[(merged_results_df["dataset"] == "antmaze-medium-explore-v0,antmaze-medium-explore80navigate-v0,antmaze-medium-explore40navigate-v0,antmaze-medium-navigate-v0") & (merged_results_df["hps"] == frozenset(set(["lr", "discount"])))]
optima = []
for name, group in data_temp.groupby(["hp.agent_name", "phase_num"]):
  marginalized_group = group.groupby(["hp.lr", "hp.discount"])["mean_normalized_goal_distance_return"].apply(lambda values: trim_mean(values, proportiontocut=0.25)).reset_index()
  t = marginalized_group[marginalized_group["mean_normalized_goal_distance_return"] == marginalized_group["mean_normalized_goal_distance_return"].max()].iloc[0][["hp.lr", "hp.discount"]]
  t["Algorithm"] = name[0]
  optima.append(t)
x_lower, x_upper, x_log = get_bounds("lr", agent_name)
y_lower, y_upper, y_log = get_bounds("discount", agent_name)
fig, ax = plt.subplots()
sns.lineplot(data=pd.DataFrame(optima), x="hp.lr", y="hp.discount", hue="Algorithm", errorbar=None, marker="o")
ax.set(xscale="log")
plt.xlim(x_lower, x_upper)
plt.ylim(y_lower, y_upper)
plt.savefig("test.png")
```
