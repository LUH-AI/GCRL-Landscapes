```python
import pandas as pd
import numpy as np
merged_training_df: pd.DataFrame = merged_training_df
merged_results_df: pd.DataFrame = merged_results_df
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
merged_training_with_iqm_df.groupby(["hp.agent_name"])["grad/value_cosine_similarity_quant0.05"].describe()
df_long[(df_long["variable"] == "grad/value_cosine_similarity")].groupby(["hp.agent_name", "quantile"])["value"].describe()
```

```python
fig, ax = plt.subplots()
merged_training_with_iqm_df["id"] = merged_training_with_iqm_df.index
df = merged_training_with_iqm_df.copy()
quant_cols = [col for col in df.columns if 'quant' in col]
id_cols = [col for col in df.columns if 'quant' not in col]

df_long = df.melt(
    id_vars=id_cols,
    value_vars=quant_cols,
    var_name='column',
    value_name='value'
)
df_long['quantile'] = df_long['column'].str.extract(r'quant([\d.]+)')[0].astype(float)

# Extract the base column name (everything before 'quant')
df_long['variable'] = df_long['column'].str.replace(r'_?quant[\d.]+', '', regex=True)

# Clean up
df_long = df_long.drop('column', axis=1)
sns.lineplot(data=df_long[df_long["variable"] == "grad/value_cosine_similarity"], x="quantile", y="value", hue="hp.agent_name", errorbar=None, marker="o")

# Normal distribution
quantiles = np.linspace(0, 1, 100)
import scipy.stats
values = scipy.stats.truncnorm(-1, 1, loc=0, scale=1).ppf(quantiles)
sns.lineplot(x=quantiles, y=values, color="black")

plt.ylabel("Value-Function Cosine Similarity")
plt.title("Value-Function Cosine Similarity Quantiles Inside of Batch")
plt.savefig("grad_cosine_similarity_quantiles.png")
plt.close()

fig, ax = plt.subplots()
# Construct PDF from quantiles
# TODO: all agents
df_long_filtered = df_long[(df_long["variable"] == "grad/value_cosine_similarity") & (df_long["hp.agent_name"] == "hiql")]
p = df_long[(df_long["variable"] == "grad/value_cosine_similarity") & (df_long["hp.agent_name"] == "qrl")]["value"].values
q = df_long[df_long["variable"] == "grad/value_cosine_similarity"]["quantile"].values
df_q = df_long_filtered.groupby(["quantile"])["value"].mean()
p = df_q.index
q = df_q.values
print(p)
print(q)
x_vals = []
pdf_vals = []

for i in range(len(p) - 1):
    x_segment = (q[i] + q[i+1]) / 2
    pdf_segment = (p[i+1] - p[i]) / (q[i+1] - q[i])
    x_vals.append(x_segment)
    pdf_vals.append(pdf_segment)

x = np.array(x_vals)
pdf = np.array(pdf_vals)
sns.lineplot(x=x, y=pdf)
plt.xlim(-1, 1)
plt.ylim(0, 2)
plt.savefig("test.png")


plt.close()


# Calculate densities for each interval
# density = ΔP / Δx
fig, ax = plt.subplots()
prob_changes = np.diff(p)
value_changes = np.diff(q)
densities = prob_changes / value_changes
x_points = (q[:-1] + q[1:]) / 2

# For plotting, we need the density at each quantile point
x_plot = np.linspace(q.min(), q.max(), 1000)
y_plot = np.interp(x_plot, q, p)
print(len(x_plot))
print(len(y_plot))
fig, ax = plt.subplots()
print(df_long[["hp.agent_name", "quantile", "value"]].groupby(["hp.agent_name", "quantile"]).describe())
mean_df = df_long[df_long["variable"] == "grad/value_cosine_similarity"].groupby(["hp.agent_name", "quantile"])["value"].mean().reset_index()
mean_df = mean_df.rename(columns={"hp.agent_name": "Algorithm"})
mean_df["Algorithm"] = mean_df["Algorithm"].str.upper()
ax = sns.lineplot(data=mean_df, x="value", y="quantile", hue="Algorithm", errorbar="ci", marker="o")

# # Overlay normal distribution
# x = np.linspace(-1, 1, 200)  # fine-grained x-values
# y = scipy.stats.norm(loc=0.075, scale=0.4).cdf(x)  # normal PDF with mean/std from your data
# plt.plot(x, y, color='red', linestyle='--', label='Normal fit')
#
# # Overlay uniform distribution
# x = np.linspace(-1, 1, 200)  # fine-grained x-values
# y = scipy.stats.uniform(loc=-1, scale=2).cdf(x)  # uniform PDF with mean/std from your data
# plt.plot(x, y, color='blue', linestyle='--', label='Uniform fit')

plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.title("Inter-Goal Gradient Alignment")
plt.xlabel("Gradient Cosine Similarity")
plt.ylabel("Cumulative Probability")
# Swap axes to get cdf
plt.savefig("gradient-alignment-cdf.png")

plt.close()

# fig, ax = plt.subplots()
# quantiles = np.linspace(0, 1, 100)
# import scipy.stats
# values = scipy.stats.norm.ppf(quantiles)
# sns.lineplot(x=np.linspace(-1, 1, 100), y=scipy.stats.norm().cdf(np.linspace(-1, 1, 100)))
# plt.xlim(-1, 1)
# plt.ylim(0, 1)
# plt.savefig("test.png")
# plt.close()

```

```python
fig, ax = plt.subplots()
sns.displot(data=merged_training_with_iqm_df, x="grad/value_cosine_similarity_mean", hue="hp.agent_name")
plt.savefig("grad_cosine_similarity_distributions.png")
```

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
