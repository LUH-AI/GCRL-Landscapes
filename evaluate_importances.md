```python
import pandas as pd
importance_df: pd.DataFrame
importance_reset_df = importance_df.reset_index()
importance_reset_df.columns
importance_reset_df = importance_reset_df[importance_reset_df["agent"] != "crl"]
```

The importance scores we currently have do not sum to one due to only looking at singular hyperparameters.
We will probably have to normalize them. Let's do this and look if that makes sense later on (TODO).

```python
importance_reset_df["mean_normalized"] = importance_reset_df["mean"] / importance_reset_df.groupby(["setting", "agent", "trainingprogress"])["mean"].transform("sum")
```

# Uniformity across Phases

## Perplexity Plot

```python
import numpy as np
perplexity_df = importance_reset_df.groupby(["setting", "agent", "trainingprogress"])["mean_normalized"].apply(lambda xs: np.prod(xs ** (-xs))).reset_index()
perplexity_df["agent"] = perplexity_df["agent"].str.upper()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="paper", style="whitegrid")
fig, ax = plt.subplots(figsize=(3.5, 2))
sns.lineplot(data=perplexity_df[perplexity_df["setting"] == "scheduled_exploration"], x="trainingprogress", y="mean_normalized", hue="agent", marker="o", ax=ax)
ax.set_xlabel("Phase")
ax.set_xticks([25, 50, 75, 100], [1, 2, 3, 4])
ax.set_ylabel("Perplexity")
ax.legend(title="Algorithm", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.ylim(1, 9)
plt.tight_layout()
plt.savefig("perplexity-scheduled.png", dpi=1200)
```

## Entropy Plot

```python
import numpy as np
entropy_df = importance_reset_df.groupby(["setting", "agent", "trainingprogress"])["mean_normalized"].apply(lambda xs: -np.sum(xs * np.log2(xs))).reset_index()

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
sns.lineplot(data=entropy_df[entropy_df["setting"] == "scheduled_exploration"], x="trainingprogress", y="mean_normalized", hue="agent", ax=ax)
plt.title("Entropy")
plt.savefig("entropy-scheduled.png")
```

## Importance Mass

```python
importance_mass_df = importance_reset_df.groupby(["setting", "agent", "trainingprogress"])["mean_normalized"].apply(lambda xs: {"top1": np.sum(sorted(xs)[-1:]), "top2": np.sum(sorted(xs)[-2:])}).reset_index()
importance_mass_df["Algorithm"] = importance_mass_df[["agent", "level_3"]].apply(lambda row: row["agent"] + row["level_3"], axis=1)
print(importance_mass_df)
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
sns.lineplot(data=importance_mass_df[importance_mass_df["setting"] == "scheduled_exploration"], x="trainingprogress", y="mean_normalized", hue="Algorithm", ax=ax)
plt.title("Importance Mass")
plt.savefig("importance_mass-scheduled.png")

```

# Ranking Stability across Phases

```python
importance_reset_df.columns
```

```python
from functools import partial
rank_df = importance_reset_df.copy()
rank_df["rank"] = rank_df.groupby(["setting", "agent", "trainingprogress"])["mean"].rank(ascending=False) - 1
rank_wide = rank_df.pivot(index=["setting", "agent", "hp"], columns='trainingprogress', values=["rank", "mean_normalized"]).reset_index()

phases = sorted(importance_reset_df["trainingprogress"].unique())
def calculate_transition_kendall_taus(phases: list[int], rank_df: pd.DataFrame) -> dict[str, float]:
  from scipy.stats import weightedtau
  from functools import partial
  def rank_to_importance(phase1: int, phase2: int, rank: int) -> float:
    """Mean of HP importance between both phases as weight"""
    one = rank_df[(rank_df[("rank", phase1)] == rank)][("mean_normalized", phase1)]
    two = rank_df[(rank_df[("rank", phase2)] == rank)][("mean_normalized", phase2)]
    return (one.mean() + two.mean()) / 2
  phase_transitions = zip(phases[:-1], phases[1:])
  return pd.Series({f"{round(phase1/100*4)}->{round(phase2/100*4)}": weightedtau(rank_df[("rank", phase1)], rank_df[("rank", phase2)], weigher=partial(rank_to_importance, phase1, phase2)).statistic for phase1, phase2 in phase_transitions}).reset_index().rename(columns={"index": "transition", 0: "kendalltau"})

kendalltau_df = rank_wide.groupby(["setting", "agent"]).apply(partial(calculate_transition_kendall_taus, phases)).reset_index()
kendalltau_df["agent"] = kendalltau_df["agent"].str.upper()

fig, ax = plt.subplots(figsize=(3, 2))
sns.lineplot(data=kendalltau_df[kendalltau_df["setting"] == "scheduled_exploration"], x="transition", y="kendalltau", hue="agent", marker="o", legend=False, ax=ax)
plt.ylim(0, 1)
ax.set_xlabel("Phase Transition")
ax.set_ylabel(r"Weighted Kendall's $\tau$")
# ax.legend(title="Algorithm")
plt.tight_layout()
plt.savefig("kendalltau-scheduled.png", dpi=1200)
plt.close()
```
