# ENVS — Advantage Data

One subfolder per environment, each containing the raw `advantages.parquet` and all files derived from it.

```
ENVS/
├── antmaze-medium/
├── antmaze-large/
└── cube/
```

---

## Raw data: `advantages.parquet`

Each file stores the full cross-goal advantage matrix A(obs_i, goal_j) evaluated at the last training checkpoint for every agent × configuration × seed combination.

| Environment | Size | Row groups | Rows |
|---|---|---|---|
| antmaze-medium | 4.1 GB | 12 800 | 838 860 800 |
| antmaze-large | 3.9 GB | 12 770 | 836 894 720 |
| cube | 3.9 GB | 12 790 | 838 205 440 |

**Schema**

| Column | Type | Description |
|---|---|---|
| `checkpoint_path` | string | Absolute path to the `.pkl` checkpoint used |
| `agent` | string | `CRL`, `GCIQL`, `GCIVL`, or `QRL` |
| `dataset` | string | Environment name (e.g. `antmaze-medium-navigate-v0`) |
| `phase` | string | Training step of the checkpoint (last phase only) |
| `seed` | string | Random seed `0`–`4` |
| `configuration` | string | Hyperparameter configuration index `0`–`63` |
| `actor` | string | Network head (`actor`, `low_actor`, `high_actor`) |
| `batch_idx` | string | Batch index `0`–`9` within the checkpoint (10 different 256-sample draws) |
| `obs_idx` | int64 | Observation index `0`–`255` |
| `goal_idx` | int64 | Goal index `0`–`255` |
| `is_positive` | bool | `True` when `obs_idx == goal_idx` (matched / diagonal pair) |
| `advantage` | float32 | A(obs_i, goal_j) — raw advantage value |

**Structure per row group:** one row group = one (agent, configuration, seed, batch\_idx) combination = 256 × 256 = 65 536 rows forming a complete advantage matrix.

**Agents per environment:**  4 agents × 1 phase × 64 configurations × 5 seeds × 10 batches = ~12 800 row groups.  Each agent has exactly one phase (the last checkpoint); the raw step numbers differ across agents.

---

## Extracted files

All derived files are produced by `ANALYSIS/extract_env.py`.

### `advantages_summary.parquet`

One row per (agent, actor, phase, seed, configuration). Aggregates all 10 batches.

| Column | Description |
|---|---|
| `agent`, `actor`, `phase`, `seed`, `configuration` | Group keys |
| `n` | Total rows (= 655 360 per group) |
| `n_positive` | Number of diagonal (matched) pairs |
| `pos_fraction` | `n_positive / n` — always 1/256 ≈ 0.0039 |
| `adv_mean`, `adv_std` | Mean and std of all advantages in the group |
| `adv_p5`, `adv_p25`, `adv_median`, `adv_p75`, `adv_p95` | Percentiles |
| `ess` | *(antmaze-medium only)* Softmax ESS without alpha — use `ess_per_batch` instead |

### `advantage_matrix_metrics.parquet`

One row per row group (one 256×256 matrix). The primary file for discrimination analysis.

| Column | Description |
|---|---|
| `agent`, `config`, `seed`, `phase`, `actor`, `batch_idx` | Group keys |
| `fr_auc` | Pr(A⁺ > A⁻) — fraction of (obs, off-diagonal goal) pairs where matched advantage wins |
| `gap_mean` | Mean of A⁺[i] − A⁻[i,j] across all off-diagonal pairs |
| `gap_std` | Std of the above gaps |
| `extractability_index` | `gap_mean / gap_std` — signal-to-noise of the advantage gap |
| `mrr` | Mean reciprocal rank of the matched goal across all 256 goals per observation |
| `prec_at_1` | Fraction of observations where the matched goal ranks first |
| `prec_at_5` | Fraction of observations where the matched goal ranks in top 5 |
| `Aplus_mean`, `Aplus_std` | Stats of diagonal advantages |
| `Aminus_std`, `adv_minus_mean` | Stats of off-diagonal advantages |

### `ess_per_batch.parquet`

One row per row group. ESS computed from AWR weights on the diagonal (matched pairs) using the per-configuration `alpha` value. **Not available for cube** (no config JSONs found).

| Column | Description |
|---|---|
| `agent`, `config`, `seed`, `phase`, `actor`, `batch_idx` | Group keys |
| `ess` | Normalised ESS in [0,1]: `(Σw̃)² / (N·Σw̃²)` where `w̃ = clip(exp(α·A⁺), ε, 100) / max(·)` |

### `advantages_histograms.npz`

200-bin histograms of the raw advantage distribution per group. Keys are formatted as `{agent}_{actor}_{phase}_{seed}_{configuration}_counts` and `…_edges`.

---

## Quick load

```python
import pandas as pd

ENV = "antmaze-medium"   # or antmaze-large, cube
BASE = f"ENVS/{ENV}"

summary = pd.read_parquet(f"{BASE}/advantages_summary.parquet")
matrix  = pd.read_parquet(f"{BASE}/advantage_matrix_metrics.parquet")
ess     = pd.read_parquet(f"{BASE}/ess_per_batch.parquet")   # not for cube
```

Reading the raw parquet efficiently (streaming, never loads full file):

```python
import pyarrow.dataset as ds

dataset = ds.dataset(f"{BASE}/advantages.parquet", format="parquet")
scanner = dataset.scanner(columns=["agent", "obs_idx", "goal_idx", "advantage"],
                           batch_size=2**17)
for batch in scanner.to_batches():
    df = batch.to_pandas()
    # process one ~131k-row chunk at a time
```
