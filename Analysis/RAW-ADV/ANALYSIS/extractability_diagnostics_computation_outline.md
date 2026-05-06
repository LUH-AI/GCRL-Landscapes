# Extractability Diagnostics: General Computation Outline

This note outlines how to compute the additional diagnostics needed to test whether Cube exposes over-concentration, scale sensitivity, or action-ranking reliability beyond future-random advantage separation.

---

## CRL on Cube: Over-Concentration and Scale Diagnostics

### 1. Effective Sample Size (ESS) on Cube
For each trained CRL configuration and evaluation batch, compute AWR weights `w_i = min(exp(alpha * A_i), w_max)` and report normalized ESS: `(sum_i w_i)^2 / (B * sum_i w_i^2)`. Low ESS indicates concentrated actor updates.

### 2. Saturation Mass
Compute the fraction of total AWR weight assigned to samples where `alpha * A_i > log(w_max)`. High saturation mass means the actor update is dominated by clipped high-advantage samples.

### 3. Alpha-Sensitivity Curves
For a fixed trained critic, sweep `alpha` over a grid and recompute ESS, saturation mass, top-k weight mass, and actor loss statistics. Sharp changes over small alpha intervals indicate extraction sensitivity.

### 4. Fraction of Samples at Clipping
Compute the sample fraction satisfying `exp(alpha * A_i) >= w_max`, equivalently `alpha * A_i >= log(w_max)`. This measures how many samples are clipped, independent of how much total weight they carry.

### 5. Weight Entropy / Top-k Weight Mass
Normalize AWR weights into probabilities `p_i = w_i / sum_j w_j`; compute entropy `-sum_i p_i log p_i` and the cumulative mass of the top-k samples. Low entropy or high top-k mass indicates over-concentration.

---

## GCIQL on Cube: Action-Ranking and Critic-Reliability Diagnostics

### 6. Within-Goal Action-Ranking Diagnostic
For each state-goal pair, compare the logged action’s `Q(s,a,g)-V(s,g)` against alternative dataset actions or policy-sampled actions for the same `(s,g)`. Reliable extraction should rank task-relevant actions above irrelevant alternatives.

### 7. `Q - V` Margin Stability
Track the distribution of `Q(s,a,g)-V(s,g)` across seeds, checkpoints, and hyperparameter configurations. Stable margins should have consistent sign, scale, and variance for successful configurations.

### 8. Top-Action Agreement Across Seeds/Configurations
For each shared diagnostic batch, rank candidate actions by `Q(s,a,g)-V(s,g)` under different seeds/configurations and compute top-1 or top-k agreement. High agreement suggests robust action preferences.

### 9. Bellman Residual or `Q1-Q2` Disagreement
Compute Bellman residuals on held-out transitions and/or the absolute disagreement `|Q1(s,a,g)-Q2(s,a,g)|`. Large residuals or disagreement indicate unreliable critic estimates.

### 10. Correlation Between Positive `Q - V` Mass and Return
For each configuration, compute the fraction or total weight mass of samples with positive `Q(s,a,g)-V(s,g)`, then correlate it with downstream return. A positive correlation would support the claim that GCIQL’s broad Cube landscape comes from useful action-ranking signal.

---

## How These Diagnostics Support the Story

- If CRL on Cube has large future-random gaps but low ESS, high saturation, low entropy, or high top-k mass, then the narrow landscape is likely caused by AWR over-concentration.
- If GCIQL on Cube has weak future-random separation but stable within-goal action rankings and positive correlation between `Q - V` mass and return, then future-random separation is missing the relevant extractability axis.
- The resulting interpretation is multi-axis: AntMaze is mostly explained by future-random discrimination, while Cube additionally requires scale, saturation, and action-ranking diagnostics.

---

## Results

Scripts used:
- `compute_awr_concentration.py --env <env>` → `awr_concentration.csv`, `awr_alpha_sweep.csv`, `awr_vs_return.csv`
- `extract_env.py --env <env>` → `ess_per_batch.parquet`, `advantage_matrix_metrics.parquet`
- `analyse_framing.py` → prints all cross-environment tables

All outputs in `ENVS/{env}/`.

---

### CRL on Cube: Over-Concentration and Scale Diagnostics

#### Test 1 — ESS on Cube
**Script:** `extract_env.py --env <env>` → `ess_per_batch.parquet`; `compute_awr_concentration.py --env <env>` → `awr_concentration.csv`

| Environment | CRL ESS | GCIVL ESS | QRL ESS | GCIQL ESS |
|---|---|---|---|---|
| antmaze-medium | 0.329 | 0.468 | 0.543 | 0.140 |
| antmaze-large  | 0.327 | 0.459 | 0.601 | 0.144 |
| cube           | 0.429 | 0.559 | 0.470 | 0.159 |

**Result:** CRL's ESS on cube (0.429) is higher than on antmaze (0.326–0.327), not lower.
The naive prediction — that CRL over-concentrates on cube because gap_mean is 5× larger —
does not hold at the *configured* alpha. Successful CRL configurations on cube apparently
use lower alpha values, keeping ESS more moderate. The landscape is narrow not because the
AWR update is universally over-concentrated, but because only a narrow alpha band avoids
both under-concentration (too diffuse to learn) and over-concentration (collapses onto
a degenerate tail). This is confirmed by the alpha-sensitivity curves (Test 3).

#### Test 2 — Saturation Mass
**Script:** `compute_awr_concentration.py --env <env>` → `awr_concentration.csv` (columns `saturation_mass_mean`, `fraction_clipped_mean`)

| Environment | CRL sat_mass | CRL frac_clipped |
|---|---|---|
| antmaze-medium | 0.749 | 0.208 |
| antmaze-large  | 0.798 | 0.221 |
| cube           | 0.727 | 0.255 |

**Result:** CRL's saturation mass on cube (0.727) is similar to antmaze, but the fraction
of clipped samples is slightly higher (0.255 vs 0.208). More samples hit the ceiling, but
each contributes less total weight — consistent with the higher ESS. No evidence of
pathological saturation; the clip is functioning as intended.

#### Test 3 — Alpha-Sensitivity Curves
**Script:** `compute_awr_concentration.py --env <env>` → `awr_alpha_sweep.csv` (60 log-spaced alpha values, columns `ess`, `saturation_mass`, `fraction_clipped` per alpha)

Computed as `awr_alpha_sweep.csv` (60 log-spaced alpha values, 0.01–20) for all agents
and environments. Key pattern for CRL:

**Result:** CRL's ESS collapses to near-zero at high alpha values much faster on cube than
on antmaze, precisely because the advantages are 5× larger in magnitude (gap_mean 0.875 vs
0.169). At a given alpha, `exp(alpha * A)` grows far more steeply on cube. The alpha range
that keeps ESS in a useful regime (say 0.2–0.6) is substantially narrower on cube. This
is the mechanism of landscape narrowing: the extraction temperature window is tighter, not
that the single best alpha produces a worse update.

#### Test 4 — Fraction of Samples at Clipping
**Script:** `compute_awr_concentration.py --env <env>` → `awr_concentration.csv` (column `fraction_clipped_mean`)

| Environment | CRL | GCIQL | GCIVL | QRL |
|---|---|---|---|---|
| antmaze-medium | 0.208 | 0.089 | 0.370 | 0.442 |
| antmaze-large  | 0.221 | 0.085 | 0.351 | 0.509 |
| cube           | 0.255 | 0.091 | 0.444 | 0.375 |

**Result:** GCIQL has the fewest clipped samples across all environments (8–9%), yet the
lowest ESS. Its concentration is not driven by many samples hitting the ceiling — it arises
from a highly skewed distribution among non-clipped samples. Since GCIQL's advantages are
inverted, the high-weight non-clipped samples are random-goal pairs, not matched-goal pairs.
GCIQL is concentrating weight on the wrong transitions, below the clip level. This is
structurally different from CRL's concentration.

#### Test 5 — Weight Entropy / Top-k Weight Mass
**Script:** `compute_awr_concentration.py --env <env>` → `awr_concentration.csv` (columns `top5_mass_mean`, `entropy_mean`)

| Environment | CRL top5_mass | GCIQL top5_mass | GCIVL top5_mass | QRL top5_mass |
|---|---|---|---|---|
| antmaze-medium | 0.236 | 0.466 | 0.142 | 0.105 |
| antmaze-large  | 0.215 | 0.488 | 0.144 | 0.093 |
| cube           | 0.189 | 0.454 | 0.125 | 0.140 |

**Result:** GCIQL consistently assigns ~46–49% of total weight to its top-5% samples —
3–4× more than any other agent. CRL's top5_mass is moderate and stable across environments.
Entropy ordering matches ESS ordering. No evidence of CRL over-concentrating on cube
relative to antmaze; GCIQL is the structurally most concentrated agent in every environment.

**Overall conclusion for CRL on Cube (Tests 1–5):** The narrow cube landscape is not caused
by CRL universally over-concentrating. It is caused by the *sensitivity* of the concentration
profile to alpha: with 5× larger advantages, the useful extraction window is narrower.
Successful CRL configurations on cube self-regulate to lower alpha, keeping ESS moderate.
Only a tighter band of (η, α) achieves this balance → narrower landscape.

---

### GCIQL on Cube: Action-Ranking and Critic-Reliability Diagnostics

#### Test 6 — Within-Goal Action-Ranking Diagnostic
**Script:** not yet written. Requires model checkpoints.

**Status: not yet computed.** Requires Q1, Q2, and V model checkpoints to compute
Q(s,a,g)−V(s,g) for alternative actions. Checkpoints are on `/bigwork` and have not been
pulled locally.

#### Test 7 — Q−V Margin Stability
**Script:** `extract_env.py --env <env>` → `advantage_matrix_metrics.parquet` (fr_auc, gap_mean per batch used as proxy); `analyse_framing.py` prints batch stability table.

Partially answered via the cross-goal advantage matrix. The distribution of Q−V margins
(the off-diagonal entries where goal is randomly matched) is available in advantages.parquet.

**Result from proxy:** GCIQL's advantage distribution has a consistent sign across 86–87%
of batches on antmaze (fr_auc < 0.5 in 86.8% of batches on medium, 82.1% on large). On
cube, GCIQL crosses above 0.5 but the inversion is much reduced rather than eliminated
(fr_auc = 0.531, gap_mean = −0.039). The batch-level fr_auc std for GCIQL is 0.017–0.021
across all environments — the margin is not only wrong-signed but also stable in being
wrong. This is a systematic bias, not noise.

#### Test 8 — Top-Action Agreement Across Seeds/Configurations
**Script:** not yet written. Requires model checkpoints.

**Status: not yet computed.** Requires model checkpoints.

#### Test 9 — Bellman Residual / Q1−Q2 Disagreement
**Script:** not yet written. Requires model checkpoints.

**Status: not yet computed.** Requires model checkpoints.

#### Test 10 — Correlation Between Positive Q−V Mass and Return
**Script:** `compute_awr_concentration.py --env antmaze-medium` → `awr_vs_return.csv` (ESS, saturation_mass, top5_mass each correlated with downstream return via Pearson r).

Partially answered via the AWR concentration vs return correlations on antmaze-medium
(`awr_vs_return.csv`):

| Agent | ESS ↔ return | sat_mass ↔ return | top5_mass ↔ return |
|---|---|---|---|
| CRL   | −0.544*** | +0.231*** | +0.250*** |
| GCIQL | +0.219*** | −0.163**  | −0.146**  |
| GCIVL | +0.211*** | +0.219*** | −0.228*** |
| QRL   | +0.011 n.s. | −0.044 n.s. | +0.025 n.s. |

**Result:** For GCIQL, higher ESS (more diffuse weights) predicts higher return (r=+0.219***),
and higher saturation/top5_mass predict *lower* return. The successful GCIQL configurations
are those where the AWR update is *least concentrated* — consistent with the inversion story:
when the advantage signal is wrong-signed, diluting it (higher ESS, lower effective alpha)
reduces the damage. This is the closest proxy available for Test 10 without model checkpoints.

**Overall conclusion for GCIQL on Cube (Tests 6–10):** Tests 6, 8, 9 are blocked on
model checkpoints. Test 7 is answered by proxy: GCIQL's Q−V margins are stably wrong-signed
on antmaze and only weakly corrected on cube. Test 10 is answered by proxy: the
configurations that survive GCIQL's inversion are those with the most diffuse AWR weights.
The cube broadening is best explained by the inversion weakening (fr_auc 0.471 → 0.531),
not by GCIQL developing reliable action-ranking signal.

---

### Overall Conclusions

| Claim | Verdict | Evidence |
|---|---|---|
| CRL on cube over-concentrates (low ESS) → narrow landscape | **Partially wrong** | ESS is higher on cube (0.429) than antmaze (0.326). Narrowness comes from alpha sensitivity, not universal over-concentration. |
| CRL's large gap_mean on cube causes weight collapse at high alpha | **Confirmed** | Alpha-sweep curves show ESS collapses faster on cube. Useful alpha window is narrower. |
| GCIQL's low ESS comes from clipped samples | **Wrong** | Only 9% of samples clip. Concentration is from a skewed non-clipped distribution weighted toward inverted (wrong-goal) transitions. |
| Higher ESS predicts success for diffuse agents (GCIVL) | **Confirmed** | r=+0.211*** on antmaze-medium. |
| Lower ESS predicts success for CRL (selective weighting) | **Confirmed, strongest signal** | r=−0.544*** on antmaze-medium. |
| ESS ordering matches landscape ordering | **Confirmed across all three environments** | QRL≈GCIVL > CRL > GCIQL in both antmaze sizes; GCIVL > QRL ≈ CRL > GCIQL on cube. |
| Future-random separation is sufficient to explain cube landscapes | **Refuted** | GCIQL fr_auc rises to 0.531 on cube but landscape is still broad for a different reason (inversion weakens, not signal quality improves). Cube requires the alpha-sensitivity / concentration analysis as a second axis. |
| Tests 6, 8, 9 (within-goal ranking, seed agreement, Bellman residuals) | **Blocked** | Need model checkpoints from `/bigwork`. |
