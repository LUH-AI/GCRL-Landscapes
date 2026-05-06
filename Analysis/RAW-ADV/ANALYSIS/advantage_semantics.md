# Advantage Semantics and the Trainability Landscape

> This document explains the three cross-goal advantage metrics (FR-AUC, gap_mean, extractability_index),
> what the antmaze/cube data shows, and why the trainability landscape changes between environments.
> All claims about goal/reward/advantage definitions are grounded in the OGBench source code.

---

## 1. What the three metrics measure

### FR-AUC

Pick one matched pair (obs_i, goal_i) and one random pair (obs_i, goal_j ≠ i).
How often does the matched pair get a higher advantage?

- **0.5** = random coin flip — the critic cannot distinguish the right goal from a wrong one
- **> 0.5** = the critic correctly ranks the matched pair higher
- **< 0.5** = the critic is *inverted* — it systematically prefers random goals over the true one

FR-AUC only counts wins and losses. It says nothing about how decisive the wins are.

### gap_mean

The average *margin* by which the matched pair wins: `E[A+(i) − A−(i,j)]`.

FR-AUC and gap_mean can diverge in opposite directions:
- **High FR-AUC, small gap**: wins 55% of comparisons by tiny margins
- **Moderate FR-AUC, large gap**: wins 52% of comparisons by large margins

AWR cares about the margin because it exponentiates: `w_i ∝ exp(α · A_i)`.
A large gap creates concentrated weights; a small gap creates diffuse weights.

### extractability_index

`gap_mean / (gap_std + ε)` — a signal-to-noise ratio.

A large average margin that is *consistent* across all observation pairs gives high EI.
A large margin that varies wildly (some pairs huge, some near zero) gives low EI.

---

## 2. OGBench code grounding — goals, rewards, and advantages

### 2a. What is a "goal" in each environment?

Goals are **not** full states. They are projections onto task-relevant subspaces.

| Environment | Goal representation | Source |
|---|---|---|
| antmaze-medium/large | 2D position `(x, y)` | `maze.py:482–484`, oracle = `np.array(self.cur_goal_xy)` |
| cube-single-play | Scaled, centered 3D object position `(x, y, z)` | `cube_env.py:796–806` |

For cube, the oracle representation is:
```python
ob.append((ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler)
```
The arm state, gripper state, and joint velocities are **not part of the goal**.

### 2b. How rewards work (both environments)

Rewards are **sparse** — there is no continuous shaping signal.

| Environment | Success condition | Reward |
|---|---|---|
| antmaze | agent `(x,y)` within 0.5 m of goal | +1 / 0 |
| cube | cube position within 0.04 m of target | 0 / −1 (one cube) |

```python
# antmaze (maze.py:453–466)
reward = 1.0 if success else 0.0

# cube (cube_env.py:808–815)
successes = self._compute_successes()       # L2 ≤ 0.04 per cube
reward = float(sum(successes) - len(successes))
```

Because rewards are sparse, the advantage signal comes entirely from the learned value/Q
functions — there is no dense shaping to fall back on.

### 2c. Goal sampling — why the diagonal is "matched"

Goals are sampled as **future states from the same trajectory** using geometric sampling
(`datasets.py:252–281`):

```python
offsets = np.random.geometric(p=1 - discount, size=batch_size)
traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
```

This is exactly what makes the diagonal of the cross-goal matrix meaningful:
- **Diagonal (obs_i, goal_i)**: the goal IS a future state from obs_i's episode — the algorithm
  was actually trained to reach this goal from this observation.
- **Off-diagonal (obs_i, goal_j)**: goal is a future state from a *different* episode — a random
  goal that obs_i was never paired with in training.

FR-AUC therefore measures whether the algorithm's advantage function has generalised
correctly from the training goal distribution: does it rank the trained-on goal higher than
an out-of-distribution random goal?

### 2d. How each algorithm defines its advantage

The four algorithms split into two structural groups:

| Algorithm | Advantage formula | Value type |
|---|---|---|
| **CRL** | `Q(s, a, g) − V(s, g)` | Contrastive (NCE inner products in learned rep space) |
| **GCIQL** | `Q(s, a, g) − V(s, g)` | IQL-style Bellman (expectile regression) |
| **GCIVL** | `V(s', g) − V(s, g)` | IQL-style V-function (no explicit Q) |
| **QRL** | `V(s', g) − V(s, g)` | Negated quasimetric distance `−d(s, g)` |

Sources: `crl.py:78`, `gciql.py:68`, `gcivl.py:69`, `qrl.py:84`.

**Group 1 (Q−V): CRL and GCIQL**
`adv = Q(s,a,g) − V(s,g)` is the advantage of taking the *specific action a* over the average
action from state s. For a matched pair, a and g were paired during training, so the action
should genuinely be better than average. For a random pair, the action is irrelevant to g,
so Q and V should be near-equal → advantage near zero. Correct ranking requires the learned
Q to properly attribute credit to matched-goal actions.

**Group 2 (TD): QRL and GCIVL**
`adv = V(s',g) − V(s,g)` is the *progress toward goal g* made by the transition s→s'.
For a matched pair, s→s' is a step in the right direction toward g by construction.
For a random pair, the step may or may not be helpful. Correct ranking requires the value
function to correctly encode "how far am I from g?" so that progress toward a matched goal
scores higher than noise.

---

## 3. What the data shows

### Antmaze (both sizes) — internally consistent picture

| Agent | fr_auc | gap_mean | EI | Landscape |
|-------|--------|----------|----|-----------|
| QRL   | 0.546–0.582 | +0.36–+0.68 | 0.19–0.25 | broad |
| GCIVL | 0.564–0.575 | +0.14–+0.14 | 0.12–0.14 | broad |
| CRL   | 0.527–0.530 | +0.17–+0.17 | 0.05–0.05 | selective/narrow |
| GCIQL | 0.471–0.472 | −0.10–−0.10 | −0.07–−0.08 | brittle |

EI rank = landscape width rank. The signal-to-noise ratio is an environment-independent
proxy for extractability regime.

### Cube — the picture shifts

| Agent | fr_auc | gap_mean | EI | Landscape (mobility) |
|-------|--------|----------|----|-----------|
| GCIVL | 0.644 | +0.160 | +0.125 | broad |
| CRL   | 0.562 | **+0.875** | **+0.164** | **narrow** |
| GCIQL | 0.531 | −0.039 | −0.028 | **broad** |
| QRL   | 0.523 | +0.167 | +0.108 | **narrow** |

The metric changes predict the landscape changes: agents whose advantage signal sharpens
get a narrower landscape; agents whose signal weakens or recovers get a broader one.

---

## 4. Why the picture changes — grounded in OGBench task structure

### CRL: contrastive representations are much sharper on cube

On **antmaze**, CRL learns a representation space where `φ(s)^T ψ(g) ≈ Q(s, g)`.
The state `s` is a 26D+ locomotion state (joint angles, velocities). The goal `g` is
a **2D position (x, y)**. The contrastive objective must bridge a large representational
gap: map a 26D locomotion state to something comparable with a 2D position vector.

On **cube**, the state `s` includes the cube's 3D position *directly*. The goal `g` is
the cube's target 3D position. The contrastive objective now only needs to match a
3D subspace of the full state to a 3D goal. The representational gap is much smaller.

Consequence: CRL's representations are far more precise on cube, so advantages are
extremely peaked. The gap_mean explodes from 0.17 to 0.875 — the matched pair wins
by a 5× larger margin. Under AWR, this means weights concentrate on a tiny fraction
of transitions → the policy update becomes hypersensitive to α → landscape narrows.

### GCIQL: inversion weakens on cube because the Q−V credit assignment problem eases

On **antmaze**, GCIQL's Q−V margin is inverted (fr_auc = 0.471, gap_mean = −0.10).
The most likely cause: on a maze, the IQL Q function assigns high values near any
goal-like region, not just the matched one. The Bellman bootstrap propagates value
from high-reward states to nearby states regardless of which goal they belong to.
The V function, trained with expectile regression on the same bootstrapped targets,
can overshoot the matched-goal Q → Q − V < 0 for matched pairs.

On **cube**, the success threshold is strict (0.04 m vs 0.5 m on antmaze) and the
reward is −1 everywhere except exact target. The IQL Bellman target is a much cleaner
single spike rather than a broad near-goal region. This limits value propagation to
"leaked" nearby states and reduces the V overshoot. FR-AUC rises from 0.471 to 0.531,
gap_mean rises from −0.10 to −0.04. GCIQL is still slightly inverted in expectation
but no longer systematically wrong → more configurations can extract a policy → broader
landscape.

### QRL: quasimetric distances degrade in cube's state space

On **antmaze**, `adv = d(s, g) − d(s', g)` is a well-defined progress signal.
Quasimetric distances compose along maze paths: if s→s' is a step toward g,
then d(s,g) > d(s',g) reliably. The maze navigation structure creates a clean
partial order over states with respect to any goal.

On **cube**, the state space is higher-dimensional and the relevant subspace
(is the cube getting closer to its target?) is harder to isolate from the arm
kinematics. The quasimetric d(s,g) must be learned from sparse reward signals
in a high-dimensional joint space. The learned distances are noisier:
`d(s, g) − d(s', g)` for a matched pair becomes similar in magnitude to
`d(s, g_rand) − d(s', g_rand)` for a random pair.

Consequence: gap_mean drops from 0.68 → 0.17 and EI from 0.25 → 0.11.
AWR weights are more uniform (higher ESS) but the advantage signal is weaker →
fewer configurations produce reliable improvement → landscape narrows.
This is the **opposite mechanism** from CRL: not "signal too concentrated" but
"signal too weak."

### GCIVL: value progress is task-agnostic and stays robust

GCIVL uses `adv = V(s', g) − V(s, g)` with a standard IQL V-function (no quasimetric
constraint). Unlike QRL, GCIVL's V is not constrained to satisfy triangle inequalities.
It is trained with a simpler expectile regression loss on sparse rewards.

On both antmaze and cube, V(s,g) can be learned via standard TD: "how much cumulative
reward do I expect from s toward g?" The transition s→s' is in the direction of g
(geometric sampling) so V increases reliably. GCIVL's fr_auc actually improves on cube
(0.564 → 0.644), suggesting that the cube reward structure (strict 0.04 m threshold,
per-cube success) is even cleaner for a standard V-function than the antmaze structure.

GCIVL is the most **task-agnostic** of the four agents: its advantage formulation
does not make strong assumptions about the geometry of the goal space.

---

## 5. The unified story

The same mechanism — **advantage semantics determine how concentrated AWR weights become**
— explains both environments. But *which* objective encodes goal-directed structure well
depends on the task's geometry.

| Task type | What "goal-directed" means | Well-matched objectives | Poorly-matched |
|---|---|---|---|
| Navigation (antmaze) | Move (x,y) along maze geodesic | QRL (quasimetric), GCIVL (V-progress) | CRL (rep gap 26D→2D), GCIQL (Q−V inversion) |
| Manipulation (cube) | Place 3D object at target | CRL (3D→3D rep, sharp), GCIVL (generic V) | QRL (quasimetric degrades), GCIQL (partial recovery) |

**The trainability landscape is not a fixed property of the algorithm.**
It is a property of how well the algorithm's inductive bias for representing goal-directed
value aligns with the task's goal structure.

The three metrics are the fingerprint of that alignment:
- **FR-AUC** detects systematic inversion (gradient sign errors)
- **gap_mean** measures the strength of the signal going into AWR weights
- **EI** measures the reliability / consistency of that signal

Together they predict landscape width without needing to run a full hyperparameter sweep.

---

## Reproducibility

```bash
# From RAW-ADV/
python ANALYSIS/analyse_framing.py          # all findings with correlation tables
python ANALYSIS/compute_afr_metrics.py --env antmaze-medium
python ANALYSIS/compute_afr_metrics.py --env antmaze-large
python ANALYSIS/compute_afr_metrics.py --env cube
```

Outputs: `ENVS/{env}/afr_metrics.csv`, `ENVS/antmaze-medium/ess_table.tex`
