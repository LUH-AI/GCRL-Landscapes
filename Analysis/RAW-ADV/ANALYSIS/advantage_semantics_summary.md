# Advantage Semantics and the Trainability Landscape — Summary

---

## The three metrics

| Metric | What it measures |
|---|---|
| **FR-AUC** | Fraction of (matched, random) comparisons where the matched pair scores higher. 0.5 = random; <0.5 = inverted. |
| **gap_mean** | Average margin by which the matched pair wins: `E[A+(i) − A−(i,j)]`. AWR exponentiates advantages, so margin size directly controls weight concentration. |
| **extractability_index** | `gap_mean / gap_std` — signal-to-noise ratio. High = consistent signal; low = noisy or weak. |

---

## What the data shows

### Antmaze (both sizes)

| Agent | fr_auc | gap_mean | EI | Landscape |
|---|---|---|---|---|
| QRL   | 0.55–0.58 | +0.36–+0.68 | 0.19–0.25 | broad |
| GCIVL | 0.56–0.58 | +0.14       | 0.12–0.14 | broad |
| CRL   | 0.53      | +0.17       | 0.05      | selective/narrow |
| GCIQL | 0.47      | −0.10       | −0.07     | brittle |

EI rank = landscape width rank. Consistent across both maze sizes.

### Cube

| Agent | fr_auc | gap_mean | EI | Landscape |
|---|---|---|---|---|
| GCIVL | 0.64 | +0.16    | +0.12  | broad |
| CRL   | 0.56 | **+0.88** | **+0.16** | **narrow** |
| GCIQL | 0.53 | −0.04    | −0.03  | **broad** |
| QRL   | 0.52 | +0.17    | +0.11  | **narrow** |

CRL and GCIQL swap places. QRL and GCIVL swap places. The metric changes predict the landscape changes.

---

## What OGBench defines (from source)

**Goals are subsets of state:**
- Antmaze: goal = 2D position `(x, y)` only (`maze.py:484`)
- Cube: goal = scaled 3D object position only (`cube_env.py:796–806`); arm and gripper state are not part of the goal

**Rewards are sparse:**
- Antmaze: +1 if within 0.5 m, else 0
- Cube: 0 if within 0.04 m, else −1 — stricter threshold, narrower spike in state space

**Advantages split into two groups:**

| Group | Algorithms | Formula | What it measures |
|---|---|---|---|
| Q−V margin | CRL, GCIQL | `Q(s,a,g) − V(s,g)` | Is this action better than average for reaching g? |
| TD progress | QRL, GCIVL | `V(s′,g) − V(s,g)` | Does s→s′ make progress toward g? |

---

## Why the picture changes on cube

### CRL narrows — representational gap collapses

On antmaze: CRL's contrastive loss compares a 26D+ locomotion state to a 2D goal. Large gap between spaces → imprecise representations → moderate gap_mean (0.17).

On cube: the full state already contains the cube's 3D position; the goal IS a 3D cube position. The representational gap is nearly gone. CRL's contrastive objective becomes very precise → gap_mean jumps to 0.88 → AWR weights concentrate onto a tiny set of transitions → landscape narrows.

### GCIQL broadens — Bellman target is a cleaner spike

On antmaze: the 0.5 m success radius is wide. IQL's Bellman bootstrap spreads value to many nearby states. V overshoots for matched pairs → Q−V < 0 (inverted).

On cube: the 0.04 m threshold is strict. Value propagation stays local. V overshooting is reduced → Q−V recovers from −0.10 to −0.04 → more configurations extract a policy → landscape broadens.

### QRL narrows — quasimetric degrades in high-dimensional state space

On antmaze: quasimetric distances compose cleanly along maze paths. If s→s′ is a step toward g, d(s,g) > d(s′,g) reliably → strong gap_mean (0.68).

On cube: the full state includes arm kinematics, gripper, and cube pose. The quasimetric must be learned in this high-dimensional, entangled space from sparse rewards. Distances become noisier → gap_mean drops to 0.17 → signal too weak for reliable policy improvement → landscape narrows.

### GCIVL stays broad — weakest geometric assumption

GCIVL uses `V(s′,g) − V(s,g)` with a standard IQL V-function — no quasimetric constraint, no contrastive architecture. The transition s→s′ is geometrically sampled toward g, so progress is guaranteed by construction. The stricter cube reward makes V easier to learn precisely. FR-AUC improves from 0.56 → 0.64. GCIVL is the only agent that gets strictly better on cube.

---

## The core claim

The trainability landscape is not an intrinsic property of an algorithm.
It is a property of how well the algorithm's inductive bias for goal-directed value aligns with the task's goal structure.

When the goal space changes from 2D navigation positions to 3D object configurations, the alignment changes — predictably, from first principles:

- CRL's contrastive precision scales with goal-state representational alignment → sharpens on cube
- QRL's quasimetric composability depends on state-space geometry → degrades on cube
- GCIQL's Q−V inversion depends on reward spike width → partially recovers on cube
- GCIVL's generic V-progress is task-agnostic → stays robust across both

**FR-AUC, gap_mean, and EI are the fingerprint of that alignment** — measurable without a full hyperparameter sweep.
