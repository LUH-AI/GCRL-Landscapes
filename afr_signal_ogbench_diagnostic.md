# AFR: Future-Random Advantage Separation for AWR Diagnostics

We want one diagnostic that is fair to all four algorithms:

- CRL: advantage is a reachability-lift / future-occupancy signal.
- QRL: advantage is quasimetric distance reduction.
- GCIVL: advantage is value progress, often interpretable as progress toward the goal.
- GCIQL: advantage is a bootstrapped action-value margin, `Q(s,a,g) - V(s,g)`.

Correlating all of these advantages with Euclidean goal-distance return is not fair, because CRL is not trained to predict distance. A better common question is:

> Does the algorithm assign higher advantage to a goal that is actually reachable in the dataset future than to a random goal?

This gives a common, self-supervised reachability-ranking diagnostic.

## Definition

For each dataset transition `(s, a, s_next)`, sample two goals:

\[
g^+ \sim p^D_{\mathrm{geom}}(\cdot \mid s,a)
\]

from the same trajectory's geometric future-goal distribution, and

\[
g^- \sim p^D_{\mathrm{rand}}(\cdot)
\]

from the random dataset goal distribution.

For an algorithm-specific advantage function

\[
A_{\mathrm{alg}}(s,a,g),
\]

compute

\[
A^+ = A_{\mathrm{alg}}(s,a,g^+),
\qquad
A^- = A_{\mathrm{alg}}(s,a,g^-).
\]

Define the **future-random advantage gap**:

\[
\boxed{
\Delta A_{\mathrm{FR}} = A^+ - A^-.
}
\]

This is the main diagnostic signal.

---

## Metrics

### Future-random AUC

\[
\boxed{
\mathrm{FR\text{-}AUC} = \Pr(A^+ > A^-)
}
\]

Interpretation:

- `0.5`: no better than random ranking.
- `> 0.5`: future goals are ranked above random goals.
- close to `1.0`: clean reachability ranking.

---

### xtractability Index

\[
\boxed{
\mathrm{EI}
=
\frac{\mathbb{E}[\Delta A_{\mathrm{FR}}]}
{\mathrm{Std}(\Delta A_{\mathrm{FR}})+\epsilon}
}
\]

Interpretation:

- high EI: strong, stable separation.
- low EI: weak or noisy separation.
- negative EI: random goals often outrank future goals.

---

### AWR weight-ratio proxy

AWR uses

\[
w = \exp(\alpha A).
\]

For a future goal and random goal, the implied weight ratio is

\[
\frac{w^+}{w^-}
=
\exp\left(\alpha(A^+ - A^-)\right)
=
\exp(\alpha\Delta A_{\mathrm{FR}}).
\]

So the log-ratio is

\[
\boxed{
\log\frac{w^+}{w^-} = \alpha\Delta A_{\mathrm{FR}}.
}
\]

This directly connects the AFR diagnostic to AWR temperature sensitivity.

If \(\Delta A_{\mathrm{FR}}\) is large and stable, the algorithm is easy to extract with many values of \(\alpha\). If \(\Delta A_{\mathrm{FR}}\) is weak or noisy, the policy extractor becomes sensitive to \(\alpha\).

---

## Why this is a fair common signal

The four algorithms use different advantage semantics, but all should rank dataset-reachable future goals above random goals if their advantages are useful.

| Algorithm | Advantage semantics | AFR expectation |
|---|---|---|
| CRL | reachability-lift / future occupancy ratio | future goals should strongly outrank random goals |
| QRL | quasimetric distance reduction | future goals should be closer / easier than random goals |
| GCIVL | value progress, roughly negative-distance progress | future goals should yield positive value progress |
| GCIQL | bootstrapped `Q - V` margin | reliable margins should rank reachable goals above random goals |

This diagnostic is especially important for CRL, because CRL estimates discounted future reachability rather than Euclidean distance. Therefore, distance-return correlation is a mismatched diagnostic for CRL, while future-vs-random separation matches its learning objective.

---

## Conceptual JAX implementation

Use this when you already have `adv_plus` and `adv_minus`.

```python
def afr_metrics_from_advantages(adv_plus, adv_minus, alpha=None, eps=1e-8):
    """
    Compute future-random advantage separation diagnostics.

    Args:
        adv_plus:  [B] advantages for future/geometric goals g+.
        adv_minus: [B] or [B, K] advantages for random goals g-.
        alpha: optional AWR temperature. If provided, logs AWR ratio metrics.
        eps: numerical stabilizer.

    Returns:
        Dict of scalar metrics.
    """
    adv_plus = jnp.asarray(adv_plus)
    adv_minus = jnp.asarray(adv_minus)

    if adv_minus.ndim == 1:
        gap = adv_plus - adv_minus
    elif adv_minus.ndim == 2:
        # adv_minus: [B, K]. Broadcast adv_plus over K random negatives.
        gap = adv_plus[:, None] - adv_minus
    else:
        raise ValueError(f"adv_minus must have ndim 1 or 2, got {adv_minus.ndim}")

    gap_mean = jnp.mean(gap)
    gap_std = jnp.std(gap) + eps
    fr_auc = jnp.mean(gap > 0.0)
    extractability_index = gap_mean / gap_std

    metrics = {
        "afr/fr_auc": fr_auc,
        "afr/gap_mean": gap_mean,
        "afr/gap_std": gap_std,
        "afr/extractability_index": extractability_index,
    }

    if alpha is not None:
        log_weight_ratio = alpha * gap
        metrics.update({
            "afr/log_weight_ratio_mean": jnp.mean(log_weight_ratio),
            "afr/log_weight_ratio_std": jnp.std(log_weight_ratio),
            "afr/frac_weight_ratio_gt_10": jnp.mean(log_weight_ratio > jnp.log(10.0)),
            "afr/frac_weight_ratio_gt_100": jnp.mean(log_weight_ratio > jnp.log(100.0)),
        })

    return metrics
```

---

## OGBench-style helper function

This assumes you can provide an algorithm-specific function with signature:

```python
def advantage_fn(params, batch, goals):
    """Return A_alg(s, a, goals) for each row in batch."""
    ...
```

The batch must contain:

- `observations`
- `actions`
- a future/geometric goal array, e.g. `actor_goals`, `value_goals`, or `goals`

For random goals, the cheapest approximation to \(p^D_{\mathrm{rand}}\) is to shuffle the future goals inside the batch. If your dataloader already has random goals, use those instead.

```python
import jax
import jax.numpy as jnp


def _tree_index(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def _tree_first_dim(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("Goal tree has no leaves.")
    return leaves[0].shape[0]


def compute_afr_metrics_ogbench(
    params,
    batch,
    rng,
    advantage_fn,
    alpha=None,
    future_goal_key="actor_goals",
    random_goal_key=None,
    num_random_goals=1,
    eps=1e-8,
):
    """
    Future-random advantage separation diagnostic for OGBench-style batches.

    Args:
        params:
            Agent/network parameters.
        batch:
            OGBench batch. Must contain observations/actions and a future goal key.
        rng:
            JAX PRNGKey.
        advantage_fn:
            Callable: advantage_fn(params, batch, goals) -> [B]
            This should compute the same actor advantage used by AWR.
        alpha:
            Optional AWR temperature.
        future_goal_key:
            Batch key containing geometric/future goals. Common candidates:
            "actor_goals", "value_goals", or "goals".
        random_goal_key:
            Optional batch key for random goals. If None, random goals are approximated
            by shuffling future goals inside the batch.
        num_random_goals:
            Number of random negatives per transition. If >1, computes [B, K] gaps.
        eps:
            Numerical stabilizer.

    Returns:
        Dict of scalar diagnostics.
    """
    if future_goal_key not in batch:
        raise KeyError(
            f"future_goal_key={future_goal_key!r} not found in batch. "
            f"Available keys: {list(batch.keys())}"
        )

    g_plus = batch[future_goal_key]
    batch_size = _tree_first_dim(g_plus)

    # A(s, a, g+)
    adv_plus = advantage_fn(params, batch, g_plus)  # [B]

    if random_goal_key is not None and random_goal_key in batch:
        g_minus_base = batch[random_goal_key]
        adv_minus = advantage_fn(params, batch, g_minus_base)
        return afr_metrics_from_advantages(
            adv_plus=adv_plus,
            adv_minus=adv_minus,
            alpha=alpha,
            eps=eps,
        )

    # Approximate p_rand^D by shuffling future goals within the batch.
    # This is usually sufficient for a lightweight diagnostic.
    rngs = jax.random.split(rng, num_random_goals)
    adv_minus_list = []

    for r in rngs:
        perm = jax.random.permutation(r, batch_size)
        g_minus = _tree_index(g_plus, perm)
        adv_minus_list.append(advantage_fn(params, batch, g_minus))  # [B]

    if num_random_goals == 1:
        adv_minus = adv_minus_list[0]
    else:
        adv_minus = jnp.stack(adv_minus_list, axis=1)  # [B, K]

    return afr_metrics_from_advantages(
        adv_plus=adv_plus,
        adv_minus=adv_minus,
        alpha=alpha,
        eps=eps,
    )
```

---

## 6. Algorithm-specific advantage functions

You can implement `advantage_fn` by reusing the exact code path that currently logs `advantage/actor`.

Pseudo-definitions:

```python
# GCIQL
# A(s,a,g) = min(Q1,Q2)(s,a,g) - V(s,g)

def gciql_advantage_fn(params, batch, goals):
    q1, q2 = critic_apply(params["critic"], batch["observations"], batch["actions"], goals)
    q = jnp.minimum(q1, q2)
    v = value_apply(params["value"], batch["observations"], goals)
    return q - v
```

```python
# GCIVL
# A(s,a,g) = V(s_next,g) - V(s,g)

def gcivl_advantage_fn(params, batch, goals):
    v = value_apply(params["value"], batch["observations"], goals)
    nv = value_apply(params["value"], batch["next_observations"], goals)
    return nv - v
```

```python
# QRL
# A(s,a,g) = d(s,g) - d(s_next,g), or the exact QRL actor-advantage path.
# If your QRL actor uses a latent dynamics model, replace next_observations
# with the predicted next latent state.

def qrl_advantage_fn(params, batch, goals):
    d_sg = qmetric_apply(params["qmetric"], batch["observations"], goals)
    d_next_g = qmetric_apply(params["qmetric"], batch["next_observations"], goals)
    return d_sg - d_next_g
```

```python
# CRL
# A(s,a,g) = f(s,a,g) - fV(s,g)

def crl_advantage_fn(params, batch, goals):
    f_sa_g = crl_q_apply(params["critic"], batch["observations"], batch["actions"], goals)
    f_v_g = crl_v_apply(params["value"], batch["observations"], goals)
    return f_sa_g - f_v_g
```

These are schematic. In the actual OGBench code, use the existing Flax/JAX `apply` calls and names from the agent implementation.

---

## 7. What to log

At each evaluation/logging step, log:

```python
metrics.update(compute_afr_metrics_ogbench(
    params=params,
    batch=batch,
    rng=rng,
    advantage_fn=agent_advantage_fn,
    alpha=config["alpha"],
    future_goal_key="actor_goals",   # adapt if needed
    random_goal_key=None,            # or use an existing random-goal key
    num_random_goals=8,
))
```

Recommended fields:

- `afr/fr_auc`
- `afr/gap_mean`
- `afr/gap_std`
- `afr/extractability_index`
- `afr/log_weight_ratio_mean`
- `afr/frac_weight_ratio_gt_10`
- `afr/frac_weight_ratio_gt_100`

---

## 8. Expected interpretation

### High FR-AUC and high EI

The advantage function cleanly separates reachable future goals from random goals. AWR extraction should be robust, because the ranking is already strong before exponentiation.

### High FR-AUC but low EI

The sign is mostly correct, but the margin is weak or noisy. This may still require careful tuning of \(\alpha\).

### Low FR-AUC or negative EI

The advantage signal is unreliable. AWR may become brittle because \(\alpha\) is forced to compensate for a weak ranking signal.

---

## 9. Hypothesis for your experiments

Expected pattern:

| Method | Expected AFR behavior | Landscape implication |
|---|---|---|
| CRL | high future-random separation, even if distance correlation is weak | stable AWR because reachability ranking is extractable |
| QRL | good separation, possibly high saturation sensitivity | stable but potentially alpha-sensitive |
| GCIVL | good separation through local value progress | stable broad-positive regime |
| GCIQL | weaker/noisier separation due to bootstrapped `Q - V` margin | stronger alpha/lr interaction |

---

## 10. Paper-ready wording

> We evaluate advantage reliability using a future-random advantage separation diagnostic. For each transition, we compare the algorithm's predicted advantage for a geometrically sampled future goal against a random dataset goal. This diagnostic is common across methods because all offline GCRL objectives rely on self-supervised goal relabeling, yet it does not privilege distance-based methods over contrastive methods. The resulting gap directly predicts AWR extractability, since the AWR weight ratio between future and random goals is \(\exp(\alpha \Delta A_{\mathrm{FR}})\). Thus, large and stable future-random gaps indicate that policy extraction should be less sensitive to the AWR temperature.
