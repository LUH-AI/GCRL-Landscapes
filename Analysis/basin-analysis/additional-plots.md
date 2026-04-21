# Advantage Weighted Regression (AWR) Diagnostics

## 1. Plot $\alpha A$, not just $A$

AWR depends on

$$w = \min(e^{\alpha A}, 100),$$

so the actor really sees

$$z = \alpha A.$$

In $z$-space, the thresholds are universal:
* $z < 0$: downweighted
* $0 < z \le \log 100$: active, unsaturated
* $z > \log 100$: clipped

This is much cleaner than using $A$ plus a median $A_{\text{clip}}$.

So I would plot the distribution of

$$z = \alpha A$$

for each method, environment, and phase. That removes ambiguity about whether the effect comes from $A$ or $\alpha$.

---

## 2. Condition on the two GCIQL basins

This is the most important diagnostic for the mobility plot. Take the two good GCIQL regions:
* the **upper** basin: moderate learning rate, high $\alpha$
* the **lower** basin: tiny learning rate, tiny $\alpha$

Then compute, **separately for each basin and phase**:

$$p_{-} = \Pr(z < 0),$$
$$p_{\text{mid}} = \Pr(0 < z \le \log 100),$$
$$p_{\text{sat}} = \Pr(z > \log 100),$$
$$\text{ESS}.$$

This will tell you whether the two basins are genuinely different operating regimes. That is much more informative than plotting median region fractions over all configs.

---

## 3. Plot weight mass by region, not just sample count by region

Right now the bars count how many samples fall into each region. But the actor update depends on **weights**, not counts.

So define normalized weights

$$\tilde{w}_i = \frac{w_i}{\sum_j w_j},$$

and compute

$$W_{-} = \sum_i \tilde{w}_i \mathbf{1}[z_i < 0],$$
$$W_{\text{mid}} = \sum_i \tilde{w}_i \mathbf{1}[0 < z_i \le \log 100],$$
$$W_{\text{sat}} = \sum_i \tilde{w}_i \mathbf{1}[z_i > \log 100].$$

This is probably the clearest actor-side plot you can make.

**Why?** Because GCIQL may have only 10% of samples in the clipped region, but those 10% might carry almost all the actor weight. Sample fractions cannot show that.

---

## 4. Plot a two-dimensional regime map

For each config, make a point with

$$x = p_{+} = \Pr(z > 0), \quad y = s = \Pr(z > \log 100 \mid z > 0),$$

and color it by return.

**Interpretation:**
* $x$: how many winners exist
* $y$: how saturated the winner set is

**Then:**
* GCIQL should occupy low-$x$, possibly high-$y$ regions
* GCIVL/QRL should occupy high-$x$ regions
* CRL should sit somewhere balanced in the middle

This would make the algorithm differences much more visually obvious.
