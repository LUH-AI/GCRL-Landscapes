# Advantage Trustworthiness Tests for AWR

## Goal
Measure whether **positive advantages are trustworthy**, i.e.:
> Do they induce a **correct, stable, and meaningful ranking** of actions for policy extraction?

We evaluate this through **ranking correctness, stability, and signal quality**.

---

## 1. Notation
Let:
* $A(s, a, g)$: advantage
* $\alpha$: AWR temperature
* $z = \alpha A$
* $w = \min(e^z, 100)$

---

## 2. Ranking Correctness

### 2.1 Top-k Precision
Let:
* $T_k^A$: top-k samples by advantage
* $T_k^R$: top-k samples by return/progress

$$\text{Precision@k} = \frac{|T_k^A \cap T_k^R|}{k}$$

**Interpretation:**
* High $\rightarrow$ advantages select truly good actions
* Low $\rightarrow$ noisy or misleading signal

### 2.2 Weighted Correlation
$$\text{Corr}(w, R)$$
or
$$\text{Spearman}(w, R)$$

**Interpretation:**
* Measures whether AWR weights emphasize useful transitions

---

## 3. Ranking Stability

### 3.1 Top-k Overlap
For configs/phases $i, j$:

$$\text{Overlap}_k(i, j) = \frac{|T_k^{(i)} \cap T_k^{(j)}|}{k}$$

**Interpretation:**
* High $\rightarrow$ stable winner set
* Low $\rightarrow$ winner churn (instability)

### 3.2 Rank Correlation
$$\rho = \text{Spearman}(A^{(i)}, A^{(j)})$$

**Interpretation:**
* Measures global ranking stability

---

## 4. AWR Signal Geometry

### 4.1 Region Definitions
Let:
$$z_{\text{clip}} = \log 100$$

**Regions:**
* Negative: $z < 0$
* Active: $0 \le z \le z_{\text{clip}}$
* Saturated: $z > z_{\text{clip}}$

### 4.2 Sample Fractions
$$p_{-} = \Pr(z < 0)$$
$$p_{\text{mid}} = \Pr(0 \le z \le z_{\text{clip}})$$
$$p_{\text{sat}} = \Pr(z > z_{\text{clip}})$$

### 4.3 Weight Mass
Normalize weights:
$$\tilde{w}_i = \frac{w_i}{\sum_j w_j}$$

Then:
$$W_{-} = \sum_i \tilde{w}_i \mathbf{1}[z_i < 0]$$
$$W_{\text{mid}} = \sum_i \tilde{w}_i \mathbf{1}[0 \le z_i \le z_{\text{clip}}]$$
$$W_{\text{sat}} = \sum_i \tilde{w}_i \mathbf{1}[z_i > z_{\text{clip}}]$$

**Interpretation:**
* $W_{\text{mid}}$: useful gradient signal
* $W_{\text{sat}}$: saturated winner mass
* $W_{-}$: ignored samples

### 4.4 Effective Sample Size (ESS)
$$\text{ESS} = \frac{(\sum_i w_i)^2}{N \sum_i w_i^2}$$

**Interpretation:**
* Low ESS $\rightarrow$ sparse, brittle learning
* High ESS $\rightarrow$ diffuse, stable learning

---

## 5. Noise-to-Signal Ratio (NSR)
Cluster samples by similar state:

$$\text{NSR} = \frac{\mathbb{E}[\text{Var}(A \mid \text{cluster})]}{\text{Var}(A)}$$

**Interpretation:**
* Low NSR $\rightarrow$ consistent local signal
* High NSR $\rightarrow$ noisy advantage

---

## 6. AWR Sensitivity

### Policy Sensitivity to Alpha
For $\alpha_i, \alpha_j$:
$$S = \text{KL}(\pi_{\alpha_i} \| \pi_{\alpha_j})$$
where:
$$\pi_{\alpha} \propto w(\alpha)$$

**Interpretation:**
* High $\rightarrow$ brittle extraction
* Low $\rightarrow$ robust extraction

---

## 7. Critic-Side Metrics (GCIQL)

### 7.1 Twin-Q Disagreement
$$D_Q = \mathbb{E}[\|Q_1 - Q_2\|]$$

### 7.2 Bellman Residual
$$R_Q = \mathbb{E}[\|Q - (r + \gamma V')\|]$$

### 7.3 Margin Reliability
$$M = \mathbb{E}\left[ \frac{(Q - V)_+}{\|Q_1 - Q_2\| + \epsilon} \right]$$

**Interpretation:**
* High $\rightarrow$ reliable positive advantages
* Low $\rightarrow$ noisy or fragile signal

---

## 8. Minimal High-Impact Test Set
If compute is limited, run:
* **Test 1:** Top-k Overlap
* **Test 2:** Spearman Rank Correlation
* **Test 3:** $\text{Corr}(\exp(\alpha A), \text{Return})$

---

## 9. Core Hypothesis
> **Trustworthy advantages induce stable and correct rankings, not just positive mass**

---

## 10. Expected Outcomes

| Method | Behavior |
| :--- | :--- |
| **GCIQL** | Sparse, unstable winners |
| **CRL** | Moderate but reliable winners |
| **GCIVL** | Broad positive support |
| **QRL** | Broad but less precise ranking |

---

## 11. Key Insight
> **Stability comes from ranking robustness, not number of positive advantages**
