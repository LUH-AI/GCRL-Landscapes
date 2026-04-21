## 10. The next decisive test

You now need one ranking-stability plot.

For a fixed batch, compute the top-$k$ winner set by $z = \alpha A$:

$$T_k = \text{TopK}_i z_i.$$

Then compute overlap across nearby configs or phases:

$$\text{Overlap}_k(i, j) = \frac{|T_k^{(i)} \cap T_k^{(j)}|}{k}.$$

### Prediction:

$$\text{Overlap}^{\text{CRL}} > \text{Overlap}^{\text{GCIQL}}.$$

### Potentially:

$$\text{Overlap}^{\text{CRL}} \ge \text{Overlap}^{\text{QRL}}$$

under alpha perturbations.

That would directly validate the claim that CRL has **reliable winners**, not merely moderate positive mass.
