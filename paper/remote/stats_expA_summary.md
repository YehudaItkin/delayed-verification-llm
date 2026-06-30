# Experiment A — Corrected Statistical Summary

**Data:** `paper/remote/expA_results.json`  
**Analysis:** `paper/remote/stats_expA.py`  
**Unit of replication:** question (n=8 after averaging the 2 seeds per cell)  
**WARNING throughout: n=8 provides low power; interpret all p-values and CIs accordingly.**

---

## 1. Question-level cell means (seed-aggregated)

| Cell                  | mean osc | SE osc | mean final  | mean conv |
|-----------------------|----------|--------|-------------|-----------|
| kappa=weak,  delta=1  | 0.9375   | 0.6299 | 0.181022    | 0.8125    |
| kappa=weak,  delta=4  | 2.3125   | 1.1454 | 0.181644    | 0.8125    |
| kappa=strong, delta=1 | 0.5625   | 0.5625 | 0.126706    | 0.8750    |
| kappa=strong, delta=4 | 3.1875   | 1.2533 | 0.126691    | 0.8750    |

SE = SD/sqrt(8) at the question level (this corrects the original pseudoreplication of n=16).

---

## 2. Oscillation-index threshold sensitivity

Oscillation index recomputed from `mean_traj` at four thresholds θ:

| θ     | weak,δ=1 | weak,δ=4 | strong,δ=1 | strong,δ=4 |
|-------|----------|----------|------------|------------|
| 0.01  | 0.9375   | 2.5000   | 0.6875     | 3.1875     |
| 0.02  | 0.9375   | 2.3125   | 0.5625     | 3.1875     |
| 0.03  | 0.9375   | 2.3125   | 0.5000     | 3.0625     |
| 0.05  | 0.8125   | 2.3125   | 0.5000     | 3.0000     |

Delta effect (osc at δ=4 minus osc at δ=1):

| θ     | weak effect | strong effect |
|-------|-------------|---------------|
| 0.01  | +1.5625     | +2.5000       |
| 0.02  | +1.3750     | +2.6250       |
| 0.03  | +1.3750     | +2.5625       |
| 0.05  | +1.5000     | +2.5000       |

**The direction of the delta effect (delta=4 increases oscillation) is consistent and positive
across all four thresholds for both kappa levels.** The effect magnitude varies modestly
(weak: 1.4–1.6; strong: 2.5–2.6), indicating threshold-robustness in sign and approximate
magnitude.

---

## 3. Within-question paired tests (n=8)

### kappa=weak

Paired differences d_q = osc(δ=4) − osc(δ=1) per question:
`+0.5, +1.0, +1.5, 0.0, −0.5, −2.0, +10.0, +0.5`

| Statistic                  | Value  |
|----------------------------|--------|
| Mean(d_q)                  | 1.3750 |
| Median(d_q)                | 0.5000 |
| SD(d_q)                    | 3.6425 |
| Wilcoxon W (n_nonzero=7)   | 8.0    |
| Wilcoxon p (two-sided)     | 0.3750 |
| Sign-flip permutation p    | 0.4408 |

**Not significant.** The large variance (driven by Q6, d=+10.0) means the data cannot rule out
a null effect.

### kappa=strong

Paired differences d_q = osc(δ=4) − osc(δ=1) per question:
`0.0, +0.5, 0.0, 0.0, +3.5, +2.0, +6.5, +8.5`

| Statistic                  | Value  |
|----------------------------|--------|
| Mean(d_q)                  | 2.6250 |
| Median(d_q)                | 1.2500 |
| SD(d_q)                    | 3.2923 |
| Wilcoxon W (n_nonzero=5)   | 0.0    |
| Wilcoxon p (two-sided)     | 0.0625 |
| Sign-flip permutation p    | 0.0577 |

**Marginally not significant** (Wilcoxon p=0.0625, permutation p=0.0577), but the sign is
consistent: all non-zero differences are positive. The Wilcoxon result is at the boundary of
the minimum achievable p-value for n_nonzero=5 (minimum possible two-sided p ≈ 0.0625).

---

## 4. (kappa × delta) Interaction — Difference-in-Differences

DiD_q = [osc(strong,δ4) − osc(strong,δ1)] − [osc(weak,δ4) − osc(weak,δ1)] per question:
`−0.5, −0.5, −1.5, 0.0, +4.0, +4.0, −3.5, +8.0`

| Statistic                  | Value  |
|----------------------------|--------|
| Mean DiD                   | 1.2500 |
| Median DiD                 | −0.2500|
| SD DiD                     | 3.7512 |
| Sign-flip permutation p    | 0.4705 |

**Not significant.** The "dose tradeoff" claim (stronger verification amplifies the delay effect)
is not supported at n=8. The DiD is highly variable across questions (range: −3.5 to +8.0),
median near zero, and the permutation p is 0.47. The effect may exist but would require
substantially more questions to detect.

---

## 5. Cluster bootstrap 95% percentile CIs

### Cell mean osc

| Cell                  | Observed | CI 2.5% | CI 97.5% |
|-----------------------|----------|---------|----------|
| kappa=weak,  δ=1      | 0.9375   | 0.0000  | 2.2500   |
| kappa=weak,  δ=4      | 2.3125   | 0.7500  | 4.6875   |
| kappa=strong, δ=1     | 0.5625   | 0.0000  | 1.6875   |
| kappa=strong, δ=4     | 3.1875   | 0.9375  | 5.5625   |

### Delta effect (δ=4 minus δ=1)

| kappa  | Observed | CI 2.5% | CI 97.5% |
|--------|----------|---------|----------|
| weak   | 1.3750   | −0.4375 | 4.0625   |
| strong | 2.6250   | 0.6250  | 4.8750   |

**Wide CIs reflect the small n.** The strong-kappa delta effect CI just excludes zero (0.625
to 4.875); the weak-kappa effect CI includes zero. These should be treated as preliminary
estimates, not confirmatory.

---

## 6. Endpoint check

| Cell                  | mean final  | mean conv |
|-----------------------|-------------|-----------|
| kappa=weak,  δ=1      | 0.181022    | 0.8125    |
| kappa=weak,  δ=4      | 0.181644    | 0.8125    |
| kappa=strong, δ=1     | 0.126706    | 0.8750    |
| kappa=strong, δ=4     | 0.126691    | 0.8750    |

Endpoint quality (`final` error and convergence rate) is essentially identical across delta
values within each kappa level. The delta effect is thus in the dynamics (oscillations during
traversal), not the endpoint.

---

## Honest interpretation

The analysis corrects pseudoreplication: the true replication unit is n=8 questions (not
n=16), increasing uncertainty substantially relative to the original report.

**What the data support:**
- The direction of the delta effect is robust and consistent: a longer delay (δ=4) produces
  more oscillations than a short delay (δ=1) across both kappa levels and all tested
  thresholds. This directional finding is replicated in every question-level analysis.
- Under strong verification (kappa=strong), the delta effect approaches but does not reach
  conventional significance: Wilcoxon p=0.0625, permutation p=0.0577, bootstrap CI
  [0.63, 4.88]. This is the most consistent signal in the data.
- Endpoint quality (final error, convergence) does not differ by delta, indicating the
  delay effect is a transient dynamic phenomenon.

**What the data do NOT support:**
- The (kappa × delta) interaction (the "dose tradeoff" claim that strong verification
  amplifies the delay effect) is not statistically supported: DiD mean=1.25, p=0.47.
  The per-question DiDs are highly variable and the median is near zero.
- The weak-kappa delta effect is not significant (p≈0.44); one outlier question (Q6,
  d=+10) dominates the mean.
- Any strong quantitative claim about effect size: bootstrap CIs are wide due to n=8.

**Power and reliability concerns:**
- n=8 questions is the fundamental bottleneck. At n=8, the Wilcoxon test has only
  ~23 distinct achievable p-values; the minimum two-sided p for n_nonzero=5 is 0.0625.
- All results should be treated as exploratory/preliminary. Replication with more questions
  is required before drawing confirmatory conclusions.
