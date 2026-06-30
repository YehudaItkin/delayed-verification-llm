# System Identification — Experiment A: Summary

**Script:** `paper/remote/sysid_expA.py`  
**Data:** `paper/remote/expA_results.json` (64 runs: 8 questions × 2 κ × 2 δ × 2 seeds)  
**Model:** scalar delayed-consensus recurrence  `e_{t+1} = a·e_t + b·e_{t−δ} + c + noise`  
**Stability criterion:** dominant root magnitude ρ̂ of `z^{δ+1} − a·z^δ − b = 0`; predicted UNSTABLE iff ρ̂ ≥ 0.98

---

## 1. Fit quality

OLS rows per run: **16** (δ=1) and **13** (δ=4; small-sample caveat).

| Subset | n runs | mean R² | median R² | range |
|--------|--------|---------|-----------|-------|
| All non-flat runs | 25 | 0.442 | 0.325 | 0.014 – 1.000 |
| δ=1, non-flat | 8 | 0.435 | 0.318 | 0.048 – 0.960 |
| δ=4, non-flat | 17 | 0.445 | 0.325 | 0.014 – 1.000 |

**39 of 64 runs are flat** (mean trajectory converges to a constant ≤ 0.02 within 1–2 rounds). Their OLS fit returns a ≈ 0, b ≈ 0, c ≈ const, ρ̂ ≈ 0 (trivially predicted STABLE). Of these 39, **38 are true negatives (TN)**; run #8 is flat (osc=0) but ends at final error = 1.0, so it is labelled observed-oscillating via the `final ≥ 0.3` rule while predicted STABLE — a **false negative (FN)**. R² is undefined (0/0) for all flat runs. All 64 runs contribute to the confusion matrix and cell aggregation; only the 25 non-flat runs yield meaningful R² values.

---

## 2. Stability prediction vs. observed oscillation

**Observed label:** osc ≥ 2 OR final ≥ 0.3 → "oscillating" (n = 21/64 runs).  
**Predicted UNSTABLE:** ρ̂ ≥ 0.98.

| | obs = OSC | obs = conv |
|---|---|---|
| **pred = UNSTABLE** | 5 (TP) | 0 (FP) |
| **pred = STABLE** | 16 (FN) | 43 (TN) |

- **Overall agreement: 48/64 = 75.0%**, Precision = 1.00, Recall = 0.24.
- The model **never false-positives** (0 FP): every run predicted unstable was observed to oscillate.
- The model **misses 16 of 21 oscillating runs** (FN). Of these 16, at least 6 end at high final error (~0.87–1.00) — genuine non-convergence (stuck near maximal error), not transient per-round wiggle — which the ρ̂ ≥ threshold rule failed to flag. The identified model has perfect precision (no FP: every run it labels unstable is truly oscillating) but low recall (0.24): it fails to detect sustained high-error / non-convergence cases.

**On non-flat runs only** (n=25, the set where the fit is non-trivial):
- Agreement = 10/25 = 40.0%; TP=5, FP=0, FN=15, TN=5. Precision = 1.00, Recall = 0.25.

**Robustness to threshold choice** (all 64 runs): agreement ranges 70–77% across four alternative rules, with Spearman r_s = 0.69–0.76 — the conclusion is stable.

---

## 3. Rank correlations (rho_hat vs. observed measures)

| Measure | Subset | Spearman r_s | p-value |
|---------|--------|-------------|---------|
| osc_raw | all 64 runs | **0.796** | < 0.0001 |
| final error | all 64 runs | 0.365 | 0.003 |
| obs_label | all 64 runs | **0.714** | < 0.0001 |
| osc_raw | non-flat only (n=25) | 0.306 | 0.137 |
| final error | non-flat only (n=25) | 0.237 | 0.254 |

The all-runs correlations are strong but **partially inflated** by the 39 trivially-flat runs that all have ρ̂ ≈ 0 and osc = 0. Among the 25 non-flat runs — where prediction is non-trivial — the Spearman r_s drops to 0.31 (p = 0.14), consistent with a weak positive trend but not individually significant given n = 25.

---

## 4. Cell-level structure

Aggregated by (κ, δ): seeds averaged within question, then across 8 questions (n_q = 8 per cell, avoiding pseudoreplication).

| Cell | mean a | mean b | mean ρ̂ | mean R² | flat runs |
|------|--------|--------|--------|---------|-----------|
| (strong, δ=1) | +0.031 | −0.071 | 0.111 | 0.427 | 13/16 |
| (strong, δ=4) | −0.144 | +0.017 | 0.415 | 0.438 | 8/16 |
| (weak, δ=1) | +0.106 | −0.062 | 0.133 | 0.441 | 11/16 |
| (weak, δ=4) | −0.029 | +0.170 | 0.474 | 0.451 | 7/16 |

**Theory-consistent orderings:**

| Check | Expected by theory | Observed | Met? |
|-------|-------------------|----------|------|
| b < 0 (restoring), both δ=1 cells | b_strong < 0, b_weak < 0 | −0.071, −0.062 | **YES** |
| \|b_strong\| > \|b_weak\| at δ=1 | |b| larger for stronger κ | \|−0.071\| > \|−0.062\| | **YES** |
| b < 0, both δ=4 cells | b_strong < 0, b_weak < 0 | +0.017, +0.170 | **NO** |
| ρ̂ increases with δ (strong) | ρ̂(d=4) > ρ̂(d=1) | 0.415 > 0.111 | **YES** |
| ρ̂ increases with δ (weak) | ρ̂(d=4) > ρ̂(d=1) | 0.474 > 0.133 | **YES** |
| Largest ρ̂ at (strong, δ=4) | max ρ̂ at (strong, δ=4) | max is (weak, δ=4) | **NO** |

The δ=4 cells have positive mean b (b ≈ +0.017 to +0.170), violating the theory's prediction b = −η·κ < 0. This is likely a fitting artefact: with only 13 regression rows and quantised discrete trajectories, the OLS constraint is under-determined, and the sign of b is not stably estimated. The delay-ordering for ρ̂ is correct (longer delay → larger ρ̂), but the kappa-ordering reverses.

---

## 5. Implied effective gains

| Cell | a | b | η·μ = 1−a | η·κ = −b | b < 0 (restoring)? |
|------|---|---|-----------|---------|-------------------|
| (strong, δ=1) | +0.031 | −0.071 | +0.969 | +0.071 | ✓ |
| (strong, δ=4) | −0.144 | +0.017 | +1.144 | −0.017 | ✗ |
| (weak, δ=1) | +0.106 | −0.062 | +0.894 | +0.062 | ✓ |
| (weak, δ=4) | −0.029 | +0.170 | +1.029 | −0.170 | ✗ |

For δ=1 cells: a is small positive, η·μ ≈ 0.9, and η·κ is positive and larger for strong kappa, consistent with theory. For δ=4 cells: b turns positive (wrong sign) in both cells, which is inconsistent with the theory interpretation of b as a restoring delay term.

---

## 6. Honest assessment

The results offer **partial, cautious support** for the paper's delayed-consensus model, with three clear limitations:

**Supportive evidence:**
- ρ̂ has 0 false positives: every run flagged as predicted-UNSTABLE was observed to oscillate.
- The broad Spearman r_s(ρ̂, osc_raw) = 0.796 across all 64 runs shows that ρ̂ tracks oscillation count monotonically.
- The delay-ordering for ρ̂ is correct: longer δ → larger ρ̂ in both κ cells.
- The b-coefficient sign and magnitude ordering are correct at δ=1 (b < 0, |b_strong| > |b_weak|).

**Limitations and concerns:**
- The all-runs Spearman is **partially inflated** by 39 trivially-flat runs; on the 25 non-flat runs the correlation is r_s = 0.31 (p = 0.14), not individually significant.
- The model **misses 16/21 oscillating runs** (recall = 0.24): the 1-D mean trajectory often looks stable even when individual rounds oscillate.
- The b-coefficients at δ=4 are **wrong-sign** (+0.017, +0.170 rather than negative), suggesting the coarse linear fit breaks down with large delay and quantised trajectories.
- **T=18 is very short** (especially δ=4: 13 rows), trajectories are coarsely quantised (the mean error takes ~77 distinct values across all runs, with each individual trajectory spanning only a small subset), and the fit is on mean trajectories over 7 agents. These are coarse estimates, not precision measurements.
- The model is 1-D and linear; LLM debate dynamics are nonlinear, discrete, and stochastic.

**Conclusion:** The coarse dynamics of debate runs with longer delay (δ=4) show larger ρ̂ and more frequent oscillation, consistent with the paper's stability prediction. At δ=1, the b-sign ordering also matches theory. These are supporting observations — the agents' coarse dynamics are *not inconsistent* with the model — but the short trajectories, quantisation artefacts, and low recall preclude claiming the agents obey the model.

---

*Generated by `paper/remote/sysid_expA.py` on 2026-06-22.*
