# Pre-registration: does delayed verification destabilize *grounded factual* multi-agent QA?

Written **before** inspecting any amplitude number (2026-06-23). Decides the paper's title/framing:
keep "Factuality" (if rectified factual oscillation is real) vs. retitle to "Belief/Consensus"
(if grounded factual QA is genuinely stable). Resolves red-team findings M1 (title) and M7
(absorbing-boundary overclaim) jointly. See [[project-redteam-blockers]].

## Theory-derived hypothesis (from the projected/absorbing map of Remark 1)
The corrected factual state obeys `p_{t+1} = max(0, p_t − α·σ(p_{t−δ}) + g)`. The absorbing boundary at
truth (p=0) cancels a limit cycle **only if** the cycle would cross zero. When the fault forcing `g`
holds the equilibrium **away** from truth, the (rectified) cycle lives entirely in p>0 and survives.

**H1 (forcing-gated factual oscillation).** Steady-state magnitude oscillation amplitude `A` of the
consensus error increases with delay `δ` **under strong forcing**, stays at the noise floor under weak
forcing, and is at the noise floor at `δ=0` for all forcing. I.e. the δ-effect on `A` is *gated* by
forcing.

## Primary metric (period-independent, fixed window)
`A(run) = std( mean_traj[ len//2 : ] )` — standard deviation of the steady-state tail (second half) of
the consensus NLI-distance trajectory. std is per-sample, so it is robust to window length and to the
period-vs-window confound that invalidated the earlier `osc_index`. Secondary: peak-to-peak of the same
tail; dominant FFT frequency of the mean-subtracted tail vs. the predicted period `2π/θ⋆(δ)`.

Noise floor `N` = `A` in converged/low-forcing cells (weak verifier, δ=1) and (in the confirmatory run)
δ=0 and temperature-0 cells.

## Confirmatory experiment F1 (to be RUN, pre-registered)
2D sweep `δ ∈ {0,1,2,4,6}` × `forcing ∈ {off, weak, strong}` × `temp ∈ {0.7, 0.0}`; T=24, n=30
PsiloQA questions the model errs on cold, 3 seeds; Qwen3.6-35B primary, replicate on 2 families if
positive. Controls: δ=0 placebo, verifier-off, temp=0. Coordinate: primary = non-negative NLI distance
(magnitude); confirmatory = signed two-pole projection `s_t=(d_wrong−d_gold)/(d_wrong+d_gold)∈[−1,1]`
to show one-sidedness.

### Decision rule (binding)
Factual oscillation is **REAL** (→ keep "Factuality"; M7 becomes the masked-instability result) iff ALL:
1. under strong forcing, `A(δ)` increases monotonically in δ (one-sided Jonckheere–Terpstra, p<0.05);
2. `A(δ=6, strong) > N + 2·SD(N)`;
3. `A(δ=0) ≈ N` for all forcing (placebo flat) **and** `A` survives temp=0.

Otherwise → grounded factual QA is **stable** (absorbing boundary works) → retitle to
"Belief/Consensus", factuality = the stabilized contrast case. **Both outcomes are publishable; the
experiment selects the framing, it cannot "fail."**

## Step 0 (EXPLORATORY — analysis of already-collected logs, NOT confirmatory)
Because these logs already exist, Step-0 analysis is exploratory pre-evidence only. It checks whether
the forcing-gated δ-scaling signature is *already visible* in `expA_big/expA_results` (weak/strong ×
δ{1,4}), `expA_push/_tqa` (strong/forceful × δ{1,6}), and `debate_exp_results` (grounded), against the
weak/converged floor. It cannot kill the rephrasing-noise confound (no temp=0) or test δ=0. A positive
Step-0 signal justifies running F1; it does not by itself decide the title.
