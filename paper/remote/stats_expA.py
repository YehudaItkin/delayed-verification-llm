"""
stats_expA.py — Robust statistical analysis for Experiment A.

Fixes the pseudoreplication issue in the original analysis:
  - The 2 seeds per (question, kappa, delta) are NOT independent.
  - Unit of replication = question (n=8).
  - We first aggregate seeds, then do all inference at question level.

Analyses:
  1. Seed aggregation → question level
  2. Oscillation-index threshold sensitivity (θ ∈ {0.01, 0.02, 0.03, 0.05})
  3. Within-question paired tests (Wilcoxon signed-rank + sign-flip permutation)
  4. (kappa × delta) interaction via difference-in-differences permutation test
  5. Cluster bootstrap: 95% CIs for cell means and delta effects
  6. Endpoint check: mean `final` and `conv` per cell

RNG is seeded deterministically via numpy.default_rng(0).
"""

import json
import numpy as np
from scipy import stats

# ─── Load and validate ──────────────────────────────────────────────────────

with open("paper/remote/expA_results.json") as f:
    data = json.load(f)

runs = data["runs"]
assert len(runs) == 64, f"Expected 64 runs, got {len(runs)}"

questions = sorted(set(r["q"] for r in runs))
kappas = ["weak", "strong"]
deltas = [1, 4]
seeds = [0, 1]

assert len(questions) == 8, f"Expected 8 questions, got {len(questions)}"

# Index runs: run_map[(q, kappa, delta, seed)] = run dict
run_map = {}
for r in runs:
    key = (r["q"], r["kappa"], r["delta"], r["seed"])
    assert key not in run_map, f"Duplicate run: {key}"
    run_map[key] = r

assert len(run_map) == 64, f"Expected 64 unique keys, got {len(run_map)}"
print("=== Sanity checks passed: 64 runs, 8 questions, 4 cells ===\n")

rng = np.random.default_rng(0)

# ─── 1. Seed aggregation → question level ───────────────────────────────────
# For each (question, kappa, delta) average osc, final, conv over the 2 seeds.
# Result shape: q_osc[q_idx, kappa_idx, delta_idx] etc.

n_q = len(questions)
n_k = len(kappas)
n_d = len(deltas)

q_osc   = np.zeros((n_q, n_k, n_d))
q_final = np.zeros((n_q, n_k, n_d))
q_conv  = np.zeros((n_q, n_k, n_d))

for qi, q in enumerate(questions):
    for ki, kappa in enumerate(kappas):
        for di, delta in enumerate(deltas):
            vals_osc   = [run_map[(q, kappa, delta, s)]["osc"]   for s in seeds]
            vals_final = [run_map[(q, kappa, delta, s)]["final"] for s in seeds]
            vals_conv  = [run_map[(q, kappa, delta, s)]["conv"]  for s in seeds]
            q_osc  [qi, ki, di] = np.mean(vals_osc)
            q_final[qi, ki, di] = np.mean(vals_final)
            q_conv [qi, ki, di] = np.mean(vals_conv)

print("=== 1. Question-level cell means (seed-aggregated) ===")
print(f"{'Cell':<25}  {'mean osc':>10}  {'SE osc':>8}  {'mean final':>12}  {'mean conv':>10}")
for ki, kappa in enumerate(kappas):
    for di, delta in enumerate(deltas):
        cell_osc = q_osc[:, ki, di]
        print(
            f"kappa={kappa}, delta={delta:<3}"
            f"  {cell_osc.mean():>10.4f}"
            f"  {cell_osc.std(ddof=1)/np.sqrt(n_q):>8.4f}"
            f"  {q_final[:, ki, di].mean():>12.6f}"
            f"  {q_conv[:, ki, di].mean():>10.4f}"
        )
print()

# ─── 2. Oscillation-index threshold sensitivity ─────────────────────────────
# Recompute osc from mean_traj at θ ∈ {0.01, 0.02, 0.03, 0.05}.
# osc(θ) = number of sign changes of diff(mean_traj[3:]) after discarding
#          diffs with |diff| ≤ θ.

THRESHOLDS = [0.01, 0.02, 0.03, 0.05]


def compute_osc(mean_traj: list, theta: float) -> int:
    """Sign-change count of diff(mean_traj[3:]) after filtering |diff| <= theta."""
    arr = np.array(mean_traj)
    d = np.diff(arr[3:])
    d_filt = d[np.abs(d) > theta]
    if len(d_filt) < 2:
        return 0
    signs = np.sign(d_filt)
    return int(np.sum(signs[1:] != signs[:-1]))


print("=== 2. Threshold sensitivity — question-level mean osc per cell ===")
header = f"{'theta':<8}" + "".join(
    f"  k={k},d={d}" for k in kappas for d in deltas
)
print(header)
print("-" * len(header))

threshold_delta_effects = {}  # theta -> {kappa: delta_effect mean}

for theta in THRESHOLDS:
    # Build question-level osc arrays for this theta
    osc_th = np.zeros((n_q, n_k, n_d))
    for qi, q in enumerate(questions):
        for ki, kappa in enumerate(kappas):
            for di, delta in enumerate(deltas):
                vals = [
                    compute_osc(run_map[(q, kappa, delta, s)]["mean_traj"], theta)
                    for s in seeds
                ]
                osc_th[qi, ki, di] = np.mean(vals)

    cell_means = [osc_th[:, ki, di].mean() for ki, k in enumerate(kappas) for di, d in enumerate(deltas)]
    row = f"{theta:<8.2f}" + "".join(f"  {m:8.4f}" for m in cell_means)
    print(row)

    # delta effects per kappa at this theta
    delta_effects = {}
    for ki, kappa in enumerate(kappas):
        d_q = osc_th[:, ki, 1] - osc_th[:, ki, 0]  # delta=4 minus delta=1
        delta_effects[kappa] = d_q.mean()
    threshold_delta_effects[theta] = delta_effects

print()
print("Delta effect (delta=4 minus delta=1) per kappa, per theta:")
print(f"{'theta':<8}  {'weak effect':>12}  {'strong effect':>14}")
for theta in THRESHOLDS:
    de = threshold_delta_effects[theta]
    print(f"{theta:<8.2f}  {de['weak']:>12.4f}  {de['strong']:>14.4f}")
print()
print("NOTE: Check sign consistency across thresholds to assess robustness.\n")

# ─── 3. Within-question paired tests (n=8) ──────────────────────────────────
# For each kappa: d_q = osc(q, delta=4) - osc(q, delta=1)  [question-level averages]

N_PERM = 10_000

print("=== 3. Within-question paired tests (n=8) ===")
print("WARNING: n=8 is small. Power is low. Interpret p-values cautiously.\n")

for ki, kappa in enumerate(kappas):
    d_q = q_osc[:, ki, 1] - q_osc[:, ki, 0]  # delta=4 minus delta=1

    print(f"--- kappa={kappa} ---")
    print(f"  Paired differences d_q (delta=4 - delta=1) per question:")
    for qi, q in enumerate(questions):
        print(f"    Q{qi}: {d_q[qi]:+.4f}")
    print(f"  Mean(d_q) = {d_q.mean():.4f}")
    print(f"  Median(d_q) = {np.median(d_q):.4f}")
    print(f"  SD(d_q) = {d_q.std(ddof=1):.4f}")

    # Wilcoxon signed-rank test
    # 'zero_method="wilcox"' (default) drops zeros from the ranking
    nonzero_mask = d_q != 0
    n_nonzero = nonzero_mask.sum()
    if n_nonzero == 0:
        print("  Wilcoxon: all differences are zero; test undefined.")
        w_stat, w_p = np.nan, np.nan
    elif n_nonzero < 2:
        print("  Wilcoxon: only 1 non-zero difference; test undefined.")
        w_stat, w_p = np.nan, np.nan
    else:
        w_result = stats.wilcoxon(d_q, zero_method="wilcox", alternative="two-sided")
        w_stat, w_p = w_result.statistic, w_result.pvalue

    print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p:.4f} (two-sided; n_nonzero={n_nonzero}/{n_q})")
    print(f"  [Note: with n=8 the Wilcoxon test has limited power and only ~23 distinct p-values]")

    # Sign-flip permutation test (two-sided)
    observed_mean = np.abs(d_q.mean())
    perm_means = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = rng.choice([-1, 1], size=n_q)
        perm_means[i] = np.abs((d_q * signs).mean())
    perm_p = (perm_means >= observed_mean).mean()

    print(f"  Sign-flip permutation (10000 perms): |mean| observed={observed_mean:.4f}, p={perm_p:.4f} (two-sided)")
    print()

# ─── 4. (kappa × delta) interaction — Difference-in-Differences ─────────────
print("=== 4. (kappa × delta) interaction: Difference-in-Differences ===")
# DiD_q = [osc(strong,δ4) - osc(strong,δ1)] - [osc(weak,δ4) - osc(weak,δ1)]
strong_idx = kappas.index("strong")
weak_idx   = kappas.index("weak")
delta1_idx = deltas.index(1)
delta4_idx = deltas.index(4)

d_strong = q_osc[:, strong_idx, delta4_idx] - q_osc[:, strong_idx, delta1_idx]
d_weak   = q_osc[:, weak_idx,   delta4_idx] - q_osc[:, weak_idx,   delta1_idx]
did_q    = d_strong - d_weak

print(f"DiD per question (strong_delta_effect - weak_delta_effect):")
for qi, q in enumerate(questions):
    print(f"  Q{qi}: {did_q[qi]:+.4f}  [d_strong={d_strong[qi]:+.4f}, d_weak={d_weak[qi]:+.4f}]")
print(f"Mean DiD = {did_q.mean():.4f}")
print(f"Median DiD = {np.median(did_q):.4f}")
print(f"SD DiD = {did_q.std(ddof=1):.4f}")

# Sign-flip permutation test (two-sided)
obs_did = np.abs(did_q.mean())
perm_did = np.empty(N_PERM)
for i in range(N_PERM):
    signs = rng.choice([-1, 1], size=n_q)
    perm_did[i] = np.abs((did_q * signs).mean())
did_p = (perm_did >= obs_did).mean()

print(f"Sign-flip permutation test: |mean DiD| observed={obs_did:.4f}, p={did_p:.4f} (two-sided)")
print("Interpretation: p tests whether the delay effect differs between kappa=strong and kappa=weak.")
print()

# ─── 5. Cluster bootstrap ────────────────────────────────────────────────────
print("=== 5. Cluster bootstrap (n=8, 10000 draws, 95% percentile CIs) ===")

N_BOOT = 10_000
q_indices = np.arange(n_q)

boot_cell_means   = np.zeros((N_BOOT, n_k, n_d))
boot_delta_effects = np.zeros((N_BOOT, n_k))

for b in range(N_BOOT):
    idx = rng.choice(q_indices, size=n_q, replace=True)
    sample = q_osc[idx, :, :]
    boot_cell_means[b]    = sample.mean(axis=0)
    boot_delta_effects[b] = sample[:, :, 1].mean(axis=0) - sample[:, :, 0].mean(axis=0)

print("Cell mean osc — 95% bootstrap percentile CIs:")
print(f"{'Cell':<25}  {'mean':>8}  {'CI_lo':>8}  {'CI_hi':>8}")
for ki, kappa in enumerate(kappas):
    for di, delta in enumerate(deltas):
        m = boot_cell_means[:, ki, di]
        lo, hi = np.percentile(m, [2.5, 97.5])
        obs_mean = q_osc[:, ki, di].mean()
        print(f"  kappa={kappa}, delta={delta:<3}  {obs_mean:>8.4f}  {lo:>8.4f}  {hi:>8.4f}")

print()
print("Delta effect (delta=4 minus delta=1) — 95% bootstrap percentile CIs:")
print(f"{'kappa':<10}  {'observed':>10}  {'CI_lo':>8}  {'CI_hi':>8}")
for ki, kappa in enumerate(kappas):
    m = boot_delta_effects[:, ki]
    lo, hi = np.percentile(m, [2.5, 97.5])
    obs_eff = (q_osc[:, ki, 1] - q_osc[:, ki, 0]).mean()
    print(f"  {kappa:<8}  {obs_eff:>10.4f}  {lo:>8.4f}  {hi:>8.4f}")

print()

# ─── 6. Endpoint check ──────────────────────────────────────────────────────
print("=== 6. Endpoint check (question-level seed-aggregated means) ===")
print(f"{'Cell':<25}  {'mean final':>12}  {'mean conv':>12}")
for ki, kappa in enumerate(kappas):
    for di, delta in enumerate(deltas):
        print(
            f"  kappa={kappa}, delta={delta:<3}"
            f"  {q_final[:, ki, di].mean():>12.6f}"
            f"  {q_conv[:, ki, di].mean():>12.4f}"
        )
print()
print("=== Analysis complete ===")
