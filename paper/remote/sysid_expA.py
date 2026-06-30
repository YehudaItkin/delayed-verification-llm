"""
sysid_expA.py
=============
System-identification step for Experiment A:
  Fit the scalar delayed-consensus model
      e_{t+1} = a*e_t + b*e_{t-delta} + c   (+ noise)
  to each run's mean_traj, compute the dominant root magnitude rho_hat,
  predict stability, and compare to observed oscillation/non-convergence.

Data: paper/remote/expA_results.json
      64 runs = 8 questions × 2 kappa × 2 delta × 2 seeds

Interpreter: /Users/yehuda/anaconda3/envs/agents_article/bin/python3
             (numpy 2.4.4, scipy 1.17.1)

Notes on flat trajectories:
  39/64 runs converge essentially instantly (nearly constant mean_traj after t=1).
  OLS on these flat series returns a≈0, b≈0, c≈const, rho≈0 — trivially STABLE,
  which is correct.  R² is undefined (0/0) for flat series and is reported as NaN.
  ALL 64 runs contribute to the confusion matrix and cell aggregation;
  R² stats are computed only on the 25 non-flat runs.
"""

import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

# ── reproducibility (no RNG used, but satisfy the requirement) ────────────────
np.random.seed(42)

# ── load data ──────────────────────────────────────────────────────────────────
DATA_PATH = "paper/remote/expA_results.json"
with open(DATA_PATH) as f:
    raw = json.load(f)

T    = raw["meta"]["T"]    # 18
runs = raw["runs"]          # 64 dicts
assert len(runs) == 64, f"Expected 64 runs, got {len(runs)}"

# ── sanity: cell balance ───────────────────────────────────────────────────────
cell_counts = defaultdict(int)
for r in runs:
    cell_counts[(r["kappa"], r["delta"])] += 1
for kappa in ("weak", "strong"):
    for delta in (1, 4):
        assert cell_counts[(kappa, delta)] == 16, (
            f"Expected 16 runs for ({kappa},{delta}), got {cell_counts[(kappa, delta)]}"
        )

# ── helper: build OLS rows ────────────────────────────────────────────────────
def fit_run(traj, delta):
    """
    Fit e_{t+1} = a*e_t + b*e_{t-delta} + c by OLS.
    Rows: t = delta … T-2  →  T-1-delta rows.
    Returns: (a, b, c, r2, n_rows)
      r2 = NaN if trajectory is flat (ss_tot < threshold).
    """
    traj = np.asarray(traj, dtype=float)
    n = len(traj)
    idx = list(range(delta, n - 1))
    if not idx:
        return np.nan, np.nan, np.nan, np.nan, 0

    Y = np.array([traj[t + 1]          for t in idx])
    X = np.array([[traj[t], traj[t - delta], 1.0] for t in idx])

    n_rows   = len(Y)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    a, b, c  = coeffs

    Y_pred   = X @ coeffs
    ss_res   = float(np.sum((Y - Y_pred) ** 2))
    ss_tot   = float(np.sum((Y - Y.mean()) ** 2))
    r2       = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    return float(a), float(b), float(c), r2, n_rows

# ── helper: dominant root magnitude ──────────────────────────────────────────
def dominant_root(a, b, delta):
    """
    Characteristic polynomial:  z^{delta+1} - a*z^delta - b = 0
    Returns magnitude of the root with the largest modulus.
    """
    deg  = delta + 1
    poly = np.zeros(deg + 1, dtype=complex)
    poly[0]   =  1.0   # z^{delta+1}
    poly[1]   = -a     # z^delta
    poly[deg] = -b     # constant term
    rts = np.roots(poly)
    return float(np.max(np.abs(rts)))

# ── stability threshold ────────────────────────────────────────────────────────
STAB_TOL = 0.02    # predict UNSTABLE if rho_hat >= 1 - STAB_TOL = 0.98

# ── observed oscillation label ─────────────────────────────────────────────────
# Primary rule: osc >= 2  OR  final >= 0.3  → labelled "oscillating"
OSC_THRESH   = 2
FINAL_THRESH = 0.3

def obs_label(run):
    return int(run["osc"] >= OSC_THRESH or run["final"] >= FINAL_THRESH)

# ════════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("SYSTEM IDENTIFICATION — delayed-consensus model fit to debate logs")
print(f"Data: {DATA_PATH}   T={T}   n_runs={len(runs)}")
print("=" * 72)

rows_per_run = {1: T - 2, 4: T - 5}   # T-1-delta
for d, nr in rows_per_run.items():
    caveat = "  [SMALL SAMPLE CAVEAT]" if nr < 15 else ""
    print(f"  delta={d}: OLS rows per run = {nr}{caveat}")
print()

# ── per-run fit ───────────────────────────────────────────────────────────────
results = []
print(f"{'#':>3}  {'kappa':>6}  {'d':>2}  {'s':>2}  "
      f"{'a':>7}  {'b':>7}  {'c':>7}  {'R²':>6}  {'rho':>6}  "
      f"{'pred':>8}  {'obs':>5}  {'osc':>4}  {'final':>6}  {'flat':>5}")
print("-" * 95)

for i, run in enumerate(runs):
    delta   = run["delta"]
    kappa   = run["kappa"]
    seed    = run["seed"]
    traj    = run["mean_traj"]

    a, b, c, r2, n_rows = fit_run(traj, delta)

    rho           = dominant_root(a, b, delta) if not np.isnan(a) else np.nan
    pred_unstable = int(rho >= 1.0 - STAB_TOL)  if not np.isnan(rho) else np.nan
    obs_osc       = obs_label(run)

    # Is this a flat (instantly-converged) trajectory?
    traj_arr = np.asarray(traj, dtype=float)
    is_flat  = bool(np.std(traj_arr[delta + 1:]) < 1e-8)

    results.append({
        "idx":           i,
        "kappa":         kappa,
        "delta":         delta,
        "seed":          seed,
        "q":             run["q"],
        "a":             a,
        "b":             b,
        "c":             c,
        "r2":            r2,
        "n_rows":        n_rows,
        "rho":           rho,
        "pred_unstable": pred_unstable,
        "obs_osc":       obs_osc,
        "osc_raw":       run["osc"],
        "final":         run["final"],
        "conv":          run["conv"],
        "is_flat":       is_flat,
    })

    def _f(x, fmt=".3f"):
        return f"{x:{fmt}}" if not (isinstance(x, float) and np.isnan(x)) else "   nan"

    pred_str = "UNSTABLE" if pred_unstable == 1 else ("STABLE" if pred_unstable == 0 else "   NA")
    obs_str  = "  OSC" if obs_osc else " conv"
    print(f"{i:>3}  {kappa:>6}  {delta:>2}  {seed:>2}  "
          f"{_f(a):>7}  {_f(b):>7}  {_f(c):>7}  {_f(r2):>6}  {_f(rho):>6}  "
          f"{pred_str:>8}  {obs_str}  {run['osc']:>4}  {run['final']:>6.3f}  "
          f"{'YES' if is_flat else 'no':>5}")

# ════════════════════════════════════════════════════════════════════════════════
# FIT QUALITY
# ════════════════════════════════════════════════════════════════════════════════
n_flat  = sum(1 for r in results if r["is_flat"])
n_nonfl = sum(1 for r in results if not r["is_flat"])
r2_nonfl = [r["r2"] for r in results if not r["is_flat"] and not np.isnan(r["r2"])]

print()
print("=" * 72)
print("FIT QUALITY")
print("=" * 72)
print(f"  Flat trajectories (ss_tot≈0, R²=NaN, rho≈0, trivially STABLE): {n_flat}/64")
print(f"  Non-flat trajectories (R² defined):                              {n_nonfl}/64")
if r2_nonfl:
    print(f"  R² on non-flat runs — mean:{np.mean(r2_nonfl):.3f}  "
          f"median:{np.median(r2_nonfl):.3f}  "
          f"min:{np.min(r2_nonfl):.3f}  max:{np.max(r2_nonfl):.3f}")

print(f"\n  R² by delta (non-flat runs only):")
for d in (1, 4):
    r2s = [r["r2"] for r in results if not r["is_flat"] and r["delta"] == d and not np.isnan(r["r2"])]
    if r2s:
        print(f"    delta={d} ({rows_per_run[d]} rows/run): n={len(r2s)}  "
              f"mean R²={np.mean(r2s):.3f}  median={np.median(r2s):.3f}  "
              f"min={np.min(r2s):.3f}  max={np.max(r2s):.3f}")

# ════════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX — ALL 64 RUNS
# ════════════════════════════════════════════════════════════════════════════════
# Flat runs have rho≈0 → pred STABLE, which is correct (they converged).
# We use ALL 64 runs for the confusion matrix.
all_pred = np.array([r["pred_unstable"] for r in results], dtype=int)
all_obs  = np.array([r["obs_osc"]       for r in results], dtype=int)
all_rho  = np.array([r["rho"]           for r in results], dtype=float)
all_osc  = np.array([r["osc_raw"]       for r in results], dtype=float)
all_fin  = np.array([r["final"]         for r in results], dtype=float)

n_all = len(results)
tp = int(np.sum((all_pred == 1) & (all_obs == 1)))
fp = int(np.sum((all_pred == 1) & (all_obs == 0)))
fn = int(np.sum((all_pred == 0) & (all_obs == 1)))
tn = int(np.sum((all_pred == 0) & (all_obs == 0)))
agreement_pct = 100.0 * (tp + tn) / n_all

print()
print("=" * 72)
print("STABILITY PREDICTION vs OBSERVED OSCILLATION  (all 64 runs)")
print(f"  Observed label: osc >= {OSC_THRESH}  OR  final >= {FINAL_THRESH}  (n_osc={int(all_obs.sum())})")
print(f"  Predicted UNSTABLE: rho_hat >= {1.0 - STAB_TOL:.2f}")
print("=" * 72)
print(f"\n  Confusion matrix:")
print(f"                    obs=OSC  obs=conv")
print(f"  pred=UNSTABLE      {tp:5d}    {fp:5d}")
print(f"  pred=STABLE        {fn:5d}    {tn:5d}")
print(f"\n  Agreement: {tp + tn}/{n_all} = {agreement_pct:.1f}%")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
if tp + fp > 0:
    print(f"  Precision = {tp / (tp + fp):.2f}")
if tp + fn > 0:
    print(f"  Recall    = {tp / (tp + fn):.2f}")
print()
print("  Note: 39 flat runs contribute TP=0, FP=0; however, run #8 is flat (osc=0)")
print("        but ends at final=1.0, so it triggers the final>=0.3 rule and is")
print("        labelled observed-oscillating while predicted STABLE -> 1 FN.")
print("        Of the 39 flat runs: 38 TN + 1 FN.")
print("  The 25 non-flat runs are where prediction power can differ from trivial.")

# Confusion on non-flat runs only
nf_pred = np.array([r["pred_unstable"] for r in results if not r["is_flat"]], dtype=int)
nf_obs  = np.array([r["obs_osc"]       for r in results if not r["is_flat"]], dtype=int)
nf_rho  = np.array([r["rho"]           for r in results if not r["is_flat"]], dtype=float)
nf_osc  = np.array([r["osc_raw"]       for r in results if not r["is_flat"]], dtype=float)
nf_fin  = np.array([r["final"]         for r in results if not r["is_flat"]], dtype=float)

tp2 = int(np.sum((nf_pred == 1) & (nf_obs == 1)))
fp2 = int(np.sum((nf_pred == 1) & (nf_obs == 0)))
fn2 = int(np.sum((nf_pred == 0) & (nf_obs == 1)))
tn2 = int(np.sum((nf_pred == 0) & (nf_obs == 0)))
agr2 = 100.0 * (tp2 + tn2) / len(nf_pred)

print()
print(f"  Non-flat runs only (n={len(nf_pred)}):")
print(f"                    obs=OSC  obs=conv")
print(f"  pred=UNSTABLE      {tp2:5d}    {fp2:5d}")
print(f"  pred=STABLE        {fn2:5d}    {tn2:5d}")
print(f"  Agreement: {tp2+tn2}/{len(nf_pred)} = {agr2:.1f}%")

# ════════════════════════════════════════════════════════════════════════════════
# SPEARMAN CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("SPEARMAN CORRELATIONS (rho_hat vs observed measures, all 64 runs)")
print("=" * 72)
sp_osc,   p_osc   = spearmanr(all_rho, all_osc)
sp_fin,   p_fin   = spearmanr(all_rho, all_fin)
sp_label, p_label = spearmanr(all_rho, all_obs)
print(f"  rho_hat vs osc_raw  : r_s = {sp_osc:.3f}   p = {p_osc:.4f}")
print(f"  rho_hat vs final    : r_s = {sp_fin:.3f}   p = {p_fin:.4f}")
print(f"  rho_hat vs obs_label: r_s = {sp_label:.3f}   p = {p_label:.4f}")

print()
print("  Non-flat runs only (n=25):")
sp_osc2,   p_osc2   = spearmanr(nf_rho, nf_osc)
sp_fin2,   p_fin2   = spearmanr(nf_rho, nf_fin)
print(f"  rho_hat vs osc_raw  : r_s = {sp_osc2:.3f}   p = {p_osc2:.4f}")
print(f"  rho_hat vs final    : r_s = {sp_fin2:.3f}   p = {p_fin2:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS — alternative observed-label thresholds
# ════════════════════════════════════════════════════════════════════════════════
print()
print("ROBUSTNESS — alternative observed-label thresholds (all 64 runs):")
print(f"  {'Rule':>30}  {'agreement%':>11}  {'Spearman r_s':>13}")
for osc_t, fin_t in [(2, 0.3), (1, 0.3), (3, 0.3), (2, 0.5), (2, 0.2)]:
    obs_alt = np.array([int(r["osc_raw"] >= osc_t or r["final"] >= fin_t) for r in results])
    tp_a = int(np.sum((all_pred == 1) & (obs_alt == 1)))
    tn_a = int(np.sum((all_pred == 0) & (obs_alt == 0)))
    agr  = 100.0 * (tp_a + tn_a) / n_all
    sp_a, _ = spearmanr(all_rho, obs_alt)
    rule = f"osc>={osc_t} OR final>={fin_t}"
    print(f"  {rule:>30}  {agr:>10.1f}%  {sp_a:>13.3f}")

# ════════════════════════════════════════════════════════════════════════════════
# CELL-LEVEL STRUCTURE
# ════════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("CELL-LEVEL STRUCTURE  (kappa × delta), all 64 runs")
print("  Aggregate: average seeds within (q,kappa,delta) → avg across questions")
print("  → n_questions=8 per cell (pseudo-replication removed)")
print("=" * 72)

# Step 1: average over seeds within (q, kappa, delta) — all 64 runs
qkd = defaultdict(list)
for r in results:
    qkd[(r["q"], r["kappa"], r["delta"])].append(r)

qkd_mean = {}
for key, grp in qkd.items():
    qkd_mean[key] = {
        "a":   np.mean([g["a"]   for g in grp]),
        "b":   np.mean([g["b"]   for g in grp]),
        "c":   np.mean([g["c"]   for g in grp]),
        "rho": np.mean([g["rho"] for g in grp]),
        # r2: only average non-nan; record n_valid
        "r2s": [g["r2"] for g in grp if not np.isnan(g["r2"])],
        "n_flat": sum(1 for g in grp if g["is_flat"]),
    }

# Step 2: aggregate by (kappa, delta) across questions
kd = defaultdict(list)
for (q, kappa, delta), v in qkd_mean.items():
    kd[(kappa, delta)].append(v)

print(f"\n  {'Cell':>18}  {'n_q':>4}  {'mean_a':>8}  {'mean_b':>8}  "
      f"{'mean_c':>8}  {'mean_rho':>9}  {'mean_R²':>8}  {'n_flat_runs':>11}")
print("-" * 88)

cell_summaries = {}
for (kappa, delta) in [("strong", 1), ("strong", 4), ("weak", 1), ("weak", 4)]:
    grp = kd.get((kappa, delta), [])
    n_q = len(grp)
    if n_q == 0:
        print(f"  ({kappa}, d={delta}): NO DATA")
        continue
    ma   = float(np.mean([g["a"]   for g in grp]))
    mb   = float(np.mean([g["b"]   for g in grp]))
    mc   = float(np.mean([g["c"]   for g in grp]))
    mrho = float(np.mean([g["rho"] for g in grp]))
    all_r2s = [v for g in grp for v in g["r2s"]]
    mr2  = float(np.mean(all_r2s)) if all_r2s else np.nan
    n_flat_total = sum(g["n_flat"] for g in grp)
    cell_summaries[(kappa, delta)] = {"a": ma, "b": mb, "c": mc, "rho": mrho, "r2": mr2, "n": n_q}
    label = f"({kappa}, d={delta})"
    r2_str = f"{mr2:.4f}" if not np.isnan(mr2) else "  NaN"
    print(f"  {label:>18}  {n_q:>4}  {ma:>8.4f}  {mb:>8.4f}  "
          f"{mc:>8.4f}  {mrho:>9.4f}  {r2_str:>8}  {n_flat_total:>11}")

# ════════════════════════════════════════════════════════════════════════════════
# THEORY-CONSISTENT ORDERING CHECKS
# ════════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("THEORY-CONSISTENT ORDERING CHECKS")
print("  (i)  b < 0 (restoring) and |b| larger for STRONG kappa")
print("  (ii) rho increases with delta; largest at (strong, d=4)")
print("=" * 72)
print()

for delta in (1, 4):
    cs_s = cell_summaries.get(("strong", delta), {})
    cs_w = cell_summaries.get(("weak",   delta), {})
    b_s = cs_s.get("b", np.nan)
    b_w = cs_w.get("b", np.nan)
    both_neg = b_s < 0 and b_w < 0
    mag_ok   = abs(b_s) > abs(b_w) if not (np.isnan(b_s) or np.isnan(b_w)) else None
    print(f"  delta={delta}: b_strong={b_s:+.4f}, b_weak={b_w:+.4f}  "
          f"both<0: {both_neg}  |b_strong|>|b_weak|: {mag_ok}")

print()
for kappa in ("strong", "weak"):
    rho1 = cell_summaries.get((kappa, 1), {}).get("rho", np.nan)
    rho4 = cell_summaries.get((kappa, 4), {}).get("rho", np.nan)
    incr = rho4 > rho1 if not (np.isnan(rho1) or np.isnan(rho4)) else None
    print(f"  {kappa}: rho(d=1)={rho1:.4f}, rho(d=4)={rho4:.4f}  "
          f"rho increases with delta: {incr}")

rho_cells = [(cell_summaries.get((k, d), {}).get("rho", np.nan), k, d)
             for k in ("strong", "weak") for d in (1, 4)]
rho_cells = [(r, k, d) for r, k, d in rho_cells if not np.isnan(r)]
if rho_cells:
    best = max(rho_cells, key=lambda x: x[0])
    print(f"\n  Largest rho: ({best[1]}, d={best[2]}) = {best[0]:.4f}  "
          f"Expected (strong, d=4): {'MATCHES' if best[1]=='strong' and best[2]==4 else 'MISMATCH'}")

# ════════════════════════════════════════════════════════════════════════════════
# IMPLIED EFFECTIVE GAINS
# ════════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("IMPLIED EFFECTIVE GAINS  (theory: a = 1 - eta*mu, b = -eta*kappa)")
print("=" * 72)
for (kappa, delta), cs in sorted(cell_summaries.items()):
    a_v = cs["a"]
    b_v = cs["b"]
    eta_mu  =  1.0 - a_v    # should be >0
    eta_kap = -b_v           # should be >0 for restoring
    b_neg_ok = b_v < 0
    print(f"  ({kappa}, d={delta}): a={a_v:+.3f}, b={b_v:+.3f}  "
          f"eta*mu={eta_mu:+.3f}  eta*kappa={eta_kap:+.3f}  "
          f"b<0 (restoring): {b_neg_ok}")

# ════════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("CAVEATS (explicit)")
print("=" * 72)
print("  1. T=18 is short; delta=4 leaves only 13 OLS rows per run.")
print("  2. Trajectories are quantised to ~{0, 0.003, 0.335, 0.667, 1.0}; residuals")
print("     partly reflect quantisation noise rather than pure linear-model error.")
print("  3. The fit is on the MEAN trajectory over 7 agents; agent-level")
print("     heterogeneity is collapsed.")
print("  4. 39/64 runs converge instantly to a constant (flat); their rho≈0 is")
print("     trivially correct but contributes no information about the model.")
print("     The 25 non-flat runs carry all the diagnostic power.")
print("  5. The model is 1-D and linear; LLM debate dynamics are nonlinear,")
print("     discrete, and stochastic.")
print("  6. Agreement % and correlations test whether the coarse dynamics are")
print("     CONSISTENT with the theory — not that agents obey it.")
print()
print("Done.")
