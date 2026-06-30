"""Question-level robust statistics for the n=30 expanded grid (expA_big_results.json).
Paired by question (1 seed). Tests: delta effect within each kappa, kappa effect, and the
kappa x delta interaction (difference-in-differences), each with Wilcoxon + sign-flip permutation
+ cluster (question) bootstrap CIs. No pseudoreplication: n = number of questions."""
import json, numpy as np
from scipy.stats import wilcoxon

R = json.load(open("expA_big_results.json"))["runs"]
qs = sorted({r["q"] for r in R})
def cell(k, d, key):  # vector over questions
    m = {r["q"]: r[key] for r in R if r["kappa"] == k and r["delta"] == d}
    return np.array([m[q] for q in qs], float)

def perm_p(diff, iters=20000, seed=0):       # sign-flip permutation on paired diffs
    rng = np.random.default_rng(seed); obs = abs(diff.mean())
    s = rng.choice([-1, 1], size=(iters, len(diff)))
    return float((np.abs((s * diff).mean(1)) >= obs - 1e-12).mean())

def boot(diff, iters=20000, seed=1):          # cluster (question) bootstrap CI of the mean diff
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, len(diff), size=(iters, len(diff)))].mean(1)
    return np.percentile(bs, [2.5, 97.5])

def report(name, a, b):                        # paired a vs b (b-a)
    d = b - a
    try: w = wilcoxon(a, b, zero_method="wilcox").pvalue
    except ValueError: w = float("nan")
    lo, hi = boot(d)
    print(f"{name:38s} mean Δ={d.mean():+.3f}  Wilcoxon p={w:.4f}  perm p={perm_p(d):.4f}  "
          f"boot95=[{lo:+.3f},{hi:+.3f}]")

print(f"n = {len(qs)} questions\n")
for key in ("osc", "final"):
    print(f"==== metric: {key} ====")
    report(f"  delta effect @weak  (d4-d1)", cell("weak", 1, key), cell("weak", 4, key))
    report(f"  delta effect @strong(d4-d1)", cell("strong", 1, key), cell("strong", 4, key))
    report(f"  kappa effect @d1 (strong-weak)", cell("weak", 1, key), cell("strong", 1, key))
    report(f"  kappa effect @d4 (strong-weak)", cell("weak", 4, key), cell("strong", 4, key))
    # interaction: (d4-d1)@strong - (d4-d1)@weak, per question
    inter = (cell("strong", 4, key) - cell("strong", 1, key)) - (cell("weak", 4, key) - cell("weak", 1, key))
    lo, hi = boot(inter)
    try: w = wilcoxon(inter).pvalue
    except ValueError: w = float("nan")
    print(f"  kappa x delta interaction (DiD)        mean={inter.mean():+.3f}  Wilcoxon p={w:.4f}  "
          f"perm p={perm_p(inter):.4f}  boot95=[{lo:+.3f},{hi:+.3f}]")
    print()
