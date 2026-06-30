"""Cross-model pre-registered check for the numeric-estimation oscillation (one per model).
Primary: one-sided amp(alpha=0.5, delta=6) > amp(alpha=0.5, delta=1), paired by question."""
import json, glob, numpy as np
from scipy.stats import wilcoxon
for f in sorted(glob.glob("expA_v3*.json")):
    if "smoke" in f: continue
    R = json.load(open(f))["runs"]; qs = sorted({r["q"] for r in R})
    aq = lambda al, d: np.array([np.mean([r["amp"] for r in R if r["q"]==q and r["alpha"]==al and r["delta"]==d]) for q in qs])
    cz = lambda al, d: np.mean([min(r["e"])<-0.02 and max(r["e"])>0.02 for r in R if r["alpha"]==al and r["delta"]==d])
    a1, a6 = aq(0.5,1), aq(0.5,6)
    try: p = wilcoxon(a6, a1, alternative="greater").pvalue
    except ValueError: p = float("nan")
    print(f"{f:26s} amp d1={a1.mean():.3f} d6={a6.mean():.3f}  p={p:.4f}  overshoot d6={cz(0.5,6):.2f}/d1={cz(0.5,1):.2f}")
