"""Exploratory all-run cross-model check for the numeric-estimation oscillation (one per model).
All-run comparison (not the movement-conditioned plan): one-sided amp(alpha=0.5, delta=6) > amp(alpha=0.5, delta=1), paired by question."""
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

# Audit both estimands; no imputation when a question has no moved runs in a cell.
for f in sorted(glob.glob("expA_v3*.json")):
    if "smoke" in f: continue
    data = json.load(open(f)); rs = data["runs"]
    convention = data.get("meta", {}).get("delay_convention", "legacy")
    lag_note = "labels 1,6 imply lags 0,5" if convention == "legacy" else "theory delays 1,6"
    pairs = []
    for question in sorted({r["q"] for r in rs}):
        cells = [[r["amp"] for r in rs if r["q"] == question and r["alpha"] == 0.5
                  and r["delta"] == d and r["moved"]] for d in (1, 6)]
        if all(cells): pairs.append([np.mean(c) for c in cells])
    x = np.asarray(pairs)
    p = wilcoxon(x[:,1], x[:,0], alternative="greater").pvalue if len(x) else None
    print(f"{f}: moved-only paired questions={len(pairs)}, p={p}; {convention}: {lag_note}")
