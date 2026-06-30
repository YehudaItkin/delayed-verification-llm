"""Step-0 EXPLORATORY analysis (see PREREG_factual_oscillation.md): is the forcing-gated delta-scaling
of magnitude oscillation already visible in the existing grounded factual-QA logs?
Primary metric A = std(mean_traj[len//2:])  (period-independent steady-state amplitude).
NOT confirmatory: existing data, no temp=0, no delta=0. Decides only whether to run F1."""
import json, glob
import numpy as np
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# factual-QA logs (mean_traj = NLI distance to gold). grounded forcing = faulty majority; B/B2 ungrounded.
FILES = {
    "debate_exp_results.json": "grounded(orig)",
    "expA_big_results.json":   "grounded(faulty=4)",
    "expA_results.json":       "grounded(faulty=4)",
    "expA_push_results.json":  "grounded(faulty=4)",
    "expA_push_tqa.json":      "grounded(faulty=4,TQA)",
    "expA_B_results.json":     "UNgrounded",
    "expA_B2_results.json":    "UNgrounded",
}
FORCE_ORDER = {"weak": 0, "orig": 1, "strong": 2, "forceful": 3}

def amp(mt):
    mt = [float(x) for x in mt if x is not None]
    if len(mt) < 6: return None
    tail = mt[len(mt)//2:]
    return float(np.std(tail)), float(max(tail) - min(tail)), float(mt[-1])

rows = []   # (file, group, kappa, delta, q, seed, A, pk2pk, final)
for fn, group in FILES.items():
    try: d = json.load(open(fn))
    except Exception: continue
    for r in d.get("runs", []):
        mt = r.get("mean_traj")
        if not mt: continue
        a = amp(mt)
        if a is None: continue
        kap = r.get("kappa", "orig")
        rows.append((fn, group, kap, r.get("delta"), r.get("q"), r.get("seed", 0), a[0], a[1], a[2]))

def med(xs):
    xs = list(xs)
    return float(np.median(xs)) if len(xs) else float("nan")

print("="*92)
print("A = std(steady-state tail of NLI-distance trajectory).  Higher A = more magnitude oscillation.")
print("="*92)
# table: per (file/group, kappa, delta): n, median A, median pk2pk, median final
print(f"{'group':<24}{'kappa':<10}{'delta':>6}{'n':>5}{'medA':>9}{'medP2P':>9}{'medFinal':>10}")
keyset = sorted({(g, k, dl) for (_, g, k, dl, *_ ) in rows},
                key=lambda t: (t[0], FORCE_ORDER.get(t[1], 9), t[2] if t[2] is not None else 0))
for (g, k, dl) in keyset:
    sub = [r for r in rows if r[1] == g and r[2] == k and r[3] == dl]
    As = [r[6] for r in sub]; P = [r[7] for r in sub]; F = [r[8] for r in sub]
    print(f"{g:<24}{str(k):<10}{str(dl):>6}{len(sub):>5}{med(As):>9.4f}{med(P):>9.4f}{med(F):>10.4f}")

print("\n" + "="*92)
print("PAIRED delta_low -> delta_high (same question+seed), per (group,kappa): is A_high > A_low?")
print("Signature H1: the delta-effect should appear under STRONG/FORCEFUL forcing, not under WEAK.")
print("="*92)
for g in dict.fromkeys(x[1] for x in rows):
    for k in sorted({r[2] for r in rows if r[1] == g}, key=lambda x: FORCE_ORDER.get(x, 9)):
        sub = [r for r in rows if r[1] == g and r[2] == k]
        ds = sorted({r[3] for r in sub if r[3] is not None})
        if len(ds) < 2: continue
        dlo, dhi = ds[0], ds[-1]
        lo = {(r[4], r[5]): r[6] for r in sub if r[3] == dlo}
        hi = {(r[4], r[5]): r[6] for r in sub if r[3] == dhi}
        keys = [kk for kk in lo if kk in hi]
        if len(keys) < 4: continue
        al = np.array([lo[kk] for kk in keys]); ah = np.array([hi[kk] for kk in keys])
        ratio = med([h/l for h, l in zip(ah, al) if l > 1e-6])
        win = int(np.sum(ah > al)); n = len(keys)
        p = ""
        if HAVE_SCIPY and np.any(ah != al):
            try: p = f"  Wilcoxon(A_hi>A_lo) p={wilcoxon(ah, al, alternative='greater').pvalue:.4f}"
            except Exception: p = ""
        print(f"{g:<24}{str(k):<9} d{dlo}->d{dhi}: medA {med(al):.4f}->{med(ah):.4f} "
              f"(x{ratio:.2f})  A_hi>A_lo in {win}/{n}{p}")

print("\nNOISE FLOOR reference = median A of weak/low-forcing converged cells (above).")
print("READ: forcing-gated delta-scaling = ratio>1 & high win-rate under strong/forceful, ~1 under weak.")
