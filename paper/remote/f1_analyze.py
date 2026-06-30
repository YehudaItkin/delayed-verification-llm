"""F1 decision-rule analysis (PREREG_factual_oscillation.md). A = std(mean_traj tail) = r['amp']."""
import json, numpy as np
from scipy.stats import wilcoxon
def med(x): return float(np.median(x)) if len(x) else float('nan')
for fname in ['f1_tqa.json','f1_psilo.json']:
    d=json.load(open(fname))['runs']
    def A(nf,dl,tp): return [r['amp'] for r in d if r['n_faulty']==nf and r['delta']==dl and abs(r['temp']-tp)<1e-9]
    def As(nf,dl,tp): return [r['amp_s'] for r in d if r['n_faulty']==nf and r['delta']==dl and abs(r['temp']-tp)<1e-9]
    print('\n'+'='*80); print(fname.upper()); print('='*80)
    print(f"{'nf':>3}{'δ':>4}{'temp':>6}{'medA':>9}{'medA_s':>9}{'n':>4}")
    for nf in [0,4]:
        for tp in [0.7,0.0]:
            for dl in [0,1,6]:
                a=A(nf,dl,tp); print(f"{nf:>3}{dl:>4}{tp:>6.1f}{med(a):>9.4f}{med(As(nf,dl,tp)):>9.4f}{len(a):>4}")
    # noise floor: forcing OFF (nf=0), all cells
    N=[r['amp'] for r in d if r['n_faulty']==0]; Nmed,Nsd=med(N),float(np.std(N)); floor=Nmed+2*Nsd
    print(f"\nNOISE FLOOR N (nf=0): median={Nmed:.4f} sd={Nsd:.4f}  -> threshold N+2sd = {floor:.4f}")
    print("\n--- PRE-REGISTERED DECISION RULE ---")
    for tp in [0.7,0.0]:
        a0,a6=A(4,0,tp),A(4,6,tp); 
        # paired by question order
        m=min(len(a0),len(a6)); x0,x6=np.array(a0[:m]),np.array(a6[:m])
        try: p=wilcoxon(x6,x0,alternative='greater').pvalue if np.any(x6!=x0) else 1.0
        except: p=float('nan')
        win=int(np.sum(x6>x0))
        c1 = med(a6)>med(a0)  # monotone-ish (delta scaling)
        c2 = med(a6)>floor    # above noise floor
        print(f"  temp={tp}: A(nf4,δ0)={med(a0):.4f} -> A(nf4,δ6)={med(a6):.4f}  (δ6>δ0 in {win}/{m}, Wilcoxon p={p:.4f})"
              f"  | δ6>floor? {c2}")
    # placebo: nf=4 delta=0 vs floor ; forcing-gating: nf=0 delta scaling
    print(f"  PLACEBO δ=0 (nf4,temp0.7) medA={med(A(4,0,0.7)):.4f} vs floor {floor:.4f}  (should be ~floor)")
    print(f"  FORCING-GATE: nf=0 δ0->δ6 (temp0.7): {med(A(0,0,0.7)):.4f}->{med(A(0,6,0.7)):.4f} (should be flat)")
