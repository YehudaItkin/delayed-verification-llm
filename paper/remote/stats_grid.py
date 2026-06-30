import json, sys, numpy as np
from scipy.stats import wilcoxon
R=json.load(open(sys.argv[1]))["runs"]
qs=sorted({r["q"] for r in R})
ks=sorted({r["kappa"] for r in R}); ds=sorted({r["delta"] for r in R})
def vec(k,d,key): m={r["q"]:r[key] for r in R if r["kappa"]==k and r["delta"]==d}; return np.array([m[q] for q in qs if q in m],float)
def perm(d,it=20000,s=0): rng=np.random.default_rng(s);o=abs(d.mean());S=rng.choice([-1,1],(it,len(d)));return float((np.abs((S*d).mean(1))>=o-1e-12).mean())
def boot(d,it=20000,s=1): rng=np.random.default_rng(s);return np.percentile(d[rng.integers(0,len(d),(it,len(d)))].mean(1),[2.5,97.5])
def rep(name,a,b):
    n=min(len(a),len(b));a,b=a[:n],b[:n];d=b-a
    try:w=wilcoxon(a,b).pvalue
    except:w=float("nan")
    lo,hi=boot(d);print(f"{name:34s} n={n} Δ={d.mean():+.3f} Wilcoxon p={w:.4f} perm={perm(d):.4f} boot95=[{lo:+.2f},{hi:+.2f}]")
print(f"questions={len(qs)} kappas={ks} deltas={ds}\n")
for key in ("osc","final","conv"):
    print(f"== {key} ==")
    for k in ks: rep(f"  delta {ds[0]}->{ds[-1]} @{k}", vec(k,ds[0],key), vec(k,ds[-1],key))
    inter=(vec(ks[-1],ds[-1],key)-vec(ks[-1],ds[0],key))-(vec(ks[0],ds[-1],key)-vec(ks[0],ds[0],key))
    try:wi=wilcoxon(inter).pvalue
    except:wi=float("nan")
    lo,hi=boot(inter);print(f"  interaction kappaxdelta (DiD)    n={len(inter)} mean={inter.mean():+.3f} Wilcoxon p={wi:.4f} perm={perm(inter):.4f} boot95=[{lo:+.2f},{hi:+.2f}]\n")
