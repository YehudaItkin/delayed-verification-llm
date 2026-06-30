"""Optimal corrector placement -- the concrete 'where'.
Greedy corrector selection on a grounded Laplacian: at each step add the node of maximal marginal
coherence reduction  Delta_i = w||M^{-1}e_i||^2 / (1 + w e_i^T M^{-1} e_i)  (a resolvent centrality).
We show it (i) beats degree- and random-placement and (ii) concentrates on the high-leverage
bridge/hub nodes. Writes ../figures/placement.png   (run from paper/)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "figure.dpi": 300})

# --- a graph where placement matters: three 5-cliques in a chain, joined by single bridge edges ---
def clique_chain(sizes=(5, 5, 5)):
    n = sum(sizes); A = np.zeros((n, n)); idx = []; s = 0
    for sz in sizes:
        idx.append(list(range(s, s + sz)))
        for a in range(s, s + sz):
            for b in range(a + 1, s + sz):
                A[a, b] = A[b, a] = 1.0
        s += sz
    for k in range(len(idx) - 1):           # bridge edge between consecutive cliques
        a, b = idx[k][-1], idx[k + 1][0]; A[a, b] = A[b, a] = 1.0
    return A, idx
A, groups = clique_chain()
n = A.shape[0]
L = np.diag(A.sum(1)) - A
kap, w = 0.3, 12.0
I = np.eye(n)

def H(R):                                   # coherence tr M(R)^{-1}
    M = L + kap * I + w * np.diag([1.0 if i in R else 0.0 for i in range(n)])
    return float(np.trace(np.linalg.inv(M)))

def marginal_centrality(R):                 # Delta_i for every i not in R (Sherman-Morrison)
    M = L + kap * I + w * np.diag([1.0 if i in R else 0.0 for i in range(n)])
    Mi = np.linalg.inv(M); g = {}
    for i in range(n):
        if i in R: continue
        v = Mi[:, i]; g[i] = w * float(v @ v) / (1 + w * float(Mi[i, i]))
    return g

def greedy(k):
    R = []
    for _ in range(k):
        g = marginal_centrality(R); R.append(max(g, key=g.get))
    return R
def by_degree(k): return list(np.argsort(-A.sum(1))[:k])
def random_curve(k, reps=300, seed=0):
    rng = np.random.default_rng(seed); ys = []
    for _ in range(reps):
        perm = rng.permutation(n); ys.append([H(list(perm[:j])) for j in range(k + 1)])
    return np.mean(ys, 0)

K = 8
gr = greedy(K); H0 = H([])
greedy_curve = [H(gr[:j]) for j in range(K + 1)]
deg = by_degree(K); deg_curve = [H(deg[:j]) for j in range(K + 1)]
rnd = random_curve(K)
cent0 = marginal_centrality([])             # leverage of each node (at R=empty)

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
# (a) coherence vs #correctors
x = range(K + 1)
ax[0].plot(x, greedy_curve, "o-", color="#0b5394", lw=1.6, ms=4, label="greedy (resolvent centrality)")
ax[0].plot(x, deg_curve, "s--", color="#cc7700", lw=1.2, ms=3.5, label="degree")
ax[0].plot(x, rnd, "^:", color="#888888", lw=1.2, ms=3.5, label="random (mean)")
ax[0].set_xlabel("number of correctors $k$"); ax[0].set_ylabel(r"residual error $\mathrm{tr}\,M(R)^{-1}$")
ax[0].set_title("(a) greedy is near-optimal", fontsize=9); ax[0].legend(frameon=False, fontsize=7)
# (b) per-node leverage; greedy's first picks marked
order = np.argsort(-np.array([cent0[i] for i in range(n)]))
bars = ax[1].bar(range(n), [cent0[i] for i in range(n)], color="#cfe2f3", edgecolor="0.6", lw=0.4)
for r in gr[:4]:
    bars[r].set_color("#0b5394")
ax[1].set_xlabel("node"); ax[1].set_ylabel("marginal centrality $\\Delta_i$")
ax[1].set_title("(b) where: high-leverage nodes", fontsize=9)
ax[1].annotate("first 4\ngreedy picks", xy=(gr[0], cent0[gr[0]]), xytext=(n*0.45, max(cent0.values())*0.7),
               fontsize=7, color="#0b5394", arrowprops=dict(arrowstyle="->", color="#0b5394", lw=0.8))
fig.tight_layout(); fig.savefig("figures/placement.png", bbox_inches="tight")
print("greedy picks (first 6):", gr[:6], " bridge/junction nodes =", [g[-1] for g in groups[:-1]] + [g[0] for g in groups[1:]])
print(f"H: empty={H0:.2f}  greedy@{K}={greedy_curve[-1]:.2f}  degree@{K}={deg_curve[-1]:.2f}  random@{K}={rnd[-1]:.2f}")


def submodularity_sweep(trials=4000, n=8, w=1.5, kap=0.3, seed=7):
    """Verify Theorem 2's submodularity is gated by the M-matrix structure (App. D).
    Marginal Delta_j(M) = w||M^-1 e_j||^2 / (1 + w (M^-1)_jj). Submodular <=> Delta_j(R) >= Delta_j(S)
    for R subset S. Holds for grounded Laplacian + kI + pins (M-matrix, M^-1 >= 0 entrywise); can FAIL
    for arbitrary PSD M0."""
    rng = np.random.default_rng(seed)
    def marg(M, j):
        Mi = np.linalg.inv(M); col = Mi[:, j]; return w * (col @ col) / (1 + w * Mi[j, j])
    def grounded_m0():
        while True:
            A = (rng.random((n, n)) < 0.4).astype(float); A = np.triu(A, 1); A = A + A.T
            L = np.diag(A.sum(1)) - A
            if np.count_nonzero(np.linalg.eigvalsh(L) < 1e-9) == 1: return L + kap * np.eye(n)
    def arb_psd():
        B = rng.uniform(-1, 1, (n, n)); return B @ B.T + kap * np.eye(n)
    def count(m0fun):
        v = 0
        for _ in range(trials):
            M0 = m0fun(); perm = rng.permutation(n)
            R = list(perm[:rng.integers(1, n - 3)])
            S = list(dict.fromkeys(R + list(perm[:rng.integers(len(R), n - 2)])))
            j = int(next(i for i in range(n) if i not in S))
            DR = np.diag([1.0 if i in R else 0.0 for i in range(n)])
            DS = np.diag([1.0 if i in S else 0.0 for i in range(n)])
            if marg(M0 + w * DR, j) < marg(M0 + w * DS, j) - 1e-9: v += 1
        return v
    g, a = count(grounded_m0), count(arb_psd)
    print(f"submodularity violations -- grounded Laplacian+kI+pins: {g}/{trials}  |  arbitrary PSD: {a}/{trials}")
    return g, a


if __name__ == "__main__" and "--sweep" in __import__("sys").argv:
    submodularity_sweep()
