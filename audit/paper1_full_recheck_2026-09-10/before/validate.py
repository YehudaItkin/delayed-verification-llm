#!/usr/bin/env python3
"""Numerical validation of the formal skeleton (skeleton.tex). Pure numpy, no GPU.

Checks:
  C1  grounded-Laplacian eigen-decoupling (Lemma 1): full sim == sum of scalar modes
  C2  delta=1 stability boundary (Theorem 1): rho(companion)<1  <=>  eta*kappa<1 AND mu_max<2/eta+kappa
  C3  pinning-delay tradeoff (Cor 1): kappa_max(delta) is decreasing in delta
  C4  steady state (eq. 9): e_inf == (L_g + kappa I)^{-1} g
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ----- deterministic RNG (no wall-clock) -----
rng = np.random.default_rng(20260621)

def connected_graph(n, extra_p=0.25):
    """Random connected graph: spanning tree + extra edges. Returns Laplacian."""
    A = np.zeros((n, n))
    perm = rng.permutation(n)
    for k in range(1, n):                      # random spanning tree -> connected
        i, j = perm[k], perm[rng.integers(0, k)]
        A[i, j] = A[j, i] = 1.0
    for i in range(n):                         # extra edges
        for j in range(i + 1, n):
            if A[i, j] == 0 and rng.random() < extra_p:
                A[i, j] = A[j, i] = 1.0
    D = np.diag(A.sum(1))
    return D - A

def grounded_laplacian(L, R):
    free = [i for i in range(L.shape[0]) if i not in R]
    return L[np.ix_(free, free)], free

def companion(A, kappa, eta, delta):
    """Lifted matrix C for x_{t+1} = A x_t - eta*kappa x_{t-delta}; state stacks delta+1 blocks."""
    nf = A.shape[0]
    B = -eta * kappa * np.eye(nf)
    N = nf * (delta + 1)
    C = np.zeros((N, N))
    C[:nf, :nf] = A
    C[:nf, delta * nf:(delta + 1) * nf] = B
    for k in range(1, delta + 1):              # shift rows: block k <- block k-1
        C[k * nf:(k + 1) * nf, (k - 1) * nf:k * nf] = np.eye(nf)
    return C

def rho(M):
    return np.max(np.abs(np.linalg.eigvals(M)))

def mode_stable(a, kappa, eta, delta):
    """All roots of z^{delta+1} - a z^delta + eta*kappa inside open unit disk?"""
    coeffs = [1.0, -a] + [0.0] * (delta - 1) + [eta * kappa]
    return np.max(np.abs(np.roots(coeffs))) < 1.0 - 1e-9

def kappa_max(a, eta, delta, hi=40.0):
    """Largest kappa with mode stable, via bisection."""
    lo, hi = 0.0, hi
    if not mode_stable(a, lo + 1e-6, eta, delta):
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mode_stable(a, mid, eta, delta) else (lo, mid)
    return 0.5 * (lo + hi)

# =========================================================================
print("=" * 70)
print("C1  Eigen-decoupling (Lemma 1)")
L = connected_graph(12)
R = [0, 4, 9]
Lg, free = grounded_laplacian(L, R)
nf = Lg.shape[0]
mu, Q = np.linalg.eigh(Lg)
eta, kappa, delta = 0.12, 3.0, 1
A = np.eye(nf) - eta * Lg
g = rng.normal(size=nf)
T = 200
def simulate(A, g, kappa, eta, delta, T, x0func=None):
    nf = A.shape[0]
    past = [np.zeros(nf) for _ in range(delta + 1)]  # past[0]=e_t ... past[delta]=e_{t-delta}
    traj = []
    for t in range(T):
        e_next = A @ past[0] - eta * kappa * past[delta] + eta * g
        traj.append(e_next.copy())
        past = [e_next] + past[:-1]
    return np.array(traj)

traj_full = simulate(A, g, kappa, eta, delta, T, None)
# decoupled sim in eigenbasis
ghat = Q.T @ g
traj_modes = np.zeros_like(traj_full)
for i in range(nf):
    a_i = 1 - eta * mu[i]
    past = [0.0] * (delta + 1)
    xs = []
    for t in range(T):
        x_next = a_i * past[0] - eta * kappa * past[delta] + eta * ghat[i]
        xs.append(x_next)
        past = [x_next] + past[:-1]
    traj_modes[:, i] = xs
traj_modes_back = traj_modes @ Q.T
err = np.max(np.abs(traj_full - traj_modes_back))
print(f"    max |full - decoupled| over {T} steps = {err:.2e}   ->  {'PASS' if err < 1e-9 else 'FAIL'}")

# =========================================================================
print("=" * 70)
print("C2  delta=1 stability boundary (Theorem 1):  rho(C)<1  <=>  eta*kappa<1 AND mu_max<2/eta+kappa")
L = connected_graph(10)
R = [1, 6]
Lg, _ = grounded_laplacian(L, R)
mu_max = np.max(np.linalg.eigvalsh(Lg))
A = lambda eta: np.eye(Lg.shape[0]) - eta * Lg
agree = tot = 0
mismatches = []
for eta in [0.05, 0.1, 0.2, 0.35, 0.5]:
    for kappa in np.linspace(0.05, 14.0, 40):
        emp = rho(companion(A(eta), kappa, eta, 1)) < 1 - 1e-9
        pred = (eta * kappa < 1.0) and (mu_max < 2.0 / eta + kappa)
        tot += 1
        if emp == pred:
            agree += 1
        elif len(mismatches) < 5:
            mismatches.append((eta, round(kappa, 2), emp, pred))
print(f"    mu_max(L_g) = {mu_max:.3f}")
print(f"    agreement empirical vs Theorem-1 prediction: {agree}/{tot} = {100*agree/tot:.1f}%"
      f"   ->  {'PASS' if agree == tot else 'FAIL'}")
if mismatches:
    print(f"    sample mismatches (eta,kappa,emp,pred): {mismatches}")

# =========================================================================
print("=" * 70)
print("C3  pinning-delay tradeoff (Cor 1): kappa_max(delta) decreasing in delta")
eta = 0.1
for a in [0.9, 0.5, 0.0]:                      # a = 1 - eta*mu, representative modes
    row = [kappa_max(a, eta, d) for d in range(1, 7)]
    dec = all(row[k] <= row[k - 1] + 1e-6 for k in range(1, len(row)))
    print(f"    a={a:+.1f}: kappa_max(delta=1..6) = "
          + " ".join(f"{v:6.3f}" for v in row)
          + f"   ->  {'decreasing PASS' if dec else 'FAIL'}")
print(f"    (note: eta*kappa<1 i.e. kappa<{1/eta:.1f} is the delta=1 ceiling; shrinks with delta)")

# =========================================================================
print("=" * 70)
print("C4  steady state (eq. 9):  e_inf = (L_g + kappa I)^{-1} g")
L = connected_graph(14)
R = [2, 7, 11]
Lg, _ = grounded_laplacian(L, R)
nf = Lg.shape[0]
eta, kappa, delta = 0.08, 2.0, 3
A = np.eye(nf) - eta * Lg
g = rng.normal(size=nf)
# ensure stable before iterating to steady state
assert rho(companion(A, kappa, eta, delta)) < 1, "unstable params; pick smaller kappa/eta"
traj = simulate(A, g, kappa, eta, delta, 5000, None)
e_sim = traj[-1]
e_pred = np.linalg.solve(Lg + kappa * np.eye(nf), g)
err = np.max(np.abs(e_sim - e_pred))
print(f"    max |sim_inf - (L_g+kI)^-1 g| = {err:.2e}   ->  {'PASS' if err < 1e-6 else 'FAIL'}")
print("=" * 70)
print("done.")
