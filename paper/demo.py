#!/usr/bin/env python3
"""Empirical demo (no GPU): does the LINEAR stability theory predict the onset of oscillation
in the NONLINEAR multi-agent belief dynamics?

Free-node error dynamics with saturating (tanh) verification, single verification delay delta,
instantaneous gossip (d=0):

    e_{t+1} = (I - eta L_g) e_t  -  eta * kappa * tanh(e_{t-delta})  +  eta * g

tanh'(0)=1, so the linearization at e=0 is exactly the paper's model
    e_{t+1} = (I - eta L_g) e_t - eta kappa e_{t-delta} + eta g,
whose dose limit is  kappa < kappa_max(a,delta) = beta_c(a,delta)/eta,  binding mode a=1-eta*mu_min(L_g).

To swap in real LLM agents, replace `verify(e)` by a call that returns each agent's
verified-vs-current discrepancy; everything else is unchanged.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260621)

# ---------- Chebyshev U and the exact dose ceiling beta_c(a,delta) ----------
def chebU(n, c):
    if n == 0: return np.ones_like(np.asarray(c, float))
    Um1, U = np.ones_like(np.asarray(c, float)), 2 * np.asarray(c, float)
    for _ in range(2, n + 1):
        Um1, U = U, 2 * c * U - Um1
    return U

def beta_c(a, delta):
    """beta_c = 1/U_{delta-1}(c), c the branch root of U_delta(c)/U_{delta-1}(c)=a."""
    if delta == 1:
        return 1.0
    lo, hi = np.cos(np.pi / (delta + 1)), np.cos(np.pi / (2 * delta + 1))  # branch in c
    f = lambda c: chebU(delta, c) / chebU(delta - 1, c) - a                 # increasing in c
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0: lo = mid
        else: hi = mid
    c = 0.5 * (lo + hi)
    return 1.0 / chebU(delta - 1, c)

# ---------- network ----------
def connected_graph(n, p=0.3):
    A = np.zeros((n, n)); perm = rng.permutation(n)
    for k in range(1, n):
        i, j = perm[k], perm[rng.integers(0, k)]; A[i, j] = A[j, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and rng.random() < p: A[i, j] = A[j, i] = 1.0
    return np.diag(A.sum(1)) - A

# ---------- nonlinear simulation ----------
def simulate(Lg, eta, kappa, delta, g, T=4000):
    nf = Lg.shape[0]; A = np.eye(nf) - eta * Lg
    past = [np.zeros(nf) for _ in range(delta + 1)]   # past[0]=e_t ... past[delta]=e_{t-delta}
    traj = np.zeros((T, nf))
    for t in range(T):
        e_next = A @ past[0] - eta * kappa * np.tanh(past[delta]) + eta * g
        traj[t] = e_next
        past = [e_next] + past[:-1]
    return traj

def late_amplitude(traj, W=400):
    tail = traj[-W:]
    return float(np.max(np.ptp(tail, axis=0)) / 2)         # half peak-to-peak, worst node

def dominant_period(traj, W=600):
    x = traj[-W:, 0]; x = x - x.mean()
    if np.ptp(x) < 1e-6: return np.inf
    f = np.fft.rfftfreq(len(x)); P = np.abs(np.fft.rfft(x))
    P[0] = 0; k = int(np.argmax(P))
    return np.inf if f[k] == 0 else 1.0 / f[k]

# ========================================================================
L = connected_graph(10)
R = [0, 5]                                   # two grounded correctors
free = [i for i in range(10) if i not in R]
Lg = L[np.ix_(free, free)]
mu = np.linalg.eigvalsh(Lg)
mu_min, mu_max = mu[0], mu[-1]
eta = 0.5 / mu_max                            # ensures eta*mu in (0,1) -> a_i in (0,1)
g = np.zeros(len(free)); g[1] = 0.04          # a faulty agent injects small bias

print(f"grounded Laplacian: n_free={len(free)}, mu in [{mu_min:.3f},{mu_max:.3f}], eta={eta:.4f}")
print(f"{'delta':>5} {'a_bind':>7} {'kappa_max(theory)':>18} {'kappa_crit(nonlin)':>19} {'ratio':>7}")

results = {}
for delta in [1, 2, 3]:
    a_bind = 1 - eta * mu_min                  # slowest grounded mode binds the dose
    kmax = beta_c(a_bind, delta) / eta
    grid = np.linspace(0.35, 1.65, 40) * kmax
    amps = np.array([late_amplitude(simulate(Lg, eta, k, delta, g)) for k in grid])
    onset_idx = np.argmax(amps > 1e-3)
    kcrit = grid[onset_idx] if amps[onset_idx] > 1e-3 else np.nan
    results[delta] = (grid, amps, kmax, kcrit, a_bind)
    print(f"{delta:>5} {a_bind:>7.3f} {kmax:>18.4f} {kcrit:>19.4f} {kcrit/kmax:>7.3f}")

# frequency check for delta=2 (single binding mode)
d2 = 2; _, _, kmax2, _, a2 = results[d2]
traj_osc = simulate(Lg, eta, 1.25 * kmax2, d2, g)
per = dominant_period(traj_osc)
# theory: marginal root e^{i theta*}, c = branch root, theta* = arccos(c); period = 2pi/theta*
lo, hi = np.cos(np.pi / 3), np.cos(np.pi / 5)
f = lambda c: chebU(2, c) / chebU(1, c) - a2
for _ in range(80):
    m = 0.5 * (lo + hi); lo, hi = (m, hi) if f(m) < 0 else (lo, m)
theta_star = np.arccos(0.5 * (lo + hi)); per_theory = 2 * np.pi / theta_star
print(f"\ndelta=2 oscillation period: measured={per:.2f}  theory 2pi/theta*={per_theory:.2f}")

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for delta in [1, 2, 3]:
    grid, amps, kmax, kcrit, _ = results[delta]
    ax[0].plot(grid / kmax, amps, "o-", ms=3, label=f"$\\delta={delta}$")
ax[0].axvline(1.0, color="k", ls="--", lw=1, label="theory $\\kappa_{\\max}$")
ax[0].set_xlabel(r"$\kappa/\kappa_{\max}(\delta)$"); ax[0].set_ylabel("late oscillation amplitude")
ax[0].set_title("(a) Oscillation onset vs. predicted dose limit"); ax[0].legend()
t0 = simulate(Lg, eta, 0.7 * kmax2, d2, g)[-300:, 0]
t1 = traj_osc[-300:, 0]
ax[1].plot(t0, label=r"$\kappa=0.7\,\kappa_{\max}$ (stable)")
ax[1].plot(t1, label=r"$\kappa=1.25\,\kappa_{\max}$ (oscillating)")
ax[1].set_xlabel("step"); ax[1].set_ylabel(r"$e_1(t)$")
ax[1].set_title(r"(b) $\delta=2$: convergence vs. oscillation"); ax[1].legend()
fig.tight_layout()
fig.savefig("figures/demo_onset.png", dpi=150)
print("\nwrote figures/demo_onset.png")
