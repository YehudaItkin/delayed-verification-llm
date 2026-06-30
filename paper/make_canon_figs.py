"""Canonical figures for the delayed-verification paper.
Fig 1: the verification dose-delay margin beta_c(1,delta) with the golden-ratio marker.
Fig 2: the two-delay (d,delta)=(1,2) stability region in the (p,q)=(eta*mu,eta*kappa) plane.
Run: python make_canon_figs.py  ->  figures/dose_delay_margin.png, figures/twodelay_region.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
    "mathtext.fontset": "cm", "figure.dpi": 300,
})
PHI_INV = (np.sqrt(5) - 1) / 2  # 0.6180339...

# ---------- Figure 1: dose-delay margin ----------
def beta_c_a1(d):                     # binding mode a=1
    return np.sin(np.pi / (2 * d + 1)) / np.sin(d * np.pi / (2 * d + 1))

def beta_c_at_a(a_target, d):         # parametric boundary, interpolate to fixed a
    th = np.linspace(1e-4, np.pi / (d + 1) - 1e-4, 4000)
    a = np.sin((d + 1) * th) / np.sin(d * th)        # decreasing in th, range (0,(d+1)/d)
    b = np.sin(th) / np.sin(d * th)
    order = np.argsort(a)
    if a_target <= a[order][0] or a_target >= a[order][-1]:
        return np.nan
    return np.interp(a_target, a[order], b[order])

fig, ax = plt.subplots(figsize=(3.4, 2.7))
dd = np.arange(1, 11)
bc1 = np.array([beta_c_a1(d) for d in dd])
for a in (0.25, 0.5, 0.75):           # faint family of other binding modes
    bb = np.array([beta_c_at_a(a, d) for d in dd])
    ax.plot(dd, bb, color="0.7", lw=0.9, zorder=1)
    j = np.where(~np.isnan(bb))[0][-1]
    ax.annotate(f"$a={a}$", (dd[j], bb[j]), color="0.55", fontsize=7,
                xytext=(2, 0), textcoords="offset points", va="center")
ax.plot(dd, bc1, "o-", color="#0b5394", lw=1.6, ms=4.5, zorder=3,
        label=r"$a=1$ (slowest grounded mode)")
ax.plot([2], [PHI_INV], "s", color="#cc0000", ms=7, zorder=4)
ax.annotate(r"$\beta_c(1,2)=\frac{\sqrt{5}-1}{2}=1/\varphi\approx0.618$",
            (2, PHI_INV), xytext=(18, 14), textcoords="offset points",
            fontsize=8, color="#cc0000",
            arrowprops=dict(arrowstyle="->", color="#cc0000", lw=0.8))
ax.set_xlabel(r"verification delay $\delta$")
ax.set_ylabel(r"critical dose $\beta_c=\eta\,\kappa_{\max}$")
ax.set_xlim(0.6, 10.4); ax.set_ylim(0, 1.05)
ax.set_xticks(dd)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)
fig.tight_layout()
fig.savefig("figures/dose_delay_margin.png", bbox_inches="tight")
plt.close(fig)
print("dose_delay_margin.png: beta_c(1,2) =", beta_c_a1(2), " (1/phi =", PHI_INV, ")")

# ---------- Figure 2: two-delay (d,delta)=(1,2) stability region ----------
d, dl = 1, 2
def stable(p, q):                     # roots of z^{dl+1} - z^{dl} + p z^{dl-d} + q
    coeffs = np.zeros(dl + 2)
    coeffs[0] = 1.0; coeffs[1] = -1.0
    coeffs[1 + d] += p; coeffs[-1] += q
    r = np.roots(coeffs)
    return np.max(np.abs(r)) < 1 - 1e-9

P = np.linspace(0, 2.6, 360); Q = np.linspace(0, 2.6, 360)
PP, QQ = np.meshgrid(P, Q)
S = np.zeros_like(PP)
for i in range(PP.shape[0]):
    for j in range(PP.shape[1]):
        S[i, j] = stable(PP[i, j], QQ[i, j])

fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.contourf(PP, QQ, S, levels=[0.5, 1.5], colors=["#cfe2f3"], zorder=0)
ax.contour(PP, QQ, S, levels=[0.5], colors=["#0b5394"], linewidths=1.3, zorder=2)
# parametric oscillatory boundary
th = np.linspace(1e-3, np.pi - 1e-3, 6000)
pc = (np.sin(dl * th) - np.sin((dl + 1) * th)) / np.sin((dl - d) * th)
qc = (np.sin((d + 1) * th) - np.sin(d * th)) / np.sin((dl - d) * th)
m = (pc >= -0.05) & (qc >= -0.05) & (pc <= 2.6) & (qc <= 2.6)
ax.plot(pc[m], qc[m], color="#0b5394", lw=1.0, ls="--", zorder=3,
        label="oscillatory boundary")
# lambda=-1 real-root line: p(-1)^d + q(-1)^dl = 2  ->  -p + q = 2
ax.plot(P, P + 2, color="#cc0000", lw=1.1, zorder=3, label=r"$\lambda=-1$ line")
ax.fill_between([], [], [], color="#cfe2f3", label="stable region")
ax.set_xlabel(r"consensus/forcing $p=\eta\mu$")
ax.set_ylabel(r"verification dose $q=\eta\kappa$")
ax.set_xlim(0, 2.6); ax.set_ylim(0, 2.6)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title(r"two delays $(d,\delta)=(1,2)$", fontsize=9)
fig.tight_layout()
fig.savefig("figures/twodelay_region.png", bbox_inches="tight")
plt.close(fig)
print("twodelay_region.png: stable-cell example stable(0.3,0.3) =", stable(0.3, 0.3))
