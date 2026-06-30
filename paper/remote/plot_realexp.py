"""Real-agent figure (rigorous numeric-estimation oscillation test).
(a) representative signed-error trajectories: the stable cell decays to truth, the unstable cells
    overshoot THROUGH zero (the Hopf signature);
(b) signed-error amplitude by (alpha, delta): instability grows with both -> the dose-delay region.
Writes ../figures/realexp.png"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm", "figure.dpi": 300})

R = json.load(open("expA_v3.json"))["runs"]
def cell(al, d): return [r for r in R if r["alpha"] == al and r["delta"] == d]
def amp(al, d): return float(np.mean([r["amp"] for r in cell(al, d)]))
def rep_traj(al, d):                          # representative run: moved, amplitude nearest cell mean
    rs = [r for r in cell(al, d) if r["moved"]] or cell(al, d)
    m = np.mean([r["amp"] for r in rs]); return min(rs, key=lambda r: abs(r["amp"] - m))["e"]

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

# (a) representative signed trajectories
cols = {(0.5, 1): ("#2e7d32", r"$\alpha{=}0.5,\delta{=}1$ (stable)"),
        (0.5, 6): ("#0b5394", r"$\alpha{=}0.5,\delta{=}6$ (oscillates)"),
        (1.5, 6): ("#c0392b", r"$\alpha{=}1.5,\delta{=}6$ (strong)")}
for (al, d), (c, lab) in cols.items():
    e = rep_traj(al, d); ax[0].plot(range(len(e)), e, color=c, lw=1.4, label=lab)
ax[0].axhline(0, color="0.5", lw=0.8, ls=":")
ax[0].text(0.5, 0.05, "truth", color="0.4", fontsize=7)
ax[0].set_xlabel("round $t$"); ax[0].set_ylabel(r"signed error $e_t=(\bar b_t-b^\star)/\mathrm{scale}$")
ax[0].set_title("(a) overshoot through zero", fontsize=9)
ax[0].legend(frameon=False, fontsize=6.8, loc="lower right")

# (b) amplitude by (alpha, delta)
deltas = [1, 6]; als = [0.5, 1.5]; x = np.arange(len(deltas)); w = 0.36
ccol = {0.5: "#7fa8d0", 1.5: "#0b5394"}
for i, al in enumerate(als):
    ax[1].bar(x + (i - 0.5) * w, [amp(al, d) for d in deltas], w, label=rf"$\alpha{{=}}{al}$", color=ccol[al])
ax[1].set_xticks(x); ax[1].set_xticklabels([rf"$\delta{{=}}{d}$" for d in deltas])
ax[1].set_ylabel("signed-error amplitude"); ax[1].set_title("(b) instability grows with delay", fontsize=9)
ax[1].legend(frameon=False, fontsize=7.5, title="gain", title_fontsize=7.5)
ax[1].annotate("a-priori\nholds $8/8$", xy=(1 - 0.18, amp(0.5, 6)), xytext=(0.15, 0.42), fontsize=7,
               color="#0b5394", arrowprops=dict(arrowstyle="->", color="#0b5394", lw=0.8))
fig.tight_layout()
fig.savefig("../figures/realexp.png", bbox_inches="tight")
print("amp:", {(al, d): round(amp(al, d), 3) for al in als for d in deltas})
