#!/usr/bin/env python3
"""Section-7 figure (variant A): the (kappa, delta) verification-dose tradeoff in real Qwen3.6-35B
debate with a wrong majority."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("/Users/yehuda/PycharmProjects/agentic_error_propagation/paper/remote/expA_results.json"))
runs = R["runs"]; kaps = ["weak", "strong"]; dels = [1, 4]
col = {"weak": "#5fa8d3", "strong": "#1b4965"}

def cell(k, d): return [r for r in runs if r["kappa"] == k and r["delta"] == d]

def qagg(k, d, key):
    """Aggregate the 2 seeds to the question level -> 8 values (the replication unit is the question)."""
    by_q = {}
    for r in cell(k, d):
        by_q.setdefault(r["q"], []).append(np.asarray(r[key], dtype=float))
    return np.array([np.mean(v, axis=0) for v in by_q.values()])  # 8 questions

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) grouped bars: oscillation index by (kappa, delta); error bars = SE across the 8 questions
x = np.arange(len(dels)); w = 0.36
for i, k in enumerate(kaps):
    qo = [qagg(k, d, "osc") for d in dels]                      # each length 8
    m = [q.mean() for q in qo]
    se = [q.std(ddof=1) / np.sqrt(len(q)) for q in qo]
    ax[0].bar(x + (i - 0.5) * w, m, w, yerr=se, capsize=4, color=col[k], label=f"{k} verification")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"$\\delta={d}$" for d in dels])
ax[0].set_ylabel("oscillation index (question mean $\\pm$ SE, $n{=}8$)")
ax[0].set_title("(a) Delay raises oscillation in both cells\n(direction threshold-robust; underpowered at $n{=}8$, interaction n.s.)")
ax[0].legend()

# (b) mean error trajectory (strong verification): delta=1 vs delta=4, question-aggregated
for d in dels:
    M = qagg("strong", d, "mean_traj")                         # 8 questions x 18 rounds
    m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(M.shape[0])
    c = "#2a9d8f" if d == 1 else "#e76f51"
    ax[1].plot(m, "o-", ms=3, color=c,
               label=(f"$\\delta=1$ (fresh): converges" if d == 1 else f"$\\delta=4$ (delayed): elevated"))
    ax[1].fill_between(range(len(m)), m - se, m + se, color=c, alpha=0.15)
ax[1].set_xlabel("debate round $t$"); ax[1].set_ylabel("mean factual error $e_t$")
ax[1].set_title("(b) Strong verification (question means, $n{=}8$)")
ax[1].legend()

fig.tight_layout()
out = "/Users/yehuda/PycharmProjects/agentic_error_propagation/paper/figures/demo_llm.png"
fig.savefig(out, dpi=150); print("wrote", out)
for k in kaps:
    for d in dels:
        c = cell(k, d)
        print(f"  {k:6s} d={d}: osc={np.mean([r['osc'] for r in c]):.2f} conv={np.mean([r['conv'] for r in c]):.2f}")
