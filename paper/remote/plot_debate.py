#!/usr/bin/env python3
"""Figure for section 7's real-LLM proof of concept (Qwen3.6-35B debate, delayed verification)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("/Users/yehuda/PycharmProjects/agentic_error_propagation/paper/remote/debate_exp_results.json"))
runs = R["runs"]; deltas = sorted({r["delta"] for r in runs})
col = {1: "#2a9d8f", 4: "#e76f51"}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

# (a) the clean example
mp = {r["delta"]: r for r in runs if r["q"].startswith("Who directed the film 'Murder")}
for d in deltas:
    tr = mp[d]["mean_traj"]
    ax[0].plot(range(len(tr)), tr, "o-", ms=4, color=col[d],
               label=(f"$\\delta=1$ (fresh): converges" if d == 1 else f"$\\delta=4$ (delayed): oscillates"))
ax[0].set_xlabel("debate round $t$"); ax[0].set_ylabel("mean factual error $e_t$")
ax[0].set_ylim(-0.05, 1.08)
ax[0].set_title("(a) Real LLM debate, one question\n(3 free + 2 stubborn-wrong agents)")
ax[0].legend(loc="center left")

# (b) oscillation index across questions
qs = sorted({r["q"] for r in runs})
x = np.arange(len(qs)); w = 0.38
osc = {d: [next(r["osc"] for r in runs if r["delta"] == d and r["q"] == q) for q in qs] for d in deltas}
ax[1].bar(x - w/2, osc[1], w, color=col[1], label=f"$\\delta=1$ (mean {np.mean(osc[1]):.2f})")
ax[1].bar(x + w/2, osc[4], w, color=col[4], label=f"$\\delta=4$ (mean {np.mean(osc[4]):.2f})")
ax[1].set_xlabel("question"); ax[1].set_ylabel("oscillation index (sign changes of $\\Delta\\bar e$)")
ax[1].set_xticks(x); ax[1].set_xticklabels([str(i + 1) for i in range(len(qs))])
ax[1].set_title("(b) Delay raises oscillation across 8 questions")
ax[1].legend()

fig.tight_layout()
out = "/Users/yehuda/PycharmProjects/agentic_error_propagation/paper/figures/demo_llm.png"
fig.savefig(out, dpi=150)
print("wrote", out)
print(f"delta=1 mean_osc={np.mean(osc[1]):.2f}  delta=4 mean_osc={np.mean(osc[4]):.2f}")
