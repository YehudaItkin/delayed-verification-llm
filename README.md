# Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement

Code and paper for a control-theoretic study of **delayed verification** in multi-agent LLM
systems: modeling the generator–verifier–critic loop as a delayed consensus over a graph with
grounded *corrector* nodes, deriving when verification stabilizes vs. destabilizes factual
consensus, and where to place correctors.

## Contents

```
paper/
  main.tex            full paper (compiles to main.pdf, ~11pp)
  skeleton.tex        math-only skeleton (theorems + proofs)
  validate.py         numerical validation of the stability theory (NumPy, no GPU)
  demo.py             synthetic nonlinear (tanh) onset/frequency demo
  figures/            generated figures
  remote/             real LLM-debate experiment (runs against a vLLM OpenAI endpoint)
    debate_exp.py     debate harness + PsiloQA calibration
    debate_expA.py    variant A: wrong-majority + (kappa, delta) sweep
    plot_*.py         figure generation
    *_results.json    experiment outputs
```

## Key results

- **Reduction.** The grounded Laplacian eigen-decouples the closed loop into scalar delay
  recurrences `x_{t+1}=a x_t - eta*kappa x_{t-delta}`.
- **Verification dose.** Closed-form critical gain `kappa_max(delta)` (Chebyshev form; exact at
  `delta=2`); too-strong or too-delayed correction destabilizes truth into oscillation. Synchronized
  gossip/verification delays are the worst case (inverse golden ratio at `delta=2`).
- **Placement.** Corrector placement is supermodular ⇒ greedy `(1-1/e)` (inherited from
  Clark–Bushnell–Poovendran leader selection).
- **Empirics.** A synthetic `tanh` loop matches the predicted onset within ~2%; a real Qwen3.6-35B
  debate reproduces the `(kappa, delta)` dose tradeoff (strong verification: best when fresh, worst
  when delayed), with threshold and variance caveats detailed in the paper.

## Reproduce

Theory (no GPU):
```bash
python paper/validate.py     # stability boundary, dose curve, steady state
python paper/demo.py         # nonlinear onset/frequency
```

Real LLM experiment (needs a vLLM OpenAI-compatible endpoint + an NLI model):
```bash
export VLLM_API_KEY=...                       # never commit this
python paper/remote/debate_exp.py calibrate   # select movable questions
python paper/remote/debate_expA.py            # (kappa, delta) sweep
```

## Status

Published as [arXiv:2606.27409](https://arxiv.org/abs/2606.27409). Methodological predecessor: *Delayed Repression and Emergent Instability in Adaptive Multi-Agent Systems* (arXiv:2605.30392).
