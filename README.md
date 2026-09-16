# Delayed Verification in Multi-Agent LLM Systems

Code, reviewed manuscript and research records for **Stability of a Consensus Model and Coherence-Based Corrector Placement**, a revision of [arXiv:2606.27409](https://arxiv.org/abs/2606.27409).

The updated [manuscript source](paper/main.tex) and [28-page PDF](paper/main.pdf) contain the corrected mathematical scope, historical experimental reanalysis, and the expanded factual study below. Publishing this repository does not replace the arXiv article; that is a separate submission. `paper/arxiv_submission.tar.gz` and `paper/skeleton.*` are historical artifacts, not the current replacement package or authoritative revised manuscript.

## Expanded factual study: 400 questions

Collected September 13-15, 2026: 200 PsiloQA questions, 200 TruthfulQA questions, 800 trajectories and 226800 requests. Three agents and four fixed erroneous peers were tested with verifier lags 0 and 5, using the recorded server artifact tagged `qwen3.8:latest`. The tag is an operational identifier; its digest is frozen in the manifest.

The complete study records are in [experiments/factual_expanded_v4](experiments/factual_expanded_v4/README.md), including:

- Full prompts, responses, request seeds, parsed outputs and agent trajectories.
- Frozen questions, source evidence, model digest and executed code.
- Both post-hoc semantic review versions, including decisions, reasons and selected sources.
- Paired outcome analysis, question clusters, all bootstrap replicates and verification reports.

Large JSON/JSONL files are gzip-compressed without changing their original contents. There are no Git LFS pointers or external data downloads in the offline reproduction path. [RESEARCH_MANIFEST.json](RESEARCH_MANIFEST.json) records original and stored checksums. Restoring the complete data uses about 0.8 GB of disk space, with additional space needed for derived analyses.

## Reproduce from this checkout

Use Python 3.11 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python reproduce.py
```

The command restores and verifies compressed files, replays all 226800 saved requests without calling a model, reapplies both frozen semantic reviews, and recomputes the outcome statistics and bootstrap arrays. Outputs go into a fresh ignored `reproduction-*` directory. To restore data without running the analysis, use `python restore_data.py` (standard library only).

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) documents further historical and mathematical checks. These calculations reproduce the preserved trajectories and labels; they are not a new independent run or independent factual adjudication.

## What the results establish

The linear stability threshold is exact for the stated symmetric consensus recurrence. Greedy placement guarantees a fraction of the optimal coherence reduction, not optimal static mean-square error. Signed numeric LLM experiments use an externally imposed correction law; they do not identify the dynamics of ordinary factual verification.

In the expanded factual study, conservative completion bounds still permit either sign of the change in binary error amplitude. Abstentions dominate the remaining unscored cells. A separate post-hoc cluster-bootstrap analysis finds lower TruthfulQA abstention frequency at lag 5; it does not establish improved accuracy or a change in error variability. Both semantic scoring versions are preserved, and neither constitutes independent human validation.

## Layout

- `paper/`: reviewed manuscript, figures, numerical checks and historical experiment drivers/data.
- `paper/lean/`: selected algebraic certificates, using Lean 4.32.2 and Std.
- `experiments/factual_expanded_v4/`: expanded study and all preserved analyses.
- `audit/`: supporting mathematical and historical reanalysis records.
- `restore_data.py`, `reproduce.py`: offline restoration and replay entry points.

Historical raw generations were not saved for some old experiments. The complete new study does not repair those missing historical responses. The full external PsiloQA candidate export and pilot history are not bundled; the frozen selected study is.
