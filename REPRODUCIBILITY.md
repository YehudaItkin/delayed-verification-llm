# Reproducibility records for the revised paper

Accompanies *Delayed Verification in Multi-Agent LLM Systems: Stability of a Consensus Model and Coherence-Based Corrector Placement*. The manuscript and repository README describe the corrections. This supplement includes the complete final expanded experiment (400 questions, 800 trajectories, 226800 requests), both frozen semantic scoring versions, outcome analysis, historical experiment archives, and selected mathematical checks.

## Layout and lossless compression

The article compiles from paper/main.tex with figures under paper/figures/. The repository keeps code and records under paper/, audit/, and experiments/. Large JSON and JSONL data files are supplied as deterministic gzip streams. They preserve every original byte, including the complete request/response log. The RESEARCH_MANIFEST.json records the original and compressed sizes and SHA-256 hashes. Allow about 0.8 GB of disk space to restore the data, plus space for derived checks.

From the repository root, with Python 3.11 or later:

```sh
python restore_data.py
```

This restores compressed files to the exact paths named in the manuscript and checks every payload hash. It preserves the compressed originals and refuses to replace a different existing file. The article PDF does not require decompression of the data.

## Full offline reproduction

With Python 3.11+, NumPy and SciPy installed:

```sh
python reproduce.py
```

The supplied results used NumPy 2.4.0. Exact numerical equality can depend on the numerical library version. The command replays all saved prompts, seeds, responses, parser outcomes and synchronous delay indices without contacting a model service, applies both frozen semantic decision sets, and recomputes all outcome statistics and bootstrap arrays. It writes a new reproduction-* directory and checks against the preserved reports. Source responses and frozen decisions remain unchanged. A passed replay verifies the recorded calculations, not factual correctness of the semantic labels or independent reproduction on a new model run.

For historical archives and harness checks, from the repository root after restoration:

```sh
python audit/paper1_full_recheck_2026-09-10/test_recheck.py
python audit/paper1_full_recheck_2026-09-10/recompute.py
```

The second script recomputes derived historical summaries beside itself. The 21 source JSON files remain unchanged. Historical archives lack raw generations for some experiments, as disclosed in the article. Smoke files are included only to preserve the earlier integrity manifest, not as additional evidence.

## Mathematical checks

From paper/lean with Lean 4.32.2 installed:

```sh
lean PlacementCheck.lean
lean DelayCheck.lean
lean ControllerCheck.lean
lean RecheckCheck.lean
lean RereadCheck.lean
```

Only Std is required. These are algebraic certificates for explicit witnesses and selected identities, not formalizations of the general stability/placement theorems or of the empirical model. Wolfram check files and stored results are under audit/series_2026-09-09/fixes_step*/. Python numerical checks include paper/validate.py, paper/verify_absorbing.py, and the boundary/delay check scripts. Plotting additionally needs Matplotlib and writes figures.

## Experimental records

The final frozen manifest and complete responses are under experiments/factual_expanded_v4/qwen38/v4_3_snapshot/results/primary/. The original executed code is preserved in the adjacent code/ directory; code_v4_3/ also contains the later offline replay and tests. The model tag is an operational server identifier, with its artifact digest recorded; no independent model-family or parameter-count certification is claimed.

semantic_review_20260915/ and semantic_review_v2_20260915/ contain the frozen decisions, reasons, selected external sources and derived trajectories. They are same-assistant post-hoc assessments, not independent human validation. outcome_analysis_20260915/ records the post-hoc protocol, paired cluster-bootstrap outputs and all saved replicates. Neither hypothetical binary completions of abstentions nor increased scoring coverage proves accuracy. Both signs of the primary amplitude contrast remain possible.

Selection protocols and frozen pool/selection records are included. The original external PsiloQA export and all pilot runs are not bundled; this package reproduces the preserved selected study, not a new download of the source population or the development history. Service authentication and personal machine settings are not needed for any offline check. Legacy model-calling drivers require separately configured endpoints if used for new runs.
