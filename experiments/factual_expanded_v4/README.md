# Expanded factual study, September 13-15, 2026

400 questions (200 PsiloQA and 200 TruthfulQA), 800 trajectories, 226800 recorded requests. Read [the repository reproduction guide](../../REPRODUCIBILITY.md) or run `python reproduce.py` from the repository root.

## Main records

- [Final protocol](code_v4_3/PROTOCOL_RU.md) and [selection/analysis plan](SELECTION_AND_ANALYSIS_RU.md).
- [Executed code](qwen38/v4_3_snapshot/code/).
- [Manifest, full request log and trajectories](qwen38/v4_3_snapshot/results/primary/). Large files have a `.gz` suffix; `python restore_data.py` restores the exact paths used by the analysis and manuscript.
- [Full technical verification](primary_v4_3_verification.json).
- [Semantic review v1](semantic_review_20260915/REPORT_RU.md) and [v2](semantic_review_v2_20260915/REPORT_RU.md), with both frozen decision sets and their provenance.
- [Post-hoc outcome analysis](outcome_analysis_20260915/REPORT_RU.md), including code, paired results and every bootstrap replicate.

All three-agent histories have 48 states. Lag 0 and lag 5 conditions share initial answers, and all questions remain in the reported comparisons. The model's server tag and artifact digest are recorded without asserting an independently verified model family or parameter count.

The semantic reviews are post-hoc judgments by the same assisting model, not independent human validation. The reported completion bounds for binary amplitude are not confidence intervals and still include both signs. The separate outcome-frequency analysis retains abstentions, unknowns and format failures in its denominator; fewer abstentions do not by themselves demonstrate more correct answers.

For files with `.gz` suffixes, the original bytes, sizes and SHA-256 checksums are in [RESEARCH_MANIFEST.json](../../RESEARCH_MANIFEST.json). No research response was removed to fit the repository.
