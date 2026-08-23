# SmolLM3 Outcome-Weighted Skill Results

| Method | Action Acc | Skill Acc | Source Skill Agree | Action Transition | Tool Exec OK | DB Hash | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| static_action | 64.62% | 71.41% | 79.32% | 42.91% | 56.31% | 63.64% | static low-rank policy baseline |
| source_discovered_cluster_binder | 62.04% | 69.79% | 77.22% | 37.23% | 56.12% | 67.27% | previous deployable discovered-skill binder |
| outcome_weighted_source_skill_binder | 63.33% | 70.92% | 78.19% | 41.49% | 53.12% | 70.91% | Source pseudo-skill binder, EXECUTE states upweighted |
| outcome_weighted_execute_selective | 64.62% | 71.41% | 79.32% | 42.91% | 52.43% | 63.64% | only override high-confidence EXECUTE; no effective override here |
| outcome_weighted_prefix_skill_binder_no_action_logits | 63.17% | 70.44% | 75.28% | 40.60% | 55.52% | 74.55% | no action logits/turn index; prefix evidence + outcome weighting |
| outcome_weighted_prefix_execute_selective | 64.62% | 71.41% | 79.32% | 42.91% | 52.43% | 63.64% | prefix selective; no effective override here |
| source_skill_rerank | 77.06% | 85.14% | 100.00% | 59.75% | 66.99% | 72.73% | Source-skill diagnostic upper bound |
| oracle_skill_rerank | 87.88% | 100.00% | 85.14% | 76.60% | 68.40% | 80.00% | oracle skill upper bound |

## Outcome-Critical Findings

- Single-class oracle decomposition shows only `Oracle Commit / Execute` changes DB Hash: `63.64% -> 80.00%` (+16.36 points).
- `outcome_weighted_prefix_skill_binder_no_action_logits` reaches `74.55%` DB Hash, above `source_skill_rerank` at `72.73%`, despite lower single-step Action Acc.
- The selective EXECUTE variants did not help in this run because the confidence gate barely opened, so they effectively matched static behavior.
- This supports the document claim that optimizing outcome-critical skill states is more useful than uniformly improving all single-step action labels.
