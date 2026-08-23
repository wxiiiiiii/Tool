# SmolLM3 Outcome-Critical Hybrid Validation

| Method | Action Acc | Tool Exec OK | DB Hash | Interpretation |
|---|---:|---:|---:|---|
| static_action | 64.62% | 56.31% | 63.64% | static low-rank policy baseline |
| outcome_weighted_source_skill_binder | 63.33% | 53.12% | 70.91% | Source pseudo-skill binder, EXECUTE upweighted |
| outcome_weighted_prefix_skill_binder_no_action_logits | 63.17% | 55.52% | 74.55% | best deployable method; no action logits or turn-index shortcut |
| outcome_weighted_prefix_execute_override | 64.78% | 54.57% | 74.55% | only EXECUTE overrides static; same DB as best full prefix binder |
| outcome_weighted_prefix_execute_confirm_override | 64.14% | 54.92% | 74.55% | only EXECUTE/CONFIRM overrides static; same DB as best full prefix binder |
| source_skill_rerank | 77.06% | 66.99% | 72.73% | Source-skill diagnostic upper bound |
| oracle_skill_rerank | 87.88% | 68.40% | 80.00% | oracle skill upper bound |

## Takeaways

- The best deployable DB Hash is still `74.55%`, but v13 shows this result can be reproduced by only allowing outcome-critical `EXECUTE` or `EXECUTE+CONFIRM` overrides.
- This narrows the mechanism: the gain is not from globally copying all Source skills, but from correcting a small number of commit/verification decisions that affect final DB state.
- Single-step Action Acc is not predictive of task outcome here: `source_skill_rerank` has much higher Action Acc (`77.06%`) but lower DB Hash (`72.73%`) than the outcome-critical prefix binder (`74.55%`).
- Next useful step is not more all-state skill tuning; it is either true closed-loop validation or an impact-weighted train-split rollout estimate for commit/verification states.
