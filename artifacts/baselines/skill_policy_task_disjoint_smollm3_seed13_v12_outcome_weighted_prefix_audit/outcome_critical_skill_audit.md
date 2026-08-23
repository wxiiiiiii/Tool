# Outcome-Critical Skill Audit

## Replay Summary

| Method | Action Acc | Tool Exec OK | DB Hash | Tasks |
|---|---:|---:|---:|---:|
| static_action | 64.62% | 56.31% | 63.64% | 55 |
| outcome_weighted_prefix_skill_binder_no_action_logits | 63.17% | 55.52% | 74.55% | 55 |
| source_skill_rerank | 77.06% | 66.99% | 72.73% | 55 |
| oracle_skill_rerank | 87.88% | 68.40% | 80.00% | 55 |

## Static vs Corrected Task-Level Audit

| Quadrant | Tasks |
|---|---:|
| F->F | 14 |
| F->S | 6 |
| S->S | 35 |

### F -> S Repair Types

| Outcome Type | Repaired Tasks | Before First DB Mutation |
|---|---:|---:|
| Commit / Execute | 5 | 4 |
| Verify / Confirm | 1 | 1 |

### S -> F Regression Types

| Outcome Type | Regressed Tasks |
|---|---:|

## Outcome-Critical Oracle Decomposition

| Oracle Setting | Action Acc | Tool Exec OK | DB Hash | DB Delta vs Static |
|---|---:|---:|---:|---:|
| Static | 64.62% | 56.31% | 63.64% | 0.00% |
| Oracle Entity / Information | 69.95% | 55.30% | 63.64% | 0.00% |
| Oracle Verify / Confirm | 71.73% | 57.45% | 63.64% | 0.00% |
| Oracle Commit / Execute | 67.53% | 54.97% | 80.00% | 16.36% |
| Oracle Recovery | 68.82% | 53.44% | 63.64% | 0.00% |
| Oracle Finish | 68.34% | 53.29% | 63.64% | 0.00% |
| Full Oracle Skill | 87.88% | 68.40% | 80.00% | 16.36% |

## Deployability Gap

| Layer | Value |
|---|---:|
| P(Target Skill = Source Skill) | 75.28% |
| P(Source Action | Correct Skill) | 100.00% |
| P(GT Action | Correct Skill) | 80.47% |
| Static DB Hash | 63.64% |
| Source Skill DB Hash | 72.73% |
| Deployable DB Hash | 74.55% |
| DB Gap Source Skill - Deployable | -1.82% |

## Go / No-Go

GO: at least one oracle skill class improves DB Hash; continue with outcome-weighted skill binding.
