# Outcome-Critical Skill Audit

## Replay Summary

| Method | Action Acc | Tool Exec OK | DB Hash | Tasks |
|---|---:|---:|---:|---:|
| static_action | 64.62% | 56.31% | 63.64% | 55 |
| source_discovered_cluster_binder | 62.36% | 56.12% | 67.27% | 55 |
| source_skill_rerank | 77.06% | 66.99% | 72.73% | 55 |
| oracle_skill_rerank | 87.88% | 68.40% | 80.00% | 55 |

## Static vs Corrected Task-Level Audit

| Quadrant | Tasks |
|---|---:|
| F->F | 18 |
| F->S | 2 |
| S->S | 35 |

### F -> S Repair Types

| Outcome Type | Repaired Tasks | Before First DB Mutation |
|---|---:|---:|
| Information / Entity Resolution | 1 | 1 |
| Recovery | 1 | 1 |

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
| P(Target Skill = Source Skill) | 77.22% |
| P(Source Action | Correct Skill) | 97.70% |
| P(GT Action | Correct Skill) | 78.45% |
| Static DB Hash | 63.64% |
| Source Skill DB Hash | 72.73% |
| Deployable DB Hash | 67.27% |
| DB Gap Source Skill - Deployable | 5.45% |

## Go / No-Go

GO: at least one oracle skill class improves DB Hash; continue with outcome-weighted skill binding.
