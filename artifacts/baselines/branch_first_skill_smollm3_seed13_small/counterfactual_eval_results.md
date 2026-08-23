# Counterfactual Response Skill Results

Skills are discovered from same-prefix Source policy response fingerprints, not from action groups.

## Setup

- Train prefixes: 80
- Eval prefixes: 80
- Observation bank entries: 160
- Num skills: 8
- Split: `{'split_unit': 'task', 'seed': 13, 'val_frac': 0.15, 'num_groups': 129, 'num_train_groups': 110, 'num_val_groups': 19, 'num_train_samples': 3996, 'num_val_samples': 619, 'decision_jsonl': 'artifacts/tau/tau_success_policy_traces_hier_v2.jsonl'}`

## Metrics

| Method | Branch Acc All | Branch Acc Write | Success/Failure Pair | Write Success/Failure Pair | Unique Action Prefix | Correct Flip | Harmful Flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source Counterfactual Response | 73.44% | 73.44% | 2.50% | 6.25% | 20.00% | 24.38% | 19.69% |
| Static Low-Rank | 68.75% | 50.00% | 80.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Oracle Source Branch Response + Broadcast | 68.75% | 50.00% | 80.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Oracle Source Branch Response + Hard Branch-First | 78.44% | 57.81% | 80.00% | 0.00% | 46.25% | 9.69% | 0.00% |
| Oracle Source Branch Response + Soft Hierarchical | 78.75% | 57.81% | 78.75% | 0.00% | 53.75% | 10.31% | 0.31% |
| Oracle Skill + Branch Operator + Broadcast | 68.75% | 50.00% | 80.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Oracle Skill + Branch Operator + Hard Branch-First | 72.50% | 57.81% | 80.00% | 0.00% | 26.25% | 3.75% | 0.00% |
| Oracle Skill + Branch Operator + Soft Hierarchical | 73.44% | 57.81% | 80.00% | 0.00% | 36.25% | 4.69% | 0.00% |
| Predicted Skill + Branch Operator + Broadcast | 68.75% | 50.00% | 80.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Predicted Skill + Branch Operator + Hard Branch-First | 72.50% | 57.81% | 80.00% | 0.00% | 25.00% | 3.75% | 0.00% |
| Predicted Skill + Branch Operator + Soft Hierarchical | 73.44% | 57.81% | 80.00% | 0.00% | 42.50% | 4.69% | 0.00% |
| Rule-Constrained Branch Ceiling | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 31.25% | 0.00% |

## Notes

- `Static Low-Rank` is the existing target-to-source latent alignment plus frozen Source head.
- `Oracle Source Branch Response` adds held-out Source branch-level response residuals to Static Low-Rank.
- `Oracle/Predicted Skill + Branch Operator` use raw Target post-pre hidden deltas and predict branch residuals, not full action-logit gaps.
- `Rule-Constrained Branch Ceiling` is a rule ceiling, not a learned Source oracle.