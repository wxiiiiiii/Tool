# Counterfactual Observation Branch Benchmark

Fixed held-out tau-bench prefixes are paired with synthetic success/failure/empty/conflict tool observations.
The main metric is whether the frozen transferred policy enters the expected response branch.

## Setup

- Prefixes: 485
- Counterfactual rows: 1940
- Tool kinds: `{'retrieve': 374, 'write': 79, 'compute': 19, 'think': 11, 'search': 2}`
- Observation counts: `{}`
- Existing-hidden fallback audit: `False`
- Split: `{'split_unit': 'task', 'seed': 13, 'val_frac': 0.15, 'num_groups': 129, 'num_train_groups': 110, 'num_val_groups': 19, 'num_train_samples': 3996, 'num_val_samples': 619, 'decision_jsonl': 'artifacts/tau/tau_success_policy_traces_hier_v2.jsonl'}`

## Results

| Method | Branch Acc All | Branch Acc Write | Observed Branch All | Observed Branch Write | Success/Failure Pair All | Pair Write | Unique Action Prefixes All | Unique Action Prefixes Write | Avg Unique Actions All | Avg Unique Actions Write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static_lowrank_policy | 70.36% | 50.00% | 70.36% | 50.00% | 83.71% | 0.00% | 0.00% | 0.00% | 1.00 | 1.00 |
| oracle_branch_constrained_policy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 2.18 | 3.00 |

## Action Distributions

### static_lowrank_policy

- `conflict`: ACT_RETRIEVE_ORDER:485
- `empty`: ACT_RETRIEVE_ORDER:485
- `failure`: ACT_RETRIEVE_ORDER:485
- `success`: ACT_RETRIEVE_ORDER:485

### oracle_branch_constrained_policy

- `conflict`: ASK_USER:253, VERIFY:232
- `empty`: ACT_RETRIEVE_ORDER:485
- `failure`: ACT_RETRIEVE_ORDER:485
- `success`: ACT_RETRIEVE_ORDER:395, STOP:64, ANSWER:15, VERIFY:11
