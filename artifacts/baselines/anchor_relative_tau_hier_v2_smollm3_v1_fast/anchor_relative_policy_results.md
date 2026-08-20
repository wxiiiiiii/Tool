# Anchor-Relative Policy Interface Results

This experiment trains the policy interface once on source anchor-relative coordinates and performs target onboarding with anchor forward/cache only.

- Source tensor: `artifacts/hidden_states/qwen3_4b_tau_success_policy_traces_hier_v2.pt`
- Target tensor: `artifacts/hidden_states/smollm3_3b_tau_success_policy_traces_hier_v2.pt`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries during target onboarding: `0`

## Best Run

| Strategy | K | Feature | Normalization | Policy | Source Val Acc | Target Acc | Agreement | Transition Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|
| transition_critical | 128 | all | center | mlp | 57.80% | 48.17% | 63.84% | 41.19% |

## All Runs

| Strategy | K | Feature | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy_critical | 128 | state | center | 60.84% | 54.89% | 47.50% | 50.29% | 64.79% | 24.27% | 44.02% |
| policy_critical | 128 | all | center | 61.56% | 58.57% | 47.45% | 51.73% | 61.37% | 25.79% | 40.51% |
| transition_critical | 128 | state | center | 58.82% | 54.71% | 44.88% | 47.98% | 57.51% | 21.40% | 34.02% |
| transition_critical | 128 | all | center | 57.80% | 55.82% | 48.17% | 51.30% | 63.84% | 24.27% | 41.19% |
