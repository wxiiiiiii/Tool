# Anchor-Relative Policy Interface Results

This experiment trains the policy interface once on source anchor-relative coordinates and performs target onboarding with anchor forward/cache only.

- Source tensor: `artifacts/hidden_states/qwen3_4b_tau_success_policy_traces_hier_v2.pt`
- Target tensor: `artifacts/hidden_states/qwen25_3b_tau_success_policy_traces_hier_v2.pt`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries during target onboarding: `0`

## Best Run

| Strategy | K | Feature | Normalization | Policy | Source Val Acc | Target Acc | Agreement | Transition Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|
| transition_critical | 128 | all | center | mlp | 57.80% | 38.76% | 49.10% | 23.75% |

## All Runs

| Strategy | K | Feature | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy_critical | 128 | state | center | 60.84% | 54.89% | 38.16% | 41.18% | 51.29% | 13.99% | 26.55% |
| policy_critical | 128 | all | center | 61.56% | 58.57% | 37.20% | 41.76% | 46.13% | 13.68% | 20.97% |
| transition_critical | 128 | state | center | 58.82% | 54.71% | 23.90% | 24.13% | 34.26% | 7.38% | 17.56% |
| transition_critical | 128 | all | center | 57.80% | 55.82% | 38.76% | 42.49% | 49.10% | 14.01% | 23.75% |
