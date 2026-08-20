# Anchor-Relative Policy Interface Results

This experiment trains the policy interface once on source anchor-relative coordinates and performs target onboarding with anchor forward/cache only.

- Source tensor: `artifacts/hidden_states/qwen3_4b_tau_success_policy_traces_hier_v2.pt`
- Target tensor: `artifacts/hidden_states/smollm3_3b_tau_success_policy_traces_hier_v2.pt`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries during target onboarding: `0`

## Best Run

| Strategy | K | Feature | Calibration | Normalization | Policy | Source Val Acc | Target Acc | Agreement | Transition Agreement |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| policy_transition_mixture | 512 | state_pool | ridge | center | mlp | 65.90% | 59.65% | 77.96% | 61.30% |

## All Runs

| Strategy | K | Feature | Calibration | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy_transition_mixture | 512 | state | ridge | center | 64.60% | 62.21% | 58.53% | 61.42% | 78.89% | 34.23% | 62.57% |
| policy_transition_mixture | 512 | state_pool | ridge | center | 65.90% | 64.33% | 59.65% | 62.43% | 77.96% | 35.60% | 61.30% |
| transition_critical | 512 | state | ridge | center | 64.31% | 63.51% | 55.86% | 58.24% | 74.71% | 32.89% | 57.96% |
| transition_critical | 512 | state_pool | ridge | center | 64.02% | 63.66% | 55.56% | 57.37% | 72.26% | 32.80% | 54.46% |
