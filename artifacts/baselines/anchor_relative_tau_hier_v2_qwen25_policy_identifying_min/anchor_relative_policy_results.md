# Anchor-Relative Policy Interface Results

This experiment trains the policy interface once on source anchor-relative coordinates and performs target onboarding with anchor forward/cache only.

- Source tensor: `artifacts/hidden_states/qwen3_4b_tau_success_policy_traces_hier_v2.pt`
- Target tensor: `artifacts/hidden_states/qwen25_3b_tau_success_policy_traces_hier_v2.pt`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries during target onboarding: `0`

## Best Run

| Strategy | K | Feature | Calibration | Normalization | Policy | Source Val Acc | Target Acc | Agreement | Transition Agreement |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| policy_transition_mixture | 512 | state_pool | ridge | center | mlp | 65.90% | 58.14% | 75.73% | 57.58% |

## All Runs

| Strategy | K | Feature | Calibration | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy_transition_mixture | 512 | state | ridge | center | 64.60% | 62.21% | 57.85% | 60.26% | 76.14% | 33.55% | 58.10% |
| policy_transition_mixture | 512 | state_pool | ridge | center | 65.90% | 64.33% | 58.14% | 60.84% | 75.73% | 34.26% | 57.58% |
| transition_critical | 512 | state | ridge | center | 64.31% | 63.51% | 56.75% | 60.55% | 74.63% | 33.04% | 56.85% |
| transition_critical | 512 | state_pool | ridge | center | 64.02% | 63.66% | 53.61% | 55.64% | 69.66% | 30.43% | 50.55% |
