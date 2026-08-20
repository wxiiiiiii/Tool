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
| transition_critical | 512 | state | ridge | center | mlp | 70.52% | 63.01% | 72.63% | 53.49% |

## All Runs

| Strategy | K | Feature | Calibration | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| transition_critical | 512 | state | none | center | 70.52% | 79.24% | 37.40% | 37.14% | 38.96% | 12.84% | 14.46% |
| transition_critical | 512 | state | ridge | center | 70.52% | 79.24% | 63.01% | 61.85% | 72.63% | 39.45% | 53.49% |
