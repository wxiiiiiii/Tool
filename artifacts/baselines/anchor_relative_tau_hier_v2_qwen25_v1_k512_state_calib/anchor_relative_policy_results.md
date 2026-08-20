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
| transition_critical | 512 | state | ridge | center | mlp | 70.52% | 56.16% | 64.44% | 43.50% |

## All Runs

| Strategy | K | Feature | Calibration | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| transition_critical | 512 | state | none | center | 70.52% | 79.24% | 34.69% | 33.38% | 37.12% | 10.42% | 12.77% |
| transition_critical | 512 | state | ridge | center | 70.52% | 79.24% | 56.16% | 56.07% | 64.44% | 32.28% | 43.50% |
