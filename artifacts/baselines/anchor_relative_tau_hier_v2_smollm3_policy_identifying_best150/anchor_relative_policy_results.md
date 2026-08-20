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
| policy_transition_mixture | 512 | state_pool | ridge | center | mlp | 71.39% | 65.83% | 74.19% | 55.21% |

## All Runs

| Strategy | K | Feature | Calibration | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy_transition_mixture | 512 | state_pool | ridge | center | 71.39% | 79.78% | 65.83% | 63.29% | 74.19% | 42.42% | 55.21% |
