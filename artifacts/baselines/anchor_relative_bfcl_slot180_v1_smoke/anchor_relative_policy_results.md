# Anchor-Relative Policy Interface Results

This experiment trains the policy interface once on source anchor-relative coordinates and performs target onboarding with anchor forward/cache only.

- Source tensor: `artifacts/hidden_states/qwen25_05b_bfcl_slot180.pt`
- Target tensor: `artifacts/hidden_states/smollm2_360m_bfcl_slot180.pt`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries during target onboarding: `0`

## Best Run

| Strategy | K | Feature | Normalization | Policy | Source Val Acc | Target Acc | Agreement | Transition Agreement |
|---|---:|---|---|---|---:|---:|---:|---:|
| policy_logdet | 64 | state | center | mlp | 88.89% | 72.78% | 75.00% | 0.00% |

## All Runs

| Strategy | K | Feature | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy_critical | 32 | state | center | 77.78% | 90.56% | 69.44% | 66.67% | 70.56% | 0.00% | 0.00% |
| policy_critical | 32 | state_delta | center | 74.07% | 88.89% | 64.44% | 62.96% | 65.56% | 0.00% | 0.00% |
| policy_critical | 32 | state_transition | center | 77.78% | 91.11% | 70.56% | 66.67% | 72.78% | 0.00% | 0.00% |
| policy_critical | 32 | all | center | 70.37% | 89.44% | 63.89% | 62.96% | 64.44% | 0.00% | 0.00% |
| policy_critical | 32 | state | whiten | 77.78% | 90.56% | 61.67% | 62.96% | 61.11% | 0.00% | 0.00% |
| policy_critical | 32 | state_delta | whiten | 70.37% | 93.89% | 60.00% | 62.96% | 60.56% | 0.00% | 0.00% |
| policy_critical | 32 | state_transition | whiten | 81.48% | 92.22% | 61.67% | 62.96% | 59.44% | 0.00% | 0.00% |
| policy_critical | 32 | all | whiten | 74.07% | 93.89% | 60.00% | 62.96% | 61.11% | 0.00% | 0.00% |
| policy_critical | 64 | state | center | 74.07% | 95.00% | 70.00% | 59.26% | 72.78% | 0.00% | 0.00% |
| policy_critical | 64 | state_delta | center | 74.07% | 94.44% | 68.89% | 70.37% | 67.78% | 0.00% | 0.00% |
| policy_critical | 64 | state_transition | center | 70.37% | 93.89% | 67.22% | 59.26% | 68.89% | 0.00% | 0.00% |
| policy_critical | 64 | all | center | 77.78% | 96.11% | 66.67% | 70.37% | 67.22% | 0.00% | 0.00% |
| policy_critical | 64 | state | whiten | 85.19% | 97.78% | 68.33% | 62.96% | 68.33% | 0.00% | 0.00% |
| policy_critical | 64 | state_delta | whiten | 77.78% | 96.67% | 60.56% | 66.67% | 60.56% | 0.00% | 0.00% |
| policy_critical | 64 | state_transition | whiten | 85.19% | 97.22% | 66.11% | 59.26% | 67.78% | 0.00% | 0.00% |
| policy_critical | 64 | all | whiten | 77.78% | 96.67% | 58.89% | 62.96% | 58.89% | 0.00% | 0.00% |
| policy_logdet | 32 | state | center | 85.19% | 90.56% | 68.89% | 59.26% | 68.33% | 0.00% | 0.00% |
| policy_logdet | 32 | state_delta | center | 74.07% | 89.44% | 62.22% | 66.67% | 64.44% | 0.00% | 0.00% |
| policy_logdet | 32 | state_transition | center | 81.48% | 89.44% | 67.78% | 59.26% | 71.67% | 0.00% | 0.00% |
| policy_logdet | 32 | all | center | 74.07% | 89.44% | 62.22% | 55.56% | 66.11% | 0.00% | 0.00% |
| policy_logdet | 32 | state | whiten | 85.19% | 93.33% | 63.33% | 48.15% | 65.56% | 0.00% | 0.00% |
| policy_logdet | 32 | state_delta | whiten | 70.37% | 92.78% | 63.33% | 51.85% | 65.00% | 0.00% | 0.00% |
| policy_logdet | 32 | state_transition | whiten | 81.48% | 94.44% | 65.00% | 55.56% | 66.67% | 0.00% | 0.00% |
| policy_logdet | 32 | all | whiten | 74.07% | 92.22% | 62.78% | 51.85% | 63.89% | 0.00% | 0.00% |
| policy_logdet | 64 | state | center | 88.89% | 96.67% | 72.78% | 66.67% | 75.00% | 0.00% | 0.00% |
| policy_logdet | 64 | state_delta | center | 74.07% | 95.00% | 62.22% | 62.96% | 62.78% | 0.00% | 0.00% |
| policy_logdet | 64 | state_transition | center | 88.89% | 96.11% | 67.78% | 59.26% | 70.56% | 0.00% | 0.00% |
| policy_logdet | 64 | all | center | 81.48% | 96.11% | 61.11% | 59.26% | 62.78% | 0.00% | 0.00% |
| policy_logdet | 64 | state | whiten | 85.19% | 96.67% | 67.78% | 62.96% | 71.11% | 0.00% | 0.00% |
| policy_logdet | 64 | state_delta | whiten | 81.48% | 97.22% | 63.89% | 62.96% | 65.00% | 0.00% | 0.00% |
| policy_logdet | 64 | state_transition | whiten | 81.48% | 96.67% | 67.22% | 55.56% | 69.44% | 0.00% | 0.00% |
| policy_logdet | 64 | all | whiten | 81.48% | 97.22% | 63.33% | 59.26% | 64.44% | 0.00% | 0.00% |
| transition_critical | 32 | state | center | 74.07% | 88.89% | 69.44% | 70.37% | 70.56% | 0.00% | 0.00% |
| transition_critical | 32 | state_delta | center | 70.37% | 90.56% | 62.78% | 62.96% | 63.89% | 0.00% | 0.00% |
| transition_critical | 32 | state_transition | center | 77.78% | 90.56% | 68.33% | 62.96% | 70.00% | 0.00% | 0.00% |
| transition_critical | 32 | all | center | 66.67% | 89.44% | 64.44% | 62.96% | 65.00% | 0.00% | 0.00% |
| transition_critical | 32 | state | whiten | 77.78% | 92.78% | 62.22% | 62.96% | 64.44% | 0.00% | 0.00% |
| transition_critical | 32 | state_delta | whiten | 77.78% | 95.00% | 58.33% | 66.67% | 61.11% | 0.00% | 0.00% |
| transition_critical | 32 | state_transition | whiten | 77.78% | 92.22% | 62.78% | 62.96% | 62.78% | 0.00% | 0.00% |
| transition_critical | 32 | all | whiten | 74.07% | 94.44% | 58.33% | 62.96% | 58.89% | 0.00% | 0.00% |
| transition_critical | 64 | state | center | 66.67% | 93.89% | 68.89% | 59.26% | 70.56% | 0.00% | 0.00% |
| transition_critical | 64 | state_delta | center | 74.07% | 94.44% | 67.78% | 66.67% | 68.89% | 0.00% | 0.00% |
| transition_critical | 64 | state_transition | center | 66.67% | 93.33% | 71.67% | 62.96% | 72.78% | 0.00% | 0.00% |
| transition_critical | 64 | all | center | 70.37% | 95.56% | 65.56% | 70.37% | 64.44% | 0.00% | 0.00% |
| transition_critical | 64 | state | whiten | 81.48% | 96.67% | 65.00% | 59.26% | 66.11% | 0.00% | 0.00% |
| transition_critical | 64 | state_delta | whiten | 77.78% | 96.67% | 61.11% | 70.37% | 61.11% | 0.00% | 0.00% |
| transition_critical | 64 | state_transition | whiten | 81.48% | 97.22% | 66.67% | 59.26% | 68.33% | 0.00% | 0.00% |
| transition_critical | 64 | all | whiten | 77.78% | 96.67% | 61.11% | 74.07% | 61.11% | 0.00% | 0.00% |
