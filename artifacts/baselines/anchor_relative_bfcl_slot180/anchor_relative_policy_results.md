# Anchor-Relative Policy Interface Results

This experiment trains the policy interface once on source anchor-relative coordinates and performs target onboarding with anchor forward/cache only.

- Source tensor: `artifacts/hidden_states/qwen25_05b_bfcl_slot180.pt`
- Target tensor: `artifacts/hidden_states/smollm2_360m_bfcl_slot180.pt`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries during target onboarding: `0`

## Best Run

| Strategy | K | Normalization | Policy | Source Val Acc | Target Acc | Source-Target Agreement |
|---|---:|---|---|---:|---:|---:|
| policy_critical | 128 | center | mlp | 77.78% | 78.89% | 81.11% |

## All Runs

| Strategy | K | Norm | Source Val | Source Full | Target Full | Target Val | Agreement |
|---|---:|---|---:|---:|---:|---:|---:|
| policy_critical | 8 | center | 66.67% | 82.78% | 40.56% | 70.37% | 35.56% |
| policy_critical | 8 | rank | 55.56% | 71.67% | 35.56% | 37.04% | 28.33% |
| policy_critical | 8 | raw | 81.48% | 77.78% | 36.11% | 40.74% | 35.00% |
| policy_critical | 8 | whiten | 66.67% | 81.67% | 43.89% | 48.15% | 42.78% |
| policy_critical | 16 | center | 66.67% | 95.00% | 57.78% | 62.96% | 57.22% |
| policy_critical | 16 | rank | 62.96% | 91.67% | 54.44% | 59.26% | 55.00% |
| policy_critical | 16 | raw | 85.19% | 93.33% | 52.22% | 48.15% | 51.11% |
| policy_critical | 16 | whiten | 81.48% | 97.22% | 53.33% | 44.44% | 55.00% |
| policy_critical | 32 | center | 77.78% | 96.67% | 63.89% | 62.96% | 65.00% |
| policy_critical | 32 | rank | 81.48% | 97.22% | 56.11% | 51.85% | 57.78% |
| policy_critical | 32 | raw | 88.89% | 98.33% | 55.56% | 48.15% | 57.22% |
| policy_critical | 32 | whiten | 74.07% | 96.11% | 62.78% | 66.67% | 63.33% |
| policy_critical | 64 | center | 74.07% | 96.11% | 69.44% | 59.26% | 72.22% |
| policy_critical | 64 | rank | 85.19% | 97.78% | 57.22% | 55.56% | 58.33% |
| policy_critical | 64 | raw | 77.78% | 96.67% | 63.89% | 48.15% | 67.22% |
| policy_critical | 64 | whiten | 85.19% | 97.78% | 65.00% | 59.26% | 67.22% |
| policy_critical | 128 | center | 77.78% | 96.67% | 78.89% | 66.67% | 81.11% |
| policy_critical | 128 | rank | 85.19% | 97.78% | 56.67% | 59.26% | 56.67% |
| policy_critical | 128 | raw | 77.78% | 96.67% | 65.56% | 66.67% | 66.11% |
| policy_critical | 128 | whiten | 77.78% | 96.67% | 75.56% | 70.37% | 77.22% |
| policy_prototypes | 8 | center | 59.26% | 83.89% | 57.22% | 66.67% | 57.78% |
| policy_prototypes | 8 | rank | 44.44% | 73.33% | 45.56% | 44.44% | 37.78% |
| policy_prototypes | 8 | raw | 66.67% | 80.00% | 51.67% | 66.67% | 50.56% |
| policy_prototypes | 8 | whiten | 55.56% | 81.11% | 44.44% | 55.56% | 44.44% |
| policy_prototypes | 16 | center | 62.96% | 94.44% | 62.78% | 70.37% | 61.67% |
| policy_prototypes | 16 | rank | 59.26% | 91.67% | 55.00% | 59.26% | 53.33% |
| policy_prototypes | 16 | raw | 70.37% | 93.89% | 66.11% | 59.26% | 66.67% |
| policy_prototypes | 16 | whiten | 66.67% | 95.00% | 61.67% | 70.37% | 61.11% |
| policy_prototypes | 32 | center | 74.07% | 96.11% | 64.44% | 48.15% | 67.22% |
| policy_prototypes | 32 | rank | 77.78% | 96.67% | 54.44% | 59.26% | 56.11% |
| policy_prototypes | 32 | raw | 81.48% | 97.22% | 65.56% | 62.96% | 66.11% |
| policy_prototypes | 32 | whiten | 81.48% | 97.22% | 63.89% | 62.96% | 64.44% |
| policy_prototypes | 64 | center | 77.78% | 96.67% | 71.67% | 62.96% | 73.89% |
| policy_prototypes | 64 | rank | 74.07% | 96.11% | 49.44% | 55.56% | 49.44% |
| policy_prototypes | 64 | raw | 81.48% | 97.22% | 65.00% | 70.37% | 64.44% |
| policy_prototypes | 64 | whiten | 81.48% | 97.22% | 70.00% | 62.96% | 70.56% |
| policy_prototypes | 128 | center | 81.48% | 97.22% | 78.33% | 66.67% | 80.00% |
| policy_prototypes | 128 | rank | 77.78% | 96.67% | 63.33% | 70.37% | 63.33% |
| policy_prototypes | 128 | raw | 77.78% | 96.67% | 61.11% | 66.67% | 61.11% |
| policy_prototypes | 128 | whiten | 81.48% | 97.22% | 72.78% | 62.96% | 74.44% |
| random | 8 | center | 66.67% | 87.78% | 46.11% | 51.85% | 44.44% |
| random | 8 | rank | 66.67% | 76.11% | 37.22% | 29.63% | 37.78% |
| random | 8 | raw | 74.07% | 86.67% | 37.78% | 44.44% | 37.78% |
| random | 8 | whiten | 77.78% | 88.33% | 49.44% | 59.26% | 47.22% |
| random | 16 | center | 77.78% | 96.67% | 60.56% | 66.67% | 59.44% |
| random | 16 | rank | 66.67% | 95.00% | 59.44% | 66.67% | 57.78% |
| random | 16 | raw | 77.78% | 96.11% | 48.89% | 55.56% | 46.67% |
| random | 16 | whiten | 85.19% | 97.78% | 65.56% | 81.48% | 64.44% |
| random | 32 | center | 81.48% | 97.22% | 63.33% | 59.26% | 63.89% |
| random | 32 | rank | 74.07% | 96.11% | 60.00% | 70.37% | 58.33% |
| random | 32 | raw | 85.19% | 97.22% | 50.00% | 59.26% | 51.11% |
| random | 32 | whiten | 81.48% | 97.22% | 68.89% | 66.67% | 69.44% |
| random | 64 | center | 74.07% | 96.11% | 63.89% | 59.26% | 64.44% |
| random | 64 | rank | 88.89% | 98.33% | 50.56% | 44.44% | 51.11% |
| random | 64 | raw | 81.48% | 97.22% | 55.56% | 59.26% | 56.67% |
| random | 64 | whiten | 77.78% | 96.67% | 66.67% | 66.67% | 66.67% |
| random | 128 | center | 77.78% | 96.67% | 71.67% | 66.67% | 73.89% |
| random | 128 | rank | 81.48% | 97.22% | 50.56% | 59.26% | 51.11% |
| random | 128 | raw | 81.48% | 97.22% | 63.89% | 70.37% | 64.44% |
| random | 128 | whiten | 77.78% | 96.67% | 70.56% | 66.67% | 72.22% |
| semantic | 8 | center | 88.89% | 90.56% | 53.33% | 62.96% | 59.44% |
| semantic | 8 | rank | 70.37% | 78.33% | 52.78% | 44.44% | 58.89% |
| semantic | 8 | raw | 85.19% | 83.33% | 53.89% | 62.96% | 61.67% |
| semantic | 8 | whiten | 81.48% | 87.78% | 47.22% | 55.56% | 46.67% |
| semantic | 16 | center | 74.07% | 95.00% | 58.33% | 55.56% | 57.22% |
| semantic | 16 | rank | 70.37% | 92.78% | 41.11% | 44.44% | 41.11% |
| semantic | 16 | raw | 81.48% | 96.11% | 48.89% | 48.15% | 48.89% |
| semantic | 16 | whiten | 77.78% | 96.67% | 60.00% | 55.56% | 61.11% |
| semantic | 32 | center | 77.78% | 96.67% | 62.22% | 55.56% | 63.33% |
| semantic | 32 | rank | 81.48% | 97.22% | 56.67% | 62.96% | 57.22% |
| semantic | 32 | raw | 81.48% | 97.22% | 60.56% | 66.67% | 62.78% |
| semantic | 32 | whiten | 81.48% | 97.22% | 61.11% | 62.96% | 63.33% |
| semantic | 64 | center | 81.48% | 97.22% | 71.67% | 62.96% | 73.33% |
| semantic | 64 | rank | 77.78% | 96.67% | 60.56% | 62.96% | 60.56% |
| semantic | 64 | raw | 81.48% | 97.22% | 63.89% | 70.37% | 64.44% |
| semantic | 64 | whiten | 77.78% | 96.67% | 66.11% | 62.96% | 67.22% |
| semantic | 128 | center | 77.78% | 96.67% | 76.11% | 66.67% | 78.33% |
| semantic | 128 | rank | 77.78% | 96.67% | 46.67% | 48.15% | 46.67% |
| semantic | 128 | raw | 77.78% | 96.67% | 60.56% | 66.67% | 60.56% |
| semantic | 128 | whiten | 77.78% | 96.67% | 68.89% | 59.26% | 71.11% |
