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
| semantic | 128 | center | linear | 81.48% | 76.11% | 77.78% |

## All Runs

| Strategy | K | Norm | Source Val | Source Full | Target Full | Target Val | Agreement |
|---|---:|---|---:|---:|---:|---:|---:|
| policy_critical | 8 | center | 66.67% | 67.78% | 31.67% | 44.44% | 37.78% |
| policy_critical | 8 | rank | 51.85% | 59.44% | 48.89% | 62.96% | 52.22% |
| policy_critical | 8 | raw | 74.07% | 65.00% | 34.44% | 37.04% | 33.89% |
| policy_critical | 8 | whiten | 74.07% | 63.33% | 47.78% | 55.56% | 55.56% |
| policy_critical | 16 | center | 74.07% | 78.33% | 58.89% | 70.37% | 56.11% |
| policy_critical | 16 | rank | 70.37% | 72.22% | 41.67% | 37.04% | 45.00% |
| policy_critical | 16 | raw | 77.78% | 77.78% | 39.44% | 40.74% | 37.78% |
| policy_critical | 16 | whiten | 66.67% | 74.44% | 57.78% | 59.26% | 53.33% |
| policy_critical | 32 | center | 81.48% | 90.00% | 65.00% | 66.67% | 66.67% |
| policy_critical | 32 | rank | 62.96% | 85.00% | 52.78% | 55.56% | 52.22% |
| policy_critical | 32 | raw | 77.78% | 87.78% | 53.89% | 55.56% | 52.22% |
| policy_critical | 32 | whiten | 70.37% | 88.89% | 59.44% | 66.67% | 60.00% |
| policy_critical | 64 | center | 81.48% | 97.22% | 69.44% | 59.26% | 72.22% |
| policy_critical | 64 | rank | 77.78% | 95.00% | 46.67% | 37.04% | 46.11% |
| policy_critical | 64 | raw | 85.19% | 96.11% | 62.22% | 59.26% | 63.89% |
| policy_critical | 64 | whiten | 81.48% | 97.22% | 65.56% | 62.96% | 66.11% |
| policy_critical | 128 | center | 85.19% | 97.78% | 75.00% | 55.56% | 77.22% |
| policy_critical | 128 | rank | 77.78% | 96.67% | 53.89% | 59.26% | 53.89% |
| policy_critical | 128 | raw | 81.48% | 97.22% | 60.00% | 59.26% | 59.44% |
| policy_critical | 128 | whiten | 81.48% | 97.22% | 73.89% | 70.37% | 74.44% |
| policy_prototypes | 8 | center | 48.15% | 61.11% | 58.89% | 62.96% | 60.00% |
| policy_prototypes | 8 | rank | 44.44% | 55.00% | 52.78% | 59.26% | 43.89% |
| policy_prototypes | 8 | raw | 59.26% | 62.78% | 55.00% | 62.96% | 48.89% |
| policy_prototypes | 8 | whiten | 59.26% | 60.00% | 47.78% | 48.15% | 52.22% |
| policy_prototypes | 16 | center | 62.96% | 79.44% | 61.11% | 62.96% | 57.78% |
| policy_prototypes | 16 | rank | 62.96% | 76.11% | 58.89% | 66.67% | 54.44% |
| policy_prototypes | 16 | raw | 62.96% | 80.00% | 52.78% | 51.85% | 51.11% |
| policy_prototypes | 16 | whiten | 66.67% | 81.11% | 60.56% | 62.96% | 61.11% |
| policy_prototypes | 32 | center | 74.07% | 93.89% | 60.56% | 51.85% | 63.33% |
| policy_prototypes | 32 | rank | 66.67% | 84.44% | 46.11% | 48.15% | 46.11% |
| policy_prototypes | 32 | raw | 74.07% | 92.22% | 63.33% | 59.26% | 63.33% |
| policy_prototypes | 32 | whiten | 77.78% | 92.78% | 63.33% | 59.26% | 65.56% |
| policy_prototypes | 64 | center | 77.78% | 96.67% | 67.78% | 55.56% | 68.89% |
| policy_prototypes | 64 | rank | 77.78% | 96.67% | 53.33% | 55.56% | 53.33% |
| policy_prototypes | 64 | raw | 88.89% | 97.78% | 63.89% | 70.37% | 65.00% |
| policy_prototypes | 64 | whiten | 81.48% | 97.22% | 67.22% | 51.85% | 68.89% |
| policy_prototypes | 128 | center | 85.19% | 97.78% | 76.11% | 55.56% | 77.22% |
| policy_prototypes | 128 | rank | 77.78% | 96.67% | 56.11% | 66.67% | 56.11% |
| policy_prototypes | 128 | raw | 77.78% | 96.67% | 45.00% | 40.74% | 44.44% |
| policy_prototypes | 128 | whiten | 81.48% | 97.22% | 75.56% | 59.26% | 76.11% |
| random | 8 | center | 77.78% | 73.33% | 52.22% | 51.85% | 51.67% |
| random | 8 | rank | 55.56% | 66.67% | 33.33% | 37.04% | 37.78% |
| random | 8 | raw | 70.37% | 73.33% | 33.33% | 37.04% | 34.44% |
| random | 8 | whiten | 74.07% | 73.33% | 47.78% | 55.56% | 50.56% |
| random | 16 | center | 70.37% | 82.78% | 62.22% | 59.26% | 60.00% |
| random | 16 | rank | 55.56% | 80.00% | 45.00% | 55.56% | 42.22% |
| random | 16 | raw | 70.37% | 84.44% | 42.22% | 48.15% | 41.67% |
| random | 16 | whiten | 66.67% | 79.44% | 56.67% | 66.67% | 60.56% |
| random | 32 | center | 70.37% | 88.33% | 70.56% | 66.67% | 71.67% |
| random | 32 | rank | 62.96% | 88.33% | 59.44% | 70.37% | 56.67% |
| random | 32 | raw | 66.67% | 88.33% | 62.22% | 70.37% | 60.00% |
| random | 32 | whiten | 66.67% | 88.89% | 68.33% | 66.67% | 68.33% |
| random | 64 | center | 81.48% | 97.22% | 65.56% | 66.67% | 65.00% |
| random | 64 | rank | 81.48% | 96.11% | 49.44% | 59.26% | 48.89% |
| random | 64 | raw | 77.78% | 96.67% | 60.56% | 70.37% | 59.44% |
| random | 64 | whiten | 77.78% | 96.67% | 65.00% | 70.37% | 62.78% |
| random | 128 | center | 85.19% | 97.78% | 72.22% | 62.96% | 73.33% |
| random | 128 | rank | 81.48% | 97.22% | 52.78% | 59.26% | 53.33% |
| random | 128 | raw | 77.78% | 96.67% | 56.11% | 51.85% | 55.00% |
| random | 128 | whiten | 81.48% | 97.22% | 75.00% | 62.96% | 75.00% |
| semantic | 8 | center | 74.07% | 74.44% | 57.78% | 66.67% | 61.67% |
| semantic | 8 | rank | 70.37% | 65.00% | 53.33% | 62.96% | 56.67% |
| semantic | 8 | raw | 81.48% | 70.00% | 49.44% | 62.96% | 58.33% |
| semantic | 8 | whiten | 81.48% | 73.89% | 51.67% | 62.96% | 51.11% |
| semantic | 16 | center | 66.67% | 79.44% | 53.33% | 44.44% | 56.11% |
| semantic | 16 | rank | 70.37% | 78.89% | 59.44% | 55.56% | 55.00% |
| semantic | 16 | raw | 66.67% | 78.89% | 51.67% | 51.85% | 54.44% |
| semantic | 16 | whiten | 62.96% | 81.11% | 55.00% | 51.85% | 60.00% |
| semantic | 32 | center | 77.78% | 92.22% | 55.00% | 59.26% | 58.89% |
| semantic | 32 | rank | 74.07% | 86.11% | 57.78% | 62.96% | 54.44% |
| semantic | 32 | raw | 85.19% | 92.78% | 44.44% | 40.74% | 45.56% |
| semantic | 32 | whiten | 77.78% | 92.22% | 56.67% | 59.26% | 58.33% |
| semantic | 64 | center | 81.48% | 97.22% | 71.11% | 62.96% | 71.67% |
| semantic | 64 | rank | 70.37% | 95.56% | 56.11% | 62.96% | 55.00% |
| semantic | 64 | raw | 81.48% | 97.22% | 62.78% | 70.37% | 63.33% |
| semantic | 64 | whiten | 81.48% | 97.22% | 71.11% | 62.96% | 73.89% |
| semantic | 128 | center | 81.48% | 97.22% | 76.11% | 59.26% | 77.78% |
| semantic | 128 | rank | 77.78% | 96.67% | 46.11% | 48.15% | 46.11% |
| semantic | 128 | raw | 77.78% | 96.67% | 48.33% | 40.74% | 48.33% |
| semantic | 128 | whiten | 88.89% | 98.33% | 72.22% | 62.96% | 73.33% |
