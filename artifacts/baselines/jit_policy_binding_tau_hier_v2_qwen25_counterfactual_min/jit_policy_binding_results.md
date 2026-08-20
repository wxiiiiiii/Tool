# Just-in-Time Non-Parametric Policy Binding

- Target: `Qwen2.5-3B`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries: `0`
- Anchor-only calibration: `ridge`

| Method | Memory | Filter | Memory Size | kNN | beta | gate | temp | Source Val | Source Full | Target Acc | Policy Agreement | Transition Agreement | Gate Open | Changed | Improved | Harmed | B Harm | C Improve |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_relative_base | none | all | 0 | 0 | 0.00 | 0.00 | 0.00 | 64.31% | 63.51% | 56.75% | 74.63% | 56.85% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| jit_counterfactual | raw | source_reliable | 2709 | 8 | 0.25 | 0.00 | 0.05 | 65.03% | 65.33% | 56.29% | 75.19% | 57.98% | 7.11% | 5.98% | 1.80% | 2.25% | 9.68% | 6.14% |

## Best

`anchor_relative_base` with `none` memory reached 56.75% target action accuracy and 74.63% policy agreement.
