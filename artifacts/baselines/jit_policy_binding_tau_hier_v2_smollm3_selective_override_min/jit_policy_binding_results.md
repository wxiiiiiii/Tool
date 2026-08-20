# Just-in-Time Non-Parametric Policy Binding

- Target: `SmolLM3-3B`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries: `0`
- Anchor-only calibration: `ridge`

| Method | Memory | Filter | Memory Size | kNN | beta | gate | temp | Source Val | Source Full | Target Acc | Policy Agreement | Transition Agreement | Gate Open | Changed | Improved | Harmed | B Harm | C Improve |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_relative_base | none | all | 0 | 0 | 0.00 | 0.00 | 0.00 | 64.31% | 63.51% | 55.86% | 74.71% | 57.96% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| jit_selective | raw | source_reliable | 2709 | 8 | 0.25 | 0.00 | 0.05 | 64.88% | 65.35% | 55.43% | 73.93% | 57.42% | 6.11% | 5.01% | 1.56% | 1.99% | 10.20% | 5.37% |

## Best

`anchor_relative_base` with `none` memory reached 55.86% target action accuracy and 74.71% policy agreement.
