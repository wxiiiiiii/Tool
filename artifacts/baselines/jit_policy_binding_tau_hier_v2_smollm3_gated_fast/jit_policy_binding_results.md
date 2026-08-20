# Just-in-Time Non-Parametric Policy Binding

- Target: `SmolLM3-3B`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries: `0`
- Anchor-only calibration: `ridge`

| Method | Memory | Memory Size | kNN | beta | gate | temp | Source Val | Source Full | Target Acc | Policy Agreement | Transition Agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_relative_base | none | 0 | 0 | 0.00 | 0.00 | 0.00 | 68.06% | 70.36% | 60.72% | 74.45% | 56.85% |
| jit_gated | raw | 3923 | 8 | 0.50 | 0.05 | 0.05 | 68.93% | 71.53% | 59.63% | 75.19% | 57.82% |

## Best

`anchor_relative_base` with `none` memory reached 60.72% target action accuracy and 74.45% policy agreement.
