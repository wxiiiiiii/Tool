# Just-in-Time Non-Parametric Policy Binding

- Target: `Qwen2.5-3B`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries: `0`
- Anchor-only calibration: `ridge`

| Method | Memory | Memory Size | kNN | beta | gate | temp | Source Val | Source Full | Target Acc | Policy Agreement | Transition Agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_relative_base | none | 0 | 0 | 0.00 | 0.00 | 0.00 | 68.06% | 70.36% | 57.44% | 71.94% | 53.09% |
| jit_gated | raw | 3923 | 8 | 0.50 | 0.05 | 0.05 | 68.93% | 71.53% | 56.29% | 71.98% | 54.90% |

## Best

`anchor_relative_base` with `none` memory reached 57.44% target action accuracy and 71.94% policy agreement.
