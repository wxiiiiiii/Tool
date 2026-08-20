# Just-in-Time Non-Parametric Policy Binding

- Target: `smollm2-bfcl`
- Target labels used: `0`
- Target gradient steps: `0`
- Source online queries: `0`
- Anchor-only calibration: `ridge`

| Method | Memory | Memory Size | kNN | beta | temp | Source Val | Source Full | Target Acc | Policy Agreement | Transition Agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor_relative_base | none | 0 | 0 | 0.00 | 0.00 | 74.07% | 84.44% | 76.67% | 81.67% | 0.00% |
| jit_residual | raw | 153 | 4 | 0.50 | 0.05 | 77.78% | 95.00% | 82.22% | 85.00% | 0.00% |
| jit_residual | capsule | 24 | 4 | 0.50 | 0.05 | 77.78% | 80.00% | 72.78% | 78.33% | 0.00% |

## Best

`jit_residual` with `raw` memory reached 82.22% target action accuracy and 85.00% policy agreement.
