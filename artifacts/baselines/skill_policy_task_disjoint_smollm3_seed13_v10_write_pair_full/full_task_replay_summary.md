# SmolLM3 Full Tau Replay With Write-Pair Calibration

This run evaluates the deployable write-pair branch calibration on held-out task-disjoint tau-bench states.

## Setup

- Target backbone: `HuggingFaceTB/SmolLM3-3B`
- Source policy: Qwen3-4B frozen policy
- Split: task-disjoint, seed 13
- Held-out samples: 619
- Held-out tasks: 55
- Grounder: semantic
- Argument generator: oracle

## Replay Results

| Method | Action Acc | Tool Grounding Tool-Only | Args Valid | Tool Exec OK | DB Hash |
|---|---:|---:|---:|---:|---:|
| Static Low-Rank | 64.62% | 64.55% | 62.46% | 56.31% | 63.64% |
| Write-Pair Calibrated v2 | 64.78% | 63.88% | 63.88% | 57.53% | 63.64% |
| Response-Operator Obs Binder + Write-Pair v2 | 64.62% | 64.21% | 64.65% | 58.25% | 63.64% |

## Interpretation

The calibrated write-pair branch prior improves executable tool-call quality, especially `Tool Exec OK`, but does not change final DB hash on this held-out task split. This matches the counterfactual diagnosis: the update fixes a local success/failure branch weakness, but final task state is still dominated by earlier grounding/action choices and trajectory-level errors.

The result supports using the write-pair prior as a targeted component rather than treating it as the full solution for closed-loop policy transfer.
