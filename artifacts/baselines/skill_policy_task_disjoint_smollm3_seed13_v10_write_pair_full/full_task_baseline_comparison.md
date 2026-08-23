# SmolLM3 Full Tau Replay Baseline Comparison

All methods use the same held-out task-disjoint split, semantic grounding, oracle argument generation, and tau-bench executor.

## Results

| Method | Type | Action Acc | Tool Grounding Tool-Only | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|
| Static Low-Rank | deployable | 64.62% | 64.55% | 56.31% | 63.64% |
| Confidence-Gated Soft Skill | deployable | 64.78% | 64.88% | 56.45% | 63.64% |
| Source-Distilled Prefix Skill Binder no action logits | deployable | 64.94% | 64.88% | 56.69% | 63.64% |
| Write-Pair Calibrated v2 | deployable | 64.78% | 63.88% | 57.53% | 63.64% |
| Response-Operator Obs Binder Residual + Write-Pair v2 | deployable | 64.62% | 64.21% | 58.25% | 63.64% |
| Source-Discovered Cluster Binder | deployable-ish | 62.36% | 59.20% | 56.12% | 67.27% |
| Response-Operator Binder Allowed | deployable-ish | 61.55% | 61.20% | 55.52% | 67.27% |
| Response-Operator Obs Binder Allowed | deployable-ish | 60.10% | 58.19% | 54.33% | 67.27% |
| Source-Discovered Cluster Oracle | diagnostic oracle | 65.59% | 63.55% | 62.41% | 67.27% |
| Response-Operator Oracle Allowed | diagnostic oracle | 65.91% | 63.88% | 59.93% | 67.27% |
| Source Skill Rerank | source-skill oracle | 77.06% | 75.25% | 66.99% | 72.73% |
| Oracle Skill Rerank | GT-skill upper | 87.88% | 81.61% | 68.40% | 80.00% |

## Takeaways

- `63.64%` DB Hash is `35/55` held-out tasks. Because DB Hash is task-level, small changes in action accuracy may not change the final count.
- The write-pair calibration improves executable tool quality (`Tool Exec OK`: `56.31% -> 58.25%`) but does not move DB Hash beyond `35/55`.
- Some deployable-ish binder methods have lower action/tool metrics but reach `67.27%` DB Hash (`37/55`), showing that task success is not perfectly correlated with single-step action accuracy.
- The headroom is clear: source-skill and oracle-skill diagnostics reach `72.73%` and `80.00%` DB Hash. This suggests the main missing component is not argument generation, but more reliable high-level skill/branch binding before concrete action selection.

## Interpretation

The current deployable method mostly improves local executable correctness, but not enough task-level decisions flip from failure to success. The strongest diagnostic evidence points toward improving cross-backbone high-level skill binding rather than further tuning the write-pair correction alone.
