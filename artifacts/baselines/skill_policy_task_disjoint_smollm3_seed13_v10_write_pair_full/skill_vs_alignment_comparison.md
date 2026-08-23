# Skill Methods vs Single-Step Alignment Baselines

Target: `SmolLM3-3B`  
Source policy: `Qwen3-4B`  
Main held-out setting: task-disjoint split, seed 13, 619 held-out decision states.  
Full task replay: 55 held-out tasks, semantic grounding, oracle arguments, tau-bench executor.

## Held-Out Single-Step Policy Comparison

These rows compare policy/action prediction on held-out decision states. This is the fairest view for asking whether skill improves over static one-step alignment.

| Method | Core Idea | Target Labels | Target Gradient | Action Acc | Transition Acc / Agreement | Notes |
|---|---|---:|---:|---:|---:|---|
| Unlock Pure Vector | one-vs-rest action directions in target hidden space | 0 | 0 | 51.05% | 27.13% trans acc | strictest pure-vector direction baseline |
| Faithful Unlock Direction | source capability directions + low-rank transfer | 0 | 0 | 62.36% | 40.43% trans acc | closer to original Unlock-style direction transfer |
| Paired Low-Rank Policy Alignment | closed-form target hidden to source policy latent map | 0 | 0 | 64.62% | 42.91% trans acc | strongest static alignment on same task-disjoint split |
| Anchor-Relative Best | anchor-relative policy coordinates + ridge calibration | 0 | 0 | 65.83% | 55.21% trans agreement | zero-training interface; from best 150-epoch anchor run |
| Best Deployable Skill Binder | source-distilled prefix skill binder, no action-logit input | 0 | 0 | 64.94% | 41.84% action trans / 51.77% skill trans | best deployable skill action result in v10 |
| Confidence-Gated Soft Skill | soft skill prior over primitive action logits | 0 | 0 | 64.78% | 42.91% action trans / 53.19% skill trans | small local gain over static action |
| Write-Pair Calibrated Skill | pre-observation write-pair calibration | 0 | 0 | 64.78% | 42.55% action trans / 52.66% skill trans | improves tool execution more than action acc |

## Full Task Replay Comparison

These rows evaluate whether action choices actually execute and reach the final DB state. This is where skill-level methods can differ from pure action accuracy.

| Method | Type | Action Acc | Tool Exec OK | DB Hash | Task Count |
|---|---|---:|---:|---:|---:|
| Static Low-Rank | deployable static alignment | 64.62% | 56.31% | 63.64% | 55 |
| Confidence-Gated Soft Skill | deployable skill prior | 64.78% | 56.45% | 63.64% | 55 |
| Source-Distilled Prefix Skill Binder no action logits | deployable skill binder | 64.94% | 56.69% | 63.64% | 55 |
| Write-Pair Calibrated v2 | deployable calibration | 64.78% | 57.53% | 63.64% | 55 |
| Response-Operator Obs Binder + Write-Pair v2 | deployable response residual | 64.62% | 58.25% | 63.64% | 55 |
| Source-Discovered Cluster Binder | deployable-ish skill clustering | 62.36% | 56.12% | 67.27% | 55 |
| Response-Operator Binder Allowed | deployable-ish response binder | 61.55% | 55.52% | 67.27% | 55 |
| Source Skill Rerank | source-skill oracle diagnostic | 77.06% | 66.99% | 72.73% | 55 |
| Oracle Skill Rerank | GT-skill upper bound | 87.88% | 68.40% | 80.00% | 55 |

## Full Replay Reference From Earlier Non-Task-Disjoint Runs

These are useful historical references, but not directly apples-to-apples with the task-disjoint 55-task split above because they use the full 4615-sample replay.

| Method | Samples / Tasks | Action Acc | Tool Exec OK | DB Hash |
|---|---:|---:|---:|---:|
| Unlock-style Latent MSE | 4615 / 362 | 71.81% | 63.19% | 68.23% |
| KL Adapter 30e | 4615 / 362 | 71.22% | 65.40% | 65.19% |
| KL Adapter rank128 60e | 4615 / 362 | 72.96% | 67.30% | 64.36% |
| KL Adapter + seq/adaptive transition | 4615 / 362 | 76.21% | 68.48% | 72.10% |
| Target-Local Probe | 4615 / 362 | 81.37% | 70.96% | 76.80% |
| Qwen Source Policy Replay | 4615 / 362 | 82.60% | 71.95% | 79.01% |

## Conclusion

The current deployable skill method does **not** yet beat the strongest single-step alignment methods on held-out Action Acc. Its best deployable Action Acc is `64.94%`, which is slightly above static low-rank (`64.62%`) but below the best Anchor-Relative result (`65.83%`) and below earlier full-replay trained adapter methods.

The stronger signal is task-level: several skill/response binders get lower Action Acc but higher DB Hash (`67.27%` vs static `63.64%`), and the source-skill/oracle-skill diagnostics reach `72.73%` and `80.00%` DB Hash. This means skill abstraction has real headroom, but the current deployable skill binder is not reliable enough yet.

So the clean current claim is:

> Single-step alignment remains stronger for local action prediction, while skill-level policy structure reveals a task-success headroom that static action accuracy misses. The next method needs to improve deployable skill binding, not just add more skill reranking on top of action logits.

