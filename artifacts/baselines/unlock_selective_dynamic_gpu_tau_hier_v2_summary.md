# Selective Dynamic Unlock Verification

This run verifies whether a state-uncertainty gated dynamic transition prior can improve over static Unlock-style low-rank alignment.

Remote GPU:

- Host: `ssh-cn-huabei1.ebcloud.com`
- GPU: `NVIDIA GeForce RTX 4090`
- PyTorch CUDA: available

## Method

`unlock_lowrank_selective_dynamic` keeps the static Unlock low-rank alignment, but applies the Source transition prior only when:

```text
target action margin <= threshold
and
source transition prior confidence >= threshold
```

This is intended to avoid the failure mode of naive dynamic correction, where every step is biased by a global `P(a_t | a_{t-1})` prior.

All runs use:

- target action labels: 0
- target gradient steps: 0
- online source queries: 0
- source/target paired hidden states: yes

## Threshold Sweep

Main sweep used `prior_confidence_threshold=0.35`.

| Target | Weight | Margin | Gate Open | Full Action Acc | Val Action Acc | Tool Val Acc |
|---|---:|---:|---:|---:|---:|---:|
| SmolLM3 | 0.35 | 0.05 | 4.02% | 75.58% | 68.06% | 71.64% |
| SmolLM3 | 0.35 | 0.10 | 8.09% | 75.04% | 67.77% | 71.64% |
| SmolLM3 | 0.35 | 0.15 | 11.97% | 74.80% | 68.06% | 72.22% |
| SmolLM3 | 0.35 | 0.20 | 15.42% | 74.73% | 68.21% | 72.22% |
| SmolLM3 | 0.05 | 0.05 | 4.02% | 75.86% | 68.93% | 72.51% |
| SmolLM3 | 0.05 | 0.10 | 8.02% | 75.84% | 68.93% | 72.51% |
| Qwen2.5 | 0.35 | 0.05 | 4.49% | 75.25% | 67.34% | 72.51% |
| Qwen2.5 | 0.35 | 0.10 | 8.86% | 74.71% | 67.05% | 71.35% |
| Qwen2.5 | 0.35 | 0.15 | 13.03% | 74.34% | 67.20% | 70.76% |
| Qwen2.5 | 0.35 | 0.20 | 16.04% | 74.04% | 67.63% | 71.35% |
| Qwen2.5 | 0.05 | 0.05 | 4.49% | 75.64% | 67.34% | 73.10% |
| Qwen2.5 | 0.05 | 0.10 | 8.84% | 75.60% | 67.20% | 72.81% |

Best conservative setting chosen for full replay:

```text
weight = 0.05
margin_threshold = 0.05
prior_confidence_threshold = 0.35
```

## Full Replay Comparison

Replay uses semantic grounding, oracle arguments, and the same tau-bench executor / DB-hash path.

| Target | Method | Action Acc | Seq Exact | Transition Pair | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|---:|
| SmolLM3 | static Unlock lowrank | 75.99% | 8.01% | 56.17% | 67.79% | 69.89% |
| SmolLM3 | naive dynamic prior | 74.50% | 4.70% | 54.39% | 67.36% | 67.40% |
| SmolLM3 | selective dynamic, w=.05/m=.05 | 75.86% | 8.01% | 56.05% | 67.76% | 69.89% |
| Qwen2.5 | static Unlock lowrank | 75.86% | 8.01% | 56.03% | 67.43% | 70.99% |
| Qwen2.5 | naive dynamic prior | 73.85% | 5.25% | 53.26% | 66.23% | 66.57% |
| Qwen2.5 | selective dynamic, w=.05/m=.05 | 75.64% | 8.01% | 55.49% | 67.19% | 70.72% |

## Conclusion

Selective dynamic correction successfully prevents the large regression caused by the naive global transition prior. However, it does not improve over static Unlock low-rank alignment:

- SmolLM3 DB Hash stays flat at `69.89%`.
- Qwen2.5 DB Hash drops slightly from `70.99%` to `70.72%`.
- Seq Exact remains `8.01%`.

This verifies that the current multi-step bottleneck is not solved by action-history transition priors, even when gated by target uncertainty. The next useful dynamic method should condition on structured observations and environment state, for example:

```text
P(a_t | a_{t-1}, tool_success, tool_error, missing_slot, db_updated, need_confirmation)
```

Rather than:

```text
P(a_t | a_{t-1})
```

This negative result is useful for the paper: static Unlock is a strong per-state action-boundary transfer baseline, while simple dynamic priors do not preserve full sequential Agent policy dynamics.
