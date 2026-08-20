# State-Conditioned Dynamic Unlock Verification

This run tests whether replacing the naive global transition prior

```text
P(a_t | a_{t-1})
```

with a coarse state-conditioned prior

```text
P(a_t | a_{t-1}, state_bucket_t)
```

improves multi-step Unlock-style policy transfer.

Remote GPU:

- Host: `ssh-cn-huabei1.ebcloud.com`
- GPU: `NVIDIA GeForce RTX 4090`

## Method

Added two baselines:

| Method | Description |
|---|---|
| `unlock_lowrank_state_dynamic` | Uses Source policy transition counts conditioned on current state bucket. |
| `unlock_lowrank_state_dynamic_fallback` | Uses state-conditioned prior when available; falls back to global transition prior otherwise. |

The state bucket is a coarse symbolic summary of the decision state:

```text
domain | turn_phase | last_tool_kind | observation_status | entity_type | user_intent
```

Examples:

```text
retail | mid | retrieve | tool_observed | order | cancel
airline | late | update | write_success | reservation | modify
```

All methods use:

- target action labels: 0
- target gradient steps: 0
- online source queries: 0
- paired source/target hidden states: yes

## Sweep Summary

Main sweep:

- `dynamic_transition_weight`: `0.05`, `0.10`
- `dynamic_margin_threshold`: `0.05`, `0.10`
- `state_dynamic_min_count`: `2`, `4`
- prior confidence threshold: `0.35`

State-conditioned coverage was high:

| Target | min_count | State Bucket Coverage |
|---|---:|---:|
| SmolLM3 | 2 | about 92.6% |
| SmolLM3 | 4 | about 86.2% |
| Qwen2.5 | 2 | about 92.3% |
| Qwen2.5 | 4 | about 85.6% |

Best replay setting:

```text
method = unlock_lowrank_state_dynamic_fallback
weight = 0.05
margin_threshold = 0.05
state_dynamic_min_count = 2
```

## Full Replay Results

Replay uses semantic grounding, oracle arguments, and the tau-bench executor / DB-hash path.

| Target | Method | Action Acc | Seq Exact | Transition Pair | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|---:|
| SmolLM3 | static Unlock | 75.99% | 8.01% | 56.17% | 67.79% | 69.89% |
| SmolLM3 | naive global dynamic | 74.50% | 4.70% | 54.39% | 67.36% | 67.40% |
| SmolLM3 | state-conditioned dynamic | 75.90% | 7.73% | 55.98% | 67.64% | 69.89% |
| Qwen2.5 | static Unlock | 75.86% | 8.01% | 56.03% | 67.43% | 70.99% |
| Qwen2.5 | naive global dynamic | 73.85% | 5.25% | 53.26% | 66.23% | 66.57% |
| Qwen2.5 | state-conditioned dynamic | 75.86% | 8.01% | 55.87% | 67.36% | 70.99% |

## Interpretation

State-conditioned dynamic correction is much safer than the naive global prior: it avoids the large DB Hash regression seen with `P(a_t | a_{t-1})`.

However, it still does not improve over static Unlock. This suggests that the bottleneck is not merely missing a better action-history transition table. The coarse state bucket is too weak to capture the actual environment feedback required for sequential agent behavior.

The next optimization should move from symbolic bucket priors to observation-sensitive transition features, such as:

```text
tool_success / tool_error / empty_result / missing_slot / write_committed / confirmation_needed
```

and use these features to correct only specific high-impact transitions:

```text
ACT -> ASK
ACT -> RETRY
ACT -> STOP
ASK -> ACT
write_success -> STOP
tool_error -> RETRY
```

This result is useful for the paper: it shows that static latent alignment is hard to beat with shallow dynamic priors, and that true sequential policy transfer needs finer-grained environment feedback modeling rather than only action sequence statistics.
