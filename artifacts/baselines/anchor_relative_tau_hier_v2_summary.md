# Anchor-Relative Tau-Hier-v2 Main Results

Goal: reduce the distillation-like target adapter setup by testing a zero-target-training anchor-relative policy interface on tau-hier-v2.

## Implementation

Added `baselines/anchor_relative_policy.py` with:

- policy anchor bank selection: `random`, `semantic`, `policy_prototypes`, `decision_boundary`, `policy_critical`, `policy_logdet`, `transition_critical`, `policy_transition_mixture`;
- model-independent relative coordinates: `sim(h(s), h(anchor_i))`;
- transition-aware features: `state`, `state_delta`, `state_transition`, `all`;
- policy-identifying pooled features: `state_pool`, `state_delta_pool`, `state_transition_pool`, `all_pool`, which append per-action mean/max similarity to source-labeled policy anchors;
- target onboarding with `Target labels = 0`, `Target gradient steps = 0`, `Source online queries = 0`;
- optional anchor-only closed-form target calibration: `none`, `ridge`, `orthogonal`.

The useful calibration in this round is `ridge`: it uses only anchor correspondence to map the target relative-coordinate frame to the source relative-coordinate frame. It does not use target action labels or target gradient training.

The latest update shifts effort away from non-parametric memory correction and into the zero-training policy interface itself. The new `policy_transition_mixture` anchor bank explicitly mixes:

- high-confidence action prototypes;
- low-margin decision-boundary anchors;
- trajectory transition anchors.

This better matches the scientific hypothesis: cross-backbone relational policy geometry should be identifiable from policy-critical anchors, not from arbitrary memory retrieval.

## Tau-Hier-v2 Results

Source: `Qwen3-4B` hidden states, source relative policy trained once.

| Target | Method | K | Feature | Calibration | Source Val Acc | Target Action Acc | Policy Agreement | Transition Agreement |
|---|---|---:|---|---|---:|---:|---:|---:|
| SmolLM3-3B | transition-critical anchors | 128 | all | none | 57.80% | 48.17% | 63.84% | 41.19% |
| SmolLM3-3B | transition-critical anchors | 512 | state | none | 70.52% | 37.40% | 38.96% | 14.46% |
| SmolLM3-3B | transition-critical anchors | 512 | state | ridge | 70.52% | 63.01% | 72.63% | 53.49% |
| Qwen2.5-3B | transition-critical anchors | 128 | all | none | 57.80% | 38.76% | 49.10% | 23.75% |
| Qwen2.5-3B | transition-critical anchors | 512 | state | none | 70.52% | 34.69% | 37.12% | 12.77% |
| Qwen2.5-3B | transition-critical anchors | 512 | state | ridge | 70.52% | 56.16% | 64.44% | 43.50% |

## Policy-Identifying Anchor Update

Fast 40-epoch diagnostic sweep, comparing the old `transition_critical` anchor bank to the new `policy_transition_mixture` anchor bank:

| Target | Strategy | K | Feature | Source Val Acc | Target Action Acc | Policy Agreement | Transition Agreement |
|---|---|---:|---|---:|---:|---:|---:|
| SmolLM3-3B | transition_critical | 512 | state | 64.31% | 55.86% | 74.71% | 57.96% |
| SmolLM3-3B | transition_critical | 512 | state_pool | 64.02% | 55.56% | 72.26% | 54.46% |
| SmolLM3-3B | policy_transition_mixture | 512 | state | 64.60% | 58.53% | 78.89% | 62.57% |
| SmolLM3-3B | policy_transition_mixture | 512 | state_pool | 65.90% | 59.65% | 77.96% | 61.30% |
| Qwen2.5-3B | transition_critical | 512 | state | 64.31% | 56.75% | 74.63% | 56.85% |
| Qwen2.5-3B | transition_critical | 512 | state_pool | 64.02% | 53.61% | 69.66% | 50.55% |
| Qwen2.5-3B | policy_transition_mixture | 512 | state | 64.60% | 57.85% | 76.14% | 58.10% |
| Qwen2.5-3B | policy_transition_mixture | 512 | state_pool | 65.90% | 58.14% | 75.73% | 57.58% |

Best-combo 150-epoch run:

| Target | Strategy | Feature | Source Val Acc | Source Full Acc | Target Action Acc | Policy Agreement | Transition Agreement |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | policy_transition_mixture | state_pool | 71.39% | 79.78% | 65.83% | 74.19% | 55.21% |
| Qwen2.5-3B | policy_transition_mixture | state_pool | 71.39% | 79.78% | 60.89% | 68.04% | 46.53% |

Compared with the previous strongest 150-epoch anchor-relative baseline, this improves SmolLM3 from `63.01%` to `65.83%` and Qwen2.5 from `56.16%` to `60.89%`, still with `Target labels = 0`, `Target gradient steps = 0`, and `Source online queries = 0`.

## Comparison To Adapter Method

| Target | Adapter-Based Ours | Anchor-Relative Ridge | Gap | Target Training |
|---|---:|---:|---:|---|
| SmolLM3-3B | 76.21% | 65.83% | -10.38 | adapter: yes; anchor-relative: no |
| Qwen2.5-3B | 75.30% | 60.89% | -14.41 | adapter: yes; anchor-relative: no |

## Takeaway

Pure anchor-relative coordinates are not sufficient on 25-action tau-hier-v2. Increasing K makes the source-side relative policy expressive enough, but direct target transfer collapses unless the target relative frame is calibrated. Anchor-only ridge calibration recovers a large part of the loss while preserving the intended onboarding regime: no target labels, no target gradient steps, and no source teacher queries during target onboarding.

The policy-identifying anchor update shows that the main zero-training bottleneck is not memory correction. A better anchor bank alone improves both SmolLM3 and Qwen2.5, which supports the idea that the interface should be built from action prototypes, decision boundaries, and policy transitions rather than arbitrary state anchors.

Memory/JIT should now be treated as an auxiliary negative study: it can increase source fidelity, but it does not reliably improve target correctness. The main next direction is to make the anchor-relative coordinates more policy-identifying, especially through richer boundary/transition coordinates and better anchor-only frame calibration.
