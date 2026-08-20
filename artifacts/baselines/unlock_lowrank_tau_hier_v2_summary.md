# Unlock Low-Rank Subspace Tau-Hier-v2 Results

Goal: test whether adding Unlock-style low-rank subspace alignment and action-direction intervention improves over the strict pure-vector Unlock baseline and Anchor-Relative.

## Method

Two new baselines were added to `baselines/hidden_policy_baselines.py`:

- `unlock_lowrank_subspace`: fit a closed-form ridge map from target hidden states to Qwen3 source policy latents, truncate the linear map to rank 64, then apply the frozen Qwen3 source action head.
- `unlock_lowrank_intervention`: apply the same low-rank subspace alignment, then add source-latent action-direction logits with weight `alpha`.

Supervision/onboarding conditions:

- Target action labels: 0
- Target gradient steps: 0
- Online source queries at evaluation: 0
- Training signal: paired source/target states and source policy latents

This is no longer a pure zero-fitting vector method: it estimates a low-rank cross-model subspace alignment from paired hidden states. It is therefore closer to an Unlock-style latent-alignment baseline than `unlock_pure_vector`.

## Results

Source: `Qwen3-4B` source policy on `tau_success_policy_traces_hier_v2`.

| Target | Method | Val Action Acc | Policy Agreement | Stored Alignment Params |
|---|---|---:|---:|---:|
| SmolLM3-3B | pure_vector_best_conf0.5 | 55.78% | 63.58% | 0 |
| SmolLM3-3B | lowrank_subspace | 68.93% | 79.77% | 524544 |
| SmolLM3-3B | lowrank_intervention_alpha0.05 | 68.93% | 79.77% | 524544 |
| SmolLM3-3B | lowrank_intervention_alpha0.10 | 68.50% | 80.06% | 524544 |
| SmolLM3-3B | lowrank_intervention_alpha0.25 | 68.06% | 78.90% | 524544 |
| SmolLM3-3B | lowrank_intervention_alpha0.50 | 67.63% | 78.03% | 524544 |
| Qwen2.5-3B | pure_vector_best_conf0.5 | 50.43% | 55.35% | 0 |
| Qwen2.5-3B | lowrank_subspace | 67.05% | 79.77% | 524544 |
| Qwen2.5-3B | lowrank_intervention_alpha0.05 | 67.49% | 80.06% | 524544 |
| Qwen2.5-3B | lowrank_intervention_alpha0.10 | 67.77% | 80.78% | 524544 |
| Qwen2.5-3B | lowrank_intervention_alpha0.25 | 66.76% | 79.48% | 524544 |
| Qwen2.5-3B | lowrank_intervention_alpha0.50 | 65.75% | 78.61% | 524544 |

## Comparison With Anchor-Relative

| Target | Anchor-Relative Best | Unlock Low-Rank Best | Difference |
|---|---:|---:|---:|
| SmolLM3-3B | 65.83% | 68.93% | +3.10 for Unlock low-rank |
| Qwen2.5-3B | 60.89% | 67.77% | +6.88 for Unlock low-rank |

## Takeaway

Adding low-rank subspace alignment substantially improves over pure-vector directions and beats the current Anchor-Relative zero-target-training result. The improvement mainly comes from subspace alignment, not from action-direction intervention: intervention is neutral on SmolLM3 and only slightly helpful on Qwen2.5 at `alpha=0.10`.

This means the earlier conclusion should be refined:

- Anchor-Relative beats strict pure-vector direction transfer.
- But low-rank paired-state subspace alignment is currently stronger than Anchor-Relative.
- The trade-off is conceptual: low-rank Unlock-style alignment relies on paired source/target hidden states and estimates a cross-model mapping, while Anchor-Relative aims for a more canonical, anchor-based policy interface with no target labels or gradients.

