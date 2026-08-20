# Unlock Pure-Vector Tau-Hier-v2 Results

Goal: replace the previous trained `unlock_style_latent_mse` adapter baseline with a pure-vector Unlock-style baseline, then compare against zero-target-training Anchor-Relative.

## Method

`unlock_pure_vector` builds one-vs-rest action/capability directions directly in the target hidden space. It uses paired target activations grouped by Qwen3 source-policy pseudo labels, then predicts actions by cosine projection onto those directions.

Supervision/onboarding conditions:

- Target action labels: 0
- Target gradient steps: 0
- Trainable target adapter parameters: 0
- Online source queries at evaluation: 0
- Stored object: target-space direction vectors, center, and bias

This is intentionally stricter than the old `unlock_style_latent_mse`, which trains a low-rank target adapter with paired source latents.

## Pure-Vector Unlock Sweep

Source: `Qwen3-4B` source policy on `tau_success_policy_traces_hier_v2`.

| Target | Method | Val Action Acc | Policy Agreement | Fit Samples | Trained Params |
|---|---|---:|---:|---:|---:|
| SmolLM3-3B | unlock_pure_vector | 54.91% | 62.57% | 3923 | 0 |
| SmolLM3-3B | unlock_pure_vector_conf0.5 | 55.78% | 63.58% | 3305 | 0 |
| SmolLM3-3B | unlock_pure_vector_conf0.7 | 55.06% | 61.71% | 2452 | 0 |
| SmolLM3-3B | unlock_pure_vector_conf0.9 | 51.45% | 56.50% | 1143 | 0 |
| Qwen2.5-3B | unlock_pure_vector | 47.98% | 53.47% | 3923 | 0 |
| Qwen2.5-3B | unlock_pure_vector_conf0.5 | 50.43% | 55.35% | 3305 | 0 |
| Qwen2.5-3B | unlock_pure_vector_conf0.7 | 49.13% | 54.62% | 2452 | 0 |
| Qwen2.5-3B | unlock_pure_vector_conf0.9 | 44.22% | 47.69% | 1143 | 0 |

## Comparison With Anchor-Relative

| Target | Pure-Vector Unlock Best | Anchor-Relative Best | Gap |
|---|---:|---:|---:|
| SmolLM3-3B | 55.78% | 65.83% | +10.05 for Anchor-Relative |
| Qwen2.5-3B | 50.43% | 60.89% | +10.46 for Anchor-Relative |

For reference, the older `unlock_style_latent_mse` result on the tau-hier-v2 hidden baseline table is `69.94%`, but it uses `147,776` trainable target-adapter parameters and is therefore not a pure-vector baseline.

## Takeaway

Anchor-Relative beats the strict pure-vector Unlock-style baseline on both targets under the same zero-target-label, zero-target-gradient onboarding regime. The result supports the current claim that raw target-space capability directions are weaker than policy-critical relative coordinates with anchor-only frame calibration.

