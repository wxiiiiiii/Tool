# Baseline Comparison: tau-bench hierarchical action vocabulary

Setup:

- Source policy: Qwen3-4B hierarchical policy, frozen action head.
- Target backbone: SmolLM3-3B hidden states.
- Data: 4,615 tau-bench success-trajectory decision steps, 25 actions.
- Validation split: 15%, seed 13.
- Replay: semantic grounding + oracle arguments, 362 tasks.

## Policy-level baselines

| Method | Val Action Acc | Val Policy Consistency | Val Tool Action Acc | Trained Params | Training Signal |
|---|---:|---:|---:|---:|---|
| Ours: policy-aware adapter | 72.54% | 78.61% | 78.07% | 147,776 | target labels + frozen source head |
| Target-local linear probe | 71.68% | 79.62% | 77.49% | 51,225 | target labels |
| Unlock-style latent MSE | 69.94% | 83.09% | 76.32% | 147,776 | paired source latents |
| Pairwise linear ridge | 68.79% | 79.62% | 72.22% | 524,544 | paired source latents |
| Fixed slice projection | 7.08% | 5.78% | 2.05% | 0 | none |
| Random linear projection | 1.30% | 1.59% | 2.63% | 0 | none |
| Native SmolLM3 prompting | 0.00% | n/a | n/a | 0 | prompt only, first 60 samples |

## tau-bench replay baselines

| Method | Action Acc | Tool Grounding Acc | Tool Exec OK | DB Hash Match |
|---|---:|---:|---:|---:|
| Ours: Qwen frozen policy + SmolLM3 adapter | 82.60% | 82.23% | 71.95% | 79.01% |
| Target-local SmolLM3 policy | 81.37% | 80.00% | 70.96% | 76.80% |
| Qwen source policy on Qwen hidden states | 78.09% | 77.51% | 68.80% | 71.82% |
| Unlock-style latent MSE | 71.81% | 73.87% | 63.19% | 68.23% |
| Label-free KL distillation | 71.22% | 70.68% | 65.40% | 65.19% |
| Label-free state+KL distillation | 71.18% | 70.46% | 65.58% | 65.19% |

## Strong-claim check: no target action labels

The original CE adapter keeps the Qwen action head frozen, but it still trains the target adapter with target action labels. To test the stronger claim, we trained two extra adapters without target labels:

- `Label-free KL distillation`: optimize only `KL(p_source || p_target)` on paired prompts.
- `Label-free state+KL distillation`: optimize `MSE(z_target, z_source) + KL(p_source || p_target)`.

Both checkpoints record `uses_target_labels_for_training = false`. They transfer some source behavior, but they do not reach the CE adapter:

| Method | Full Action Acc | DB Hash Match | Target Labels Used For Training |
|---|---:|---:|---|
| CE policy-aware adapter | 82.60% | 79.01% | yes |
| Label-free KL distillation | 71.22% | 65.19% | no |
| Label-free state+KL distillation | 71.18% | 65.19% | no |

Conclusion: the current strong claim is not verified yet. The safer claim is that frozen-head target adapters are effective and efficient, but the best current adapter still benefits from target action labels.

Takeaway:

- Random/fixed projections collapse, showing the adapter is not a meaningless projection.
- Pairwise latent alignment and Unlock-style MSE preserve source predictions reasonably well, but underperform policy-aware adapter on ground-truth action accuracy and replay success.
- Target-local probe is competitive but requires training a target-specific action classifier; the proposed method keeps the source action head frozen and still wins on validation action accuracy and DB hash match.
- Native prompting is slow and format-unstable in this setup, even on a 60-sample smoke test.
- The stronger label-free policy-transfer claim needs more work: source-policy distillation works partially, but trails the target-label CE adapter by 11.38 action-accuracy points and 13.82 DB-match points.
