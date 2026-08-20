# Hidden Policy Baseline Results

- samples: 4615
- actions: 25
- validation split seed: 13

| Method | Val Action Acc | Val Policy Consistency | Val Tool Action Acc | Trained Params | Training Signal |
|---|---:|---:|---:|---:|---|
| ours_policy_aware_adapter | 72.54% | 78.61% | 78.07% | 147776 | target_labels_ce_frozen_source_head |
| target_local_linear_probe | 71.68% | 79.62% | 77.49% | 51225 | target_labels |
| unlock_style_latent_mse | 69.94% | 83.09% | 76.32% | 147776 | paired_source_latents_mse |
| pairwise_linear_ridge | 68.79% | 79.62% | 72.22% | 524544 | paired_source_latents |
| fixed_slice_projection | 7.08% | 5.78% | 2.05% | 0 | none |
| random_linear_projection | 1.30% | 1.59% | 2.63% | 0 | none |
