# Hidden Policy Baseline Results

- samples: 180
- actions: 3
- validation split seed: 13

| Method | Val Action Acc | Val Policy Consistency | Val Tool Action Acc | Trained Params | Training Signal |
|---|---:|---:|---:|---:|---|
| target_local_linear_probe | 70.37% | 74.07% | nan% | 2883 | target_labels |
| ours_policy_aware_adapter | 70.37% | 74.07% | nan% | 32864 | target_labels_ce_frozen_source_head |
| pairwise_linear_ridge | 66.67% | 70.37% | nan% | 61504 | paired_source_latents |
| unlock_style_latent_mse | 66.67% | 77.78% | nan% | 65664 | paired_source_latents_mse |
| random_linear_projection | 37.04% | 40.74% | nan% | 0 | none |
