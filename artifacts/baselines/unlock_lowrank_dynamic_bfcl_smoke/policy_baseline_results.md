# Hidden Policy Baseline Results

- samples: 90
- actions: 3
- validation split seed: 13

| Method | Val Action Acc | Val Policy Consistency | Val Tool Action Acc | Trained Params | Training Signal |
|---|---:|---:|---:|---:|---|
| unlock_lowrank_subspace | 76.92% | 84.62% | nan% | 61504 | paired_source_latents_lowrank_closed_form |
| unlock_lowrank_dynamic | 76.92% | 84.62% | nan% | 61504 | paired_source_latents_lowrank_closed_form_plus_source_transition_prior |
