# Hidden Policy Baseline Results

- samples: 180
- actions: 3
- validation split seed: 13

| Method | Val Action Acc | Val Policy Consistency | Val Tool Action Acc | Trained Params | Training Signal |
|---|---:|---:|---:|---:|---|
| unlock_lowrank_subspace | 66.67% | 70.37% | nan% | 61504 | paired_source_latents_lowrank_closed_form |
| unlock_lowrank_intervention | 66.67% | 70.37% | nan% | 61504 | paired_source_latents_lowrank_closed_form_plus_action_direction_intervention |
