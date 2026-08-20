# Hidden Policy Baseline Results

- samples: 4615
- actions: 25
- validation split seed: 13

| Method | Val Action Acc | Val Policy Consistency | Val Tool Action Acc | Trained Params | Training Signal |
|---|---:|---:|---:|---:|---|
| unlock_lowrank_subspace | 67.05% | 79.77% | nan% | 524544 | paired_source_latents_lowrank_closed_form |
| unlock_lowrank_intervention | 66.76% | 79.48% | nan% | 524544 | paired_source_latents_lowrank_closed_form_plus_action_direction_intervention |
