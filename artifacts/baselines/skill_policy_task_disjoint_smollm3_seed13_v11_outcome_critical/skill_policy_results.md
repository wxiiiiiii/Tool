# Skill Policy Validation

- held-out samples: 619
- split: `task`

## Skill Taxonomy

| Skill | Actions |
|---|---|
| COLLECT_INFO | ACT_RETRIEVE_USER, ACT_RETRIEVE_ORDER, ACT_RETRIEVE_PRODUCT, ACT_RETRIEVE_CATALOG, ACT_RETRIEVE_RESERVATION, ACT_RETRIEVE_AIRPORT, ACT_SEARCH_FLIGHT, ACT_COMPUTE |
| CONFIRM | VERIFY |
| EXECUTE | ACT_UPDATE_ORDER_CANCEL, ACT_UPDATE_ORDER_ITEMS, ACT_UPDATE_ORDER_PAYMENT, ACT_UPDATE_ORDER_ADDRESS, ACT_UPDATE_USER_ADDRESS, ACT_UPDATE_RESERVATION_BOOK, ACT_UPDATE_RESERVATION_CANCEL, ACT_UPDATE_RESERVATION_FLIGHT, ACT_UPDATE_RESERVATION_BAGGAGE, ACT_UPDATE_RESERVATION_PASSENGER, ACT_SEND_CERTIFICATE |
| FINISH | ANSWER, STOP |
| RECOVER_OR_ELICIT | ASK_USER, THINK, TRANSFER |

## Results

| Method | Action Acc | Skill Acc | Boundary Skill Acc | Skill-Action Gap | Source Action Agree | Source Skill Agree | Same-Skill Wrong Action | Action Transition | Skill Transition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source_policy | 80.61% | 85.14% | 83.80% | 4.52% | 100.00% | 100.00% | 4.52% | 65.96% | 72.87% |
| static_action | 64.62% | 71.41% | 67.87% | 6.79% | 73.83% | 79.32% | 6.79% | 42.91% | 53.19% |
| predicted_skill_rerank | 64.14% | 71.41% | 66.84% | 7.27% | 73.02% | 78.84% | 7.27% | 42.02% | 52.84% |
| soft_skill_prior | 64.30% | 71.24% | 67.35% | 6.95% | 73.34% | 78.84% | 6.95% | 42.02% | 52.13% |
| confidence_gated_soft_skill | 64.78% | 71.57% | 68.12% | 6.79% | 73.99% | 79.48% | 6.79% | 42.91% | 53.19% |
| selective_hard_skill_rerank | 64.78% | 71.57% | 68.12% | 6.79% | 73.99% | 79.48% | 6.79% | 42.91% | 53.19% |
| temporal_soft_skill_prior | 62.68% | 69.63% | 64.27% | 6.95% | 71.57% | 77.22% | 6.95% | 40.96% | 50.71% |
| prefix_defined_oracle_skill | 49.60% | 58.48% | 55.78% | 8.89% | 52.99% | 61.07% | 8.89% | 30.32% | 39.18% |
| prefix_defined_random_oracle | 13.89% | 40.87% | 38.56% | 26.98% | 14.05% | 40.39% | 26.98% | 5.67% | 20.57% |
| prefix_latent_skill_binder | 50.08% | 59.61% | 58.87% | 9.53% | 54.12% | 62.84% | 9.53% | 31.21% | 40.07% |
| prefix_latent_skill_binder_no_prev | 49.60% | 59.29% | 58.35% | 9.69% | 53.63% | 62.52% | 9.69% | 30.67% | 39.54% |
| source_discovered_cluster_oracle | 65.59% | 71.57% | 70.18% | 5.98% | 74.64% | 79.16% | 5.98% | 45.04% | 52.48% |
| source_discovered_cluster_binder | 62.04% | 69.79% | 66.58% | 7.75% | 71.41% | 77.22% | 7.75% | 37.23% | 49.29% |
| response_operator_oracle_allowed | 65.91% | 72.21% | 69.15% | 6.30% | 76.25% | 80.78% | 6.30% | 43.79% | 52.84% |
| response_operator_binder_allowed | 61.55% | 68.82% | 63.50% | 7.27% | 70.92% | 75.93% | 7.27% | 38.65% | 48.23% |
| observation_type_allowed | 55.74% | 63.49% | 59.38% | 7.75% | 61.55% | 69.14% | 7.75% | 34.04% | 43.79% |
| observation_type_residual | 64.62% | 71.41% | 67.87% | 6.79% | 73.83% | 79.32% | 6.79% | 42.91% | 53.19% |
| response_operator_obs_oracle_allowed | 64.94% | 71.41% | 68.38% | 6.46% | 75.61% | 80.29% | 6.46% | 42.20% | 50.89% |
| response_operator_obs_binder_allowed | 60.10% | 68.82% | 64.52% | 8.72% | 69.79% | 76.25% | 8.72% | 36.35% | 47.70% |
| response_operator_obs_oracle_residual | 64.62% | 71.24% | 67.61% | 6.62% | 73.83% | 79.16% | 6.62% | 42.91% | 52.84% |
| response_operator_obs_binder_residual | 64.30% | 71.24% | 67.61% | 6.95% | 73.83% | 79.16% | 6.95% | 42.20% | 52.66% |
| write_pair_calibrated_v2 | 64.78% | 71.24% | 68.12% | 6.46% | 72.54% | 77.71% | 6.46% | 42.55% | 52.66% |
| response_operator_obs_binder_residual_plus_write_pair_v2 | 64.62% | 70.92% | 67.61% | 6.30% | 72.37% | 77.38% | 6.30% | 42.20% | 51.95% |
| response_operator_oracle_residual | 64.94% | 71.57% | 67.87% | 6.62% | 74.15% | 79.48% | 6.62% | 43.09% | 53.37% |
| response_operator_binder_residual | 64.30% | 71.24% | 67.61% | 6.95% | 73.83% | 79.16% | 6.95% | 41.67% | 52.66% |
| response_operator_gated_residual | 64.30% | 71.24% | 67.61% | 6.95% | 73.83% | 79.16% | 6.95% | 41.67% | 52.66% |
| response_operator_soft_residual | 64.62% | 71.41% | 67.87% | 6.79% | 73.83% | 79.32% | 6.79% | 42.91% | 53.19% |
| source_distilled_skill_binder | 64.30% | 70.92% | 66.58% | 6.62% | 74.15% | 79.32% | 6.62% | 41.84% | 51.06% |
| source_distilled_soft_skill_binder | 64.62% | 71.57% | 67.61% | 6.95% | 73.67% | 79.16% | 6.95% | 42.91% | 53.19% |
| source_distilled_skill_binder_no_turn | 63.81% | 70.27% | 65.55% | 6.46% | 73.67% | 78.68% | 6.46% | 41.67% | 50.53% |
| source_distilled_skill_binder_no_prev | 64.46% | 71.24% | 67.61% | 6.79% | 74.47% | 79.81% | 6.79% | 41.49% | 50.89% |
| source_distilled_skill_binder_no_action_logits | 64.62% | 71.57% | 68.64% | 6.95% | 74.15% | 79.48% | 6.95% | 42.02% | 52.66% |
| source_distilled_prefix_skill_binder | 64.14% | 70.76% | 66.58% | 6.62% | 73.83% | 79.16% | 6.62% | 41.49% | 50.89% |
| source_distilled_prefix_soft_skill_binder | 64.14% | 71.08% | 66.84% | 6.95% | 73.34% | 78.84% | 6.95% | 42.38% | 52.66% |
| source_distilled_prefix_skill_binder_no_action_logits | 64.94% | 71.73% | 67.87% | 6.79% | 74.15% | 79.64% | 6.79% | 41.84% | 51.77% |
| source_distilled_prefix_only_skill_binder | 64.30% | 71.08% | 67.35% | 6.79% | 74.15% | 79.64% | 6.79% | 42.02% | 51.60% |
| source_distilled_action_binder | 62.04% | 68.17% | 64.27% | 6.14% | 72.21% | 76.41% | 6.14% | 38.12% | 47.34% |
| source_distilled_action_soft_binder | 64.62% | 71.41% | 67.35% | 6.79% | 74.31% | 79.00% | 6.79% | 42.38% | 52.84% |
| source_distilled_random_group_binder | 62.36% | 69.31% | 65.04% | 6.95% | 71.57% | 77.38% | 6.95% | 41.31% | 51.06% |
| oracle_train_skill_binder | 64.14% | 71.24% | 69.15% | 7.11% | 73.67% | 79.00% | 7.11% | 41.84% | 51.77% |
| oracle_train_soft_skill_binder | 64.14% | 71.08% | 67.61% | 6.95% | 73.18% | 78.68% | 6.95% | 41.49% | 51.77% |
| source_skill_rerank | 77.06% | 85.14% | 83.80% | 8.08% | 91.92% | 100.00% | 8.08% | 59.75% | 72.87% |
| oracle_skill_rerank | 87.88% | 100.00% | 100.00% | 12.12% | 77.54% | 85.14% | 12.12% | 76.60% | 100.00% |
| oracle_only_collect_info | 69.95% | 79.00% | 74.29% | 9.05% | 76.09% | 82.55% | 9.05% | 49.65% | 63.30% |
| oracle_only_confirm | 71.73% | 78.51% | 75.32% | 6.79% | 76.09% | 81.42% | 6.79% | 53.01% | 63.83% |
| oracle_only_execute | 67.53% | 75.28% | 72.49% | 7.75% | 73.67% | 79.64% | 7.75% | 46.10% | 57.80% |
| oracle_only_finish | 68.34% | 75.12% | 73.52% | 6.79% | 74.47% | 79.97% | 6.79% | 45.04% | 56.56% |
| oracle_only_recover_or_elicit | 68.82% | 77.71% | 75.84% | 8.89% | 72.54% | 78.84% | 8.89% | 48.58% | 62.23% |
