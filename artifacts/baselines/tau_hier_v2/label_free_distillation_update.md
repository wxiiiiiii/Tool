# Label-free distillation update

Goal: do not optimize the CE target-label adapter. Improve label-free source-policy distillation:

```text
h_source -> E_source -> W_pi -> p_source
h_target -> E_target -> W_pi -> p_target
loss uses p_source / source policy geometry only, no target action labels
```

Implemented in `baselines/source_policy_distillation_adapter.py`:

- temperature distillation;
- confidence-weighted and confidence-filtered KL;
- policy-rowspace subspace alignment, using the frozen action head row-space;
- centered-logit decision-margin alignment;
- trajectory transition-aware distillation via adjacent decision logit-delta alignment;
- adaptive/importance-aware transition weighting based on source-teacher action type;
- sliding-window sequence distillation over 3-step policy dynamics;
- trajectory error decomposition for DB-hash failures;
- scaling via `--max-samples`;
- diagnostics: Source-Target Policy Agreement, Correct-Teacher Retention, Incorrect-Teacher Agreement, transition agreement, and correct-teacher transition retention.

## 4.6K success-only policy trace

| Method | Target Labels Used | Action Acc | Tool Action Acc |
|---|---|---:|---:|
| Source Qwen policy on Qwen hidden states | no target adapter | 78.09% | 81.53% |
| KL distill, rank 64, 30e | no | 71.22% | 73.87% |
| KL distill, rank 128, 60e | no | 72.96% | 74.76% |
| KL distill + transition w=0.1, rank 128, 60e | no | 74.11% | 76.69% |
| KL distill + transition w=0.2, rank 128, 60e | no | 74.84% | 79.64% |
| KL + transition w=0.2, high-conf transition fine-tune | no | 75.43% | 80.56% |
| KL + transition w=0.2, high-conf KL+transition fine-tune | no | 75.67% | 79.89% |
| KL + transition w=0.15 | no | 74.95% | 78.75% |
| KL + adaptive transition w=0.15 + 3-step sequence | no | 76.21% | 80.19% |
| Confidence-weighted KL, T=2 | no | 69.80% | not replayed |
| Confidence-filtered KL, conf >= 0.55 | no | 69.36% | not replayed |
| KL + strong decision margin | no | 70.38% | not replayed |
| KL + policy-rowspace subspace | no | 70.66% | not replayed |

Best policy-fidelity result so far: rank128 KL + adaptive transition-aware distillation at `transition_weight=0.15`, plus 3-step sequence distillation.

## Source-teacher diagnostics

Full success-only set for rank128 KL:

| Metric | Value |
|---|---:|
| Source action accuracy | 78.09% |
| Target action accuracy | 72.96% |
| Source-Target Policy Agreement | 86.11% |
| Correct-Teacher Retention | 90.43% |
| Incorrect-Teacher Agreement | 70.72% |

Interpretation:

- The adapter preserves most source decisions.
- The remaining gap to Source comes partly from not retaining all correct teacher decisions.
- It also copies many incorrect teacher decisions, so blindly maximizing agreement is not enough.

After adding transition-aware distillation:

| Metric | Rank128 KL | + Transition w=0.1 | + Transition w=0.2 |
|---|---:|---:|---:|
| Target action accuracy | 72.96% | 74.11% | 74.84% |
| Source-Target Policy Agreement | 86.11% | 86.91% | 88.58% |
| Correct-Teacher Retention | 90.43% | 91.43% | 92.84% |
| Incorrect-Teacher Agreement | 70.72% | 70.82% | 73.39% |
| Transition Agreement | 73.64% | 75.38% | 77.62% |
| Correct-Teacher Transition Retention | 81.49% | 84.29% | 85.75% |
| Incorrect-Teacher Transition Agreement | 62.07% | 62.25% | 65.62% |

Increasing the transition loss from 0.1 to 0.2 improves the main fidelity metrics without target labels: policy agreement rises by +2.47 points over the rank128 KL baseline, transition agreement rises by +3.98 points, and correct-teacher retention rises by +2.41 points. The tradeoff is that incorrect-teacher agreement also rises, so the target adapter is preserving the source policy more faithfully, including some source mistakes.

## Teacher-error filtering and confidence gating

Source confidence is correlated with teacher correctness, but naive one-stage confidence gating did not improve transfer. On source-policy diagnostics, confidence >= 0.8 keeps 48.52% of states and raises teacher accuracy on kept states to 94.37%, but direct filtering removes too much coverage for adapter training.

One-stage attempts:

| Method | Labels Used | Val Action Acc | Val Policy Agreement | Val Transition Agreement | Notes |
|---|---|---:|---:|---:|---|
| Transition confidence soft gate, gamma=0.5 | no | 71.53% | 84.10% | 71.43% | worse than ungated w=0.2 |
| Transition confidence soft gate, gamma=1.0 | no | 71.82% | 84.39% | 70.33% | worse than ungated w=0.2 |
| KL + transition soft gate, gamma=0.5 | no | 72.25% | 84.25% | 69.23% | worse than ungated w=0.2 |
| Transition hard filter, conf >= 0.5 | no | 70.95% | 83.96% | 72.53% | keeps 71.15% transition pairs |
| Transition hard filter, conf >= 0.6 | no | 70.66% | 83.38% | 69.23% | keeps 54.71% transition pairs |
| Oracle correct-teacher filter | yes | 71.68% | 82.95% | 71.43% | diagnostic upper bound attempt; removing all teacher mistakes also removes useful boundary coverage |

Two-stage high-confidence fine-tuning works better: first train the full `transition_weight=0.2` adapter, then start from that checkpoint and run 15 low-LR epochs with high-confidence transition filtering.

| Method | Labels Used | Action Acc | Policy Agreement | Correct-Teacher Retention | Incorrect-Teacher Agreement | Transition Agreement |
|---|---|---:|---:|---:|---:|---:|
| KL + transition w=0.2 | no | 74.84% | 88.58% | 92.84% | 73.39% | 77.62% |
| w=0.2 -> high-conf transition fine-tune | no | 75.43% | 89.79% | 93.73% | 75.77% | 80.13% |
| w=0.2 -> high-conf KL+transition fine-tune | no | 75.67% | 90.27% | 94.03% | 76.85% | 81.19% |

This improves policy fidelity without target labels, but it still increases incorrect-teacher agreement. The optimization is therefore better described as confidence-stabilized policy preservation, not solved teacher-error removal.

## Adaptive transition and sequence distillation

Following the Pareto sweep, fixed `transition_weight=0.15` is better than both 0.125 and 0.175 on the full success-only diagnostic, and slightly better than 0.2 on action/policy agreement before replay. The stronger update is to combine:

- `transition_weight=0.15`;
- adaptive/importance-aware transition weights from source-teacher action types;
- 3-step sliding-window sequence distillation with `sequence_weight=0.05`.

Full diagnostic:

| Method | Labels Used | Action Acc | Policy Agreement | Correct-Teacher Retention | Incorrect-Teacher Agreement | Transition Agreement |
|---|---|---:|---:|---:|---:|---:|
| KL + transition w=0.15 | no | 74.95% | 88.71% | 92.90% | 73.79% | 78.02% |
| KL + transition w=0.2 | no | 74.84% | 88.58% | 92.84% | 73.39% | 77.62% |
| w=0.2 -> high-conf KL+transition fine-tune | no | 75.67% | 90.27% | 94.03% | 76.85% | 81.19% |
| KL + adaptive transition w=0.15 + 3-step sequence | no | 76.21% | 91.16% | 94.87% | 77.94% | 82.67% |

The sequence-adaptive setting is the first setting that enters the 91% policy-agreement regime while also matching the prior best DB-hash result. As before, higher fidelity still raises incorrect-teacher agreement, so this is a stronger policy-preservation result rather than a teacher-error-removal result.

## Entropy confidence and transition-delta weighting

The slide version of confidence-aware distillation defines teacher confidence as
`1 - H(p_source) / log |A|`. The original implementation already supported
confidence-aware KD, but used normalized top-1 probability by default. We added
`--confidence-mode entropy` to match the slide formula exactly, and added
`--transition-delta-strength` as an optional source centered-logit transition
magnitude factor for adaptive transition weighting.

Implementation status:

| Component | Status | Notes |
|---|---|---|
| Confidence-aware KD | implemented | now supports `max_prob`, `entropy`, and `margin` confidence modes |
| Transition-aware distillation | implemented | centered-logit adjacent transition loss |
| Adaptive transition weighting | implemented | action semantics + confidence endpoints; now optionally source transition-delta magnitude |
| Short-horizon sequence distillation | implemented | sliding window, current best uses `K=3` |

New SmolLM3 test results:

| Method | Full Action Acc | Full Tool Acc | Val Action Acc | Val Policy Agreement | Val Transition Agreement | Replay |
|---|---:|---:|---:|---:|---:|---|
| previous best: adaptive `tw=0.15` + seq3, 60e | 76.21% | 80.19% | 72.98% | 84.68% | 73.63% | DB 72.10% |
| entropy confidence + delta weighting, 30e | 70.12% | 72.07% | 69.65% | 81.07% | 68.13% | not replayed |
| delta weighting only, 30e | 70.23% | 73.08% | 71.10% | 82.08% | 67.03% | not replayed |

Interpretation: matching the entropy-confidence formula and adding transition
delta weighting did not improve accuracy or policy fidelity in this setting. The
likely reason is that one-stage confidence reweighting reduces useful boundary
coverage, while transition-delta weighting over-emphasizes large logit movements
that are not necessarily the transitions most useful for executor success. These
features remain available for ablation, but the main method should keep the
previous adaptive transition + seq3 objective.

## Scaling

The tau historical files provide 4,615 success-only decision states and 9,018 all-trajectory decision states. There is not enough native tau data here for true 10K/25K scaling without adding more trajectories or augmentation.

| Training data | Eval data | Action Acc | Tool Action Acc |
|---|---|---:|---:|
| 4.6K success-only | 4.6K success-only | 72.96% | 74.76% |
| 5K all-trajectories | 4.6K success-only | 68.30% | 72.91% |
| 9K all-trajectories | 4.6K success-only | 71.29% | 73.29% |

More all-trajectory states did not improve success-policy preservation. The likely reason is that failed trajectories contain lower-quality or inconsistent teacher decision states.

## Replay

Semantic grounding + oracle arguments:

| Method | Action Acc | Tool Grounding Acc | Tool Exec OK | DB Hash Match |
|---|---:|---:|---:|---:|
| KL distill, rank64, 30e | 71.22% | 70.68% | 65.40% | 65.19% |
| KL distill, rank128, 60e | 72.96% | 71.47% | 67.30% | 64.36% |
| KL distill + transition w=0.1, rank128, 60e | 74.11% | 73.17% | 69.40% | 72.10% |
| KL distill + transition w=0.2, rank128, 60e | 74.84% | 78.33% | 65.91% | 69.61% |
| w=0.2 -> high-conf transition fine-tune | 75.43% | 78.61% | 65.96% | 70.99% |
| w=0.2 -> high-conf KL+transition fine-tune | 75.67% | 79.02% | 66.71% | 70.72% |
| KL + transition w=0.15 | 74.95% | 79.63% | 68.07% | 70.99% |
| KL + adaptive transition w=0.15 + 3-step sequence | 76.21% | 80.22% | 68.48% | 72.10% |

Transition-aware distillation fixes the earlier replay regression at `transition_weight=0.1`: DB hash rises from 64.36% to 72.10%, while preserving the no-target-label training setting. `transition_weight=0.2` gives the best policy fidelity and tool grounding, but its executor DB hash drops to 69.61%, suggesting that stronger transition matching can improve action choice while exposing more argument/execution sensitivity downstream.

High-confidence fine-tuning partially recovers the executor regression of `transition_weight=0.2`: DB hash rises from 69.61% to 70.99% for the transition-only fine-tune, while action accuracy remains above the original `w=0.2` setting. It still does not beat `w=0.1` on DB hash, so the best executor setting and best fidelity setting are currently different.

After sequence-adaptive distillation, the best fidelity and best utility are no longer cleanly separated: DB hash reaches 72.10%, matching the previous `w=0.1` utility-best result, while action accuracy and policy fidelity are substantially higher.

## Cross-backbone target: Qwen2.5-3B-Instruct

Tested the same label-free transfer setup with `Qwen/Qwen2.5-3B-Instruct` as a new target backbone:

```text
Qwen3-4B source hidden -> source adapter -> frozen W_pi
Qwen2.5-3B target hidden -> target tiny adapter -> same frozen W_pi
```

The first run used the same SmolLM3-best no-target-label method: rank128 low-rank MLP adapter, KL distillation from the frozen Qwen3 source policy, adaptive transition weight `0.15`, and 3-step sequence distillation. Target BFCL/tau action labels are not used for training.

That setting underfit Qwen2.5: it transferred local action behavior partially, but transition fidelity was much lower than SmolLM3. A Qwen2.5-specific no-label sweep showed that the useful fix is not prompt tuning, temperature weighting, or margin matching; it is a slightly larger target adapter with stronger transition distillation. The best Qwen2.5 setting so far is rank256 low-rank MLP, adaptive transition weight `0.2`, and 3-step sequence distillation.

Full-set diagnostic:

| Target Backbone / Setting | Target Labels Used | Action Acc | Source-Target Policy Agreement | Correct-Teacher Retention | Transition Agreement |
|---|---|---:|---:|---:|---:|
| SmolLM3-3B | no | 76.21% | 91.16% | 94.87% | 82.67% |
| Qwen2.5-3B, rank128 `tw=0.15` | no | 73.11% | 86.20% | 90.40% | 74.02% |
| Qwen2.5-3B, rank256 `tw=0.2` | no | 75.30% | 89.21% | 93.40% | 79.14% |

Replay with semantic grounding + oracle arguments:

| Target Backbone / Setting | Action Acc | Tool Grounding Acc | Tool Exec OK | DB Hash Match |
|---|---:|---:|---:|---:|
| SmolLM3-3B | 76.21% | 80.22% | 68.48% | 72.10% |
| Qwen2.5-3B, rank128 `tw=0.15` | 73.11% | 77.14% | 65.55% | 66.57% |
| Qwen2.5-3B, rank256 `tw=0.2` | 75.30% | 78.72% | 66.87% | 69.61% |

Qwen2.5-3B transfers part of the frozen Qwen3 policy behavior. The optimized rank256 adapter closes a meaningful part of the gap: +2.19 action-accuracy points, +3.01 policy-agreement points, +5.12 transition-agreement points, and +3.04 DB-hash points over the first rank128 setting. It still trails SmolLM3 on transition fidelity and final DB state, so Qwen2.5 is now a useful stress test for cross-backbone scalability.

Qwen2.5 sweep summary:

| Setting | Action Acc | Policy Agreement | Transition Agreement | Notes |
|---|---:|---:|---:|---|
| rank128 `tw=0.15` + seq3 | 73.11% | 86.20% | 74.02% | first transfer |
| rank128 T=2 + confidence + `tw=0.2` | 72.98% | 86.37% | 74.51% | better KL/delta, no action gain |
| rank128 `tw=0.25` + seq4 | 73.43% | 85.87% | 73.81% | stronger sequence not enough |
| rank128 margin + `tw=0.2` | 74.04% | 87.26% | 75.73% | helps, but less than capacity |
| rank256 `tw=0.2` + seq3 | 75.30% | 89.21% | 79.14% | best full diagnostic and replay |
| rank256 margin + `tw=0.2` | 74.34% | 88.86% | 78.56% | margin hurts action |
| rank256 T=2 + confidence + `tw=0.2` | 74.04% | 87.39% | 75.90% | not the main fix |
| rank256 `tw=0.25` + seq4 | 74.95% | 89.10% | 78.89% | lower DB hash, 66.57% |
| rank512 `tw=0.2` + seq3 | validation worse | validation worse | validation worse | over-capacity / unstable |

Qwen2.5 error decomposition:

| First Error Type | Failed Tasks |
|---|---:|
| wrong_policy_action | 80 |
| execution_error | 19 |
| wrong_write_action | 4 |
| missed_stop_or_transfer | 4 |
| stop_or_transfer_too_early | 2 |
| write_execution_error | 1 |

Most common first-error transitions:

| Transition | Failed Tasks |
|---|---:|
| ASK_USER->ACT_RETRIEVE_USER | 24 |
| ACT_RETRIEVE_USER->VERIFY | 12 |
| ACT_RETRIEVE_ORDER->ASK_USER | 11 |
| ACT_RETRIEVE_USER->ACT_RETRIEVE_USER | 8 |
| ACT_RETRIEVE_RESERVATION->THINK | 6 |

Interpretation: Qwen3-4B -> Qwen2.5-3B-Instruct provides a useful same-family transfer test. The first failure was caused mostly by insufficient policy-state alignment capacity and weak transition preservation, not argument generation, because replay used oracle arguments. Rank256 fixes a substantial part of this while still training only the target adapter and using no target action labels. The remaining gap is concentrated in low-frequency airline/update actions and ask/retrieve/verify transitions, so the next scalable improvement should be adaptive rank/regularization plus action-frequency-balanced distillation, not another SmolLM3-only hyperparameter.

## Error decomposition

Trajectory decomposition on the sequence-adaptive replay:

| First Error Type | Failed Tasks |
|---|---:|
| wrong_policy_action | 75 |
| execution_error | 15 |
| missed_stop_or_transfer | 4 |
| stop_or_transfer_too_early | 3 |
| wrong_write_action | 3 |
| write_execution_error | 1 |

Most common first-error transitions among failed tasks:

| Transition | Failed Tasks |
|---|---:|
| ASK_USER->ACT_RETRIEVE_USER | 20 |
| ACT_RETRIEVE_USER->VERIFY | 13 |
| ACT_RETRIEVE_ORDER->VERIFY | 10 |
| ACT_RETRIEVE_ORDER->ASK_USER | 9 |
| ACT_RETRIEVE_USER->ACT_RETRIEVE_USER | 8 |

High conditional DB-failure transitions are dominated by state-changing operations, such as `VERIFY->ACT_UPDATE_*`, `ACT_UPDATE_*->ANSWER`, and repeated update transitions. This supports the next optimization direction: keep sequence-adaptive policy distillation, but add executor/memory-aware calibration around state-changing tools.

## Conclusion

The strongest updated claim is:

> Without target action labels, a tiny target adapter can preserve a substantial part of the frozen Qwen policy behavior, reaching 72.96% action accuracy vs. the Qwen source policy's 78.09% on the same success-only decision benchmark.

Updated after transition-aware distillation:

> Without target action labels, a tiny target adapter can preserve a substantial part of the frozen Qwen policy behavior. The best fidelity setting reaches 76.21% action accuracy, 91.16% Source-Target Policy Agreement, and 82.67% trajectory transition agreement vs. the Qwen source policy.

Not yet verified:

> Label-free adapter can almost losslessly preserve Source behavior, or maximize policy fidelity and executor DB success with the same hyperparameter.

Next work should keep the sequence-adaptive checkpoint as the main method and focus on executor/memory calibration for state-changing transitions. The next target is not another generic transition-weight sweep, but reducing failures around `VERIFY->ACT_UPDATE_*`, repeated update actions, and retrieve/verify loops while preserving the 91% policy-agreement regime.
