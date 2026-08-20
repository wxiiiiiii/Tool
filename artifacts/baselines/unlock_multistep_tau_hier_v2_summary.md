# Unlock Multi-Step Tau-Hier-v2 Evaluation

Goal: test whether Unlock-style latent/subspace alignment transfers a complete sequential Agent policy, rather than only a static action/capability direction.

## Interpretation Of Unlock

Unlock/Master-Key style transfer is best understood as latent capability-direction or subspace alignment. It does not directly align final task results, DB state, or full multi-step agent trajectories. In this project, the closest implementation is:

```text
h_T -> low-rank alignment -> z_Qwen -> frozen Qwen action head -> universal action
```

This evaluates static action transfer. To test the multi-step setting, we additionally replay predicted action sequences through tau-bench grounding, constrained arguments, tool execution, and DB hash evaluation.

## Methods

- `unlock_lowrank_subspace`: static low-rank paired activation alignment into the Qwen3 source policy latent space.
- `unlock_lowrank_intervention`: low-rank alignment plus source-latent action-direction logit intervention.
- `unlock_lowrank_dynamic`: low-rank alignment plus source-policy transition prior during trajectory-ordered decoding.

All three use:

- Target action labels: 0
- Target gradient steps: 0
- Online source queries at evaluation: 0
- Paired source/target hidden states: yes

## Full Tau Replay Results

Replay uses `arg-generator=oracle`, `grounder=semantic`, and the same tau-bench executor/DB hash path as prior experiments.

| Target | Method | Action Acc | Seq Exact | Transition Pair Acc | First Div Step | DB Hash | Tool Exec OK |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | unlock_lowrank_subspace | 75.99% | 8.01% | 56.17% | 4.04 | 69.89% | 67.79% |
| SmolLM3-3B | unlock_lowrank_intervention | 75.32% | 6.63% | 55.21% | 3.95 | 70.72% | 65.56% |
| SmolLM3-3B | unlock_lowrank_dynamic | 74.47% | 4.70% | 54.36% | 3.89 | 67.40% | 67.34% |
| Qwen2.5-3B | unlock_lowrank_subspace | 75.86% | 8.01% | 56.03% | 4.14 | 70.99% | 67.43% |
| Qwen2.5-3B | unlock_lowrank_intervention | 75.30% | 7.46% | 55.23% | 4.10 | 72.38% | 65.25% |
| Qwen2.5-3B | unlock_lowrank_dynamic | 73.85% | 5.25% | 53.26% | 3.91 | 66.57% | 66.23% |

Reference replay numbers already present in the repo:

| Method | Action Acc | Tool Exec OK | DB Hash |
|---|---:|---:|---:|
| Qwen3 source policy | 78.09% | 68.80% | 71.82% |
| Ours Qwen3->SmolLM3 adapter | 82.60% | 71.95% | 79.01% |
| Previous `unlock_style_latent_mse` | 71.81% | 63.19% | 68.23% |

## Findings

1. `unlock_lowrank_subspace` is much stronger than the previous MSE-style Unlock baseline on static action and replay DB hash.
2. `unlock_lowrank_intervention` lowers action accuracy slightly but improves final DB Hash on both targets:
   - SmolLM3: `69.89% -> 70.72%`
   - Qwen2.5: `70.99% -> 72.38%`
3. The naive trajectory Markov prior, `unlock_lowrank_dynamic`, hurts both action accuracy and DB Hash. This is important: sequential Agent policy dynamics are not captured by simply adding a source transition prior.
4. Sequence exact match remains very low, around `5%--8%`, even when action accuracy is around `75%`. This directly shows the limitation of treating multi-step tool-use policy as independent static action alignment.

## Conclusion

Unlock-style latent alignment transfers static action behavior well, but it does not by itself solve multi-step Agent policy preservation. The useful improvement comes from action-direction logit intervention at the executor outcome level, while naive transition-prior decoding is insufficient. The next method should optimize policy transitions with state-dependent confidence and environment feedback, rather than applying a global Markov transition prior.

