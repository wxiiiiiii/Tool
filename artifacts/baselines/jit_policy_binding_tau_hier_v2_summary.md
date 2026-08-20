# JIT / Skill-Pro Inspired Policy Binding Results

Goal: test whether a training-free non-parametric policy memory can improve cross-backbone policy binding beyond anchor-only ridge calibration.

## Method

The implementation adapts the ideas from JitRL and Skill-Pro to Universal Action logits, rather than token logits or natural-language skill execution.

Pipeline:

1. Build the source-side anchor-relative policy interface once from Qwen3-4B.
2. Build a portable source policy memory over anchor-relative coordinates.
3. For a target state, retrieve similar source memory entries in relational policy space.
4. Apply a non-parametric action-logit correction:

```text
z_target'(s, a) = z_target(s, a) + beta * Delta_memory(s, a)
```

No target labels, no target adapter training, and no online source model queries are used.

Two memory variants were implemented:

- `raw`: JitRL-style raw source policy memory over source decision states.
- `capsule`: Skill-Pro-style compressed source policy capsules by action prototype.

A trust-region variant `jit_gated` was also added. It only applies memory correction when retrieved memory is action-compatible with the base policy or sufficiently more confident.

The latest version adds a stricter selective/counterfactual gate:

- `source_reliable` memory keeps only source-correct, high-confidence, high-margin entries.
- `jit_selective` only intervenes when target base and retrieved memory disagree, target base is low-margin, memory neighbors are consistent, and retrieved memory has a stronger action margin than the base policy.
- The report now includes gate/change/improve/harm rates and a four-quadrant diagnostic:

```text
                         Source correct   Source wrong
Target base correct             A              B
Target base wrong               C              D
```

`B Harm` measures teacher-error propagation; `C Improve` measures useful JIT correction.

## Tau-Hier-v2 Fast Results

All runs use Qwen3-4B as source, `transition_critical` anchors, `K=512`, `state` features, and anchor-only ridge calibration.

| Target | Method | Source Val | Target Acc | Policy Agreement | Transition Agreement | Target Labels | Target Gradient |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | anchor-relative ridge base | 68.06% | 60.72% | 74.45% | 56.85% | 0 | 0 |
| SmolLM3-3B | JIT residual raw memory | 68.50% | 58.79% | 75.71% | 58.83% | 0 | 0 |
| SmolLM3-3B | JIT gated raw memory | 68.93% | 59.63% | 75.19% | 57.82% | 0 | 0 |
| Qwen2.5-3B | anchor-relative ridge base | 68.06% | 57.44% | 71.94% | 53.09% | 0 | 0 |
| Qwen2.5-3B | JIT residual raw memory | 68.50% | 56.12% | 72.26% | 55.21% | 0 | 0 |
| Qwen2.5-3B | JIT gated raw memory | 68.93% | 56.29% | 71.98% | 54.90% | 0 | 0 |

## Reliability-Gated Follow-Up

The next optimization added `jit_reliable`, a target-label-free trust gate based on retrieval similarity margin, neighborhood action entropy, base-policy uncertainty, and retrieved-policy margin. Hyperparameters are still selected on source validation only.

| Target | Method Selected | Target Acc | Policy Agreement | Transition Agreement | Observation |
|---|---|---:|---:|---:|---|
| SmolLM3-3B | anchor-relative ridge base | 60.72% | 74.45% | 56.85% | Reliable memory candidates did not beat base under source-val selection |
| Qwen2.5-3B | anchor-relative ridge base | 57.44% | 71.94% | 53.09% | Reliable memory candidates did not beat base under source-val selection |

The full reports are:

- `artifacts/baselines/jit_policy_binding_tau_hier_v2_smollm3_reliable_fast/jit_policy_binding_results.md`
- `artifacts/baselines/jit_policy_binding_tau_hier_v2_qwen25_reliable_fast/jit_policy_binding_results.md`

## Selective / Counterfactual JIT Follow-Up

This diagnostic run used a smaller `40`-epoch source relative policy and a narrow sweep to test whether stricter selective correction reduces teacher-error propagation. It should be compared within this section, not against the 80/150-epoch headline rows above.

| Target | Method | Filter | Target Acc | Policy Agreement | Transition Agreement | Gate Open | Changed | Improved | Harmed | B Harm | C Improve |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | anchor-relative base | all | 55.86% | 74.71% | 57.96% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| SmolLM3-3B | selective JIT | source_reliable | 55.43% | 73.93% | 57.42% | 6.11% | 5.01% | 1.56% | 1.99% | 10.20% | 5.37% |
| Qwen2.5-3B | anchor-relative base | all | 56.75% | 74.63% | 56.85% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen2.5-3B | selective JIT | source_reliable | 56.27% | 75.08% | 57.82% | 7.41% | 6.11% | 1.82% | 2.30% | 9.68% | 6.23% |

Compared with the earlier less-strict selective diagnostic on SmolLM3, the stricter override-margin gate reduced `Harmed` from `4.57%` to `1.99%` and `B Harm` from `23.13%` to `10.20%`. The cost is that `Improved` also fell from `3.16%` to `1.56%`, so target accuracy remains slightly below the base. This confirms the diagnosis: selective JIT can suppress teacher-error propagation, but the current gate is conservative and still does not identify enough C-class cases where target is wrong and source memory is useful.

Full reports:

- `artifacts/baselines/jit_policy_binding_tau_hier_v2_smollm3_selective_override_min/jit_policy_binding_results.md`
- `artifacts/baselines/jit_policy_binding_tau_hier_v2_qwen25_selective_override_min/jit_policy_binding_results.md`

## Counterfactual Stability-Gated JIT

The next optimization adds a target-unsupervised self-consistency signal. Before accepting a memory correction, the base policy is evaluated under random anchor-feature dropout. JIT is allowed only when the base action is unstable under this perturbation, the base and memory actions disagree, and the retrieved memory still has a stronger margin. This approximates the desired condition:

```text
correct only when Target is uncertain and Memory is reliable
```

It still uses `0` target labels, `0` target gradient steps, and `0` source online queries.

| Target | Method | Memory Filter | Target Acc | Policy Agreement | Transition Agreement | Gate Open | Changed | Improved | Harmed | B Harm | C Improve |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | base | all | 55.86% | 74.71% | 57.96% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| SmolLM3-3B | counterfactual JIT | source_reliable | 55.43% | 74.11% | 57.65% | 5.89% | 4.90% | 1.52% | 1.95% | 10.20% | 5.20% |
| SmolLM3-3B | counterfactual JIT | source_correct | 55.49% | 74.41% | 57.89% | 7.74% | 5.89% | 1.80% | 2.17% | 12.24% | 6.14% |
| Qwen2.5-3B | base | all | 56.75% | 74.63% | 56.85% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen2.5-3B | counterfactual JIT | source_reliable | 56.29% | 75.19% | 57.98% | 7.11% | 5.98% | 1.80% | 2.25% | 9.68% | 6.14% |
| Qwen2.5-3B | counterfactual JIT | source_correct | 56.36% | 75.45% | 58.22% | 8.26% | 6.13% | 1.89% | 2.28% | 11.61% | 6.58% |

The self-consistency gate makes the correction slightly more conservative than the earlier selective gate, but it does not improve target accuracy over the base. Relaxing memory from `source_reliable` to `source_correct` increases useful C-class repairs, but it also increases B-class teacher-error propagation. This confirms that the key missing signal is not simply source confidence or target perturbation instability. The unsupervised arbiter needs environment-side evidence: whether the corrected abstract action grounds to a valid tool, preserves local trajectory consistency, and survives environment perturbations.

Full reports:

- `artifacts/baselines/jit_policy_binding_tau_hier_v2_smollm3_counterfactual_min/jit_policy_binding_results.md`
- `artifacts/baselines/jit_policy_binding_tau_hier_v2_smollm3_counterfactual_source_correct_min/jit_policy_binding_results.md`
- `artifacts/baselines/jit_policy_binding_tau_hier_v2_qwen25_counterfactual_min/jit_policy_binding_results.md`
- `artifacts/baselines/jit_policy_binding_tau_hier_v2_qwen25_counterfactual_source_correct_min/jit_policy_binding_results.md`

For reference, the stronger 150-epoch anchor-ridge run reached:

| Target | Anchor-relative Ridge | Adapter-Based Ours |
|---|---:|---:|
| SmolLM3-3B | 63.01% | 76.21% |
| Qwen2.5-3B | 56.16% | 75.30% |

## Takeaway

The JIT/Skill-Pro inspired policy memory increased source-side validation accuracy and improved policy/transition agreement, but did not improve target action accuracy in this POC. This means the retrieved source policy memory is useful for making the target mimic the source policy distribution more closely, but the memory also transfers source-side or binding-side mistakes. In other words, policy fidelity and task accuracy are not identical.

The most useful result is diagnostic: a source-validation-selected memory logit boost improves fidelity but not target action accuracy. Even a conservative reliability gate does not yet beat anchor-ridge, which suggests the missing component is not just a better pointwise gate. The next version should learn or derive a target-unsupervised selector from target-side consistency signals, e.g. whether memory correction preserves trajectory smoothness, environment execution validity, and neighborhood agreement under perturbations.

The selective/counterfactual follow-up narrows this further: the bottleneck is discriminating B from C without target labels. Source confidence and target perturbation instability reduce harm but do not find enough useful C cases. The next useful optimization is an environment-aware arbitration score that opens the gate only when a correction improves local trajectory consistency or environment-binding validity under perturbations, instead of using policy-logit confidence alone.

Given the low C-class repair ceiling, memory/JIT is now best treated as an auxiliary negative study rather than the main path. The stronger next result comes from improving the zero-training anchor-relative representation itself: `policy_transition_mixture + state_pool` raises SmolLM3 target action accuracy to `65.83%` and Qwen2.5 to `60.89%` without target labels, target gradients, or online source queries. See `artifacts/baselines/anchor_relative_tau_hier_v2_summary.md`.
