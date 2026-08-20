# Policy IR Sufficiency and Method Ablation

This report completes the requested non-new-backbone experiments for Policy IR
minimal sufficiency and method-component ablation. All runs use the same frozen
Qwen3 policy, the same target adapters, the same `api_rename` environment, the
same 1,000 tau-bench states, and oracle arguments. Documents in `paper/` are
used only as research references.

## Policy IR Minimal Sufficiency

The experiment progressively increases the information exposed to the
Environment Binder:

- `ir_act_only`: only knows that the action is tool-like.
- `ir_operation`: knows operation type such as retrieve/update/search.
- `ir_operation_object`: adds the object/entity such as order/user/product.
- `ir_full`: adds effect/constraint/precondition information.
- `behavior_verify_ir`: full behavior-aware capability matcher, including
  hard-negative effect checks and state-conditioned verification.

| Target | IR Level | Functional Grounding | Recall@3 | Recall@5 | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | ACT only | 62.20% | 68.79% | 72.83% | 37.03% | 40.00% |
| SmolLM3-3B | operation | 62.20% | 71.87% | 73.03% | 37.03% | 40.00% |
| SmolLM3-3B | operation + object | 62.30% | 70.91% | 73.99% | 37.22% | 41.33% |
| SmolLM3-3B | operation + object + effect + constraint | 72.70% | 76.49% | 79.38% | 57.52% | 44.00% |
| SmolLM3-3B | behavior-aware full binder | 74.50% | 86.32% | 88.05% | 60.15% | 57.33% |
| Qwen2.5-3B-Instruct | ACT only | 61.90% | 66.28% | 71.10% | 36.68% | 42.67% |
| Qwen2.5-3B-Instruct | operation | 61.90% | 69.75% | 71.48% | 36.68% | 42.67% |
| Qwen2.5-3B-Instruct | operation + object | 62.00% | 69.17% | 71.48% | 36.87% | 44.00% |
| Qwen2.5-3B-Instruct | operation + object + effect + constraint | 72.00% | 74.76% | 77.26% | 56.95% | 44.00% |
| Qwen2.5-3B-Instruct | behavior-aware full binder | 73.90% | 85.55% | 86.32% | 59.85% | 60.00% |

Interpretation: `ACT`, operation-only, and operation+object are not sufficient.
Adding effect/constraint information substantially improves grounding and
execution, but it still does not recover DB Hash. The minimal useful interface
needs behavior-aware compatibility, not just a richer tuple. This supports the
claim:

```text
minimal sufficient Policy IR = abstract action + behavior/effect/precondition compatibility
```

## Method Component Ablation

This experiment ablates the components of the current `behavior_verify_ir`
Environment Binder:

- `behavior_only_ir`: behavior capability matching only.
- `behavior_no_verify_ir`: behavior + retrieval, without schema/precondition
  verification.
- `behavior_no_retrieval_ir`: behavior + verification, without retrieval prior.
- `behavior_verify_ir`: behavior + retrieval + verification.

| Target | Binder Variant | Functional Grounding | Recall@3 | Recall@5 | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | behavior only | 73.80% | 84.01% | 85.36% | 56.77% | 41.33% |
| SmolLM3-3B | behavior + retrieval, no verify | 79.90% | 85.93% | 88.25% | 68.42% | 41.33% |
| SmolLM3-3B | behavior + verify, no retrieval | 73.00% | 85.16% | 87.86% | 58.08% | 44.00% |
| SmolLM3-3B | behavior + retrieval + verify | 74.50% | 86.32% | 88.05% | 60.15% | 57.33% |
| Qwen2.5-3B-Instruct | behavior only | 73.30% | 82.08% | 84.01% | 56.37% | 44.00% |
| Qwen2.5-3B-Instruct | behavior + retrieval, no verify | 79.40% | 84.39% | 86.13% | 68.34% | 44.00% |
| Qwen2.5-3B-Instruct | behavior + verify, no retrieval | 72.30% | 84.39% | 86.32% | 57.53% | 44.00% |
| Qwen2.5-3B-Instruct | behavior + retrieval + verify | 73.90% | 85.55% | 86.32% | 59.85% | 60.00% |

Interpretation: the ablation is useful precisely because the best single-step
grounding is not the best DB Hash. Removing verification gives higher
single-step grounding and tool execution, but final DB Hash stays low. The full
method is the only variant that preserves final state substantially better,
which means the verifier is not merely improving surface selection; it is
filtering behaviorally incompatible or state-inconsistent calls.

## Takeaway

The useful method is not simply a stronger tool selector. It needs all three:

```text
retrieval prior + behavior-aware capability matching + state/precondition verifier
```

