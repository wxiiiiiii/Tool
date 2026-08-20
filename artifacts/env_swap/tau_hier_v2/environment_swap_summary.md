# Environment / Tool-Registry Swap Summary

This experiment tests whether the current framework has two-way decoupling:

```text
Backbone hidden state -> Target adapter -> frozen policy -> Universal Policy IR
Universal Policy IR -> Environment Binder -> concrete tau-bench tool
```

Documents in `paper/` are used only as research references. The experiment keeps
the Target adapter, frozen policy, and universal policy action vocabulary fixed.
Only the backend tool registry is changed.

## Implementation

Added `baselines/tau_environment_swap.py`.

The script replays fixed policy predictions under synthetic tau-bench tool
registry variants:

- `identity`: original tool registry.
- `tool_order_permutation`: same tools, different order / slots.
- `api_rename`: same functions, opaque API names such as `env_api_03`.
- `schema_field_rename`: same functions, renamed schema fields such as
  `arg_0_order_id`.
- `api_alias`: same functions, versioned API aliases such as
  `get_order_details_v2`.
- `one_to_many`: same function exposed through multiple endpoint aliases.

Compared binders:

- `policy_ir`: our Universal Policy IR -> Environment Binder. It binds the
  predicted abstract policy action to the environment's current concrete tool.
- `metadata_ir`: a harder metadata-only binder. It uses the predicted Universal
  Policy IR plus current tool name/description/schema metadata, but does not use
  the original-name alias map for tool selection.
- `retrieval_ir`: an improved metadata-only binder. It still does not use the
  original-name alias map for tool selection, but adds action profiles and
  state-aware retrieval rules over tool metadata.
- `description_only`: ignores policy IR and chooses a tool by lexical
  description/query overlap.
- `concrete_tool_name`: freezes original concrete tool names.
- `concrete_tool_slot`: freezes original concrete tool slots.

All runs use oracle arguments to isolate policy/grounding/executor effects from
argument generation.

## Full Identity Check

SmolLM3 full 4,615-state replay with `identity + policy_ir` matches the previous
main replay:

| Target | Variant | Binder | Samples | Action Acc | Grounding Acc | Tool Exec OK | DB Hash |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | identity | policy_ir | 4,615 | 76.21% | 80.22% | 68.48% | 72.10% |

This confirms the new environment-swap replay is aligned with the original
tau-bench replay path.

## SmolLM3 Environment Swap, 1,000 States

The 1,000-state subset has higher action accuracy than the full set due to
sample composition. Environment robustness should therefore be read by comparing
rows within this table.

| Variant | Binder | Action Acc | Grounding Acc | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|
| identity | policy_ir | 79.00% | 83.40% | 73.87% | 70.67% |
| tool_order_permutation | policy_ir | 79.00% | 83.40% | 73.87% | 70.67% |
| api_rename | policy_ir | 79.00% | 83.40% | 73.87% | 70.67% |
| schema_field_rename | policy_ir | 79.00% | 83.40% | 73.87% | 70.67% |
| api_alias | policy_ir | 79.00% | 83.40% | 73.87% | 70.67% |
| one_to_many | policy_ir | 79.00% | 83.40% | 73.87% | 70.67% |
| identity | description_only | 79.00% | 1.60% | 1.60% | 26.67% |
| api_rename | description_only | 79.00% | 1.40% | 1.40% | 24.00% |
| api_rename | concrete_tool_name | 79.00% | 48.10% | 0.00% | 16.00% |
| tool_order_permutation | concrete_tool_slot | 79.00% | 45.30% | 16.35% | 16.00% |

## Backbone + Environment Swap, 1,000 States

This uses the Qwen3 source policy transferred to Qwen2.5-3B-Instruct, then swaps
the backend tool environment.

| Target | Variant | Binder | Action Acc | Grounding Acc | Tool Exec OK | DB Hash |
|---|---|---|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | identity | policy_ir | 79.00% | 82.80% | 73.55% | 72.00% |
| Qwen2.5-3B-Instruct | api_rename | policy_ir | 79.00% | 82.80% | 73.55% | 72.00% |
| Qwen2.5-3B-Instruct | api_rename | concrete_tool_name | 79.00% | 48.10% | 0.00% | 16.00% |
| Qwen2.5-3B-Instruct | tool_order_permutation | concrete_tool_slot | 79.00% | 45.60% | 16.99% | 16.00% |

## Metadata-Only Environment Binder

The previous `policy_ir` binder is an upper-bound constructed binder: it uses a
known alias map from swapped endpoints back to the original tau-bench tools. To
test a more realistic lightweight binder, we added `metadata_ir`, which selects
tools only from the Universal Policy IR and current tool metadata.

| Target | Variant | Binder | Action Acc | Grounding Acc | Tool Exec OK | DB Hash |
|---|---|---|---:|---:|---:|---:|
| SmolLM3-3B | identity | metadata_ir | 79.00% | 52.00% | 30.26% | 41.33% |
| SmolLM3-3B | api_rename | metadata_ir | 79.00% | 51.90% | 30.08% | 40.00% |
| SmolLM3-3B | schema_field_rename | metadata_ir | 79.00% | 51.90% | 30.08% | 40.00% |
| SmolLM3-3B | api_alias | metadata_ir | 79.00% | 52.00% | 30.26% | 41.33% |
| SmolLM3-3B | one_to_many | metadata_ir | 79.00% | 52.00% | 30.26% | 41.33% |
| Qwen2.5-3B-Instruct | identity | metadata_ir | 79.00% | 52.40% | 28.76% | 42.67% |
| Qwen2.5-3B-Instruct | api_rename | metadata_ir | 79.00% | 52.40% | 28.76% | 42.67% |
| Qwen2.5-3B-Instruct | schema_field_rename | metadata_ir | 79.00% | 52.40% | 28.76% | 42.67% |
| Qwen2.5-3B-Instruct | api_alias | metadata_ir | 79.00% | 52.40% | 28.76% | 42.67% |
| Qwen2.5-3B-Instruct | one_to_many | metadata_ir | 79.00% | 52.40% | 28.76% | 42.67% |

Interpretation: `metadata_ir` is environment-invariant, but not yet strong
enough in absolute utility. It keeps nearly identical performance across API
rename, schema rename, aliasing, and one-to-many variants, which means it is not
dependent on concrete tool tokens. However, it trails the constructed
`policy_ir` binder by roughly 31 grounding points and 28-30 DB-hash points on
the 1,000-state subset. This identifies Environment Binding as the next major
research component.

## Retrieval-Style Environment Binder

We then added `retrieval_ir`, which remains alias-free for selection but uses
stronger action profiles and state-aware metadata scoring. This is a more
realistic lightweight Environment Binder than the constructed `policy_ir` upper
bound.

| Target | Variant | Binder | Action Acc | Exact Grounding | Functional Grounding | Tool Exec OK | DB Hash |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | identity | retrieval_ir | 79.00% | 67.20% | 67.20% | 45.86% | 45.33% |
| SmolLM3-3B | api_rename | retrieval_ir | 79.00% | 67.50% | 67.50% | 46.43% | 45.33% |
| SmolLM3-3B | schema_field_rename | retrieval_ir | 79.00% | 67.10% | 67.10% | 45.68% | 44.00% |
| SmolLM3-3B | one_to_many | retrieval_ir | 79.00% | 51.30% | 67.50% | 46.43% | 45.33% |
| Qwen2.5-3B-Instruct | identity | retrieval_ir | 79.00% | 66.70% | 66.70% | 45.17% | 44.00% |
| Qwen2.5-3B-Instruct | api_rename | retrieval_ir | 79.00% | 66.90% | 66.90% | 45.56% | 44.00% |
| Qwen2.5-3B-Instruct | schema_field_rename | retrieval_ir | 79.00% | 66.60% | 66.60% | 44.98% | 42.67% |
| Qwen2.5-3B-Instruct | one_to_many | retrieval_ir | 79.00% | 51.40% | 66.90% | 45.56% | 44.00% |

`retrieval_ir` improves over `metadata_ir` by about 14-16 grounding points and
15-17 tool-execution points, while staying stable under API rename and schema
rename. In one-to-many environments, exact endpoint-name grounding drops because
the binder may select an equivalent endpoint alias, but functional grounding
remains stable.

This narrows the gap to the constructed `policy_ir` upper bound, but does not
close it: DB hash remains around 44-45% vs. 70-72% for `policy_ir` on the same
1,000-state subset.

## Full Retrieval Binder Check

To make sure the `retrieval_ir` result is not a 1,000-state subset artifact, we
also ran full 4,615-state replays for the two most important variants:
`identity` and `api_rename`. These runs use the same frozen policy predictions,
the same Target adapter, and oracle arguments; only the environment binding is
changed.

| Target | Variant | Binder | Samples | Action Acc | Functional Grounding | Tool Exec OK | DB Hash |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | identity | retrieval_ir | 4,615 | 76.21% | 66.76% | 45.22% | 48.90% |
| SmolLM3-3B | api_rename | retrieval_ir | 4,615 | 76.21% | 66.96% | 45.59% | 48.62% |
| Qwen2.5-3B-Instruct | identity | retrieval_ir | 4,615 | 75.30% | 65.48% | 44.13% | 47.51% |
| Qwen2.5-3B-Instruct | api_rename | retrieval_ir | 4,615 | 75.30% | 65.68% | 44.50% | 47.51% |

The full replay confirms the same trend as the 1,000-state matrix: changing
concrete API names does not materially reduce policy action accuracy, functional
grounding, execution success, or final DB hash. The remaining gap is therefore
not caused by environment-name brittleness; it is mainly the absolute strength of
the lightweight binder and the upstream policy/action errors.

## Binder Bottleneck Diagnostic

We added an Oracle Binder, Oracle@K diagnostics, a first typed reranker, a typed
verifier, and a behavior-aware verifier in
`environment_binding_diagnostics.md`. On the 1,000-state `api_rename` subset,
`oracle_tool` reaches 100% DB Hash for both SmolLM3 and Qwen2.5, while
`retrieval_ir` remains around 44-45% DB Hash. This confirms that the current
main bottleneck is Environment Binding rather than executor correctness.

The retrieval candidate analysis shows a large Recall@5-to-Recall@1 gap:
SmolLM3 has 76.11% Recall@5 but 49.52% Recall@1; Qwen2.5 has 73.60% Recall@5
but 47.59% Recall@1. The first `typed_rerank_ir` improves Recall@3 but does not
improve DB Hash by itself. The verifier-backed `typed_verify_ir` improves DB
Hash from 45.33% to 52.00% on SmolLM3 and from 44.00% to 50.67% on Qwen2.5.
The stronger `behavior_verify_ir` further improves DB Hash to 57.33% on SmolLM3
and 60.00% on Qwen2.5. Oracle@3 reaches about 80-81% DB Hash and Oracle@5
reaches about 84-85%, showing that the remaining gap is mainly top-K candidate
discrimination and behavior-aware verification.

## Interpretation

The current evidence supports a first version of two-way decoupling:

- Model Binding: already shown by Qwen3 -> SmolLM3 and Qwen3 -> Qwen2.5
  transfers with the same frozen source policy.
- Environment Binding: with the same fixed policy predictions, `policy_ir`
  preserves grounding, execution, and DB hash under tool rename, tool order
  permutation, schema field rename, API aliasing, and one-to-many endpoint
  aliases.
- Metadata-only binding is stable under environment swaps. The stronger
  `retrieval_ir` version improves substantially over plain metadata matching,
  but still remains below the constructed `policy_ir` upper bound. The current
  system proves the interface decomposition more strongly than it proves fully
  automatic environment onboarding.
- Concrete tool-name and tool-slot policies fail under the expected swaps:
  rename breaks name binding, and order permutation breaks slot binding.
- Direct description matching is much weaker, showing that the result is not
  simply because tool descriptions are enough.

This is not yet a final proof for arbitrary environments: the current
environment binder is a lightweight constructed binder over tau-bench aliases,
not a learned binder for a genuinely unseen third-party API. But it directly
validates the intended decomposition under controlled tool-registry shifts:

```text
Policy portability != tool-token portability
```

The next stronger experiment should replace the constructed alias map with a
learned or retrieval-based Environment Binder trained only on tool metadata, and
then test a genuinely semantically equivalent but independently implemented tool
environment.
