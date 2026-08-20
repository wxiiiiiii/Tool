# Environment Binding Diagnostics

This diagnostic follows the next-step plan: decide whether the current bottleneck
is the Environment Binder or the Policy IR itself. The experiment fixes the
Target adapter, frozen policy, Universal Policy IR, tau-bench tasks, executor,
and oracle arguments. Only the binder is changed.

## Code Changes

Updated `baselines/tau_environment_swap.py` with:

- `oracle_tool`: diagnostic upper-bound binder that uses the ground-truth
  concrete tool in the swapped environment. This isolates binder/executor
  capacity from retrieval errors.
- `typed_rerank_ir`: retrieve -> typed rerank binder. It first retrieves top-5
  candidates, then reranks by typed behavioral compatibility:
  `(operation, object, intended effect, constraints)`.
- `typed_verify_ir`: retrieve -> typed rerank -> verify binder. It adds a
  lightweight verifier over required schema evidence, side-effect compatibility,
  and state/action preconditions.
- `behavior_verify_ir`: behavior-aware capability matcher. It adds explicit
  hard-negative checks for operation/object/effect compatibility, for example
  separating `get_order`, `modify_order`, `cancel_order`, `return_order`, and
  `exchange_order`.
- `oracle_top3` / `oracle_top5`: diagnostic binders that execute the correct
  tool only if it appears in the retriever's top-K candidates; otherwise they
  fall back to the retrieval top-1. These measure how much error is due to
  top-K recall versus candidate discrimination.
- Binder diagnostics: `functional_recall_at_1/3/5_tool_only`,
  `functional_mrr_tool_only`, and `candidate_original_tools_top5`.

## 1,000-State API-Rename Diagnostic

All rows below use `api_rename`, where concrete tool names are replaced with
opaque names such as `env_api_03`.

| Target | Binder | Action Acc | Functional Grounding | Recall@1 | Recall@3 | Recall@5 | MRR | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | retrieval_ir | 79.00% | 67.50% | 49.52% | 66.09% | 76.11% | 60.36% | 46.43% | 45.33% |
| SmolLM3-3B | typed_rerank_ir | 79.00% | 67.70% | 49.90% | 71.87% | 76.11% | 60.92% | 47.18% | 45.33% |
| SmolLM3-3B | typed_verify_ir | 79.00% | 72.60% | 59.34% | 74.57% | 76.11% | 67.14% | 59.30% | 52.00% |
| SmolLM3-3B | behavior_verify_ir | 79.00% | 74.50% | 63.01% | 86.32% | 88.05% | 74.52% | 60.15% | 57.33% |
| SmolLM3-3B | oracle_top3 | 79.00% | 76.10% | 49.52% | 66.09% | 76.11% | 60.36% | 60.15% | 80.00% |
| SmolLM3-3B | oracle_top5 | 79.00% | 81.30% | 49.52% | 66.09% | 76.11% | 60.36% | 69.36% | 84.00% |
| SmolLM3-3B | oracle_tool | 79.00% | 100.00% | 49.52% | 66.09% | 76.11% | 60.36% | 94.61% | 100.00% |
| Qwen2.5-3B-Instruct | retrieval_ir | 79.00% | 66.90% | 47.59% | 64.35% | 73.60% | 58.39% | 45.56% | 44.00% |
| Qwen2.5-3B-Instruct | typed_rerank_ir | 79.00% | 66.50% | 46.82% | 69.56% | 73.60% | 58.15% | 45.17% | 44.00% |
| Qwen2.5-3B-Instruct | typed_verify_ir | 79.00% | 71.50% | 56.45% | 72.83% | 73.60% | 64.64% | 57.68% | 50.67% |
| Qwen2.5-3B-Instruct | behavior_verify_ir | 79.00% | 73.90% | 61.08% | 85.55% | 86.32% | 72.83% | 59.85% | 60.00% |
| Qwen2.5-3B-Instruct | oracle_top3 | 79.00% | 75.60% | 47.59% | 64.35% | 73.60% | 58.39% | 59.85% | 81.33% |
| Qwen2.5-3B-Instruct | oracle_top5 | 79.00% | 80.40% | 47.59% | 64.35% | 73.60% | 58.39% | 68.53% | 85.33% |
| Qwen2.5-3B-Instruct | oracle_tool | 79.00% | 100.00% | 47.59% | 64.35% | 73.60% | 58.39% | 94.61% | 100.00% |

For `oracle_tool`, Recall@K columns are still computed from the retrieval
candidate list for diagnosis; the executed tool itself is oracle-selected. For
`oracle_top3` and `oracle_top5`, Recall@K is also computed from the original
retrieval candidate list, because these rows measure the retrieval-stage upper
bound.

## Interpretation

The Oracle Binder result is decisive: when the correct concrete tool is supplied,
DB Hash reaches 100% on this 1,000-state subset. Therefore the executor,
argument replay path, and environment swap machinery are not the bottleneck.
The main bottleneck is Environment Binding.

The retrieval analysis shows that the correct tool is often present but not
ranked first. For SmolLM3, Recall@5 is 76.11% while Recall@1 is only 49.52%;
for Qwen2.5, Recall@5 is 73.60% while Recall@1 is 47.59%. This means the next
improvement should focus on reranking and verification, not only stronger
retrieval.

The first typed reranker is partially useful but not sufficient. It improves
SmolLM3 Recall@3 from 66.09% to 71.87% and Qwen2.5 Recall@3 from 64.35% to
69.56%, but it does not improve DB Hash by itself. After adding a verifier,
`typed_verify_ir` raises SmolLM3 DB Hash from 45.33% to 52.00% and Qwen2.5 DB
Hash from 44.00% to 50.67%. This supports the diagnosis that the bottleneck is
not only candidate retrieval, but also state-conditioned verification of whether
a candidate tool is executable and behaviorally compatible with the Policy IR.

The behavior-aware capability matcher gives the next improvement. Compared with
`typed_verify_ir`, `behavior_verify_ir` increases SmolLM3 DB Hash from 52.00%
to 57.33% and Qwen2.5 DB Hash from 50.67% to 60.00%. This supports the core
claim that semantic similarity is not enough: the binder must distinguish
behavioral compatibility, especially side effects and preconditions.

Oracle@K exposes the remaining loss. With the original retrieval candidates,
Oracle@3 already reaches 80.00% DB Hash on SmolLM3 and 81.33% on Qwen2.5;
Oracle@5 reaches 84.00% and 85.33%. Therefore the largest remaining gap is no
longer executor correctness, but candidate discrimination and verification
inside the top-K set.

## Next Code Target

The implemented `typed_verify_ir` is the first version of the three-stage
binder:

```text
Policy IR -> retrieve top-5 -> typed rerank -> verifier -> concrete tool
```

The next improvement should make the verifier less heuristic by adding an
explicit state/memory object and learned hard-negative calibration for tool
families such as `get_order`, `update_order`, and `cancel_order`.

The next stronger version should train or calibrate a listwise/pairwise
capability ranker using hard negatives:

```text
score(policy_ir, correct_tool) > score(policy_ir, hard_negative_tool) + margin
```

The training signal should use tool metadata, schema, side-effect annotations,
precondition annotations, and synthetic capability pairs, not Target task action
labels.
