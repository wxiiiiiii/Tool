# Complete Decoupling Experiment Report

This report answers which experiments from the proposed plan have been run, which
ones are still necessary, and what the current results show.

Documents in `paper/` are used only as research references. The experiments
below use the same frozen Qwen3 source policy and Policy IR unless otherwise
noted.

## Experiment Checklist

| Group | Experiment | Status | Necessary? | Current conclusion |
|---|---|---|---|---|
| 3 | Backbone + environment double swap | Done for SmolLM3 and Qwen2.5 over six environment variants | Yes | Stable under simultaneous target-backbone and environment changes on the tested variants |
| 4 | Policy IR sufficiency / granularity | Done for ACT-only, operation, operation+object, full structured IR, and behavior-aware IR | Yes | Rich behavior-aware IR/binding is necessary; coarse IR is insufficient |
| 5 | Counterfactual / perturbation invariance | Done for API rename, tool-order permutation, schema-field rename, API alias, one-to-many alias | Yes | Functional grounding and DB Hash are stable; exact endpoint names should not be the primary metric under one-to-many |
| 6 | Sequential preservation | Done with transition agreement, trajectory exact, prefix, edit similarity, first divergence, and transition-pair metrics | Yes | Ours preserves policy dynamics much better than baselines, though full exact trajectories remain hard |
| 7 | Strong baselines and ablations | Done for concrete tool ID, description selection, retrieval/typed/behavior binders, ASA, EvoTool, native prompting, target-local/latent baselines, and method-component ablations | Yes | Ours/binder stack is stronger on policy fidelity; environment binding remains the main bottleneck |
| Future | Gemma/Llama target backbones | Not yet run in this matrix | Yes for final paper claim | Needed to fully support broad backbone invariance beyond SmolLM3/Qwen2.5 |
| Future | Observation reformatting and true 1->N composition | Not yet implemented | Yes for stronger environment claim | Current `one_to_many` is alias/functionally equivalent, not a true multi-step refinement environment |

## Backbone Swap: Fixed Environment

Full 4,615-state tau-bench replay with oracle arguments.

| Backbone / Policy Path | Samples | Action Acc | Tool Grounding | Tool Exec OK | DB Hash |
|---|---:|---:|---:|---:|---:|
| Qwen3 source policy | 4,615 | 78.09% | 80.91% | 68.80% | 71.82% |
| Qwen3 -> SmolLM3 adapter | 4,615 | 76.21% | 80.22% | 68.48% | 72.10% |
| Qwen3 -> Qwen2.5 adapter | 4,615 | 75.30% | 78.72% | 66.87% | 69.61% |

Interpretation: replacing the backbone while freezing the policy keeps the final
DB state close to the Qwen3 source-policy level. This supports Model Binding.

## Double Swap Matrix

1,000-state tau-bench replay with `behavior_verify_ir`, oracle arguments, fixed
frozen policy, fixed Policy IR, and swapped backend tool environments.

| Target Backbone | Environment Variant | Functional Grounding | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|
| SmolLM3-3B | identity | 74.80% | 60.53% | 57.33% |
| SmolLM3-3B | tool_order_permutation | 74.80% | 60.53% | 57.33% |
| SmolLM3-3B | api_rename | 74.50% | 60.15% | 57.33% |
| SmolLM3-3B | schema_field_rename | 76.20% | 62.22% | 57.33% |
| SmolLM3-3B | api_alias | 74.80% | 60.53% | 57.33% |
| SmolLM3-3B | one_to_many | 74.80% | 60.53% | 57.33% |
| Qwen2.5-3B-Instruct | identity | 74.20% | 60.23% | 60.00% |
| Qwen2.5-3B-Instruct | tool_order_permutation | 74.20% | 60.23% | 60.00% |
| Qwen2.5-3B-Instruct | api_rename | 73.90% | 59.85% | 60.00% |
| Qwen2.5-3B-Instruct | schema_field_rename | 75.70% | 61.97% | 60.00% |
| Qwen2.5-3B-Instruct | api_alias | 74.20% | 60.23% | 60.00% |
| Qwen2.5-3B-Instruct | one_to_many | 74.20% | 60.23% | 60.00% |

Interpretation: the performance drop is approximately decomposable. Swapping the
backbone and swapping the environment do not interact catastrophically on these
controlled perturbations. This is the strongest current evidence for two-way
decoupling:

```text
backbone binding != environment binding
```

## Policy IR / Binder Sufficiency

1,000-state `api_rename` diagnostic. This compares increasingly rich binding
interfaces.

Detailed report: `ir_sufficiency_and_ablation_report.md`.

| Target | Binder / IR Level | Functional Grounding | Recall@3 | Recall@5 | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | description_only | 1.40% | n/a | n/a | 1.40% | 24.00% |
| SmolLM3-3B | concrete_tool_name | 48.10% | n/a | n/a | 0.00% | 16.00% |
| SmolLM3-3B | ACT only | 62.20% | 68.79% | 72.83% | 37.03% | 40.00% |
| SmolLM3-3B | operation | 62.20% | 71.87% | 73.03% | 37.03% | 40.00% |
| SmolLM3-3B | operation + object | 62.30% | 70.91% | 73.99% | 37.22% | 41.33% |
| SmolLM3-3B | operation + object + effect + constraint | 72.70% | 76.49% | 79.38% | 57.52% | 44.00% |
| SmolLM3-3B | typed_verify_ir | 72.60% | 74.57% | 76.11% | 59.30% | 52.00% |
| SmolLM3-3B | behavior_verify_ir | 74.50% | 86.32% | 88.05% | 60.15% | 57.33% |
| SmolLM3-3B | oracle_top3 | 76.10% | 66.09% | 76.11% | 60.15% | 80.00% |
| SmolLM3-3B | oracle_top5 | 81.30% | 66.09% | 76.11% | 69.36% | 84.00% |
| SmolLM3-3B | oracle_tool | 100.00% | 66.09% | 76.11% | 94.61% | 100.00% |
| Qwen2.5-3B-Instruct | ACT only | 61.90% | 66.28% | 71.10% | 36.68% | 42.67% |
| Qwen2.5-3B-Instruct | operation | 61.90% | 69.75% | 71.48% | 36.68% | 42.67% |
| Qwen2.5-3B-Instruct | operation + object | 62.00% | 69.17% | 71.48% | 36.87% | 44.00% |
| Qwen2.5-3B-Instruct | operation + object + effect + constraint | 72.00% | 74.76% | 77.26% | 56.95% | 44.00% |
| Qwen2.5-3B-Instruct | typed_verify_ir | 71.50% | 72.83% | 73.60% | 57.68% | 50.67% |
| Qwen2.5-3B-Instruct | behavior_verify_ir | 73.90% | 85.55% | 86.32% | 59.85% | 60.00% |
| Qwen2.5-3B-Instruct | oracle_top3 | 75.60% | 64.35% | 73.60% | 59.85% | 81.33% |
| Qwen2.5-3B-Instruct | oracle_top5 | 80.40% | 64.35% | 73.60% | 68.53% | 85.33% |
| Qwen2.5-3B-Instruct | oracle_tool | 100.00% | 64.35% | 73.60% | 94.61% | 100.00% |

Interpretation: richer behavior-aware binding improves over semantic retrieval.
Oracle@K shows a large remaining top-K discrimination gap. This supports the
claim:

```text
semantic similarity != behavioral compatibility
```

The strict minimal-sufficient-IR test now shows that coarse IR is not sufficient.
Operation and operation+object add little to final DB state; effect/constraint
information improves grounding but still needs behavior-aware capability
matching and verification to preserve final state.

## Perturbation Invariance

With `behavior_verify_ir`, API rename, tool-order permutation, schema-field
rename, API aliases, and one-to-many aliases do not materially change DB Hash.
The one-to-many row lowers exact tool-name grounding but keeps functional
grounding and DB Hash stable, which is why functional grounding is the correct
metric for environment changes.

## Sequential Preservation

Full 4,615-state policy traces.

Detailed report: `sequential_preservation_report.md`.

| Target | Action Acc | Source Policy Agreement | Transition Agreement | Trajectory Exact | Prefix@3 | Edit Similarity | Avg First Divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | 76.21% | 91.16% | 82.67% | 8.01% | 57.18% | 76.66% | 3.97 |
| Qwen2.5-3B-Instruct | 75.30% | 89.21% | 79.14% | 6.91% | 57.73% | 75.81% | 4.02 |

Interpretation: transition agreement is high relative to baselines, but exact
trajectory match is strict and remains low because one early divergence can make
the whole trajectory non-exact. For future 1->N environments, raw concrete tool
sequence exact match should be replaced by abstract Policy IR sequence
equivalence plus final state equivalence.

## Baselines

| Target | Method | Target Labels | Samples | Action Acc | Source Agreement | Transition Agreement | DB Hash |
|---|---|---|---:|---:|---:|---:|---:|
| SmolLM3-3B | Ours, tiny adapter + frozen policy | no | 4,615 | 76.21% | 91.16% | 82.67% | 72.10% |
| SmolLM3-3B | ASA target activation steering | calibration labels | 4,615 | 50.81% | 55.08% | 30.66% | n/a |
| SmolLM3-3B | EvoTool official theta | no | 60 | 20.00% | 23.33% | 3.57% | n/a |
| SmolLM3-3B | Native prompting | no | 60 | 0.00% | n/a | n/a | n/a |
| Qwen2.5-3B-Instruct | Ours, tiny adapter + frozen policy | no | 4,615 | 75.30% | 89.21% | 79.14% | 69.61% |
| Qwen2.5-3B-Instruct | ASA target activation steering | calibration labels | 4,615 | 46.72% | 49.56% | 24.83% | n/a |
| Qwen2.5-3B-Instruct | EvoTool official theta | no | 20 | 35.00% | 35.00% | 11.11% | n/a |
| Qwen2.5-3B-Instruct | Native prompting | no | 60 | 23.33% | n/a | n/a | n/a |

## What Is Still Necessary

The following experiments are still worth doing for the final paper:

1. Add Gemma/Llama target adapters to the same matrix. The current two-way
   decoupling evidence is strong for SmolLM3 and Qwen2.5, but broader backbone
   invariance requires another model family.
2. Implement true 1->N composition where one abstract action maps to a concrete
   program such as `get_order -> update_status(cancelled)`. The current
   `one_to_many` is an alias/equivalent-endpoint stress test, not full
   compositional refinement.
3. Add observation reformatting and canonical observation normalization. This is
   needed to prove sequential equivalence when environments return different
   observation schemas.
4. Replace the heuristic behavior matcher with a calibrated/listwise capability
   ranker trained on synthetic hard negatives, without Target task labels.
