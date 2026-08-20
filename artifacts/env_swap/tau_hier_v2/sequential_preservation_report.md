# Sequential Preservation Report

This report expands sequential preservation beyond single-step action accuracy.
It evaluates full 4,615-state tau-bench policy traces grouped into 362
trajectories. The comparison is against the expected abstract policy action
sequence in the tau traces. Existing Source-policy Agreement and Transition
Agreement are included from the main policy-fidelity evaluation.

## Metrics

- `Trajectory Exact Match`: the whole abstract action sequence is identical.
- `Prefix@K`: the first K decisions match before any divergence.
- `Avg First Divergence Step`: average first index where the predicted abstract
  action differs from the expected trace. Higher is better.
- `Edit Similarity`: `1 - edit_distance / max_length` over abstract action
  sequences.
- `Transition-Pair Exact`: adjacent abstract action pairs match at the same
  positions.

## Results

| Target | Action Acc | Source Policy Agreement | Transition Agreement | Trajectory Exact | Prefix@1 | Prefix@3 | Prefix@5 | Avg First Divergence | Edit Similarity | Transition-Pair Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | 76.21% | 91.16% | 82.67% | 8.01% | 94.48% | 57.18% | 34.53% | 3.97 | 76.66% | 56.71% |
| Qwen2.5-3B-Instruct | 75.30% | 89.21% | 79.14% | 6.91% | 94.75% | 57.73% | 34.25% | 4.02 | 75.81% | 55.35% |

## Interpretation

The high Source Policy Agreement and Transition Agreement show that the target
adapter preserves source-policy dynamics much better than prompting or
activation-steering baselines. However, exact trajectory match is strict and
remains low: one early error makes the entire trajectory non-exact. The edit
similarity around 76% gives a more graded view and is closer to the single-step
accuracy.

For future compositional environments, raw concrete API sequence equality should
not be the target metric. A single abstract action may refine to multiple
concrete calls in a new environment. The right sequential metric should be:

```text
abstract Policy IR sequence equivalence + final DB state equivalence
```

