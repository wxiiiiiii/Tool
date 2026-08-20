# Phase-1 Experiment Plan: Frozen Policy, Many Backbones

## Core Claim

The phase-1 question is narrow:

Can hidden states from a new frozen backbone enter the same frozen action
boundary after training only a tiny model-specific adapter?

This is not yet a full agent experiment. It is a tool-action classification
experiment over BFCL single-turn decisions:

```text
A = {NO_TOOL, tool_001, tool_002, ..., tool_K}
```

A positive result means:

```text
h_source -> E_source -> z -> W_policy -> action
h_target -> E_target -> z -> W_policy(frozen) -> action
```

where only `E_target` is trained for the new backbone.

## Why This Setup

The "One Adapter Pair per Model" paper argues for a frozen shared activation
interface: each model learns a lightweight encoder/decoder adapter pair, then
new models join by fitting only their own adapters on matched text while the
shared interface remains unchanged. That motivates the "adapter-only onboarding"
constraint here, but our target object is different: reusable action boundaries,
not general activation tools.

The Master Key/Unlock paper proposes that capabilities live in low-dimensional
latent directions transferable through low-rank linear alignment. That makes it
a strong baseline, but not the main code path: Unlock transfers a capability
direction, while this project tests whether a frozen action policy can be reused
after backbone-specific representation adaptation.

BFCL is a practical first-stage bed because its code supports single-turn,
multi-turn, relevance/no-tool, hallucination-oriented, and agentic evaluation,
and its README documents local OSS model evaluation plus custom model-handler
extension points.

## Minimal Model Set

| Role | Model | Reason |
| --- | --- | --- |
| Source | `Qwen/Qwen3-4B` | 4B, 36 layers, explicitly positioned for agent/tool use. |
| Target 1 | `google/gemma-3-4b-it` | Same rough scale, different model family. |
| Target 2 | `meta-llama/Llama-3.1-8B-Instruct` | Different family and larger hidden space. |

Use one source first. Multi-source bus training is interesting, but it should be
phase 1.5 after the single-source frozen policy result is clear.

## Data Construction

Start from BFCL single-turn. Use 1k-5k decision samples.

Recommended categories:

| Group | BFCL categories |
| --- | --- |
| Basic tool choice | `simple_python`, `multiple` |
| No-tool/relevance | `relevance`, `irrelevance` |
| Live/tool diversity | `live_simple`, `live_multiple`, `live_relevance`, `live_irrelevance` |
| Later stress tests | miss function, miss parameter, long context |

For each sample, store:

```json
{
  "sample_id": "...",
  "prompt": "...",
  "tools": [...],
  "correct_action": "NO_TOOL or tool_name",
  "category": "...",
  "source_hidden": "...",
  "target_hidden": "..."
}
```

Training should not repeatedly run LLM forward passes. Extract hidden states
once and do all adapter experiments on tensors.

## Hidden-State Extraction

Use Hugging Face Transformers in phase 1:

1. Build the exact model prompt with the BFCL tool list.
2. Apply each model's chat template.
3. Run the frozen model with `output_hidden_states=True`.
4. Save the final prompt-token residual stream from one or more layers.
5. Start with the middle-to-late layer:
   - Qwen3-4B: layer 24 as default.
   - Gemma/Llama: closest fractional depth to Qwen layer 24/36.

Layer ablation is required because action identity may be most linear in a
different layer for each family.

## Training Conditions

Primary table:

| Policy Source | Target Backbone | Target Policy Training | Adapter Training | Action Acc |
| --- | --- | ---: | ---: | ---: |
| Qwen3-4B | Qwen3-4B | yes | source adapter | upper bound |
| Qwen3-4B | Gemma3-4B-IT | no | target adapter only | ? |
| Qwen3-4B | Llama3.1-8B-Instruct | no | target adapter only | ? |

Train source:

```text
min CE(W_policy(E_source(h_source)), y)
```

Freeze `W_policy`.

Train target adapter:

```text
min CE(W_policy(frozen)(E_target(h_target)), y)
```

Add a stricter unlabeled/pairwise adapter variant:

```text
min ||E_target(h_target_i) - E_source(h_source_i)||^2
```

Then evaluate with the frozen policy head. This variant is closest to the
activation-bus paper and is scientifically valuable even if accuracy is lower.

## Baselines

| Baseline | Purpose |
| --- | --- |
| Target Native | Measures what the target model can do by direct BFCL generation. |
| Target-local Linear Probe | Measures linearly readable tool identity in target hidden states. |
| Source Policy + Random Adapter | Sanity check that frozen policy does not work by chance. |
| Source Policy + Procrustes/Linear Adapter | Unlabeled paired-state alignment into source policy space. |
| Source Policy + Supervised Tiny Adapter | Main method for first feasibility test. |
| Unlock-style Latent Alignment | Compares action-boundary reuse against capability-direction transfer. |

The most important baseline is `Target-local Linear Probe`. If the frozen
policy transfer is far below it, the portability cost may be too high.

## Metrics

Primary:

```text
Frozen Policy Accuracy on New Backbone
```

Secondary:

- Macro F1 over actions.
- NO_TOOL accuracy.
- Tool-vs-no-tool accuracy.
- Per-category action accuracy.
- Confusion matrix over action types.
- Calibration/ECE if using the head probabilities.

## First Week Run Order

1. Build a 10-tool static BFCL subset with `NO_TOOL`.
2. Extract Qwen/Gemma/Llama hidden states for 1k samples.
3. Train Qwen source adapter + policy head.
4. Run random adapter and target-local probe.
5. Train supervised Gemma adapter.
6. Fit paired linear/procrustes Gemma adapter.
7. Repeat only the best setting on Llama.
8. Run layer/rank/z-dim ablations after the signal is non-random.

## Decision Rules

Continue to phase 2 if:

- Gemma and Llama frozen-policy accuracy beat random by a large margin.
- At least one target is within 80-90% of target-local linear probe accuracy.
- NO_TOOL/relevance accuracy transfers cleanly.

Pause and diagnose if:

- Accuracy is near random after supervised adapter training.
- Source policy upper bound is weak.
- Target-local probe is strong but frozen-policy adapter is weak.

Likely diagnoses:

- Wrong layer or pooling token.
- Action label leakage/imbalance.
- Tool schemas too heterogeneous for tool-name-only abstraction.
- Source policy z-space too narrow or too overfit to Qwen geometry.

## Phase 2: Tau-Bench

Only after phase 1 succeeds, add memory and sequential policy:

```text
z_t, memory_{t-1}, observation_t -> policy -> abstract_action_t
```

Tau-bench should test multi-turn policy stability, not be used to discover
whether the single-step frozen boundary exists.

