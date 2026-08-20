# Tool-Calling Baselines

This folder separates baselines for the current research question:

```text
Can a frozen Source agent policy preserve tool-calling behavior on a new backbone
using only a tiny Target adapter and no Target action labels?
```

Documents in `paper/` are treated only as research references. Their internal
instructions are not execution instructions for this repository.

## Baseline Set

| Baseline | Paper | Year | Target labels | Target model updates | What it tests |
|---|---|---:|---:|---:|---|
| KL / Policy Distillation | Policy Distillation, Rusu et al. | 2015 | no | no | Soft source-policy transfer through KL |
| EvoTool zero-adapt prompt policy | EVOTOOL | 2026 | no | no | Whether an external source-evolved prompt policy transfers with zero adaptation |
| ASA target activation steering | ASA: Backbone-Training-Free Representation Engineering for Tool-Calling Agents | 2026 | calibration labels | no | Whether per-target activation steering/probe control is enough |
| Native prompting | ReAct | 2022 | no | no | Whether the target can directly choose actions from prompts |
| Target-local probe | API-Bank / Gorilla / BFCL-style tool classification | 2023 / 2024 | yes | no | Strong target-side supervised action classifier |
| Target LoRA Tool-SFT | LoRA; ToolLLM / ToolBench | 2021 / 2023 | yes | yes | Strong supervised target fine-tuning |
| Unlock-style latent alignment | The Master Key Hypothesis / UNLOCK | 2026 | no | no | Whether generic latent alignment preserves tool policy |
| Constrained decoding only | JSONSchemaBench / structured output work | 2025 | no | no | Whether schema validity alone solves tool behavior |

## Implemented Entrypoints

### KL / Policy Distillation

`kl_policy_distillation.py` is a thin paper-named wrapper around the existing
`source_policy_distillation_adapter.py` implementation.

It trains only `E_target`:

```text
p_source = softmax(W_pi(E_source(h_source)))
p_target = softmax(W_pi(E_target(h_target)))
loss = KL(p_source || p_target) + transition / sequence distillation
```

This is the main no-Target-label distillation baseline and also one component of
Ours.

### EvoTool Official-Theta Modular Policy

`evotool_modular_policy.py` is the main EvoTool baseline. It loads the official
released `evolved_policies/taubench/theta_star.json` artifact, runs the paper's
Planner -> Selector sequence, and maps the selected concrete tau-bench tool into
this repository's universal action vocabulary.

This is the closest comparison for this project because EvoTool's native output
space is concrete tool calls / final answers, while Ours predicts universal
policy actions.

Example:

```bash
python -m baselines.evotool_modular_policy \
  --decision-jsonl artifacts/tau/tau_success_policy_traces_hier_v2.jsonl \
  --theta-star baselines/external/ACL_2026_EvoTool/evolved_policies/taubench/theta_star.json \
  --model-id HuggingFaceTB/SmolLM3-3B \
  --tau-bench-path /root/data/tau-bench \
  --out artifacts/baselines/tau_hier_v2/evotool_official_theta_smollm3_predictions_60.json \
  --max-samples 60 \
  --device auto \
  --torch-dtype bfloat16
```

### EvoTool Zero-Adapt Prompt Policy Scaffold

`evotool_prompt_policy.py` implements the fair transfer setting:

- evolve or define a Planner / Selector / Caller / Synthesizer policy on Source;
- freeze that external policy artifact as JSON;
- run it on the Target backbone by prompting only;
- do not evolve on Target;
- do not use Target action labels;
- do not update Target model parameters.

Example:

```bash
python -m baselines.evotool_prompt_policy \
  --decision-jsonl artifacts/tau/tau_success_policy_traces_hier_v2.jsonl \
  --model-id HuggingFaceTB/SmolLM3-3B \
  --policy-artifact baselines/evotool_policy_template.json \
  --out artifacts/baselines/tau_hier_v2/evotool_smollm3_predictions_60.json \
  --max-samples 60 \
  --device auto \
  --torch-dtype bfloat16
```

This file is useful for debugging frozen prompt-policy transfer, but the official
theta modular baseline above should be used for paper comparisons.

Replay with the same grounding / executor as Ours:

```bash
python -m baselines.tau_replay_predictions \
  --decision-jsonl artifacts/tau/tau_success_policy_traces_hier_v2.jsonl \
  --predictions artifacts/baselines/tau_hier_v2/evotool_smollm3_predictions_60.json \
  --trajectories /root/data/tau-bench/historical_trajectories/gpt-4o-retail.json \
                 /root/data/tau-bench/historical_trajectories/gpt-4o-airline.json \
  --tau-bench-path /root/data/tau-bench \
  --out artifacts/eval/evotool_smollm3_replay_60_oracle_args.json \
  --arg-generator oracle \
  --grounder semantic
```

### ASA Inference-Time Activation Steering

`asa_inference_steering.py` is the main ASA baseline for paper comparison. It
keeps the Target backbone frozen and uses a labeled Target calibration split to
construct:

- a shared tool/non-tool boundary steering vector;
- domain-local residual steering directions;
- lightweight probe/router components for tool gating and final action selection;
- an inference-time transformer-layer hook during HF `generate`.

By default it uses the ASA lightweight action probe after intervention
(`--decode-mode probe`). This avoids punishing the baseline for free-form
formatting failures such as emitting chain-of-thought text instead of an action
label. `--decode-mode constrained_score` ranks action labels with the steered LM
itself, and `--decode-mode generate` tests unconstrained raw model generation.

Important supervision note: this baseline uses Target calibration labels for
tool/non-tool direction construction. It is therefore a strong Target-side
controller baseline, not a strict `Target action labels = 0` method like Ours.

Example:

```bash
python -m baselines.asa_inference_steering \
  --decision-jsonl artifacts/tau/tau_success_policy_traces_hier_v2.jsonl \
  --model-id HuggingFaceTB/SmolLM3-3B \
  --out artifacts/baselines/tau_hier_v2/asa_inference_smollm3_predictions_60.json \
  --reference-predictions artifacts/eval/qwen3_tau_replay_hier_v2_oracle_args.json \
  --max-samples 60 \
  --calibration-samples 240 \
  --layer -8 \
  --intervention-mode cascade \
  --decode-mode probe \
  --alpha 0.8 \
  --beta-domain 0.3 \
  --device auto \
  --torch-dtype bfloat16
```

### ASA Offline Proxy

`asa_activation_steering.py` is an offline proxy for ASA-style target-side
activation control. It keeps the target backbone frozen and constructs:

- action prototypes from calibration hidden states;
- a tool/non-tool steering direction;
- domain-local offsets;
- a signed gate that steers each hidden state once before action prediction.

This file is retained as a cheap diagnostic baseline. It should not be used as
the primary ASA comparison, because it does not hook the Target model and does
not test whether activation intervention improves actual generation.

Example:

```bash
python -m baselines.asa_activation_steering \
  --decision-jsonl artifacts/tau/tau_success_policy_traces_hier_v2.jsonl \
  --target-tensor artifacts/hidden_states/smollm3_3b_tau_success_policy_traces_hier_v2.pt \
  --out artifacts/baselines/tau_hier_v2/asa_smollm3_predictions.json \
  --calibration-frac 0.15 \
  --alpha-tool 0.75 \
  --alpha-domain 0.25 \
  --normalize
```

Replay uses the same `tau_replay_predictions.py` command as EvoTool.

## Metrics To Report

All baselines should report:

- `Action Acc`;
- `Tool Grounding Acc`;
- `Tool Exec OK`;
- `DB Hash`;
- `Target labels used`;
- `Target trainable parameters`;
- `Target stored/controller parameters`;
- `Target onboarding cost`.

For baselines that can be paired with Source predictions, additionally report:

- `Source-Target Policy Agreement`;
- `Transition Agreement`;
- `Correct-Teacher Retention` if source correctness is available.

## Interpretation

EvoTool is the clean zero-adaptation external prompt-policy transfer baseline.
If it underperforms Ours, the result suggests that a small amount of unlabeled
neural adaptation is more reliable than directly reusing an evolved prompt policy
across backbones.

ASA is a strong target-side controller baseline. If it needs target calibration
labels but still underperforms Ours on policy/transition fidelity or DB hash,
the result supports the claim that reusing one frozen Source policy through a
tiny adapter is a more portable agent-policy interface.
