# Universal Agent Policy

Small independent research scaffold for testing whether a frozen tool-action
policy can preserve behavior across LLM backbones through tiny per-backbone
adapters.

The first-stage experiment is intentionally offline:

1. Extract BFCL single-turn decision hidden states once per model.
2. Train a source adapter and policy head on Qwen3-4B hidden states.
3. Freeze the policy head.
4. Train only target adapters for Gemma/Llama hidden states.
5. Compare against target-native, target-local probe, random adapter,
   linear/procrustes adapter, and Unlock-style latent-alignment baselines.

See [docs/experiment_plan.md](C:/Users/Wang/Desktop/M/cross-model/docs/experiment_plan.md)
for the full experimental design.

## Offline Tensor Contract

Each model's extracted hidden states should be stored as `.pt` or `.npz`:

```python
{
    "h": Tensor[num_samples, hidden_dim],
    "y": Tensor[num_samples],              # integer action id
    "sample_id": list[str],                # optional, for paired alignment
    "action_names": list[str],             # optional
    "category": list[str],                 # optional BFCL category
}
```

The first minimal action space should be:

```text
NO_TOOL, tool_001, tool_002, ..., tool_K
```

Argument generation and constrained decoding are deliberately out of scope for
phase 1. The metric is frozen-policy action accuracy.

## Agent Runtime

The phase-2 scaffold extends the frozen-policy experiment into a minimal
single-turn agent:

```text
h_t^M -> E_M -> z_t -> frozen W_policy -> abstract action
abstract action + current tools -> grounded tool
grounded tool + prompt -> argument JSON
argument JSON + schema -> constrained executable tool call
```

The first implementation uses a slot-level universal action space:

```text
NO_TOOL, SELECT_TOOL_0, SELECT_TOOL_1, ...
```

This is a better fit for BFCL than global tool names because BFCL supplies a
fresh tool list per sample and many concrete function names appear only once.
The grounding layer maps `SELECT_TOOL_i` to the i-th concrete tool for the
current sample.

Key runtime files:

- `universal_agent_policy/policy/action_space.py`: universal slot actions.
- `universal_agent_policy/runtime/grounding.py`: abstract action to concrete
  tool mapping.
- `universal_agent_policy/runtime/policy_runtime.py`: load frozen source policy
  and per-backbone target adapter checkpoints.
- `universal_agent_policy/runtime/argument_generation.py`: empty, oracle, and
  Hugging Face JSON argument generators.
- `universal_agent_policy/runtime/constrained_decode.py`: JSON-schema
  constrained parsing, type coercion, required-field checks, enum checks, and
  unknown-field dropping.
- `universal_agent_policy/runtime/agent.py`: combines policy, grounding,
  argument generation, and constrained decoding.
- `universal_agent_policy/eval/bfcl_agent.py`: offline BFCL agent evaluation
  over saved hidden-state tensors.

## Example Commands

Extract hidden states from a BFCL-derived JSONL file:

```powershell
python -m universal_agent_policy.extract_hidden_states `
  --model-id Qwen/Qwen3-4B `
  --input-jsonl artifacts/bfcl/phase1_decisions.jsonl `
  --out artifacts/hidden_states/qwen3_4b_bfcl_phase1.pt `
  --layer 24 `
  --device cuda
```

Train the source policy and target adapters:

```powershell
python -m universal_agent_policy.train_source_policy `
  --tensor artifacts/hidden_states/qwen3_4b_bfcl_phase1.pt `
  --out-dir artifacts/policies/qwen3_4b `
  --z-dim 256 `
  --adapter low_rank_mlp `
  --rank 64

python -m universal_agent_policy.train_target_adapter `
  --tensor artifacts/hidden_states/gemma3_4b_it_bfcl_phase1.pt `
  --source-policy artifacts/policies/qwen3_4b/source_policy.pt `
  --out-dir artifacts/adapters/gemma3_4b_it `
  --adapter low_rank_mlp `
  --rank 64

python -m universal_agent_policy.procrustes_baseline `
  --source-tensor artifacts/hidden_states/qwen3_4b_bfcl_phase1.pt `
  --target-tensor artifacts/hidden_states/gemma3_4b_it_bfcl_phase1.pt `
  --source-policy artifacts/policies/qwen3_4b/source_policy.pt `
  --out-dir artifacts/baselines/gemma3_4b_it_procrustes
```

Evaluate the single-turn agent with oracle arguments:

```powershell
python -m universal_agent_policy.eval.bfcl_agent `
  --decision-jsonl artifacts/bfcl/slot_decisions_small.jsonl `
  --hidden-tensor artifacts/hidden_states/smollm2_360m_bfcl_slot180.pt `
  --source-policy artifacts/policies/qwen25_05b_slot180/source_policy.pt `
  --target-adapter artifacts/adapters/smollm2_360m_slot180/target_adapter.pt `
  --arg-generator oracle `
  --out artifacts/eval/smollm2_slot180_oracle_agent.json
```

Use `--arg-generator hf --arg-model-id <model>` to test real argument
generation. Use `oracle` only as an upper-bound setting that isolates policy
and grounding from argument-generation errors.

## Files

- `configs/phase1_bfcl_small.yaml`: recommended phase-1 settings.
- `universal_agent_policy/`: offline adapter/policy training code.
- `docs/experiment_plan.md`: research design, metrics, ablations, and risks.
