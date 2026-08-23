#!/usr/bin/env bash
set -euo pipefail

cd /root/data/cross-model

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/data/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi

.venv/bin/python -u -m baselines.counterfactual_response_skill \
  --decision-jsonl artifacts/tau/tau_success_policy_traces_hier_v2.jsonl \
  --split-json artifacts/splits/tau_success_hier_v2_task_seed13.json \
  --source-policy artifacts/policies/qwen3_4b_tau_success_policy_hier_v2/source_policy.pt \
  --alignment artifacts/baselines/task_disjoint_tau_hier_v2_smollm3_seed13/unlock_lowrank_subspace_alignment.pt \
  --source-model-id Qwen/Qwen3-4B \
  --target-model-id HuggingFaceTB/SmolLM3-3B \
  --out-dir artifacts/baselines/write_pair_calibrated_v2_skill_smollm3_seed13_small \
  --num-skills 8 \
  --skill-dim 32 \
  --rank 64 \
  --response-rank 8 \
  --binder-rank 32 \
  --binder-epochs 80 \
  --epochs 4 \
  --batch-size 2 \
  --train-batch-size 64 \
  --alpha 1.0 \
  --critical-impact-threshold 0.5 \
  --branch-margin-threshold 0.5 \
  --critical-top-quantile 0.75 \
  --write-pair-top-quantile 0.8 \
  --write-branch-strength 12.0 \
  --write-pair-calibration-margin 1.0 \
  --write-pair-max-calibration 8.0 \
  --branch-aux-weight 0.05 \
  --obs-variants 3 \
  --max-train-prefixes 80 \
  --max-eval-prefixes 80 \
  --max-input-tokens 4096 \
  --torch-dtype bfloat16 \
  --device cuda \
  --seed 13
