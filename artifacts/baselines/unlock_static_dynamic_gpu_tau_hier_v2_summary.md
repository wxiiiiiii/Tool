# Unlock Static vs Multi-Step Dynamics, GPU Remote Run

Remote host: `ssh-cn-huabei1.ebcloud.com`

GPU check:

- PyTorch: `2.7.0+cu126`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4090`

## Question

This experiment tests whether an Unlock-style latent/subspace transfer baseline solves full sequential Agent policy transfer, or mainly transfers static per-state action boundaries.

Unlock is not result alignment. In this setting it is implemented as latent/subspace alignment:

```text
h_target -> low-rank alignment -> Qwen policy latent -> frozen action head -> action
```

The stricter question is whether this also preserves:

```text
state_t -> action_t -> observation_{t+1} -> state_{t+1} -> action_{t+1}
```

## Methods

| Method | Description | Uses trajectory optimization? |
|---|---|---|
| `unlock_lowrank_subspace` | Static paired hidden-state low-rank alignment into Qwen3 policy latent space | No |
| `unlock_lowrank_intervention` | Static low-rank alignment plus fixed action-direction logit intervention | No state-dependent trajectory optimization |
| `unlock_lowrank_dynamic` | Static low-rank alignment plus global Source transition prior | Yes, but only naive global Markov prior |

All methods use no target action labels, no target gradient steps, and no online source queries during evaluation.

## Full Results

Replay uses semantic grounding, oracle arguments, and the same tau-bench executor / DB-hash path.

| Target | Method | Action Acc | Seq Exact | Transition Pair Exact | Next Acc On Expected Flip | Tool Exec OK | DB Hash |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | `unlock_lowrank_subspace` | 75.99% | 8.01% | 56.17% | 73.36% | 67.79% | 69.89% |
| SmolLM3-3B | `unlock_lowrank_intervention` | 75.32% | 6.63% | 55.21% | 72.51% | 65.56% | 70.72% |
| SmolLM3-3B | `unlock_lowrank_dynamic` | 74.50% | 4.70% | 54.39% | 71.24% | 67.36% | 67.40% |
| Qwen2.5-3B | `unlock_lowrank_subspace` | 75.86% | 8.01% | 56.03% | 73.53% | 67.43% | 70.99% |
| Qwen2.5-3B | `unlock_lowrank_intervention` | 75.30% | 7.46% | 55.23% | 72.98% | 65.25% | 72.38% |
| Qwen2.5-3B | `unlock_lowrank_dynamic` | 73.85% | 5.25% | 53.26% | 70.51% | 66.23% | 66.57% |

## Findings

1. Static Unlock-style low-rank alignment is a strong per-state action baseline: it reaches about `75%--76%` Action Acc on both Target backbones.
2. The same methods have very low sequence exact match, only `4.70%--8.01%`. This shows that good per-state action prediction does not imply complete trajectory-level policy preservation.
3. Fixed action-direction intervention slightly improves final DB Hash on both targets, but does not improve sequence preservation:
   - SmolLM3 DB Hash: `69.89% -> 70.72%`
   - Qwen2.5 DB Hash: `70.99% -> 72.38%`
4. The naive dynamic transition prior hurts both targets:
   - SmolLM3 DB Hash: `69.89% -> 67.40%`
   - Qwen2.5 DB Hash: `70.99% -> 66.57%`

## Conclusion

Unlock-style latent/subspace transfer should be reported as a strong static action-boundary baseline, not as a full sequential Agent policy transfer method. Its limitation is visible in trajectory diagnostics: high Action Acc and reasonable DB Hash coexist with very low Seq Exact and only moderate transition preservation. This supports the paper claim that complex tool-use behavior is not a single capability direction, but a conditional composition of policy modes driven by environment feedback.

GPU helps the hidden/subspace prediction stage, but not much for tau replay, because replay is dominated by Python environment execution, oracle argument lookup, constrained decoding, and DB hash computation.
