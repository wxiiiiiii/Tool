# Baselines

This folder contains standalone baselines for evaluating Universal Agent Policy transfer.

## Hidden-state baselines

`hidden_policy_baselines.py` compares cheap baselines that reuse paired source/target hidden states:

- `fixed_slice_projection`: deterministic non-trained projection into the shared policy space.
- `random_linear_projection`: random frozen adapter into the shared policy space.
- `pairwise_linear_ridge`: closed-form pairwise latent alignment from target hidden states to source policy states.
- `unlock_pure_vector`: pure-vector Unlock-style baseline. It uses source-policy pseudo labels on paired calibration states to build one-vs-rest action/capability directions directly in the target hidden space, then predicts by cosine projection. It uses no target action labels, no target gradient steps, no trained target adapter, and no online source queries at evaluation time.
- `unlock_lowrank_subspace`: Unlock-style low-rank closed-form subspace alignment from target hidden states to source policy states, followed by the frozen source action head.
- `unlock_lowrank_intervention`: `unlock_lowrank_subspace` plus source-latent action-direction logit intervention.
- `unlock_lowrank_dynamic`: `unlock_lowrank_subspace` plus a source-policy transition prior applied during trajectory-ordered decoding.
- `unlock_style_latent_mse`: low-rank MLP trained only to match source policy latents, approximating an Unlock-style low-rank latent-alignment baseline.
- `target_local_linear_probe`: target-only supervised action classifier.
- `ours_policy_aware_adapter`: the proposed tiny adapter trained with target labels and a frozen source policy head.

The two most important metrics are:

- `accuracy_val`: ground-truth action accuracy on the held-out split.
- `policy_consistency_val`: agreement with the source policy predictions on paired examples.

## Native backbone baseline

`native_backbone_prompting.py` asks a target HF model to directly emit one action name from the vocabulary. It is expensive because it performs generation per decision row, so use `--max-samples` for smoke tests.

## Tool-calling policy baselines

See `tool_calling_baselines.md` for the newer tool-policy baseline set and the
paper mapping.

- `kl_policy_distillation.py`: paper-named wrapper for the existing KL/source-policy distillation baseline.
- `evotool_prompt_policy.py`: EvoTool-style zero-adaptation external prompt-policy transfer baseline.
- `evotool_modular_policy.py`: EvoTool official-theta modular Planner -> Selector baseline, using released `theta_star.json` artifacts.
- `asa_activation_steering.py`: ASA offline proxy over saved target hidden states; useful for cheap diagnostics, not the main paper-faithful ASA result.
- `asa_inference_steering.py`: ASA-style inference-time activation steering baseline with target calibration labels, a tool-boundary/domain controller, signed gate, and HF layer hook during generation.
- `tau_replay_predictions.py`: replay any precomputed baseline predictions with the same tau-bench grounding, constrained decoding, executor, and DB-hash metrics used by Ours.

These baselines intentionally expose their supervision conditions in their JSON
summaries, especially whether they use Target action labels or only unlabeled
Target states.
