# Baseline Accuracy Comparison

All results use the tau-bench hierarchical action vocabulary. Documents in
`paper/` are used only as method references.

## Full 4,615-State Comparison

| Target | Method | Target labels used | Target trainable params | Samples | Action Acc | Source Policy Agreement | Transition Agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | Ours, rank128 KL+adaptive transition+seq3 | no | tiny adapter | 4,615 | 76.21% | 91.16% | 82.67% |
| SmolLM3-3B | ASA target activation steering | calibration labels | 0 | 4,615 | 50.81% | 55.08% | 30.66% |
| Qwen2.5-3B-Instruct | Ours, rank256 KL+adaptive transition+seq3 | no | tiny adapter | 4,615 | 75.30% | 89.21% | 79.14% |
| Qwen2.5-3B-Instruct | ASA target activation steering | calibration labels | 0 | 4,615 | 46.72% | 49.56% | 24.83% |

ASA here is an offline target-side activation-controller proxy: action
prototypes, a tool/non-tool steering direction, domain offsets, and a signed
gate are built from target calibration hidden states. It does not update the
backbone, but it does use target calibration labels, so it is not a strict
no-target-label baseline.

## 60-State Prompt Baselines

| Target | Method | Target labels used | Target trainable params | Samples | Action Acc | Source Policy Agreement | Transition Agreement | DB Hash |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM3-3B | Ours full-result reference | no | tiny adapter | 4,615 | 76.21% | 91.16% | 82.67% | 72.10% |
| SmolLM3-3B | ASA inference-time steering + probe | calibration labels | 0 | 60 | 58.33% | 56.67% | 37.50% | 75.00% |
| SmolLM3-3B | EvoTool official theta modular policy | no | 0 | 60 | 20.00% | 23.33% | 3.57% | n/a |
| SmolLM3-3B | EvoTool template/scaffold prompt policy | no | 0 | 60 | 8.33% | 8.33% | 0.00% | n/a |
| SmolLM3-3B | Native prompting | no | 0 | 60 | 0.00% | n/a | n/a | n/a |
| Qwen2.5-3B-Instruct | Ours full-result reference | no | tiny adapter | 4,615 | 75.30% | 89.21% | 79.14% | 69.61% |
| Qwen2.5-3B-Instruct | EvoTool official theta modular policy | no | 0 | 20 | 35.00% | 35.00% | 11.11% | n/a |
| Qwen2.5-3B-Instruct | EvoTool template/scaffold prompt policy | no | 0 | 60 | 21.67% | 21.67% | 3.57% | n/a |
| Qwen2.5-3B-Instruct | Native prompting | no | 0 | 60 | 23.33% | n/a | n/a | n/a |

EvoTool official theta uses the released `evolved_policies/taubench/theta_star.json`
artifact from the official repository. The baseline runs the paper's modular
Planner -> Selector sequence, maps the selected concrete tau-bench tool into
this repository's universal action vocabulary, and does not perform target
evolution, target-label training, or target parameter updates. These are smoke
tests because modular prompt-policy generation is much slower than hidden-state
adapter evaluation: each decision state requires multiple LLM generations.

The older template/scaffold rows are kept only as implementation diagnostics and
should not be treated as the main EvoTool baseline.

ASA inference-time steering is the stronger ASA comparison than the offline
proxy: it builds a target-side tool boundary, domain residuals, signed gate, and
action probe from 240 labeled SmolLM3 calibration states, then applies a
transformer-layer hook at inference. It restores the ASA baseline from the raw
generation failure mode, but it still has much lower Source-policy and
transition fidelity than Ours while using target calibration labels.

## Takeaways

- Ours is currently much stronger than both the full-state ASA proxy and the
  60-state ASA inference-probe baseline on policy fidelity, even though ASA uses
  target calibration labels.
- Zero-adaptation prompt-policy transfer is weak in this strict action-vocabulary
  interface. The official EvoTool theta improves over the initial scaffold, but
  it is still not competitive with neural policy transfer through a tiny adapter.
- Qwen2.5 native/EvoTool prompting can emit action labels more often than
  SmolLM3, but it still remains far below Ours.
- The largest advantage of Ours is not only action accuracy; it is preserving
  source policy dynamics. Transition agreement is 82.67% / 79.14% for Ours vs.
  37.50% for formal ASA inference-probe on the 60-state SmolLM3 smoke test,
  30.66% / 24.83% for the full-state ASA proxy, and near-zero for EvoTool smoke
  tests.
