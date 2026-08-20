# Zero-Training Anchor-Relative Comparison

Setup: BFCL slot180 paired saved activations. Source is `qwen25_05b_bfcl_slot180`; target is `smollm2_360m_bfcl_slot180`. The anchor-relative method trains a canonical relative policy once on source coordinates, then onboards the target with anchor forward/cache only.

| Method | Target Labels | Target Gradient Steps | Source Online Queries At Target Onboarding | Target Full Acc | Target Val Acc | Source-Target Agreement | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Anchor-relative, MLP, policy-critical K=128 center | 0 | 0 | 0 | 78.89% | 66.67% | 81.11% | Best zero-training run |
| Anchor-relative, linear, semantic K=128 center | 0 | 0 | 0 | 76.11% | 59.26% | 77.78% | Linear source-side policy readout |
| Random linear projection | 0 | 0 | 0 | 33.33% | 37.04% | 33.89% | No target training, no policy anchors |
| Pairwise linear ridge | 0 labels, paired states | yes | offline paired source latents | 95.00% | 66.67% | 95.56% | Trains target mapping |
| Unlock-style latent MSE | 0 labels, paired states | yes | offline source latents | 95.00% | 66.67% | 96.67% | Trains target adapter |
| Target-local linear probe | yes | yes | 0 | 95.56% | 70.37% | 96.11% | Trains target classifier |
| Ours policy-aware adapter | yes | yes | 0 | 95.56% | 70.37% | 96.11% | Trains target adapter |

Takeaway: the anchor-relative interface is not yet as accurate as target-trained adapters on this small BFCL setup, but it changes the onboarding regime: no target labels, no target optimization, and no source teacher calls after the source-side policy artifact is built. The result is above random/no-anchor transfer and shows that policy anchor geometry carries cross-backbone signal.

