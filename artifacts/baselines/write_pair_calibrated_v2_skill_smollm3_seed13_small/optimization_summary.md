# SmolLM3 Write-Pair Branch Optimization Summary

This run tests an observation-conditioned write-pair correction on top of the static Low-Rank policy alignment.

## Change

`Calibrated Write-Pair Prior` was strengthened from a partial `finish > recover/verify` margin to a full branch margin:

- write success: `finish` must beat every non-finish branch;
- write failure: `recover/verify` must beat irreversible `finish/execute`;
- the rule uses only observable prefix information: last tool family and counterfactual observation status.

`Predicted Pair Operator + Calibrated Write-Pair Prior` combines the learned pair-contrastive residual with this observation-conditioned branch prior.

## Results

| Method | Branch Acc All | Branch Acc Write | Write Success Branch | Write Failure Branch | Write S/F Pair | Write Correct Flip | Write Harmful Flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static Low-Rank | 68.75% | 50.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| Calibrated Write-Pair Prior v1 | 71.88% | 57.81% | 0.00% | 100.00% | 0.00% | 7.81% | 0.00% |
| Predicted Pair + Calibrated Prior v1 | 72.50% | 60.94% | 12.50% | 100.00% | 12.50% | 10.94% | 0.00% |
| Calibrated Write-Pair Prior v2 | 72.81% | 62.50% | 18.75% | 100.00% | 18.75% | 12.50% | 0.00% |
| Predicted Pair + Calibrated Prior v2 | 76.25% | 79.69% | 87.50% | 100.00% | 87.50% | 29.69% | 0.00% |
| Oracle Correct Write Branch | 76.88% | 82.81% | 100.00% | 100.00% | 100.00% | 32.81% | 0.00% |

## Takeaway

The learned pair operator alone still cannot reliably discover `write success -> finish`, but when the observation-conditioned branch constraint is made explicit, it recovers most of the oracle headroom without introducing harmful write flips. This supports the current diagnosis: the bottleneck is not the low-rank static policy alignment or the hierarchical decoder, but the missing observation-conditioned branch semantics for irreversible write actions.
