# Stable KAFA Ablation Results - GSM8K Seed 42

All three runs completed 30 rounds without `nan`, `RuntimeError`, or shape mismatch.

## Evaluation Curve

| Experiment | R4 | R9 | R14 | R19 | R24 | R29 | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kafa_stable_full` | 1.447548 | 1.440856 | 1.410989 | 1.325940 | 1.196528 | 1.274947 | 1.196528 |
| `kafa_wo_client_relevance` | 1.447593 | 1.440595 | 1.410919 | 1.325791 | 1.195915 | 1.266087 | 1.195915 |
| `kafa_wo_dirty_filter` | 1.446944 | 1.438476 | 1.402371 | 1.306002 | 1.181852 | 1.316312 | 1.181852 |

Primary metric is `client_summarized_weighted_avg.test_avg_loss` from `eval_results.log`.

## Final Best Summary

| Experiment | Best Weighted Avg | Best Individual | Last Round |
|---|---:|---:|---:|
| `kafa_stable_full` | 1.196528 | 1.193581 | 1.274947 |
| `kafa_wo_client_relevance` | 1.195915 | 1.192951 | 1.266087 |
| `kafa_wo_dirty_filter` | 1.181852 | 1.179223 | 1.316312 |

## Filter Behavior

| Experiment | Typical Kept | Main Filter Reason | Dirty Recall |
|---|---:|---|---:|
| `kafa_stable_full` | 83%-85% | mostly inconsistent supervision | 84%-93% |
| `kafa_wo_client_relevance` | 85%-87% | inconsistent supervision only | 84%-93% |
| `kafa_wo_dirty_filter` | 97%-100% | low relevance only | 0%-3% |

## Interpretation

- The stable communication backbone works: all runs remain stable through 30 rounds.
- Client-space relevance filtering has very small impact in this setting. Removing it slightly improves the best weighted loss: `1.196528 -> 1.195915`.
- Dirty supervision filtering improves measured dirty recall but hurts the current loss metric. Removing it gives the best weighted loss: `1.181852`.
- The last round is worse than the best round in all runs, so checkpoint selection matters. The best round is consistently around Round 24.

## Recommended Next Step

Use `kafa_wo_dirty_filter` as the next performance baseline if optimizing only test loss.

Run at least two more seeds for confirmation:

```bash
python federatedscope/main.py --cfg exp/kafa_matrix/gsm8k/kafa_wo_dirty_filter/seed_42/config.yaml seed 43
python federatedscope/main.py --cfg exp/kafa_matrix/gsm8k/kafa_wo_dirty_filter/seed_42/config.yaml seed 44
```

For cleaner seed-specific bookkeeping, copy the config to `seed_43` and `seed_44` directories and update `seed`, `outdir`, `expname`, and `save_to`.

