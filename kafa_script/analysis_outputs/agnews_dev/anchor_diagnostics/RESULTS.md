# Anchor and consistency diagnostics

Status: `benchmark_dev`, `paper_result: false`; seed 42 only.

| Model | Anchor AUROC | Loss AUROC | Semantic-margin AUROC | Rank-ensemble AUROC |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 0.857 | 0.761 | 0.749 | 0.886 |
| SmolLM2-360M | 0.658 | 0.781 | 0.754 | 0.785 |
| TinyLlama-1.1B | 0.644 | 0.675 | 0.678 | 0.747 |

The table measures conflict versus clean and excludes off-task examples so
that trivial task-schema differences do not inflate the result. Higher scores
mean more likely harmful. The rank ensemble averages within-client percentile
ranks of negative anchor cosine, loss, and semantic inconsistency.

The experiment rejects the hypothesis that LoRA anchor-gradient geometry is a
universally strongest detector: it wins on Qwen but loses to simpler signals
on SmolLM2 and TinyLlama. The ensemble is more stable but is not yet validated
across seeds, severities, anchor sizes, or final held-out data.
