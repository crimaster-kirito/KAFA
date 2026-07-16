# AG News Development Baseline: Local Qwen2.5-0.5B

Evidence label: `benchmark_dev` (not final holdout and not a paper result).

## Frozen setting

- Model: Qwen2.5-0.5B, weight SHA-256 `88c142...ed342`
- Seeds: 42, 43, 44
- Per seed: 256 balanced client-pool examples
- Evaluation: all 800 frozen development examples
- LoRA: rank 4, alpha 8, `q_proj` and `v_proj`
- Training: 40 optimizer steps, batch 8, learning rate 0.001
- Evaluation: constrained four-label likelihood, batch 32, BF16

## Result

| Metric | Mean | Sample SD |
|---|---:|---:|
| Accuracy | 85.79% | 0.51 pp |
| Macro-F1 | 85.66% | 0.50 pp |
| Correct-label NLL | 0.4566 | 0.0205 |
| ECE | 0.0327 | 0.0042 |
| Training wall time | 2.99 s | -- |

The unfine-tuned model evaluated at 48.63% accuracy with the frozen batch-32
evaluator. The result establishes a reproducible local-only development
baseline. It does not yet answer whether federation or heterogeneous transfer
helps.

## Numerical evaluation constraint

An earlier seed-42 run used evaluation batch 16 and produced 47.75% initial
accuracy instead of 48.63% under BF16. The run was replaced by a batch-32 rerun
before aggregation. All compared development methods must use the same frozen
evaluation batch size and dtype; otherwise sub-percentage comparisons are not
valid. Final holdout evaluation should additionally test FP32 label scoring or
per-example consistency.

