# Same-family federated LoRA development results

Status: **development evidence only** (`paper_result: false`). These runs are
used to validate the implementation and choose paper hypotheses. They are not
held-out paper results.

## Frozen setting

- Model: Qwen2.5-0.5B, weight SHA-256
  `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- Manifest SHA-256:
  `073419bb7b444f358da469c330af26abdb005fbc8745b602f83b40b38e28e0e5`.
- Three disjoint clients, 256 balanced AG News examples per client.
- Five rounds, eight local optimizer steps per client and round, rank-4 LoRA
  on `q_proj` and `v_proj`; 120 total client optimizer steps.
- Evaluation uses all 800 frozen development examples with BF16 and batch 32.
- Both methods serialize 1,113,859-byte adapters and account for 33,415,770
  bytes of total bidirectional communication under the script's convention.

## Results

Values are mean +/- sample standard deviation across seeds 42, 43, and 44.

| Aggregation | Accuracy | Macro-F1 | Label NLL | ECE |
|---|---:|---:|---:|---:|
| Factor FedAvg | 86.71 +/- 2.90 | 86.53 +/- 3.06 | 0.4360 +/- 0.0394 | 0.0315 +/- 0.0147 |
| Product average + rank-4 SVD | 86.79 +/- 2.57 | 86.66 +/- 2.71 | 0.4356 +/- 0.0424 | 0.0303 +/- 0.0130 |

Paired accuracy deltas for product-SVD minus factor FedAvg are +0.50, -0.88,
and +0.62 percentage points. The mean is +0.08 points with a sample standard
deviation of 0.83 points. The sign is not stable.

## Interpretation and decision

Product-space aggregation does **not** show a reliable development advantage.
It must not be presented as an accuracy-improving contribution. It remains a
useful diagnostic baseline for separating LoRA factorization ambiguity from
model-family and data-quality heterogeneity.

The seed-43 product-SVD wall-clock value (534.5 s) is invalid for runtime
comparison because it ran concurrently with another multithreaded SVD process,
causing severe CPU oversubscription. Isolated product-SVD runs took 18.8--19.7
s of recorded training/aggregation time, versus 9.8 s for the isolated
factor-FedAvg run. Timing comparisons require a separately controlled rerun
with fixed CPU thread counts. Accuracy and loss values remain valid because
each process used disjoint model state and deterministic data partitions.

The earlier local-only development runs are not a budget-matched comparator:
they used 256 examples and 40 optimizer steps, whereas these federated runs use
768 unique examples and 120 client optimizer steps. No local-versus-federated
superiority claim is supported by this table.
