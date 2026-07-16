# Real within-client mixture mechanism study

Status: **development evidence only** (`benchmark_dev`, `paper_result: false`).
All values below use seed 42. They establish whether the phenomenon is worth
studying; they do not establish a final cross-seed paper claim.

## Controlled construction

Every condition uses the same 256 training slots, 40 optimizer steps, batch
size 8, rank-4 LoRA, and the same 800-example frozen AG News development set.
The underlying AG News selection has SHA-256
`5cf4955362932c653ccf219eb73fe28fefc69f5fe08b6c619e8b0343d3526d59`.

- Clean samples use the correct AG News label.
- Conflict samples use the deterministic non-identity permutation
  `label -> (label + 1) mod 4`.
- Off-task samples are IMDb reviews with a genuine sentiment instruction and
  `Negative`/`Positive` answer. They are not arbitrarily assigned news labels.
- The severe joint mixture contains 103 clean, 102 conflict, and 51 off-task
  examples (40% requested conflict and 20% requested off-task).

## Cross-family severe-mixture result

| Client model family | Clean accuracy | Joint-mixture accuracy | Change | Clean macro-F1 | Joint macro-F1 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 85.50 | 63.75 | -21.75 | 85.51 | 63.59 |
| SmolLM2-360M | 85.25 | 62.75 | -22.50 | 85.36 | 60.23 |
| TinyLlama-1.1B | 88.38 | 62.50 | -25.88 | 88.32 | 62.45 |

All changes are percentage points. The effect appears in all three model
families, but the table contains one development seed and therefore supplies
no confidence interval or significance test.

## Qwen contamination decomposition

| Condition | Accuracy | Macro-F1 | Label NLL | ECE |
|---|---:|---:|---:|---:|
| Clean | 85.50 | 85.51 | 0.4727 | 0.0284 |
| 20% off-task only | 84.25 | 84.23 | 0.5419 | 0.0556 |
| 40% conflict only | 61.87 | 57.99 | 1.0932 | 0.1674 |
| 40% conflict + 20% off-task | 63.75 | 63.59 | 1.0554 | 0.2276 |

The dominant harm in this setting is in-task label conflict, not ordinary
off-task multitask training. The joint condition is not monotonically worse in
accuracy than conflict-only at this seed, so no additive or interaction claim
is supported. Calibration nevertheless worsens strongly in the joint mixture.

## Research decision

The next method gate is sample-level conflict identification on real models.
A useful method must separate clean from permuted-label samples without using
the hidden contamination labels, preserve the clean regime within 2 points,
and improve downstream AG News utility under matched samples and optimizer
steps. Cross-model aggregation is premature until this local identification
mechanism is demonstrated.

## Seed-42 identification and routing development

Three label-free signals were measured at LoRA initialization: class-anchor
gradient cosine, per-example loss, and the canonical-label semantic
inconsistency margin. No single signal dominated across architectures. For
conflict-versus-clean discrimination, an equal-weight within-client percentile
rank ensemble reached AUROC 0.886 on Qwen, 0.785 on SmolLM2, and 0.747 on
TinyLlama. These remain single-seed diagnostic results.

A fixed 50% global filter failed the clean-safety gate: Qwen clean accuracy
fell from 85.50 to 72.37, largely because Science recall collapsed to 9.5%.
This baseline is rejected.

The revised development rule uses two safeguards selected on seed 42:

1. filter only if the 25th percentile of raw anchor cosine is below 0.7; and
2. when filtering, retain an equal quota for each claimed target label.

| Qwen seed-42 method | Clean accuracy | Severe-joint accuracy | Severe macro-F1 | Severe ECE |
|---|---:|---:|---:|---:|
| Naive local LoRA | 85.50 | 63.75 | 63.59 | 0.2276 |
| Fixed 50% rank filter | 72.37 | 73.00 | 70.56 | 0.0606 |
| Adaptive balanced rank | 85.50 | 79.75 | 78.94 | 0.0472 |
| Oracle clean-only (unattainable reference) | -- | 83.88 | 83.71 | 0.1064 |

The adaptive balanced rule recovers 16.0 percentage points over naive training
and leaves clean accuracy unchanged in this development run. It retains 98
clean and 30 conflict examples, versus 91 clean and 37 conflict examples for
the unbalanced filter. The 4.13-point gap to the oracle remains material.

This is a **candidate**, not a paper result. The gate threshold, quantile, keep
fraction, and class quota were inspected on development seed 42. They must be
frozen before running seeds 43/44 and later disjoint holdout seeds. Failure on
those runs requires reporting the negative result rather than retuning per
seed.

## Cross-family gate calibration result

The initial absolute gate (`client q25 < 0.7`) is rejected as architecture
dependent: it would incorrectly trigger on clean SmolLM2 (q25 0.571) and clean
TinyLlama (q25 0.616). We therefore evaluated a development-only relative
variant, whose gate is

`client class-conditional anchor-cosine q25 / clean-anchor-reference q25 < 0.8`.

The clean anchor reference is computed locally from the same frozen anchor set
for each model. This preserves the identical score, keep fraction, and class
balancing rule, while avoiding an assumption that raw LoRA-gradient cosine has
the same scale across architectures.

| Family, seed 42 | Clean gate ratio / action | Clean accuracy | Severe gate ratio / action | Naive severe | Relative adaptive severe | Change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 0.999 / keep all | 85.50 | 0.418 / route | 63.75 | 79.75 | +16.00 |
| SmolLM2-360M | 0.968 / keep all | 85.25 | 0.456 / route | 62.75 | 77.38 | +14.63 |
| TinyLlama-1.1B | 1.033 / keep all | 88.38 | 0.573 / route | 62.50 | 73.50 | +11.00 |

This establishes only a three-family, **one-seed development** mechanism
result. The 0.8 ratio threshold was chosen after inspecting these development
diagnostics, so the table cannot establish generalization or statistical
significance. A valid final study must freeze v2, run disjoint seeds and a
held-out test split, and compare equal-byte/equal-step heterogeneous
federated aggregation against compatible baselines.

## Risks and required follow-up

1. Run development seeds 43 and 44 before selecting thresholds.
2. Evaluate multiple conflict/off-task severities rather than only the severe
   corner.
3. Freeze thresholds, then use disjoint holdout seeds and a held-out test split.
4. Report confidence intervals and paired tests; do not treat the three model
   families as independent random seeds.
5. ECE is based on normalized canonical-answer sequence losses and should be
   described exactly; TinyLlama's high clean ECE shows accuracy alone is
   insufficient.
