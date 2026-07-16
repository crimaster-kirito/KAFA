# KAFA Joint-Heterogeneity Benchmark Specification

Version: 0.3 (AG News, TREC-Coarse, and GSM8K evaluations complete), 2026-07-15

## Purpose

This benchmark tests whether conclusions drawn from standard federated LLM
fine-tuning survive when model-family heterogeneity and within-client task/data
mixtures occur together. It is an evaluation contribution first; KAFA-Router
is one baseline and is not presumed to win.

## Research questions

1. Do methods ranked highly under ordinary label-skew retain their ranking when
   individual clients contain mixtures of relevant, off-task, and conflicting
   examples?
2. How much of an apparent gain comes from compatible model coordinates rather
   than transferable task knowledge?
3. Can a method remain useful when corruption severity is unknown, without
   sacrificing clean-regime performance or minority-client utility?
4. Are accuracy gains still present after matching communication bytes,
   trainable parameters, optimizer steps, and wall-clock budget?

## Frozen smoke-test models

All initial models are public decoder-only models small enough for repeated
single-GPU execution:

| Role | Model | Family | Intended use |
|---|---|---|---|
| Server | `Qwen/Qwen2.5-1.5B` | Qwen2 | target global model |
| Client A | `Qwen/Qwen2.5-0.5B` | Qwen2 | same-family size heterogeneity |
| Client B | `HuggingFaceTB/SmolLM2-360M` | SmolLM2 | cross-family/tokenizer |
| Client C | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama | cross-family/tokenizer |

The smoke test may begin with Client A to validate plumbing, but the benchmark
claim requires all three client families. Model licenses and exact immutable
revisions must be recorded when downloaded.

## Tasks

### Stage A: AG News controlled classification

- Target capability: four-way topic classification.
- Clean examples: correctly labeled AG News samples.
- Off-task examples: IMDb sentiment examples converted to the same instruction
  envelope but not to AG News labels.
- Conflicting examples: AG News samples with a deterministic non-identity label
  permutation.
- Mixture axes: conflict in `{0, 0.2, 0.4, 0.6}` and off-task in
  `{0, 0.2}`, excluding sums greater than or equal to one.
- Primary metric: test accuracy parsed from constrained label outputs.
- Secondary metrics: macro-F1, NLL over canonical label choices, ECE,
  worst-client accuracy, clean-capability retention, communication bytes,
  wall time, peak allocated GPU memory, and trainable parameter count.

### Stage B: GSM8K controlled reasoning

- Target capability: grade-school mathematical reasoning.
- Off-task examples: unrelated instruction-following samples.
- Conflicting examples: correct questions paired with deterministically
  corrupted final answers; rationale corruption is evaluated separately.
- Primary metric: exact match of normalized final numeric answer.
- Secondary metrics: answer-choice NLL where available, invalid-output rate,
  retained general-capability metrics, communication, time, and memory.

Stage B starts only if Stage A produces a reproducible and informative method
ranking. This is a compute gate, not a narrowing of the final benchmark claim.

Stage A has now produced a reproducible negative ranking: under the frozen
severe-mixture AG News holdout, routed heterogeneous distillation loses to
public-anchor-only adaptation on all three seeds. Stage B therefore tests
whether that diagnostic survives numerical reasoning; its data partition and
development protocol are frozen in `gsm8k_benchmark_manifest_v1.json` and
`gsm8k_stage_b_development_protocol_v1.json`. Neither artifact authorizes use
of GSM8K official-test labels before a separate final protocol is frozen.

The official-test protocols are now complete. A separate secondary curve uses
`gsm8k_secondary_holdout_manifest_v1.json`: 1,024 examples sampled from the
GSM8K training split after excluding every anchor, development example, and
client-pool row. Its protocol fixes clean, 20% conflict + 10% off-task, and
40% conflict + 20% off-task levels at seeds 72--74. This secondary holdout may
support robustness-curve analysis but must not be described as the official
GSM8K test set.

### Stage A2: TREC-Coarse independent server-task replication

- Target capability: six-way question answer-type classification.
- Allocation: 72 public anchors, 72 development questions, and three disjoint
  balanced 120-question client pools after exact raw-line duplicate removal.
- Client mixture: 40% cyclic label conflict and 20% IMDb off-task traffic,
  apportioned per class; the full 500-question official test is untouched by
  development evaluation.
- Primary metric: constrained-output accuracy, analyzed two-sided over five
  designated official seeds.  Macro-F1, canonical class NLL, and ECE are
  secondary; their nominal p-values are not multiplicity-adjusted.
- Outcome: routed-minus-anchor accuracy is +0.40 pp with interval
  [-1.74,+2.54] and p=0.6306.  ECE falls by 0.0373 in every seed, supporting a
  calibration diagnostic but no accuracy-improvement claim.
- Provenance boundary: the final protocol was frozen after documented
  instrumentation repairs and before official evaluation.  The public source
  license is recorded as unknown and pretraining contamination is possible.

## Required baselines

1. Local-only and centralized upper/reference bounds.
2. Homogeneous FedAvg and factor-wise FedAvg (explicitly marked invalid under
   arbitrary LoRA gauge, retained as a common-practice baseline).
3. Product-space/SVD LoRA aggregation (FLoRA/iFLoRA-style).
4. A model-heterogeneous federated-distillation baseline.
5. FedBiOT, because this repository derives from it.
6. FedMKT/FedProxy-compatible baselines where released implementations and
   assumptions permit a fair comparison.
7. Simple hard anchor filtering.
8. Fixed and adaptive KAFA routing as diagnostic baselines.

If an external method cannot operate across different tokenizers or model
families, report that incompatibility; do not silently substitute a weaker
setting.

## Experimental controls

- Development seeds and final seeds are disjoint.
- Main tables use at least three seeds; controlled mechanism studies use 20 or
  more when cheap.
- Hyperparameters are selected only on development splits/seeds.
- Test labels never create anchors, thresholds, prompts, or early stopping.
- All compared methods use the same client data, number of optimizer steps,
  and evaluation prompts.
- Report both equal-round and equal-byte comparisons.
- Confidence intervals accompany means. Paired tests operate on matched seeds;
  multiple primary comparisons receive Holm correction.
- Raw per-run logs are immutable inputs to generated tables.

## Claim gates

An algorithmic claim requires a method to:

1. beat its strongest compatible baseline on the primary metric with corrected
   `p < 0.05` in at least two nontrivial settings;
2. avoid more than 2 pp degradation in the clean setting;
3. improve or preserve worst-client utility;
4. remain favorable under equal-byte accounting; and
5. reproduce on at least two model-family configurations and both task stages.

If these gates fail but the benchmark reliably reverses method rankings or
reveals unsupported assumptions, the paper must be framed as an empirical
evaluation/benchmark paper and must release the protocol and artifacts.

## Evidence labels

Every output must carry one of:

- `mechanism_synthetic`: controlled synthetic-data computation;
- `smoke_real`: real model/data run used for plumbing, not a paper claim;
- `benchmark_dev`: real development experiment;
- `benchmark_holdout`: frozen final experiment;
- `synthetic_preview`: fabricated trend visualization, prohibited from claims.
