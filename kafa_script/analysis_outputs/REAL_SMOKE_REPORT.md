# KAFA Real-Model Smoke Report

Date: 2026-07-14

## Evidence status

All rows below are real computations using pretrained weights and real AG News
records selected by `benchmark_manifest_v1.json`. They validate the pipeline;
they are **not paper results** because the smoke split contains only 16 training
and 8 test examples and uses one seed.

Manifest SHA-256:
`073419bb7b444f358da469c330af26abdb005fbc8745b602f83b40b38e28e0e5`.

## Results

| Role/model | Weight SHA-256 | Params | Trainable | Loss before -> after | Test accuracy before -> after | Peak GPU | Adapter bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| Client Qwen2.5-0.5B | `88c142...ed342` | 494,303,104 | 270,336 | 2.5961 -> 0.6526 | 75.0% -> 75.0% | 1.81 GB | 1,115,203 |
| Client SmolLM2-360M | `7aaff6...e540f` | 362,230,720 | 409,600 | 3.7439 -> 0.3078 | 75.0% -> 75.0% | 1.23 GB | 1,683,647 |
| Client TinyLlama-1.1B | `6e6001...14933` | 1,100,611,584 | 563,200 | 6.7385 -> 1.4664 | 75.0% -> 25.0% | 2.80 GB | 2,283,867 |
| Server Qwen2.5-1.5B | `a961db...ec796` | 1,544,259,072 | 544,768 | 1.9629 -> 0.9728 | 50.0% -> 62.5% | 4.26 GB | 2,218,655 |

Exact JSON files contain full hashes and predictions.

## Verified properties

- Immutable model revisions and weight hashes are recorded in the manifest.
- Qwen, SmolLM, and Llama tokenizers all pass the same constrained-label
  likelihood evaluator.
- LoRA updates only the declared adapter parameters.
- Serialized adapter sizes, wall time, and peak allocated GPU memory are
  measured rather than estimated.
- Every run uses the same content-addressed train/test record IDs.

## Important warning

TinyLlama's training loss fell by 78%, while its eight-example test accuracy
fell from 75% to 25%. The sample is too small for a statistical conclusion, but
it demonstrates why training loss cannot be used as a paper metric and why
model-family-specific negative transfer must be evaluated. It also directly
invalidates the repository's earlier practice of comparing methods mainly via
loss curves.

## Next gate

The next experiment must use the frozen development split, multiple seeds, and
at least the following baselines:

1. local-only per model;
2. same-family LoRA FedAvg;
3. a product-space/SVD same-family aggregation baseline;
4. heterogeneous function-space distillation through the anchor set.

Only after these baselines run end to end should KAFA-specific routing or
prototype components be reintroduced.

