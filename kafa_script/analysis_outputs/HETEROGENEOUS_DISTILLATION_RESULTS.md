# Heterogeneous canonical-probability distillation: development evidence

Unless explicitly marked as the frozen official-test comparison below, results
in this report are development-only (`paper_result: false`), seed 42, one
communication round, Qwen2.5-1.5B server, and three disjoint 256-example
clients (Qwen2.5-0.5B, SmolLM2-360M, TinyLlama-1.1B). Clients exchange only
four canonical AG News probabilities on 128 public anchors; no LoRA tensor is
transmitted or merged. The server uses rank-4 LoRA, 40 server steps, learning
rate `1e-4`, and public-anchor hard supervision weight 1.

## Compatibility smoke

An extremely small run (four client and four server steps; 32 anchors and 80
development examples) verified end-to-end cross-tokenizer probability exchange.
At server learning rate `1e-3`, both pure and anchor-supervised distillation
collapsed the server to approximately 25--26% accuracy. Lowering only the
server learning rate to `1e-4` retained/improved it from 75.00% to 76.25%.
This is a stability observation, not a performance result.

## One-round disjoint-client development comparison

| Server objective / client condition | Final accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| Initial server | 75.13 | 71.64 | Same initialization for all rows |
| Public-anchor-only | **81.25** | **80.48** | No client probability term |
| Uniform teacher, clean clients | 80.38 | 79.85 | -0.88 pp versus anchor-only |
| Uniform teacher, 40% conflict + 20% off-task clients | 75.25 | 73.46 | -6.00 pp versus anchor-only |
| Anchor-accuracy-weighted teacher, same mixed clients | 75.50 | 74.01 | -5.75 pp versus anchor-only |
| Routed mixed clients, uniform teacher, distillation weight 1.0 | 79.13 | 78.39 | -2.13 pp versus anchor-only |
| Routed mixed clients, uniform teacher, distillation weight 0.25 | 80.63 | 79.94 | -0.63 pp versus anchor-only |

The anchor-accuracy weighting used public-anchor accuracies 0.383, 0.414, and
0.727 for Qwen, SmolLM2, and TinyLlama respectively, yielding softmax weights
0.030, 0.041, and 0.929 (temperature 0.1). It modestly beats uniform mixed
distillation by 0.25 points but does not overcome the anchor-only reference.

The routed runs use the same three disjoint severe-mixture clients and the
same 10,875-byte three-client probability payload as the unfiltered teacher
runs. All three clients trigger the relative gate and retain 128 examples:
Qwen retains 97 clean / 31 conflict; SmolLM2 and TinyLlama each retain 89
clean / 39 conflict. No off-task sample receives a target-label quota after a
gate triggers. Routing raises public-anchor accuracy of the local teachers to
0.789, 0.766, and 0.836. Reducing the server distillation coefficient from 1
to 0.25 improves final server accuracy by 1.50 points, but it still does not
beat anchor-only training in this development seed.

## Two-round routed development comparison

The multi-round runner retains client LoRA states and server optimizer state
between rounds. Under the same seed-42 severe-mixture partition, two rounds of
40 client and 40 server steps produce the following development endpoints:

| Objective | Final accuracy | Macro-F1 | Payload bytes | Wall time |
|---|---:|---:|---:|---:|
| Public-anchor-only, two rounds | **86.88** | **86.66** | 7,250* | 85.6 s |
| Routed uniform distillation, weight 0.25 | 86.38 | 86.17 | 21,750 | 411.6 s |

\*The anchor-only objective has zero effective client contribution; the
recorded 7,250 bytes are an implementation artifact from a minimal dummy
client pass and must not be treated as a fair equal-byte comparison.

The routed endpoint is 0.50 points below anchor-only, despite its stronger
server calibration (NLL 0.676 versus 0.687 and ECE 0.027 versus 0.030). The
method also has approximately 4.8x wall time because it computes exact
per-example LoRA gradients. Per-round local teachers improve, but the average
teacher still does not provide sufficient additional target information to
outperform public-anchor supervision. This is a negative development result,
not an algorithmic win.

## Decision

Neither uniform nor simple capability-weighted probability distillation may be
presented as a useful heterogeneous federated method. The correct current
interpretation is a negative baseline result: public-anchor supervision is
stronger than the unfiltered client teachers under this protocol, and noisy
client knowledge can harm the server materially. Routing demonstrably narrows
this gap, but the current candidate still fails the required net-benefit gate.

The runner is nevertheless a verified baseline artifact. A candidate method
must beat the public-anchor-only objective under matched server updates, client
steps, anchors, bytes, and disjoint data; outperforming only uniform
distillation is insufficient. Required next gates are (1) route or calibrate
client contributions using only permitted local/public signals, (2) repeat at
least three frozen seeds, (3) compare equal-byte settings, and (4) use an
official held-out test split before paper claims.

## Routed heterogeneous plumbing smoke

The relative balanced router was directly integrated into the heterogeneous
runner. Under external GPU contention, a deliberately reduced smoke used a
Qwen2.5-0.5B server, 32 anchors, 80 development examples, four client/server
steps, and evaluation batch size 1. All three disjoint mixed clients triggered
the model-relative gate (ratios 0.326, 0.442, 0.455) and each retained 128
class-balanced target-task samples. The server changed from 47.50% to 58.75%.

This confirms the complete local-route -> canonical-probability-upload ->
heterogeneous-server-distillation path. It is **not comparable** to the table
above because it changes server size, anchor count, step budget, evaluation
size, and batch size. A full 1.5B routed run remains pending GPU availability.

## Multi-round state-machine smoke

`run_heterogeneous_distillation_rounds_dev.py` was validated with two rounds,
one Qwen2.5-0.5B client, one Qwen2.5-0.5B server, one local/server step per
round, four anchors, and four development examples. The second round restored
the first round's client LoRA state, and total client-to-server payload was
exactly two 1,641-byte uploads (3,282 bytes). This verifies state retention and
per-round accounting only; it is far too small for a performance comparison.

## Frozen official-test comparison (primary result)

The development-selected two-round protocol was frozen before this evaluation
and then executed on the full 7,600-example official AG News test set for
seeds 142--144. To make local-compute and payload accounting comparable, the
public-anchor-only control was re-run with the same two rounds, 40 client
steps, 40 server steps, and three heterogeneous clients. The control's server
distillation coefficient remains zero, so its client messages are deliberately
ignored by the server objective.

| Metric | Public-anchor-only | Routed distillation | Routed minus anchor-only |
|---|---:|---:|---:|
| Accuracy (mean ± SD) | 84.34 ± 0.62 | 83.78 ± 0.54 | -0.56 pp [-0.76, -0.36] 95% t CI |
| Macro-F1 (mean ± SD) | 84.28 ± 0.75 | 83.67 ± 0.63 | -0.61 pp [-0.91, -0.31] 95% t CI |
| Label NLL | 0.834 ± 0.049 | 0.713 ± 0.025 | -0.121 |
| ECE | 0.054 ± 0.017 | 0.016 ± 0.005 | -0.039 |

Every paired accuracy difference is negative (-0.58, -0.47, and -0.63 pp).
The predeclared one-sided paired test for routed accuracy greater than
anchor-only accuracy gives t(2) = -12.095, p = 0.9966. The routed run costs
2.01x the mean wall time (561.3 s versus 279.8 s); recorded payload is matched
at 21,750 bytes. Better NLL/ECE does not offset the failure of the predeclared
accuracy and macro-F1 objective. This is a completed and reproducible
**negative primary result**, not evidence for a new federation algorithm. Full
checksums, protocol verification, and per-seed records are in
`FINAL_HETEROGENEOUS_RESULTS.md`.
