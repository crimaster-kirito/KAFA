# KAFA Research Audit and CCF-A Roadmap

Status: initial audit (2026-07-14). Claims below distinguish repository evidence
from hypotheses. Synthetic preview values are not experimental evidence.

## 1. Intended research problem

KAFA targets federated parameter-efficient fine-tuning when the server owns a
large language model and clients train smaller, heterogeneous models. The
current design combines:

1. cross-model LoRA projection;
2. server-generated instruction and representation prototypes;
3. prototype alignment and soft-label distillation;
4. relevance filtering for unrelated client samples;
5. incremental prototype-bank updates.

The potentially publishable question is not simply whether these components
reduce training loss. It is whether knowledge can be transferred across model
families and non-IID clients without exposing the server model, while improving
task quality at a defensible privacy/communication/compute cost.

## 2. Evidence currently available

- The repository contains one complete RawFedAvg/Alpaca loss curve ending at
  test loss 1.0513.
- KAFA logs contain per-client training losses, but no comparable validation or
  test metric. They therefore do not establish a KAFA improvement.
- The six-task comparison tables and plots are explicitly synthetic previews.
  They must never be reported as measured results.
- The local Llama-2 directory contains configuration files but no model weights.
- The machine exposes one roughly 48 GiB RTX 4090. This supports meaningful
  small/medium-model experiments, but the proposed full 7B server plus separate
  client models requires quantization, CPU offload, sequential simulation, or a
  smaller controlled model suite.

## 3. Methodological risks

### 3.1 The present novelty is a bundle, not yet a scientific contribution

Projection, prototypes, distillation, filtering, and FedAvg are individually
standard ideas. A CCF-A submission needs a precise failure mode, a method that
directly addresses it, and evidence that simpler alternatives do not.

### 3.2 Cross-model LoRA mapping is under-specified

`LoRAMapper` mainly renames layer indices and does not generally reconcile
incompatible tensor shapes. The separate projection path uses trainable/random
linear maps, but the repository does not yet establish identifiability,
stability, or preservation of the induced LoRA update. The documentation also
describes `Q = P^{-1}` although a rectangular projection has no ordinary
inverse. A pseudoinverse, learned bi-directional map, or function-space mapping
must be defined and tested.

### 3.3 Evaluation metrics are insufficient

Language-model loss alone is not a credible main metric across GSM8K, SQuAD,
classification, and instruction following. Required task metrics include exact
match/accuracy for GSM8K, EM/F1 for SQuAD, accuracy/macro-F1 for classification,
and established generation or judge-based metrics with a documented protocol.

### 3.4 Privacy and threat-model claims are not yet justified

Hidden prototypes, soft labels, and incremental banks can leak information.
The paper must state exactly what server and clients observe and must avoid
calling the method privacy-preserving unless leakage is measured or protected.

### 3.5 Current experiment matrix is broader than the evidence budget

Six tasks times twelve methods times three seeds is expensive and includes
baselines that may not be valid under heterogeneous model architectures. A
smaller, correctly matched matrix should be validated before scaling.

## 4. Proposed first defensible research hypothesis

**Hypothesis:** In federated fine-tuning with architecturally heterogeneous
server/client language models, aligning updates in a shared function/semantic
space is more stable than directly mapping LoRA factors, especially under
label/task skew and irrelevant local data.

This reframes KAFA around a testable failure mode: LoRA factors are
non-identifiable (different factor pairs can represent the same update), so
factor-wise averaging or projection can be ill-posed. The next method iteration
should align model behavior or the composed low-rank update rather than assume
raw LoRA factors have a shared coordinate system.

## 5. Minimum credible experimental ladder

1. Unit tests on synthetic linear layers: shape correctness, round-trip error,
   invariance to equivalent LoRA factorizations, and aggregation behavior.
2. End-to-end tiny causal-LM experiment with two architectures and 3--10
   clients, comparing local-only, homogeneous FedAvg, naive heterogeneous LoRA
   mapping, Offsite-Tuning-style transfer, and the revised KAFA method.
3. Two representative real tasks first: one classification task and one
   reasoning/generation task, each with IID and several non-IID strengths.
4. Three seeds, confidence intervals, paired significance tests, convergence,
   wall-clock time, peak memory, and measured communication bytes.
5. Ablations only after the full method beats strong matched baselines.
6. Scale to additional tasks/models only when the core claim survives.

## 6. Paper gate

Drafting prose can start early, but result claims are gated on real logs. A
submission-ready paper requires:

- a literature-backed novelty statement;
- mathematically valid cross-model mapping;
- task-appropriate evaluation;
- strong and fair baselines;
- repeated real experiments with uncertainty;
- explicit threat model and limitations;
- artifact-level reproducibility.

## 7. Evidence ledger update (2026-07-15)

### Verified holdout results

- On AG News, the frozen two-round heterogeneous routed distillation protocol
  is **worse** than its strict public-anchor-only control by 0.56 percentage
  points on average across official-test seeds 142--144. It lowers the
  true-label answer-suffix loss and 10-bin ECE
  but fails the primary accuracy and macro-F1 gate and costs 2.01x wall time.
  The raw AG JSON key `label_nll` is a historical misnomer: the implementation
  records token-normalized target-answer loss rather than normalized class NLL.
- On GSM8K with a Qwen2.5-0.5B client, a public-anchor-trained Qwen2.5-1.5B
  teacher selector improves severe-mixture exact match from 3.49% to 6.82%
  (+3.34 pp) and clean exact match from 6.62% to 7.91% (+1.29 pp) on all 1,319
  official test questions at seeds 158--160.
- On the frozen non-Qwen holdout, the same selector yields positive severe
  differences for SmolLM2 (+1.59 pp) and TinyLlama (+2.02 pp), and both clean
  safety gates hold. Nevertheless, both multiplicity-corrected primary tests
  have Holm-adjusted p=0.0714 and their descriptive intervals cross zero. The
  predeclared post-observation general cross-family claim therefore **fails**.

### Verified mechanism diagnostics, not final claims

- An independent public-anchor teacher separates injected numerical answer
  conflicts with development AUROC 0.923--0.989; local client loss is markedly
  weaker in the tested Qwen setting.
- Frozen full-development studies show positive teacher-selection effects for
  Qwen, SmolLM2, and TinyLlama, but development evidence cannot override the
  failed cross-family holdout gate.

### Active hypotheses

- A frozen post-hoc development comparison finds higher severe mean EM for
  the teacher than numeric-random and client-small-loss selectors in both
  non-Qwen families. This supports, but does not confirm, that its signal adds
  information beyond the numeric interface and subset size; the cross-family
  official-test gate remains failed.
- A frozen 1,024-example secondary holdout shows nonnegative mean teacher gains
  at medium and severe contamination for Qwen, SmolLM2, and TinyLlama while
  satisfying the per-seed clean-safety rule. None of nine jointly corrected
  contrasts is significant; this is descriptive curve evidence only.
- The benchmark may still support an evaluation contribution if strong robust
  noisy-label baselines reproduce the observed reversal between server
  distillation and local sample filtering.

### Submission risks that remain open

- Teacher-loss selection is not novel: Superfiltering, Delora, and related
  small-loss/noisy-label methods substantially overlap it.
- The strongest positive GSM8K result is a centralized local-adaptation
  endpoint, not a complete federated aggregation result.
- Label conflicts are controlled synthetic interventions.  The frozen 18-run
  Hausa weak-supervision matrix now covers 50.37% naturally occurring label
  disagreement: oracle-rate Co-Teaching reduces mean development accuracy by
  11.61--19.08 pp across all three families.  The predeclared gate fails, so
  the clean test split remains unused and this is development-only evidence.
- A separately frozen 18-run Yoruba matrix tests a milder 33.28% natural weak-
  label error rate.  Co-Teaching effects are mixed (+0.53, +2.65, and -4.23 pp
  for Qwen, SmolLM2, and TinyLlama), all intervals cross zero, and no Holm-
  adjusted improvement test is significant.  Its gate also fails and its
  clean test split remains unused.
- A faithful Delora comparison remains out of scope because its published
  positive-set construction and relabeling use GPT-4o.  FedClean assumes a
  shared finite classifier and homogeneous parameter aggregation.  The frozen
  compatible two-peer Co-Teaching matrix is complete: even with the true
  contamination fraction, it has mixed-sign effects, no Holm-significant
  contaminated contrast, and costs 2.84--9.32x naive training wall time.
  On Hausa natural weak labels it is uniformly harmful and costs 1.69--1.80x.
  On Yoruba it is mixed and costs 1.64--1.89x.
- Privacy/leakage measurement and a complete federated positive method remain
  missing.  In contrast, submission formatting is no longer missing: the paper
  and separate reproducibility checklist compile with the unchanged official
  AAAI-27 anonymous style.

### Artifact and verification status

- `paper/main.pdf` compiles to six US-Letter pages under the official AAAI-27
  anonymous template with embedded Type 1 fonts and all 22 citations resolved.
  It has no compiler, BibTeX, layout, or reference diagnostics.  It includes a vector framework figure and a
  two-panel evidence plot regenerated from validated raw JSON.
- `kafa_script/analysis_outputs/HAUSA_NATURAL_NOISE_RESULTS.md` validates all
  18 paired records, raw-artifact hashes, multiplicity correction, costs, and
  the failed development gate.  Every evaluation hash matches `dev.tsv` and
  differs from the non-evaluated `test.tsv` hash.
- `kafa_script/analysis_outputs/YORUBA_NATURAL_NOISE_RESULTS.md` independently
  validates 18 lower-noise records and the failed Yoruba development gate;
  every evaluation hash matches Yoruba `dev.tsv` and excludes `test.tsv`.
- `kafa_script/analysis_outputs/AGNEWS_POST_OBSERVATION_SEED_REPLICATION.md`
  validates ten new records against the pre-run protocol digest.  All five
  new routed-minus-anchor accuracy differences are negative; their mean is
  -0.80 pp with descriptive seed-level 95% interval [-1.18, -0.42] pp.  The
  predeclared directional consistency heuristic passes, but it is explicitly
  not an alpha-controlled test and reuses the already accessed official test.
  The combined eight-seed -0.71 pp estimate remains descriptive and does not
  replace the original three-seed analysis.
- The AG `label_nll` implementation was audited against its evaluator and is
  reported in the manuscript as target-answer loss; ECE remains the calibrated
  canonical-probability metric.  Raw records and the frozen protocol are not
  retroactively rewritten.
- `paper/ReproducibilityChecklist.pdf` compiles separately, as required by the
  AAAI-27 submission instructions.  The official author-kit archive SHA-256 is
  recorded in `paper/README.md`.
- The research-specific unit tests pass (`11 passed`) when invoked as
  `.venv/bin/python -m pytest -q tests/test_capability_router.py
  tests/test_lora_projection_diagnostics.py
  tests/test_statistical_summarizers.py`.  After installing the repository-declared
  `protobuf` and `grpcio` dependencies, the repository-wide legacy suite also
  completes collection successfully (`75 tests collected`).  This is a
  collection audit rather than a claim that every expensive legacy integration
  test has been executed.
- `kafa_script/NATURAL_NOISE_PROTOCOL_CLARIFICATION.md` corrects an imprecise
  sentence in the frozen natural-noise protocols: clean training labels derive
  the corpus-level oracle forget rate and selection-purity audit, while
  per-example clean labels do not enter scoring, gradients, or model input.
  The original protocols are retained rather than silently rewritten.
- Protocol and record hashes establish workspace-internal consistency.  Until
  the artifact is placed in an immutable commit/tag or external timestamping
  service, they are not independent evidence of preregistration chronology.

## 8. Final manuscript and second-server-task update (2026-07-15)

- The earlier one-server-task evidence gap is resolved by a separately sourced
  TREC-Coarse experiment with six classes, three heterogeneous clients, a
  Qwen2.5-1.5B server, five designated official seeds, and a complete 5-by-2
  anchor-only/routed matrix.  The final protocol was frozen after documented
  instrumentation repairs and before official-test evaluation.
- TREC routed distillation changes accuracy by +0.40 pp with a descriptive
  95% seed interval [-1.74,+2.54] and two-sided paired p=0.6306.  It therefore
  does not establish an accuracy benefit.  Secondary ECE falls by 0.0373 in
  every seed; its nominal p=0.0008125 is unadjusted for metric multiplicity.
- The TREC artifact records official source-file hashes, task/manifest/runner/
  model hashes, every mixture row, simplex payload, serialized wire payload,
  and all final arguments.  Its 70-example duplicate removal and 504 selected
  records are audited for class balance and zero overlap.  The source's
  published license is recorded as unknown, and pretraining contamination
  cannot be ruled out.
- `paper/main.pdf` is now seven US-Letter pages with four tables, two vector
  figures, and 23 resolved references.  The complete submission build passes
  anonymity, font embedding, citation, split/hash, headline-number, protocol,
  and 70 focused research tests.  Visual inspection found and corrected the
  only material layout defect, an over-wide Table 4 caption.
- The remaining CCF-A risks are scientific rather than mechanical: the central
  contribution is diagnostic rather than algorithmic; neither server task
  shows an accuracy gain; privacy leakage is unmeasured; the strongest GSM8K
  positive result is local adaptation; and the artifact has not yet been put
  in an immutable public commit or independently timestamped archive.
- The frozen TREC argument ledger contains an inert `teacher_temperature`
  field that the runner records but does not apply.  This does not change the
  verified payloads or results; it is disclosed in
  `kafa_script/TREC_PROTOCOL_CLARIFICATION.md` rather than retroactively
  rewriting the frozen protocol.

## 9. Active CCF-A evidence strengthening (2026-07-15)

- The manuscript and original result artifacts remain intact while two new
  protocols are active.  No result from either protocol enters the paper until
  its complete matrix passes reconstruction-based verification.
- AG News four-arm decomposition uses anchor-only, clean-unfiltered,
  severe-unfiltered, and severe-routed server transfer.  Its final protocol
  SHA-256 is `9dd4d2696780eb67491dba440507639a9651dddbd991c8e94f892c09ee49b4ec`.
  The 4,000 balanced evaluation records are excluded from every client in the
  new runner, but they are not globally untouched: reconstruction found 303
  IDs in at least one prior server-client allocation across seeds 42--44 and
  142--149.  The authorized scope is therefore post-primary, run-disjoint,
  same-task resampling—not independent confirmation.
- Two AG protocol defects were handled fail-closed.  A pre-record anchor audit
  divided by zero; its log is retained.  A future nominal freeze timestamp led
  to four invalidated JSONs.  The later historical-exposure audit led to eleven
  more invalidated development JSONs and 23 retained JSON/log artifacts.  None
  of those values may be used for tuning, gating, inference, or paper claims.
- A stronger prospective role-reversal protocol makes IMDb sentiment the
  target and AG News the off-task source.  The protocol SHA-256 is
  `169f8348f9823276aa196e4b0d4b7c9fe0a5b7bebf5f76b923d05b8ace34a10e`.
  Before freeze, no analysis-output JSON/log referenced the IMDb test path or
  hash.  Manifest construction only hashed and counted its 25,000 raw lines;
  development uses a required nonexistent sentinel path.  IMDb train and task
  semantics were previously used as off-task inputs, so the strongest allowed
  label is “prospective untouched-test role-reversal extension,” not a
  dataset-naive or externally preregistered replication.
- The IMDb protocol declares one primary contrast only: severe-routed minus
  severe-unfiltered accuracy across ten common seeds, using a two-sided paired
  t-test and 95% seed interval.  Anchor/clean contrasts, calibration metrics,
  costs, and the net contrast are descriptive.  A narrow confirmation requires
  both increased audit clean fraction and a strictly negative primary interval
  with two-sided p<.05; failure must be reported without adding seeds.
