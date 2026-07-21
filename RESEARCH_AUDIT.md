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
  target and AG News the off-task source.  After a pre-record loader repair,
  the final protocol SHA-256 is
  `fbfc13064eca36163397017521474dde6aeda7c4304215b97491845c14ee9166`.
  Before freeze, no analysis-output JSON/log referenced the IMDb test path or
  hash.  Manifest construction only hashed and counted its 25,000 raw lines;
  development uses a required nonexistent sentinel path.  IMDb train and task
  semantics were previously used as off-task inputs, so the strongest allowed
  label is “prospective untouched-test role-reversal extension,” not a
  dataset-naive or externally preregistered replication.
- The first IMDb development attempt failed before partition resolution because
  the generic JSONL loader omitted manifest source-line metadata.  No JSON or
  model metric was written and the nonexistent test sentinel remained absent.
  The failure log is retained; the final runner deduplicates canonical training
  records while preserving their original one-indexed source lines.
- The IMDb protocol declares one primary contrast only: severe-routed minus
  severe-unfiltered accuracy across ten common seeds, using a two-sided paired
  t-test and 95% seed interval.  Anchor/clean contrasts, calibration metrics,
  costs, and the net contrast are descriptive.  A narrow confirmation requires
  both increased audit clean fraction and a strictly negative primary interval
  with two-sided p<.05; failure must be reported without adding seeds.

## 10. Resource-aware matrix execution and current test boundary (2026-07-16)

- Concurrent AG severe-unfiltered and IMDb severe-routed execution exhausted
  the device: the routed process requested another 4.64 GiB with only 3.19 GiB
  free.  The failed IMDb log is retained and no JSON was produced.  The same
  seed/method subsequently completed when run alone, so this is classified as
  an infrastructure failure rather than an algorithmic outcome.
- Both matrix schedulers now drain all ordinary jobs before gradient routing.
  Anchor and unfiltered arms retain up to two-way concurrency.  Two concurrent
  IMDb unfiltered jobs reached approximately 40.6 GiB used and 100% sampled GPU
  utilization, empirically ruling out a third concurrent model on this device.
  Scheduling changes do not alter any frozen model, data, or analysis setting.
- IMDb development seed 311 is complete.  Severe routed reaches 91.6% accuracy
  versus 89.9% for severe uniform and raises the aggregate selected clean
  fraction to 72.7% from the 103/256 (40.2%) severe baseline.  This is
  implementation-only development evidence: the positive +1.7 pp direction
  runs against the frozen negative official-test hypothesis and is neither a
  paper result nor grounds to change that hypothesis.
- The complete IMDb development matrix subsequently passed the reconstruction
  verifier (12/12 records; protocol SHA-256
  `fbfc13064eca36163397017521474dde6aeda7c4304215b97491845c14ee9166`).
  Routing triggered in 18/18 client-rounds and raised aggregate selected clean
  fraction to 72.5%.  Mean development accuracy is 90.03% for severe uniform
  and 91.33% for severe routed (+1.30 pp); by the frozen protocol this direction
  is implementation-only and cannot tune, stop, or reverse the official test.
- The first full verifier invocation exposed the same source-line metadata
  mismatch as the earlier pre-record runner audit: it used the generic JSONL
  loader while the manifest keys canonical-deduplicated original line numbers.
  No record was invalidated or changed.  The verifier now imports the frozen
  runner's audited line-preserving training loader, has a regression test, and
  passes all matrix reconstruction checks.  This is verifier instrumentation,
  not a protocol or result change.
- The expanded focused research regression suite passes all 134 tests,
  including scheduler exclusivity, duplicate-lock rejection, fail-closed IMDb
  paper-asset generation, and LaTeX smoke checks.  A repository-wide exploratory invocation
  reports 94 passes, 54 failures, and 4 collection errors, primarily because
  legacy integration tests require absent optional packages such as
  `torchvision` and `matplotlib`, while `test_tinyllama_training.py` exposes
  script functions as fixture-based pytest tests.  Therefore no repository-wide
  green-suite claim is made; the focused suite and legacy dependency boundary
  remain separately reported.
- A reviewer-style confounding audit identified that the original IMDb routed
  and severe-unfiltered arms change selected-set size as well as composition.
  Before any IMDb official model evaluation, a separate secondary addendum was
  frozen (SHA-256
  `47175564a023ea59e0137c847bdf747e3ec70b436ae52bc51a0f92f9b2670d64`).
  It holds K=128, claimed-label quotas, local steps, severe mixture, and all
  server settings fixed across deterministic target-eligible random-K,
  forced routed-K, and audit-only maximum-purity-K controls.  The last arm has
  103 clean plus 25 conflict rows and is explicitly not called oracle-clean.
  Two accuracy contrasts form one secondary two-sided Holm family; they cannot
  alter or confirm the original primary.  The addendum was designed after IMDb
  development was observed but while both official output directories remained
  empty, so its strongest allowed label is prospective secondary mechanistic
  evidence on the same still-unopened official test.
- The matched-control development matrix passed its reconstruction verifier
  (9/9) and the routed equivalence audit (3/3).  Mean accuracies are 89.70%
  random-K, 91.33% routed-K, and 90.97% maximum-purity-K.  Routed-K exceeds
  random-K in every development seed (+1.4, +1.8, +1.7 pp), whereas the arm
  with the highest attainable clean fraction (80.5%) does not exceed routed-K
  (0.0, -0.3, -0.8 pp).  These are development-only observations and cannot
  license the frozen secondary official hypotheses.
- Immediately before official launch, both IMDb official directories contained
  zero files and no official runner was active.  The two verified development
  reports, three verifier hashes, and this boundary were serialized in
  `kafa_script/imdb_official_start_audit_v1.json` at
  `2026-07-16T10:12:35Z`.  The original 10-by-4 official matrix then became the
  first authorized process to verify and decode the 25,000-row IMDb test.
- IMDb official seed 321 completed all four original arms without failure.
  Anchor, clean uniform, severe uniform, and severe routed accuracies are
  91.868%, 91.864%, 91.312%, and 91.608%, respectively.  The single-seed
  routed-minus-severe difference is +0.296 pp, opposite the frozen negative
  confirmation direction; it is not an interim test and cannot stop or modify
  the remaining nine seeds.  Routed audit clean fraction is 71.6%, while its
  canonical score ECE is 0.0079 versus 0.0780 for severe uniform.  The complete
  paired family remains required before any paper claim.
- IMDb official seed 322 also completed all four arms without failure.  Anchor,
  clean uniform, severe uniform, and severe routed accuracies are 91.920%,
  91.840%, 91.216%, and 91.608%, respectively.  Its routed-minus-severe
  difference is +0.392 pp and routed audit clean fraction is 71.2% (547/768).
  Score ECE is 0.0079 for routed versus 0.0752 for severe uniform.  These are
  run-integrity observations only: no two-seed aggregate, interim test, early
  stopping rule, or protocol change is authorized.  The complete ten-seed
  paired family remains required.

## 11. Manuscript inferential-role and final-asset contract audit (2026-07-16)

- A reviewer-style manuscript audit found that the legacy AG News comparison
  estimates routed-minus-anchor (D--A), whereas the TREC and prospective IMDb
  routing estimand is D--C.  The main text and framework therefore no longer
  call the legacy AG result cross-task ``consistency'' evidence.  It is labelled
  prior task-specific context; only the new AG four-arm matrix can supply a
  directly comparable D--C contrast, and even that remains post-primary
  resampling rather than independent confirmation.
- The framework's former TREC-to-IMDb-to-AG arrows incorrectly implied an
  execution chronology.  They were removed and replaced by three independent
  inferential-role boxes.  The evidence ledger is now a single-column table
  fixed before Results and explicitly referenced in the text.  The class-count
  symbol is now L, matched subset size remains K=128, and the GSM8K local
  retention count is K_local.
- The cross-task asset generator now parses and binds the frozen TREC, IMDb
  original, IMDb matched-size, and AG protocol semantics before loading any
  records.  Verifier receipts must match local protocol hashes, phase, record
  count, seeds, and methods.  Exact sign-flip sensitivities use integer traces
  of the recorded confusion matrices rather than rounded floating accuracy.
- The IMDb primary claim gate is explicit and fail-closed: routed clean
  fraction must exceed 103/256, the paired D--C interval must be strictly
  negative, and the two-sided paired p-value must be below .05.  Original and
  matched routed records additionally require a separate conditional
  implementation-integrity receipt; that receipt is not a fourth primary-gate
  component and is not statistical equivalence.  A failed primary gate still
  produces the complete result but cannot produce confirmatory wording.  AG direction
  relative to TREC is stored as a descriptive field, never as its evidence
  class.
- Generated tables and figures distinguish primary, descriptive,
  post-primary, and secondary contrasts.  A current-input manifest binds the
  generator, protocols, verifiers, all 150 expected records, and every
  published asset hash; the submission build rejects stale assets when inputs
  are incomplete or changed.  Synthetic contract tests pass 22/22, the related
  TREC/IMDb/AG regression set passes 61/61, and the complete focused submission
  build passes 134/134 tests.
- The rebuilt anonymous working submission contains six US-Letter main pages,
  three supplement pages, one current main table, one current vector framework
  figure, 29 resolved references, embedded fonts, and no anonymity leak.  The
  supplement log is free of warning/overfull/underfull diagnostics.  Final
  IMDb/AG numerical assets, the outcome-dependent abstract language, and the
  Conclusion remain intentionally absent until all frozen matrices verify.
  This snapshot is superseded by the 2026-07-20 sections below after the IMDb
  matrices, generated assets, Conclusion, and expanded supplement were rebuilt.

## 12. IMDb complete-matrix verification and post-run verifier repair (2026-07-20)

- The original IMDb official matrix completed all 40 frozen records and passed
  its reconstruction verifier with protocol SHA-256
  `fbfc13064eca36163397017521474dde6aeda7c4304215b97491845c14ee9166`
  and runner SHA-256
  `95853ad21d850e9fe68464b01b3a179b0d487cb6a8c66e647bc6b08041792459`.
  The matched-K secondary matrix completed all 30 frozen records and passed its
  verifier with protocol SHA-256
  `47175564a023ea59e0137c847bdf747e3ec70b436ae52bc51a0f92f9b2670d64`
  and runner SHA-256
  `6d132fad0b7b97511c123b9ea7a3834b2b80700d21e8aba5c303361c4ba4f068`.
- Only after both matrix verifiers passed, the routed relationship verifier
  frozen at official start (SHA-256
  `2c8678f7bcca3c00fb304ab4e9774bdea04f8b9bb5e9eeb996798baf8d1f18dd`)
  exposed an over-strong post-run instrumentation assumption: it required
  whole-record equality even when the original conditional gate retained 256
  examples while the frozen matched arm deliberately forced K=128.  This
  assumption was not part of either scientific protocol and could not hold on
  an authorized gate rejection.
- The verifier was repaired after all official records existed, so this is
  explicitly a **post-record instrumentation repair**, not a pre-evaluation
  change.  The repaired verifier (SHA-256
  `5518b1773c63a5b147e16bb1993a4058d00c05ef246b765ca7f7684d0e422e3c`)
  checks exact functional identity only before a local gate fork, proves that
  the original gate decision is the sole authorized first divergence, and
  continues validating each arm's frozen branch invariants after a fork.  It
  partitions all 60 client-rounds into 55 exact pre-fork triggered rounds, 4
  first authorized gate divergences, and 1 post-local-fork round that is not
  equivalence-eligible.  Six of ten seed trajectories are fully identical;
  four have the protocol-authorized forced-gate divergence.  The receipt is
  conditional implementation equivalence, not full-trajectory or statistical
  equivalence.
- No raw JSON, seed, arm, metric, statistic, protocol, or runner was modified.
  The IMDb primary gate remains exactly the three frozen scientific clauses:
  clean fraction above 103/256, a strictly negative D--C 95% interval, and a
  two-sided paired p-value below .05.  The conditional relationship receipt is
  stored separately as an integrity record and cannot change that gate.
- After all three integrity checks passed, the official primary estimate was
  D--C = +0.291 pp (95% CI [+0.072, +0.510]); paired two-sided t-test
  p=0.014940 and exact sign-flip sensitivity p=0.017578.  Routing raised the
  audited clean fraction from 40.23% in severe uniform to 66.53%, but the
  interval is positive rather than strictly negative.  Therefore the frozen
  narrow routing-harm confirmation gate **fails** even though the observed
  effect is significantly positive; the complete result must be reported as a
  prospective directional reversal, not confirmation.
- In the frozen secondary Holm family, routed-K minus target-eligible random-K
  is +0.323 pp (95% CI [+0.170, +0.477]), raw p=0.001022 and Holm-adjusted
  p=0.002044.  Maximum-purity-K minus routed-K is +0.047 pp (95% CI
  [-0.100, +0.195]), raw and Holm-adjusted p=0.487230.  These matched-K tests
  are secondary mechanism probes and cannot upgrade or replace the failed
  primary confirmation gate.
- The verified summaries are
  `kafa_script/analysis_outputs/IMDB_SERVER_ROLE_REVERSAL_OFFICIAL.md`
  (SHA-256
  `74bfda83d4d8e350a93a26972cbe1d4ca1c9bd173f399b3337ed5686e45dd5ea`)
  and `kafa_script/analysis_outputs/IMDB_MATCHED_CONTROL_OFFICIAL.md`
  (SHA-256
  `b165a41e868ffd9175a328aceb32dfd49e709457ea52a5e533e7e5e48d04dbf3`).
  Six official paper assets were then generated atomically under
  `paper/generated/`; their machine-readable ledger SHA-256 is
  `82f2ce28ec5e95c4e545e6ed3e2fb41abb63982568c5b100a8f983b48157680c`.
  The focused IMDb/cross-task contract suite passes 39/39 tests.

## 13. IMDb manuscript integration and submission-window audit (2026-07-20)

- The verified IMDb result block, table, and vector figure are now included in
  `paper/main.tex`.  The abstract, Results, Discussion, Limitations, and an
  independent Conclusion were rewritten around the failed negative-direction
  gate rather than the originally hypothesized outcome.  The manuscript states
  that the TREC routing direction did not transport to the protocol-frozen
  IMDb setting and treats the sign reversal as setting-specific heterogeneity,
  not universal routing benefit.
- The rebuilt anonymous manuscript is seven US-Letter pages, with References
  beginning on page 6.  It contains two tables, two vector figures, and 29
  resolved references.  The structural verifier passes font embedding,
  anonymity, citation, split/hash, and page-boundary checks; the reported-number
  verifier passes against the generated IMDb ledger and earlier raw sources.
- The official AAAI-27 timetable was rechecked on 2026-07-20.  The abstract is
  due 2026-07-21 23:59 UTC-12, the full paper 2026-07-28 23:59 UTC-12, and
  supplementary material/code 2026-07-31 23:59 UTC-12.  A registration-ready
  title, 183-word abstract, keywords, evidence boundary, and human-action
  checklist are stored in `paper/OPENREVIEW_SUBMISSION.md`.  Author identity,
  order, affiliations, conflicts, and account actions remain human-only fields.
- AG News secondary four-arm execution was independently resumed from its four
  already verified records after the old IMDb relationship verifier had stopped
  the coordinator.  This AG study remains post-primary task-specific resampling
  and cannot change the prospective IMDb primary result or its failed gate.

## 14. Formatting repair, selector-component preparation, and AG progress (2026-07-20)

- The current anonymous submission build verifies as seven US-Letter main pages
  and four supplement pages, with References beginning on main page 6 and 31
  resolved references.  The supplement now includes a closest-work boundary
  table and an explicit post-record IMDb verifier-repair disclosure.  The
  structural verifier and reported-number verifier pass after these edits.
- Scientific wording was narrowed to avoid overclaiming: the manuscript reports
  setting-specific heterogeneity rather than task-level dependence, internally
  frozen evaluation rather than external preregistration, and no statistically
  detectable additional gain for the audit-only maximum-purity control rather
  than a proof that higher purity cannot help.
- To address the strongest remaining novelty risk, a separate post-hoc
  selector-component ablation was specified in
  `kafa_script/imdb_selector_component_ablation_plan_v1.md`.  It is explicitly
  secondary and reused-test.  New CPU-side implementation files
  `kafa_script/imdb_selector_component_ablation.py` and
  `kafa_script/verify_imdb_selector_component_ablation.py` define the
  gradient-only, loss-only, and margin-only matched-K selector semantics and a
  fail-closed structural verifier without modifying the frozen IMDb primary or
  matched-K runners.  The selector/verifier unit tests pass 9/9; the broader
  IMDb selector/matched/generator smoke set passes 23/23.
- AG News secondary execution completed and verified all 40 records over seeds
  281--290 and four arms.  The verified post-primary resampling result is:
  B--A \(+0.01\) pp (95% CI \([-0.28,+0.29]\), Holm \(p=.9541\)),
  C--B \(-2.04\) pp (\([-2.44,-1.65]\), Holm \(p<10^{-4}\)), and
  D--C \(+1.46\) pp (\([+1.06,+1.86]\), Holm \(p<10^{-4}\)); the routed
  net relative to anchor-only remains descriptive at \(-0.58\) pp
  (\([-0.80,-0.35]\)).  The generated summary is
  `kafa_script/analysis_outputs/AGNEWS_SERVER_ABLATION_SECONDARY.md`.
  This remains post-primary task-specific evidence and must not be used as
  independent confirmation.
- The cross-task server-audit assets were generated from the now-complete
  150-record input set: TREC 40, IMDb original 40, IMDb matched-K 30, and AG
  News 40.  `kafa_script/generate_server_audit_assets.py --verify-only`
  certifies the current `paper/generated/server_audit_*` assets.  A generator
  instrumentation assumption was repaired after AG completion: TREC/IMDb
  confusion-matrix traces are now checked with float32-level tolerance, while
  AG News records without confusion matrices reconstruct integer correct counts
  from accuracy times evaluation size with a fail-closed integer tolerance.
  Raw records, protocols, and reported statistics were not modified.
- After AG integration, the main text was compressed and the IMDb role-reversal
  heading was narrowed to "protocol-frozen" to avoid implying external
  preregistration.  The OpenReview registration packet now includes the AG
  secondary result as bounded post-primary context rather than listing it as
  running.  A fresh `paper/build_submission.sh` run passes, and the final
  structural verifiers reported a seven-page anonymous main PDF, four-page
  supplement, two-page reproducibility checklist, two main tables, two vector
  main figures, 31 resolved references, embedded fonts, anonymous PDF metadata,
  and 137 research-related tests passed before the selector-component
  integration.  Additional selector/server-audit tests passed 33/33.
- A submission-facing risk checklist was added at
  `paper/SUBMISSION_RISK_CHECKLIST.md`.  It records verified mechanical state,
  human-only OpenReview fields, supported claims, must-not-claim boundaries,
  and the five highest remaining reviewer risks: diagnostic novelty, narrow
  scope, post-primary chronology for TREC/AG, absent privacy measurement, and
  artifact packaging risk.  The evidence ledger caption was also narrowed to
  describe internal frozen chronology rather than "prospective" chronology.
  Another then-current `paper/build_submission.sh` run passed with 137 tests,
  the supplemental selector/server-audit tests passed 33/33, and
  `git diff --check`, `paper/verify_submission.py`, and
  `paper/verify_reported_numbers.py` pass.
- An anonymous artifact packager was added at
  `paper/package_anonymous_artifact.py`.  Its default dry-run and build modes
  collect only paper files, generated assets, protocols/manifests, focused
  runners/verifiers/tests, and verified JSON/Markdown analysis outputs while
  excluding logs, failure directories, invalidated records, runtime logs,
  model weights, caches, git state, and local-report files with absolute paths.
  The then-current build produced `paper/artifacts/KAFA_anonymous_artifact.zip`
  with 732 manifest files, SHA-256
  `602c5b51e1323c430024c2b4a7f4d6efe5dcc797a1d39732aa6e1c4805cd8ec7`.
  Archive-name and text scans found zero logs/failure/invalidated/weight/git
  entries and zero local-path, repository-owner, or obvious credential-string
  findings.

## 15. Selector-component ablation execution started (2026-07-20)

- The post-hoc IMDb selector-component ablation matrix was launched in
  `kafa_script/analysis_outputs/imdb_selector_component_ablation/official_evaluation`
  with the frozen 10-seed by 3-arm plan: `gradient_only_k`, `loss_only_k`,
  and `margin_only_k`.  This is reused-test secondary diagnostic evidence and
  is not available for manuscript conclusions until the full 30-record matrix
  verifies.
- The first seed's three arms completed and each individual JSON record passed
  `verify_record` structural checks.  For seed 321 only, accuracies are:
  gradient-only 91.248%, loss-only 91.636%, and margin-only 91.532%.
  Audit-only selected clean/conflict counts across two rounds and three
  clients are 450/318, 528/240, and 539/229 respectively.  These single-seed
  values are sanity checks, not statistical conclusions.
- Peak allocation for the completed component runs is 27.09 GiB, matching the
  prior IMDb matched-K peak.  Because a 49 GiB RTX 4090 cannot safely host two
  such runs when both enter peak allocation, the matrix is intentionally
  running sequentially despite intermittent low-utilization phases.
- A dedicated summary generator,
  `kafa_script/summarize_imdb_selector_component_ablation.py`, was added.  It
  will verify the 30 component records, reuse the existing verified
  `severe_routed_k` matched-K records as the full-router reference, and report
  the three post-hoc full-minus-component paired contrasts with Holm
  adjustment.  Synthetic summarizer tests passed, and
  `paper/package_anonymous_artifact.py` included the summarizer in the planned
  artifact bundle.  The artifact from Section 14 was therefore treated as a
  prior build pending the completed component matrix.

## 16. Selector-component ablation completed and reported as post-hoc diagnostic (2026-07-21)

- The IMDb selector-component ablation completed the full 10-seed by 3-arm
  matrix.  `kafa_script/verify_imdb_selector_component_ablation.py` verifies
  all 30 records with classification
  `post_hoc_secondary_reused_test`.  Mean accuracies are 91.30% for
  `gradient_only_k`, 91.50% for `loss_only_k`, and 91.36% for
  `margin_only_k`; selected clean/conflict totals are 4559/3121, 5171/2509,
  and 5496/2184 respectively.
- `kafa_script/summarize_imdb_selector_component_ablation.py` generated
  `kafa_script/analysis_outputs/IMDB_SELECTOR_COMPONENT_ABLATION_OFFICIAL.md`.
  It reuses the existing verified `severe_routed_k` matched-K records as the
  full-router reference.  The three post-hoc paired contrasts are:
  full-minus-gradient \(+0.16\) pp (95% CI \([+0.00,+0.32]\), raw
  \(p=0.044963\), Holm \(p=0.134889\)); full-minus-loss \(-0.04\) pp
  (\([-0.17,+0.09]\), Holm \(p=0.504867\)); and full-minus-margin
  \(+0.10\) pp (\([-0.00,+0.21]\), raw \(p=0.050929\), Holm \(p=0.134889\)).
  No full-minus-component contrast survives Holm correction, so this result
  bounds attribution rather than identifying a single necessary component.
- The supplement now includes a short post-hoc selector-component diagnostic
  section.  `paper/verify_reported_numbers.py` was extended to recompute these
  supplement numbers from raw JSON and now reports that AG News,
  primary/post-primary TREC, GSM8K, Co-Teaching, severity, Hausa, Yoruba, and
  IMDb selector-component numbers all match the manuscript sources.  The
  selector-component verifier, summarizer, and tests are now part of
  `paper/build_submission.sh`.  A fresh full submission build verifies the
  anonymous seven-page main PDF, four-page supplement, two-page checklist,
  reported numbers, TREC/IMDb/AG matrices, generated assets, and 147
  research-related tests.
- The anonymous artifact was rebuilt after the selector-component update.
  `paper/package_anonymous_artifact.py --dry-run` reports 765 manifest files,
  and `--build` writes `paper/artifacts/KAFA_anonymous_artifact.zip` with
  SHA-256 `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.
  A zip-level text scan found no old commit fragments, local absolute path,
  original Git URL, repository-owner/user string, or obvious credential string.

## 17. Reviewer-risk wording pass and internal attack notes (2026-07-21)

- A reviewer-risk wording pass removed ambiguous causal and confirmation
  language from the manuscript without changing reported results.  The router
  score definition now refers to candidate-answer loss rather than causal
  answer loss, discovery/descriptive evidence is not described as upgrading to
  confirmation, and the supplement figure caption refers to pre-specified
  gates rather than confirmatory gates.
- A fresh `paper/build_submission.sh` after these manuscript edits verifies
  the anonymous seven-page main PDF, four-page supplement, two-page checklist,
  reported numbers, complete TREC/IMDb/AG matrices, generated assets, and 147
  research-related tests.  The anonymous artifact was rebuilt from this state
  with 765 manifest files and SHA-256
  `977fabf89abaa6e5960f87873828aa61f82a2c44e430f0623885b0f14104b9a8`; a
  zip-level text scan again found no old commit fragments, local absolute path,
  original Git URL, repository-owner/user string, or obvious credential string.
- `paper/REVIEWER_ATTACK_NOTES.md` was added as an internal, non-artifact
  preparation file.  It records likely objections around optimizer novelty,
  mixed signs, non-independent TREC/AG evidence, purity as an incomplete
  mechanism, synthetic contamination, absent privacy guarantees, CCF-A
  positioning, and narrow scope.

## 18. OpenReview registration packet tightened (2026-07-21)

- The AAAI-27 official pages were rechecked on 2026-07-21.  They list
  deadlines as 2026-07-21 23:59 UTC-12 for abstracts, 2026-07-28 23:59
  UTC-12 for full papers, and 2026-07-31 23:59 UTC-12 for supplementary
  material and code.  The submission instructions state that submissions are
  accepted only through OpenReview and not by email or other mechanisms.
- The OpenReview registration abstract in `paper/OPENREVIEW_SUBMISSION.md` was
  compressed from 196 to 183 words by the local token counter while preserving
  the TREC, IMDb, matched-K, and AG evidence boundaries.
- Because `paper/OPENREVIEW_SUBMISSION.md` is included in the anonymous
  artifact, the artifact was rebuilt.  The current
  `paper/artifacts/KAFA_anonymous_artifact.zip` contains 765 manifest files,
  766 zip entries, and has SHA-256
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.  The
  zip-level text scan again found no old commit fragments, local absolute path,
  original Git URL, repository-owner/user string, or obvious credential string.

## 19. Day-of-submission guardrails added (2026-07-21)

- `paper/OPENREVIEW_SUBMISSION.md` and `paper/SUBMISSION_RISK_CHECKLIST.md`
  now include submission-day guardrails from the AAAI-27 instructions: do not
  submit a placeholder title or abstract; avoid substantial abstract changes
  between the abstract and full-paper deadlines; finalize authors because AAAI
  states that authors may not be added after submission; submit through
  OpenReview rather than email; and upload the reproducibility checklist in the
  designated separate checklist field rather than merging it into the main PDF.
- Because both submission-facing files are part of the anonymous artifact, the
  artifact was rebuilt.  The current `paper/artifacts/KAFA_anonymous_artifact.zip`
  contains 765 manifest files, 766 zip entries, and has SHA-256
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.  The
  zip-level text scan found no old commit fragments, local absolute path,
  original Git URL, repository-owner/user string, obvious credential string, or
  internal reviewer notes.

## 20. Human copy-paste submission fields prepared (2026-07-21)

- `paper/OPENREVIEW_COPYPASTE_FIELDS.md` was added as a non-artifact helper for
  the human submitter.  It contains the title, 183-word abstract, keywords,
  upload-file list, current artifact SHA-256, manuscript AI-assistance
  disclosure text, and blank human-only fields for authors, affiliations,
  conflicts, funding, data/code availability, simultaneous-submission answer,
  and policy answers.
- The helper was checked against `paper/OPENREVIEW_SUBMISSION.md`; both
  abstracts count to 183 words with the local counter.  The current anonymous
  artifact SHA-256 remains
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`, and
  `OPENREVIEW_COPYPASTE_FIELDS.md` is not present inside the artifact.

## 21. Submission file manifest prepared (2026-07-21)

- `paper/SUBMISSION_FILE_MANIFEST.md` was added as a non-artifact helper for
  upload verification.  It records the current pages, byte sizes, and SHA-256
  hashes for `paper/main.pdf`, `paper/supplement.pdf`,
  `paper/ReproducibilityChecklist.pdf`, and
  `paper/artifacts/KAFA_anonymous_artifact.zip`, plus the intended OpenReview
  upload placement for each file.
- The manifest hashes were checked against the current files.  The anonymous
  artifact SHA-256 remains
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.
  `SUBMISSION_FILE_MANIFEST.md`, `OPENREVIEW_COPYPASTE_FIELDS.md`, and
  `REVIEWER_ATTACK_NOTES.md` are not present inside the artifact.

## 22. Goal-level completion audit prepared (2026-07-21)

- `paper/COMPLETION_AUDIT.md` was added as a non-artifact project-management
  audit.  It maps the original goal to concrete evidence for the research
  framing, implementation, experiment matrices, English paper, supplement,
  checklist, figures/tables, numerical consistency, anonymous PDF state,
  focused build, anonymous artifact, OpenReview copy-paste fields, upload
  manifest, and risk boundary documents.
- The audit explicitly marks human author metadata, conflicts/funding,
  OpenReview policy answers, and actual OpenReview submission as not complete.
  Therefore the technical package is mechanically ready for human submission
  review, but the overall goal should remain active until the human-only fields
  and actual submission are completed.
- `COMPLETION_AUDIT.md` is not present inside the anonymous artifact, and the
  artifact SHA-256 remains
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.

## 23. Human-field worksheet prepared (2026-07-21)

- `paper/OPENREVIEW_HUMAN_FIELDS_WORKSHEET.md` was added as a non-artifact
  helper for collecting the remaining human-only OpenReview fields.  It
  includes tables/prompts for author list and order, affiliations, contact
  author, conflicts, funding and acknowledgments, data/code availability,
  simultaneous-submission/prior-publication answers, AI-assistance policy
  answers, and final upload confirmation.
- The worksheet records the current anonymous artifact SHA-256
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16` for
  human upload checking.  It is not present inside the anonymous artifact, and
  adding it did not change the artifact SHA.

## 24. Final one-command submission preflight added (2026-07-21)

- `paper/final_submission_preflight.py` was added as a non-artifact helper.  It
  verifies that the OpenReview registration and copy-paste abstracts are both
  183 words and identical, checks the page counts, byte sizes, and SHA-256
  hashes for the main PDF, supplement, checklist, and anonymous artifact,
  verifies that non-artifact helper files are absent from the artifact zip, and
  repeats the artifact text scan for old commit fragments, local paths, original
  Git URL, repository-owner/user strings, and obvious credential strings.
- The command `.venv/bin/python paper/final_submission_preflight.py` passes and
  reports `final_submission_preflight_ok`.  `paper/verify_submission.py` and
  `paper/verify_reported_numbers.py` also pass after this addition.  The
  artifact SHA-256 remains
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16` because
  the new preflight helper is not included in the anonymous artifact.

## 25. One-page human handoff index added (2026-07-21)

- `paper/SUBMISSION_HANDOFF_INDEX.md` was added as a non-artifact one-page
  entry point for the human submitter.  It orders the handoff workflow as:
  run `paper/final_submission_preflight.py`, fill
  `paper/OPENREVIEW_HUMAN_FIELDS_WORKSHEET.md`, copy fields from
  `paper/OPENREVIEW_COPYPASTE_FIELDS.md`, verify upload hashes with
  `paper/SUBMISSION_FILE_MANIFEST.md`, read risk boundaries, and avoid uploading
  local helper files.
- `paper/final_submission_preflight.py`, `paper/verify_submission.py`, and
  `paper/verify_reported_numbers.py` pass after this addition.  The handoff
  index is not present inside the anonymous artifact, and the artifact SHA-256
  remains
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.

## 26. Post-submission receipt template added (2026-07-21)

- `paper/OPENREVIEW_SUBMISSION_RECEIPT.md` was added as a non-artifact helper
  for recording the OpenReview paper ID, URL, submission/modified times,
  submitter account, human-field confirmations, uploaded file hashes,
  confirmation email/screenshot locations, and post-submit caution items.
- `paper/SUBMISSION_HANDOFF_INDEX.md` now points to this receipt template in
  its after-submission section, and `paper/final_submission_preflight.py` now
  checks that the receipt template is excluded from the anonymous artifact.
  The preflight, submission verifier, and reported-number verifier pass after
  this addition.  The receipt template is not present inside the anonymous
  artifact, and the artifact SHA-256 remains
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.

## 27. Dirty-worktree handoff prepared (2026-07-21)

- `paper/WORKTREE_HANDOFF.md` was added as a non-artifact maintenance guide for
  the intentionally dirty worktree.  It separates OpenReview upload files,
  local-only helper/audit files, submission/package infrastructure, paper source
  and generated assets, AG/IMDb experiment outputs, selector-component code and
  tests, known verification commands, current artifact identity, and remaining
  human-only work.
- `paper/final_submission_preflight.py` now checks that `WORKTREE_HANDOFF.md`
  is excluded from the anonymous artifact.  The final preflight, submission
  verifier, and reported-number verifier pass after this addition.  The
  worktree handoff is not present inside the anonymous artifact, and the
  artifact SHA-256 remains
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`.
