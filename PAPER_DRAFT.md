# Selection Purity Is Not a Sufficient Proxy for Transfer Utility: Auditing Joint Model and Data Heterogeneity in Language-Model Adaptation

**Status: complete seven-page anonymous draft; CCF-A evidence strengthening is
active.** Frozen AG News, TREC-Coarse, and GSM8K evaluations, a
three-level severity study, a compatible Co-Teaching baseline, and Hausa and
Yoruba natural weak-label development studies are complete.  The seven-page
AAAI-27 manuscript, two vector figures, four tables, 23 references, separate
reproducibility checklist, and artifact validators are built.  The evidence
supports an evaluation/diagnostic paper, not a positive new federated method;
privacy evaluation and an algorithmically novel endpoint remain absent.  A
post-primary AG News four-arm analysis on a newly frozen 4,000-record,
client-disjoint holdout is running under a fail-closed protocol; no result from
that matrix is part of the draft until the complete matrix validates.

## Abstract

Federated adaptation of language models is often evaluated either with
homogeneous client architectures or with clean client data. In practice, both
assumptions may fail simultaneously: clients can use different model families
and tokenizers while their local corpora mix task-relevant, off-task, and
label-conflicting examples. We introduce a controlled benchmark protocol for
this joint heterogeneity and use it to audit common aggregation assumptions.
On real AG News and IMDb mixtures, severe in-task label conflict causes a
21.75--25.88 point local accuracy drop across three decoder-only model
families in a development study. Direct factor averaging is not invariant to
LoRA factorization, while tokenizer-agnostic canonical-probability distillation
can be worse than public-anchor-only training under contaminated clients.

We study a capability-conditioned local routing mechanism based on
class-conditional anchor-gradient agreement, loss, and label-consistency
signals. A model-relative gate avoids the architecture-dependent scale of raw
gradient cosine. In development experiments, routing preserves the clean
condition and improves severe-mixture local adaptation for Qwen, SmolLM2, and
TinyLlama. On a frozen three-seed, two-round comparison on the full AG News
test set, routed heterogeneous distillation is 0.56 percentage points less
accurate than public-anchor-only server adaptation despite better calibration.
On a separately frozen six-class TREC-Coarse official evaluation, routed
distillation changes accuracy by +0.40 points with a 95% seed interval of
[-1.74,+2.54], while secondary ECE falls by 0.0373 in every seed.  This
replicates a calibration pattern but not an accuracy improvement.
On a second frozen GSM8K study, a public-anchor teacher filter improves Qwen
severe-mixture exact match by 3.34 points without a clean penalty. The same
endpoint yields positive severe differences for SmolLM2 (+1.59 points) and
TinyLlama (+2.02 points), but neither family passes the predeclared,
post-observation Holm-corrected holdout gate (adjusted (p=0.0714)). On naturally weak-labelled
Hausa development data, oracle-rate Co-Teaching lowers accuracy by
11.61--19.08 points across all three families and fails its predeclared gate;
the clean test split remains unused. These results motivate a
falsifiable evaluation protocol rather than a premature universal aggregation
claim.

## 1. Introduction

Most federated LLM adaptation methods assume that a client update can be
meaningfully combined with other client updates. That assumption has two
separate failure modes. First, adapters from different architectures,
dimensions, and tokenizers do not share a natural parameter coordinate system.
Second, a local update can be harmful even when its architecture is compatible,
because a client may contain mislabeled target examples or unrelated tasks.
Evaluating only one dimension can make an aggregation mechanism look robust
while concealing the other.

This work asks a narrower and more useful question: *what survives when both
forms of heterogeneity occur together?* We begin by auditing the KAFA codebase
and its original claims. The audit finds that the legacy projection path lacks
an invariant basis for arbitrary LoRA factors and that existing plots are not
evidence for real heterogeneous LLM performance. We therefore build a
reproducible protocol around real checkpoints, recorded model hashes, serialized
data manifests, constrained label scoring, and evidence labels.

Our contribution is primarily an evaluation and diagnosis framework. A local
router is investigated as a candidate mitigation, but it is retained only when
it satisfies clean-regime safety and strong compatible baselines. This stance
is essential because public anchors alone can outperform unfiltered
heterogeneous teacher distillation in our current development results.

## 2. Related Work

**Heterogeneous federated language models.** Parameter-compatible methods such
as HetLoRA and FLoRA address rank heterogeneity or factor aggregation, whereas
FedMKT and model-heterogeneous distillation exchange behavior across different
models. These studies establish that cross-model transfer and function-space
distillation are not new by themselves. Our focus is the interaction between
model-family mismatch and mixed-quality data *within* each client, evaluated
against a public-anchor-only endpoint that can expose negative transfer.

**Noisy-label learning and data selection.** Small-loss selection, FedClean,
and Delora detect or correct noisy labels; Superfiltering shows that one
language model can select instruction data for another. Accordingly, our
public-anchor teacher score is a deliberately simple diagnostic baseline, not
a novel filtering algorithm. The open empirical question is whether the same
training-only trust signal remains safe across incompatible client model
families under simultaneous label conflict and off-task traffic.

**Evaluation position.** Existing FL work commonly varies statistical
heterogeneity while holding architectures fixed, or varies architectures on
nominally task-consistent data. We treat their Cartesian product as the object
of study and require clean-regime safety, public-anchor-only comparisons,
official holdouts, compute/communication accounting, and explicit negative
results. The intended contribution is an evaluation finding and benchmark
protocol unless a future method clears strong robust-FL baselines.

## 3. Problem Setting

There are clients \(c \in \{1,\ldots,C\}\), each with a decoder-only model
\(f_c\), tokenizer \(\tau_c\), and local dataset \(D_c\). The server model
\(f_s\) can differ in architecture and tokenizer. Each \(D_c\) contains a
mixture of clean target-task examples, label-conflicting target-task examples,
and off-task examples. The server and clients share a public target-task anchor
set \(A\), whose text and labels are available under the benchmark protocol.

For architecture-agnostic exchange, client \(c\) sends a probability vector
over a canonical task-specific \(K\)-label output space for each anchor text, not hidden
states, vocabulary logits, or LoRA tensors. The server can distill the average
or weighted probability target using its own tokenizer. This exchange is
compatible by construction but is not automatically beneficial.

## 4. Capability-Conditioned Local Routing

For a target-task sample \(z\) with claimed class \(y\), let \(g(z)\) denote
the gradient of its causal answer loss with respect to local LoRA parameters.
Let \(a_y\) be the class-conditional mean anchor gradient. We use the cosine
agreement

\[
s_g(z) = \frac{g(z)^\top a_y}{\lVert g(z)\rVert\lVert a_y\rVert}.
\]

We combine three harmful-direction ranks within a client: negative anchor
agreement, answer loss, and canonical-label inconsistency (the claimed-label
loss minus the best canonical-label loss). Off-task samples have no claimed
target label and are excluded when a target-label quota is activated. The
combined score is used only to order local examples. Clean-versus-conflict
annotations are hidden and audit-only. Off-task membership is known from the
explicit task interface; those examples have no claimed target class and are
excluded whenever the gate triggers.

Raw cosine scales vary sharply across model families. We therefore gate routing
using a relative statistic,

\[
r_c = Q_{0.25}\big(\{s_g(z):z\in D_c\}\big) /
      Q_{0.25}\big(\{s_g(a):a\in A\}\big).
\]

If \(r_c\) is below a fixed development threshold, the client retains an equal
quota of low-risk samples per claimed target class; otherwise it trains on all
examples. This class balancing is necessary: a global difficulty filter can
remove an entire class and destroy clean-task utility.

### 4.1 Public-anchor teacher endpoint for open-answer tasks

The class-conditional gradient construction requires a small canonical label
space and does not transfer directly to numerical reasoning. For an open-answer
task, we instead train an independent teacher (f_T) only on public anchors
(A). For each supplied client pair (z=(x,\tilde y)), the teacher assigns

\[
s_T(z)=-\frac{1}{|\tilde y|}\log p_{f_T}(\tilde y\mid x).
\]

After rejecting answers outside the task's canonical numerical interface, the
client retains the (K) examples with smallest (s_T) and performs the same
number of optimizer steps as the unfiltered baseline. The teacher never sees
contamination-kind annotations, client validation labels, or test examples.
This is a function-space trust endpoint: its scalar score is meaningful across
client tokenizers and architectures, while its parameters and gradients are
not transferred. It also has a clear limitation: scoring every candidate with
an additional model is substantially more expensive than naive adaptation.

## 5. Experimental Protocol

### Data and models

Stage A uses AG News four-way classification as the target capability. IMDb
reviews form real off-task examples; deterministic non-identity AG News label
permutations form conflicts. The frozen manifest separates 256 anchors,
800 development examples, client pools, and official test data. Client models
are Qwen2.5-0.5B, SmolLM2-360M, and TinyLlama-1.1B; the heterogeneous server
is Qwen2.5-1.5B. Exact revisions and safetensor hashes are recorded in the
artifact.

Stage B uses the official GSM8K split with a frozen training-only partition of
128 public anchors, 512 development questions, and three disjoint 256-example
client pools. Deterministically rotated numerical answers create in-task
conflicts and frozen SQuAD question--answer examples create off-task traffic.
The teacher is Qwen2.5-1.5B; clients are Qwen2.5-0.5B, SmolLM2-360M, and
TinyLlama-1.1B. Evaluation uses greedy generation followed by canonical
decimal normalization and exact match.

### Metrics and controls

We report constrained-output accuracy, macro-F1, token-normalized target-answer
loss for the AG evaluator, 10-bin ECE from canonical probabilities,
per-class recall, communication bytes, wall time, and peak GPU allocation.
All main comparisons must match local optimizer steps, anchors, public data,
and server updates. Final experiments require disjoint development and
holdout seeds, official test evaluation, paired tests, confidence intervals,
and multiple-comparison correction.

## 6. Evidence and Negative Results

This section is a factual log, not a final results section.

1. Severe mixtures (40% conflict, 20% off-task) reduce local AG News accuracy
   by 21.75--25.88 points across Qwen, SmolLM2, and TinyLlama at seed 42.
   Conflict dominates ordinary off-task interference.
2. On Qwen development seeds 42--44, a fixed adaptive balanced router improves
   severe local accuracy over naive mixed training by +16.00, +30.00, and
   +3.37 points. The clean path is not activated and exactly matches the clean
   baseline. This is still only three development seeds.
3. An absolute raw-cosine gate is architecture dependent and incorrectly
   triggers on clean SmolLM2 and TinyLlama. The relative anchor-normalized gate
   avoids this error in the seed-42 three-family development check.
4. One-round disjoint-client heterogeneous probability distillation does not
   beat public-anchor-only server training: 81.25% anchor-only, 80.38% with
   clean uniform teachers, 75.25% with severe mixed uniform teachers, and
   75.50% with simple anchor-accuracy weighting. Relative local routing raises
   severe mixed distillation to 80.63% at a low server distillation weight,
   but remains below anchor-only. In a two-round development run, routed
   distillation reaches 86.38% versus 86.88% for anchor-only while taking
   approximately 4.8x the wall time. These are negative baseline findings and
   rule out a claim that simple distillation solves the problem.
5. **Frozen holdout result.** The development-selected two-round severe-mixture
   protocol was frozen before evaluating seeds 142--144 on all 7,600 official
   AG News test examples. A strict public-anchor-only control executed the
   same two rounds, 40 client steps/round, 40 server steps/round, and recorded
   payload; it assigns zero weight to the teacher term. Routed distillation
   has 83.78 ± 0.54% accuracy versus 84.34 ± 0.62% for anchor-only, a paired
   difference of -0.56 pp (95% t CI [-0.76, -0.36]). All three seed-wise
   differences are negative. It lowers target-answer loss (0.713 versus 0.834) and ECE
   (0.016 versus 0.054), but costs 2.01x the wall time. Accuracy and macro-F1
   both decline; no joint decision threshold for these two metrics was
   predeclared. A separately frozen post-observation check adds seeds 145--149
   without retuning: all five differences are negative, with mean -0.80 pp
   and descriptive interval [-1.18, -0.42] pp. Its directional-consistency
   heuristic passes, but it reuses the accessed official test and is not an
   alpha-controlled test. This is a diagnostic accuracy--calibration trade-off,
   not an algorithmic contribution.
6. **Stage-B development mechanism check.** A separate frozen GSM8K partition
   and answer-normalization evaluator have been implemented. On a 64-question
   development subset and Qwen2.5-0.5B, clean final-answer adaptation averages
   6.77% exact match over seeds 52--54, while a 40% deterministic answer
   conflict plus 20% frozen SQuAD off-task mixture averages 2.08% (paired
   delta -4.69 pp). This is low-powered development evidence that the
   joint-mixture stressor transfers beyond topic classification; it is not a
   heterogeneous method result and is excluded from any final claim.
7. **Stage-B router diagnostic.** A 16-anchor relative-gradient router is
   clean-safe on all three GSM8K development seeds (it does not trigger under
   clean data), but raises severe EM only from 2.08% to 3.65%; the paired
   development interval crosses zero. It also retains 25--31 of 51 off-task
   examples. This rules out carrying the current router threshold into a
   GSM8K final protocol.
8. Adding a numeric-answer interface check reduces retained off-task examples
   to 2--5 of 51 and is exactly clean-safe, but its severe EM advantage is
   still inconclusive (+1.04 pp; interval crosses zero). Thus task-interface
   filtering alone cannot resolve numerical label conflict.
9. A subsequent loss-only numeric selector was motivated by an audit AUROC of
   0.756 for conflict detection, but still retained 45--49 of 102 conflicts
   and produced an inconclusive +1.04 pp severe EM difference. Local loss is
   not a sufficient conflict-trust signal under this protocol.
10. A public-anchor-trained independent 1.5B teacher changes this result in an
    initial 64-question development check: its conflict-loss AUROC is
    0.923--0.989 across three seeds, and teacher-ranked numerical adaptation
    improves severe EM from 2.08% to 5.73% (+3.65 pp; descriptive interval
    [+1.40,+5.89]) while satisfying the predeclared clean-safety bound. This
    is positive development evidence only; a full-development replication and
    heterogeneous server comparison remain required.
11. The separately seeded 512-question development replication passes its
    advancement rule: severe EM is 6.58% versus 2.21% for naive adaptation
    (+4.36 pp; descriptive interval [+0.18,+8.55]). On the frozen 1,319-question
    official test, teacher-ranked adaptation improves severe EM from 3.49% to
    6.82% (+3.34 pp) and clean EM from 6.62% to 7.91% (+1.29 pp) across three
    seeds. These are controlled single-task results, not yet evidence of a
    federated or cross-family advantage.
12. **Frozen cross-family GSM8K holdout.** After a separate development gate,
    SmolLM2-360M and TinyLlama-1.1B were evaluated on all 1,319 official test
    questions at seeds 164--166. Teacher ranking improves severe EM from 2.58%
    to 4.17% for SmolLM2 (+1.59 pp) and from 1.42% to 3.44% for TinyLlama
    (+2.02 pp); all six severe paired differences are positive and both clean
    safety gates hold. However, the descriptive 95% intervals cross zero and
    both Holm-adjusted one-sided tests have adjusted p=0.0714. The predeclared general
    cross-family claim therefore fails.
13. **Matched selection controls (post-hoc development).** Holding the retained
    set at 128 examples and client optimization at 40 steps, the teacher has
    higher severe mean EM than numeric-random and client-small-loss selectors
    for both non-Qwen families. For TinyLlama, its gains are +1.56 pp over
    numeric-random and +1.50 pp over client-small-loss; for SmolLM2 they are
    +0.78 and +0.98 pp. These controls support a teacher-signal interpretation
    on development data, but they neither repair the failed holdout gate nor
    substitute for Delora/FedClean-style baselines.
14. **Efficiency diagnostic.** On a matched 64-question severe-mixture profile,
    the teacher endpoint takes 1.52x the naive SmolLM2 wall time and 1.58x the
    naive TinyLlama wall time. Serial teacher scoring of 256 candidates costs
    about 8.1 seconds, and the measured teacher path peaks at 5.57 GiB allocated
    GPU memory. This cost is part of the method boundary, not free preprocessing.
15. **Frozen intermediate-severity secondary holdout.** A new 1,024-example
    GSM8K training-split holdout was frozen with zero overlap against anchors,
    development questions, and all client pools. Across clean, 20% conflict +
    10% off-task, and 40% conflict + 20% off-task conditions, all three client
    families satisfy the clean-safety rule and have nonnegative mean teacher
    gains at both contaminated levels. Qwen gains rise from +2.60 pp at medium
    to +4.07 pp at severe; SmolLM2 gains are +1.24/+1.01 pp and TinyLlama gains
    +0.42/+0.65 pp. None of the nine family-by-severity tests is significant
    after joint Holm correction, so this is a descriptive robustness curve,
    not evidence of nine confirmatory effects.
16. **Frozen AG News Co-Teaching development baseline.** A 54-run matrix
    compares naive single-model LoRA with a two-peer Co-Teaching ensemble on
    clean, 40% conflict, and 40% conflict + 20% off-task mixtures for all three
    client families. Co-Teaching receives the true contamination fraction as
    an oracle forget rate. It is not uniformly robust: contaminated mean gains
    range from +9.29 pp (TinyLlama conflict-only) to -17.67 pp (TinyLlama joint),
    and none of six one-sided contrasts survives Holm correction (all adjusted
    p=1.0). It costs 2.84--9.32x the naive training wall time and serializes
    exactly twice the adapter bytes. This is development-only strong-baseline
    evidence, not an official-test claim.
17. **Frozen Hausa natural weak-label development baseline.** An 18-run matrix
    trains on all 2,045 distant-supervision labels (50.37% disagreement with
    released clean labels) and evaluates on the same 290 clean development
    examples. Despite receiving the measured noise rate as an oracle forget
    rate, Co-Teaching lowers accuracy by 11.61 pp for Qwen, 17.13 pp for
    SmolLM2, and 19.08 pp for TinyLlama. The predeclared improvement gate fails
    (all Holm-adjusted p=1.0), so the 582-example clean test split remains
    unused. Co-Teaching costs 1.69--1.80x wall time and twice the adapter bytes.
18. **Frozen Yoruba natural weak-label development replication.** A second
    18-run matrix covers 1,340 headlines with 33.28% natural weak-label error.
    Co-Teaching changes accuracy by +0.53 pp for Qwen, +2.65 pp for SmolLM2,
    and -4.23 pp for TinyLlama; every interval crosses zero and no improvement
    test survives Holm correction. The predeclared gate fails, the 379-example
    clean test split remains unused, and cost is 1.64--1.89x wall time plus
    twice the adapter bytes.

## 7. Required Work Before Submission

The submission version must not claim a new algorithm until it meets all of
the following gates:

- Add credible cross-architecture and robust-FL comparators, match useful
  communication rather than only recorded payload, and pre-register primary
  contrasts with Holm correction.
- Extend the completed natural weak-label development result to a separately
  justified holdout protocol only if a future method passes a newly frozen
  development gate; the present Co-Teaching gate failed and forbids test use.
- Retain the current framing only if these experiments continue to show that
  public anchors dominate heterogeneous teachers. A positive algorithm claim
  is not supported by the present evidence.

## 8. Limitations

The current router uses gradients and public anchors, which are computationally
expensive and do not provide privacy protection. The controlled conflict
construction is useful for causal diagnosis but cannot represent every form of
real annotation error. Current heterogeneous evidence covers two rounds and
one frozen severe-mixture AG News configuration. The open-answer teacher
endpoint adds a second model and serial per-example scoring (1.52--1.58x wall
time in the diagnostic profile). No claim about privacy,
production deployment, or broad generalization is supported.

## Reproducibility Pointers

- `paper/main.tex`, `paper/main.pdf`, and `paper/references.bib`: official
  AAAI-27 anonymous-format submission draft, compiled PDF, and verified
  citation keys.
- `paper/ReproducibilityChecklist.tex` and `.pdf`: separately compiled official
  AAAI-27 reproducibility checklist.
- `paper/figures/framework_source.tex` and `framework.pdf`: editable vector
  evaluation framework and its compiled paper asset.
- `kafa_script/generate_paper_figures.py`, `paper/figures/evidence_data.tex`,
  and `paper/figures/evidence.pdf`: validated raw-JSON-to-vector-plot pipeline,
  generated coordinates, and compiled two-panel evidence figure.
- `BENCHMARK_SPEC.md`: protocol and claim gates.
- `kafa_script/benchmark_manifest_v1.json`: frozen sample manifest.
- `kafa_script/run_agnews_mixture_dev.py`: controlled local mixtures.
- `kafa_script/diagnose_anchor_gradient_separation.py`: routing diagnostics.
- `kafa_script/run_heterogeneous_distillation_dev.py`: heterogeneous baseline
  and routed pipeline.
- `kafa_script/heterogeneous_routed_final_protocol_v2.json`: frozen final
  comparison protocol and claim gate.
- `kafa_script/gsm8k_benchmark_manifest_v1.json` and
  `kafa_script/gsm8k_stage_b_development_protocol_v2.json`: frozen Stage-B
  reasoning data partition and development-only protocol.
- `kafa_script/analysis_outputs/GSM8K_DEVELOPMENT_REPORT.md`: development-only
  numerical-answer mixture check and its explicit validity boundary.
- `kafa_script/analysis_outputs/HAUSA_NATURAL_NOISE_RESULTS.md`: validated
  natural weak-label development result and failed test-release gate.
- `kafa_script/analysis_outputs/YORUBA_NATURAL_NOISE_RESULTS.md`: validated
  lower-noise language replication and failed test-release gate.
- `kafa_script/analysis_outputs/GSM8K_FINAL_RESULTS.md`: frozen official-test
  summary and paired per-seed gains.
- `kafa_script/analysis_outputs/GSM8K_CROSS_FAMILY_FINAL_RESULTS.md`: frozen
  two-family holdout summary with Holm-corrected claim decision.
- `kafa_script/analysis_outputs/GSM8K_MATCHED_SELECTION_BASELINES.md`: post-hoc
  matched development controls for subset size, numeric filtering, and local
  small-loss selection.
- `kafa_script/analysis_outputs/GSM8K_EFFICIENCY_PROFILE.md`: end-to-end and
  component wall-time diagnostic for the teacher endpoint.
- `kafa_script/analysis_outputs/GSM8K_INTERMEDIATE_SEVERITY_RESULTS.md`: frozen
  1,024-example secondary-holdout curves and nine-test Holm audit.
- `kafa_script/analysis_outputs/AGNEWS_COTEACHING_RESULTS.md`: frozen 54-run
  strong compatible baseline, selection audit, cost ratios, and Holm decision.
- `kafa_script/analysis_outputs/FINAL_HETEROGENEOUS_RESULTS.md`: verified
  three-seed official-test result, raw-file hashes, and negative-result
  analysis.
- `kafa_script/analysis_outputs/AGNEWS_POST_OBSERVATION_SEED_REPLICATION.md`:
  separately reports the five new seeds, protocol digest, directional
  heuristic, and descriptive eight-seed estimate without relabeling the reused
  official test as an untouched holdout.
