# Literature Collision Map and Revised Positioning

Last checked: 2026-07-14. This is a living research document, not paper prose.
Links point to primary paper pages.

## Bottom line

The repository's original story is not sufficiently novel in 2026. Three
central ideas already have close precedents:

- server LLM/client SLM mutual transfer: [FedMKT (COLING
  2025)](https://aclanthology.org/2025.coling-main.17/);
- a compressed proxy SLM, conflict-aware federated optimization, and fusion
  back into a proprietary LLM: [FedProxy (ACL
  2026)](https://aclanthology.org/2026.acl-long.794/);
- cross-architecture LoRA mapping using base-weight subspaces:
  [Cross-LoRA (2025)](https://arxiv.org/abs/2508.05232).

Consequently, KAFA cannot credibly claim the first heterogeneous LLM federated
transfer framework, the first large/small-model knowledge exchange method, or
the first cross-model LoRA projection method.

## Closest work and what it removes from our novelty claim

| Work | Setting/mechanism | Collision with current KAFA | Required response |
|---|---|---|---|
| [FedBiOT](https://arxiv.org/abs/2406.17706) | Server compresses an LLM; clients tune adapters; bi-level optimization handles public/private distribution mismatch | KAFA is built on this code and setting | Treat as direct baseline and infrastructure, not novelty |
| [HetLoRA (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.717/) | Heterogeneous client LoRA ranks, local rank pruning, sparsity-weighted aggregation | Resource/rank heterogeneity | Do not claim generic heterogeneous LoRA support |
| [FLoRA](https://arxiv.org/abs/2409.05976) | Stacking aggregation avoids cross terms when aggregating heterogeneous LoRAs | Shows factor-wise FedAvg is mathematically wrong | Include as a required aggregation baseline and diagnostic |
| [FedMKT (COLING 2025)](https://aclanthology.org/2025.coling-main.17/) | Selective mutual knowledge transfer between a server LLM and heterogeneous client SLMs | Bidirectional large/small-model transfer | Differentiate at the granularity of routed knowledge and failure setting |
| [FedDQC (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.791/) | Client-side instruction-response quality control | Filtering low-quality local instructions | Relevance filtering alone is not novel |
| [FedProxy (ACL 2026)](https://aclanthology.org/2026.acl-long.794/) | Common proxy SLM, parameter-conflict mitigation, training-free fusion into LLM | Nearly subsumes the original proxy + heterogeneity story | Must evaluate against it or move to a distinct threat/failure model |
| [Cross-LoRA](https://arxiv.org/abs/2508.05232) | SVD/Frobenius-optimal cross-model LoRA transfer | Subspace-based LoRA projection across model families | Random P/Q projection is not defensible as the main method |
| [FedMDFG/ReT-FHD line](https://proceedings.mlr.press/v267/qi25a.html) | Model-heterogeneous federated distillation | Logit-space heterogeneous transfer | Include a strong federated-distillation baseline |
| [DAFL (ICML 2021)](https://proceedings.mlr.press/v139/zhu21b.html) | Data-free knowledge distillation for heterogeneous FL | Prototype/distillation exchange without compatible parameters | Do not claim heterogeneous distillation itself |
| [On-Demand FL (AISTATS 2024)](https://proceedings.mlr.press/v238/jeong24a.html) | Select clients to match a target distribution | Relevance-aware participation | Distinguish sample/capability routing from client selection |
| [Superfiltering (ACL 2024)](https://aclanthology.org/2024.acl-long.769/) | A weaker language model filters instruction data used to tune a stronger model | Cross-size model scoring for data selection is already established | Do not claim teacher-loss filtering or cross-size scoring as novelty |
| [Delora (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.792/) | Separate LoRA-based noisy-label detector decouples sample selection from PEFT | Independent detector plus small-loss selection closely overlaps the Stage-B mechanism | Treat as a required noisy-label baseline and position Stage B as evaluation evidence |
| [FedClean (ICML 2025)](https://proceedings.mlr.press/v267/jiang25m.html) | Clean-sample selection, label correction, and robust aggregation in noisy-label FL | Stronger robust-FL machinery than simple filtering | A federated robustness claim requires direct comparison |

## Revised problem candidate

### Capability-conditioned safe transfer under partial task mismatch

The candidate setting has all three properties simultaneously:

1. server and clients use architecturally different language models;
2. each client contains a mixture of target-relevant, complementary, and
   conflicting/off-task examples rather than being wholly relevant or wholly
   irrelevant;
3. the server must improve a specified target capability without suppressing
   useful minority-client capabilities or exposing its full model.

The research question is:

> Can compact semantic anchors estimate the *directional utility* of individual
> client examples/updates for a target capability, so that heterogeneous-model
> knowledge is routed rather than indiscriminately merged?

This differs from client selection because useful and harmful examples can
coexist on the same client. It differs from generic data-quality filtering
because relevance is conditional on a server target capability. It differs
from parameter-conflict merging because routing is decided before or during
local optimization and must work across incompatible model parameter spaces.

This remains a **hypothesis**, not an established novelty claim. A focused
search and pilot must still falsify or support it.

## Candidate method: KAFA-Router (working name)

The method should avoid direct random projection of LoRA factors. A defensible
minimal design is:

1. The server constructs a small capability anchor set containing inputs,
   teacher logits (or top-k logit sketches), and calibrated uncertainty.
2. Each client estimates per-example compatibility by the alignment between
   the example's local gradient/signature and its gradient on capability
   anchors. This creates positive, neutral, and conflicting groups.
3. Local optimization routes gradients with uncertainty-aware weights rather
   than hard MSE rejection. Conflicting updates are projected or isolated in a
   private adapter; complementary updates enter a shared adapter.
4. The server aggregates *composed updates* or function-space predictions, not
   independently averaged LoRA factors. Heterogeneous architectures use anchor
   behavior as their common interface.
5. The server distills the aggregated capability delta into its model and
   checks retained performance on a general-capability guard set.

The method earns its complexity only if the pilot shows that sample-level
routing outperforms client-level selection, FedProxy-style conflict merging,
and ordinary distillation at matched compute and communication.

## Falsifiable claims and required evidence

| Claim | Evidence needed | Failure criterion |
|---|---|---|
| Sample-level routing prevents negative transfer | Sweep off-task mixture from 0% to 80%; report target and retained-capability metrics | No advantage over loss/quality filtering or client selection |
| Anchors bridge model families | At least three genuinely different client families/tokenizers | Benefit only within one architecture family |
| Routing is communication-efficient | Measured serialized bytes and wall time, including anchors | Improvement disappears under matched communication |
| The mechanism is calibrated | AUROC/AUPRC for identifying injected conflicting samples plus reliability plots | Scores do not identify known conflicts |
| Shared knowledge does not erase minority skills | Worst-client and per-capability metrics | Mean improves while minority capability collapses |

## Initial experimental scope

Start with two tasks and controlled mixtures, not the existing 216-run matrix:

- classification: AG News target with IMDb/off-domain and label-conflict
  injections;
- reasoning: GSM8K target with irrelevant instruction data and controlled
  corrupted rationales/answers.

Model families should be small enough for repeated experiments on one 48 GiB
GPU. Candidate client models must be selected after license, tokenizer, and
download availability checks. Every main result requires at least three seeds.

## July 2026 collision update and pivot decision

Two additional candidate contributions are also no longer defensible as novel:

- gauge-invariant LoRA aggregation is directly addressed by
  [GLoRA](https://arxiv.org/abs/2605.06733) and rotational alignment by
  [FedRot-LoRA](https://arxiv.org/abs/2602.23638);
- reconstruct-aggregate-refactorize LoRA is implemented by
  [iFLoRA (ACL Findings 2026)](https://aclanthology.org/2026.findings-acl.284/).

A security reframing is similarly crowded: [Safe-FedLLM (ACL
2026)](https://aclanthology.org/2026.acl-long.1120/) already uses probe-based
step/client/update discrimination for malicious federated LLM clients.

The preregistered controlled pilot also failed to show that KAFA's gradient
router beats a simple hard cosine filter under severe corruption. The remaining
credible path is therefore an **evaluation-and-analysis contribution** focused
on joint axes that current papers usually isolate:

1. model-family heterogeneity;
2. within-client mixtures of useful, off-task, and conflicting examples;
3. unknown corruption severity;
4. target utility, general-capability retention, worst-client utility, and
   actual communication/compute measured together.

KAFA-Router may serve as an adaptive baseline, not the assumed winning method.
The publishability of this path depends on showing that the proposed protocol
exposes failures or reverses conclusions that standard IID/non-IID evaluation
misses. If it does not, the project needs a more fundamental problem pivot.

## Public-anchor teacher result: novelty boundary

The Stage-B public-anchor teacher endpoint is empirically promising, but the
mechanism is not independently novel. Superfiltering already establishes
weak/strong cross-model data filtering, and Delora explicitly decouples a
noisy-label detector from PEFT. Its defensible role is therefore one of the
following, in decreasing order of ambition:

1. a simple endpoint that reveals a reversal between centralized sample
   filtering and heterogeneous server distillation under the new joint-mixture
   benchmark;
2. a cross-family robustness baseline whose behavior is measured with frozen
   clean-safety and multiplicity gates;
3. evidence motivating a future federated method that must still beat Delora,
   FedClean, and matched filtering controls.

The paper must not call teacher-loss ranking a new algorithm. A publishable
evaluation claim would instead require that the controlled benchmark exposes
failures missed by standard evaluation and that the conclusions survive
strong noisy-label and federated-filtering baselines.

The separately sourced TREC-Coarse server experiment strengthens task coverage
without changing that novelty boundary.  Its accuracy effect is null within
seed uncertainty (+0.40 pp, interval crossing zero), while its secondary ECE
improvement repeats the AG News calibration direction.  The defensible claim
is therefore a cross-task accuracy--calibration diagnostic, not a superior
aggregation algorithm or a generally beneficial router.

## Strong-baseline compatibility audit

Reading the published algorithms narrows what can be claimed as a faithful
comparison. Delora learns clean/noisy LoRA modules for finite-label
classification and then trains with selected and relabeled samples. FedClean
assumes clients share a global classifier, uses agreement between annotation
and predicted class, fits per-sample-loss GMMs, relabels with finite-class
predictions, and aggregates parameter-compatible models over multiple rounds.
Neither algorithm directly defines how to relabel an unbounded numerical
answer or aggregate incompatible Qwen/SmolLM/Llama parameters.

Therefore:

- reporting either name on GSM8K after replacing its detector, relabeling, and
  aggregation would be misleading;
- numeric-random and unadapted-client small-loss controls are compatible and
  implemented, but are weaker than the full published methods;
- a faithful FedClean/Delora comparison remains possible on the finite-label,
  homogeneous-client AG News slice and should be treated as a separate
  experiment, not silently generalized to the heterogeneous open-answer task;
- the paper must explicitly distinguish "not applicable under the stated
  interface" from "outperformed." No current evidence shows that KAFA beats
  Delora or FedClean.
