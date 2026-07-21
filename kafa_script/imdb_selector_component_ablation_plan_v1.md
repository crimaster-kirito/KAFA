# IMDb selector-component ablation plan v1

Status: post-hoc secondary design, drafted after observing the official IMDb
primary and matched-K results. This plan must not be described as prospective,
predeclared, or confirmatory.

## Purpose

The matched-K IMDb addendum shows that the full routed score outperforms
target-eligible random-K, but it does not identify which score component carries
the useful ranking signal. This post-hoc ablation decomposes the three ranking
signals used by the full router:

- gradient agreement with same-label public anchors;
- local training loss;
- canonical semantic margin.

## Frozen comparison family

Use the same official IMDb task, severe mixture, anchors, client/server models,
round count, optimizer settings, K=128, and 64-per-claimed-label quotas as the
matched-K addendum.

Re-use the existing `severe_routed_k` official records as the full-router
reference. Add three new selector arms over the same ten official seeds
321--330:

- `gradient_only_k`: rank by `P(-anchor_cosine)` only;
- `loss_only_k`: rank by `P(loss)` only;
- `margin_only_k`: rank by `P(semantic_margin)` only, with off-task rows
  ineligible under the same claimed-label quota rule.

The family contains exactly three paired contrasts:

- full routed-K minus gradient-only-K;
- full routed-K minus loss-only-K;
- full routed-K minus margin-only-K.

Use two-sided paired Student-t tests over the ten common seeds and Holm-adjust
these three p-values. Confidence intervals are unadjusted paired seed
intervals. Accuracy is the primary metric for this secondary family; macro-F1,
canonical score NLL, answer loss, ECE, composition, payload, wall time, and peak
GPU allocation are descriptive.

## Evidence boundary

This is a post-hoc reused-test secondary diagnostic. It can strengthen the
mechanistic interpretation of the observed IMDb matched-K result, but it cannot
alter the original IMDb primary gate, confirm the TREC-derived direction, or
support a broad algorithmic dominance claim.

## Execution policy

Run in a separate output directory:

`kafa_script/analysis_outputs/imdb_selector_component_ablation/official_evaluation`

Do not overwrite or modify:

- `kafa_script/analysis_outputs/imdb_server_role_reversal/official_evaluation`;
- `kafa_script/analysis_outputs/imdb_matched_control_addendum/official_evaluation`;
- `kafa_script/imdb_server_role_reversal_protocol_v1.json`;
- `kafa_script/imdb_matched_control_addendum_protocol_v1.json`.

The runner and verifier for this plan must be content-addressed in a separate
protocol before launching official records. Records are accepted only if all 30
new seed-arm JSON files verify and the existing ten `severe_routed_k` reference
records still verify under the original matched-K verifier.
