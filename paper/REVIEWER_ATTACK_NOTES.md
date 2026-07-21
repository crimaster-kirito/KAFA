# Internal reviewer-attack notes

This file is an internal preparation aid and is not part of the anonymous
artifact package.  It records the most likely review objections and the safest
response boundaries for the current manuscript.

## 1. "This is not a new federated optimizer."

Correct.  The manuscript should keep presenting the contribution as an
evaluation/audit paper.  The core novelty claim is the evidence-labelled
decomposition of clean-transfer utility, contamination sensitivity, routing
intervention, and matched-size selection order under heterogeneous decoder-only
clients.  Do not pivot to optimizer novelty.

## 2. "The result is too negative or mixed."

The mixed signs are the point.  TREC gives a reused-test negative D--C effect;
IMDb, under the frozen role-reversal protocol, gives a positive D--C effect;
AG secondary is positive on D--C but remains below anchor-only on D--A.  The
safe claim is setting- and estimand-specific transfer utility, not universal
failure or universal success.

## 3. "TREC and AG are not independent confirmation."

Concede the boundary.  TREC four-arm is post-primary reused-test diagnostic.
AG secondary is post-primary, run-disjoint same-task resampling with disclosed
overlap history.  Only the IMDb role reversal carries the protocol-frozen
cross-setting evaluation role, and even it is not external preregistration.

## 4. "Purity is not sufficient as a mechanism."

Agree.  The paper does not claim purity is irrelevant.  The matched-K IMDb
result shows routed ordering is useful relative to random-K, while
maximum-purity-K does not show a detectable additional gain.  The selector
component diagnostic does not isolate a Holm-significant single-component
replacement gap.  Mechanism attribution remains bounded, not solved.

## 5. "Synthetic contamination is unrealistic."

The correct defense is controlled identifiability of estimands, not realism.
Conflict and off-task interventions are deliberately explicit so B--A, C--B,
and D--C can be separated and audited.  The limitation is already disclosed:
the study does not sample real organizations or natural client populations.

## 6. "No privacy guarantee."

Concede fully.  The paper should never claim privacy preservation.  It only
avoids moving raw private client text in the simulated protocol and explicitly
states that logits, losses, anchors, and hashes provide no formal privacy.

## 7. "Why is this CCF-A level?"

The strongest positioning is not raw algorithmic novelty; it is rigorous
evidence hygiene around a common but under-tested assumption: cleaner selected
client data does not automatically imply higher server utility under
heterogeneous probability transfer.  The value is in protocol-frozen
role-reversal evidence, complete matrices, matched-K controls, content-addressed
artifacts, and explicit separation of discovery, secondary, and evaluation
roles.

## 8. "The paper is too narrow."

Accept the scope.  Do not overclaim population generalization.  The paper
audits three server tasks and three client model families; the conclusion is a
stress-test warning and reproducible diagnostic protocol, not a universal
benchmark result.

## Current safest one-sentence claim

Under heterogeneous decoder-only client-to-server probability transfer, higher
audited clean fraction is neither a sufficient nor a direction-stable proxy for
server accuracy; complete, evidence-labelled intervention matrices are needed
to distinguish clean-transfer utility, contamination effects, and routing
effects.
