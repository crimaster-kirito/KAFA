# OpenReview copy-paste fields

This file is for the human submitter.  It is intentionally not included in the
anonymous artifact package.

## Venue

AAAI-27 Main Technical Track.

## Title

Auditing Selection Purity and Transfer Utility under Joint Model and Data
Heterogeneity

## Abstract

A canonical probability interface makes cross-architecture exchange
well-defined across language models with different parameters and tokenizers,
but cleaner selected client data need not yield more useful server supervision.
We introduce a four-arm audit that separates clean-transfer utility,
contamination sensitivity, and the complete routing intervention, plus
matched-size comparisons that separate selection order from subset size. A
reused-test TREC decomposition finds that routing raises audited clean fraction
from 40.0% to 64.6% but reduces severe-uniform accuracy by 1.42 points. A
protocol-frozen role reversal on the 25,000-review IMDb test reverses that
direction: routing raises clean fraction from 40.2% to 66.5% and improves
severe-uniform accuracy by 0.29 points, so the frozen negative-direction gate
fails. At matched K=128, routed selection exceeds target-eligible random
selection by 0.32 points, while the audit-only maximum-attainable-purity arm
does not establish an additional gain from higher purity. A post-primary AG
News resampling also gives a positive routed-minus-severe effect but remains
below anchor-only. The opposite TREC and IMDb signs, together with AG's split
between D-C and D-A, show that transfer utility is setting- and
estimand-specific.

Local counter: 183 words.

## Keywords

- federated learning
- heterogeneous language models
- knowledge distillation
- data selection
- noisy labels
- empirical evaluation

## Upload files

- Main paper PDF: `paper/main.pdf`
- Supplementary PDF: `paper/supplement.pdf`
- Reproducibility checklist PDF: `paper/ReproducibilityChecklist.pdf`
- Optional anonymous artifact: `paper/artifacts/KAFA_anonymous_artifact.zip`

Current artifact SHA-256:

`948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`

## Human-only fields to fill

Do not infer these from the repository.

- Authors:
- Author order:
- Affiliations:
- Contact author:
- Conflicts of interest:
- Funding acknowledgments:
- Data/code availability selection:
- Simultaneous-submission answer:
- AI-assistance/policy answer:

## AI-assistance disclosure text already in the manuscript

OpenAI Codex assisted with code drafting, experiment orchestration, artifact
validation, and manuscript editing.  The human authors are responsible for
the study design and submission, verified the cited sources and reported
results against the raw artifacts, and take full responsibility for the
paper's content.

## Submission-day guardrails

- Submit through OpenReview only.
- Do not use a placeholder title or abstract.
- Do not substantially change the abstract between the abstract and full-paper
  deadlines.
- Finalize author identities and conflicts before submission; AAAI states that
  authors may not be added after submission.
- Upload the reproducibility checklist as the separate checklist file, not
  merged into the main paper.
- Upload only anonymous PDFs and the anonymous/code-safe artifact bundle.
