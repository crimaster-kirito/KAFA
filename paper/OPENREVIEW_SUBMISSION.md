# AAAI-27 OpenReview registration packet

## Time-critical human action

Create the paper entry in the AAAI-27 OpenReview site before the abstract
deadline: **2026-07-21 23:59 UTC-12 (AoE)**.  The full paper deadline is
2026-07-28 23:59 UTC-12 and the supplementary-material/code deadline is
2026-07-31 23:59 UTC-12.  Source:
<https://aaai.org/conference/aaai/aaai-27/>.

The human submitter must supply the complete author list, author order,
institutional affiliations, conflicts, contact author, and all required policy
answers.  Do not infer these fields from the repository.
AAAI states that submissions are accepted only through OpenReview, not by
email or any other mechanism.
Do not submit a placeholder title or abstract, and do not substantially change
the abstract between the abstract and full-paper deadlines.  Author identities
and conflicts must be finalized carefully because AAAI states that authors may
not be added after submission.

## Title

Auditing Selection Purity and Transfer Utility under Joint Model and Data
Heterogeneity

## Abstract for registration

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

## Suggested keywords

- federated learning
- heterogeneous language models
- knowledge distillation
- data selection
- noisy labels
- empirical evaluation

## Files to upload or prepare

- Main paper PDF: `paper/main.pdf` (anonymous; 7 US-Letter pages; references
  start on page 6).
- Supplementary PDF: `paper/supplement.pdf` (anonymous; 4 pages).
- Reproducibility checklist: `paper/ReproducibilityChecklist.pdf` (2 pages;
  upload separately in the designated checklist field, not merged into the
  main paper).
- Optional artifact/code package: use the repository state containing
  `paper/generated/server_audit_current_inputs.json`,
  `paper/generated/server_audit_ledger.json`, and the raw JSON matrices under
  `kafa_script/analysis_outputs/`.  Do not upload non-anonymous local paths,
  shell histories, or private credentials.
- Preferred anonymous artifact zip:
  `paper/artifacts/KAFA_anonymous_artifact.zip` (765 manifest files; record
  the final SHA-256 outside the archive after the last build).

Before uploading, run from the repository root:

```bash
bash paper/build_submission.sh
.venv/bin/python paper/package_anonymous_artifact.py --dry-run
.venv/bin/python paper/package_anonymous_artifact.py --build
```

The registration abstract above is 183 words by the local whitespace/token
counter.  The latest successful full build verified the anonymous PDFs, reported numbers,
complete TREC/IMDb/AG matrices, generated assets, and 147 research-related
tests after adding the selector-component diagnostic verifier, summarizer, and
tests.  The current artifact build wrote 765 manifest files and passed
archive-name plus sensitive text preflight scans; an additional zip text scan
found no local absolute path, original Git URL, repository-owner/user string,
old commit fragment, or obvious credential string.

## Human-only fields

The submitter must fill these manually in OpenReview:

- complete author list and order;
- institutional affiliations;
- contact author;
- conflicts of interest;
- funding acknowledgments, if any;
- data/code availability choices;
- policy questions, including AI-assistance disclosure consistency.

The manuscript already includes an AI-assistance disclosure in the main paper.
Do not infer human fields from git metadata, local paths, or repository owner
names.

## Risk checklist

Before final submission, read `paper/SUBMISSION_RISK_CHECKLIST.md`.  It lists
the verified mechanical state, human-only fields, supported claims, claims that
must not be made, and the highest remaining reviewer risks.

## Evidence status at preparation time

- TREC four-arm values above: verified post-primary reused-test evidence.
- IMDb four-arm values above: verified complete 10-by-4 protocol-frozen
  matrix; the frozen negative-direction primary gate failed.
- IMDb matched-size matrix: all 30 records verified; conditional implementation
  relationship audit passed and the secondary Holm summary is included above.
- AG four-arm resampling: all 40 records verified. It is post-primary
  run-disjoint same-task resampling, not independent confirmation, and is
  included only as bounded secondary context.
