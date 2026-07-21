# Goal-level completion audit

This file is for project management and human handoff.  It is not included in
the anonymous artifact package.

Objective being audited: complete a CCF-A-oriented research plan, method
implementation, rigorous experiments, and submit-ready paper draft for the KAFA
project, while separating verified results, unverified hypotheses, and
submission risks.

## Summary judgment

The technical paper package is mechanically ready for human submission review:
the manuscript, supplement, checklist, focused build, reported-number verifier,
and anonymous artifact have all been generated and verified.  The overall goal
should not be marked complete yet because OpenReview author metadata, conflicts,
funding, data/code availability choices, simultaneous-submission answers, and
policy answers are human-only fields that cannot be inferred or completed from
the repository.

## Requirement-by-requirement audit

| Requirement | Current status | Evidence |
|---|---|---|
| Research framing suitable for a CCF-A-style submission | Verified draft exists | `paper/main.tex`, `paper/main.pdf`, `paper/REVIEWER_ATTACK_NOTES.md` |
| Method implementation and experiment runners | Verified in focused scope | `kafa_script/*server*`, IMDb matched-K and selector-component runners/verifiers, `paper/build_submission.sh` |
| Rigorous experiment matrices | Verified complete for named matrices | TREC four-arm 40, IMDb original 40, IMDb matched-K 30, IMDb selector-component 30, AG secondary 40; see `paper/report.md` and raw `kafa_script/analysis_outputs/` records |
| English paper draft | Verified built PDF | `paper/main.pdf`, 7 US-Letter pages, references begin on page 6 |
| Supplementary material | Verified built PDF | `paper/supplement.pdf`, 4 pages |
| Reproducibility checklist | Verified built PDF | `paper/ReproducibilityChecklist.pdf`, 2 pages |
| Figures and tables | Verified by submission checker | Main paper has 2 tables, 2 vector figures, 31 resolved references |
| Reported numerical consistency | Verified | `.venv/bin/python paper/verify_reported_numbers.py` passes |
| Anonymous PDF state | Verified | `.venv/bin/python paper/verify_submission.py` passes PDF text/metadata and embedded-font checks |
| Focused submission build | Verified | `bash paper/build_submission.sh` passed with 147 research-related tests |
| Anonymous artifact | Verified | `paper/artifacts/KAFA_anonymous_artifact.zip`, 765 manifest files, 766 zip entries, SHA-256 `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16` |
| Artifact helper-file exclusion | Verified | `OPENREVIEW_COPYPASTE_FIELDS.md`, `SUBMISSION_FILE_MANIFEST.md`, and `REVIEWER_ATTACK_NOTES.md` are not inside the zip |
| OpenReview copy-paste fields | Prepared | `paper/OPENREVIEW_COPYPASTE_FIELDS.md`, 183-word abstract |
| Upload-file manifest | Prepared | `paper/SUBMISSION_FILE_MANIFEST.md` with pages, byte sizes, and SHA-256 hashes |
| Human handoff index | Prepared | `paper/SUBMISSION_HANDOFF_INDEX.md` |
| Post-submission receipt template | Prepared | `paper/OPENREVIEW_SUBMISSION_RECEIPT.md` |
| Worktree handoff | Prepared | `paper/WORKTREE_HANDOFF.md` |
| Final one-command local preflight | Verified | `.venv/bin/python paper/final_submission_preflight.py` passes |
| Submission risk boundary | Prepared | `paper/SUBMISSION_RISK_CHECKLIST.md` and `paper/REVIEWER_ATTACK_NOTES.md` |
| Human author metadata | Not complete | Requires human author list, order, affiliations, contact author |
| Conflicts and funding | Not complete | Requires human knowledge; do not infer from git or repository owner |
| OpenReview policy answers | Not complete | Requires human submitter choices and responsibility |
| Actual OpenReview submission | Not complete | Must be performed by human submitter through OpenReview |

## Verified claims that can be made

- This is an evaluation/audit contribution, not a new optimizer.
- A canonical probability interface handles model/tokenizer heterogeneity but
  does not guarantee useful server transfer.
- TREC reused-test four-arm evidence is hypothesis-generating and has a
  negative D--C routing effect.
- IMDb protocol-frozen role reversal reverses the TREC-derived D--C direction
  and fails the frozen negative-direction gate.
- IMDb matched-K evidence supports useful routed ordering over random-K, while
  maximum-purity-K does not show a detectable additional gain.
- AG secondary evidence is bounded same-task context, not independent
  confirmation.
- IMDb selector-component evidence is post-hoc diagnostic evidence; no
  full-minus-component contrast survives Holm correction.

## Claims that remain unsafe

- Do not claim external preregistration.
- Do not claim privacy preservation, secure aggregation, or leakage protection.
- Do not claim a new state-of-the-art optimizer.
- Do not claim that higher purity is useless or always helpful.
- Do not claim task-population generalization from the three audited tasks.
- Do not claim the entire legacy repository test suite is green.

## Current handoff files

- Copy-paste form fields: `paper/OPENREVIEW_COPYPASTE_FIELDS.md`
- One-page handoff index: `paper/SUBMISSION_HANDOFF_INDEX.md`
- Post-submission receipt template: `paper/OPENREVIEW_SUBMISSION_RECEIPT.md`
- Worktree maintenance handoff: `paper/WORKTREE_HANDOFF.md`
- Upload hash manifest: `paper/SUBMISSION_FILE_MANIFEST.md`
- One-command local preflight: `.venv/bin/python paper/final_submission_preflight.py`
- Risk checklist: `paper/SUBMISSION_RISK_CHECKLIST.md`
- Internal reviewer attack notes: `paper/REVIEWER_ATTACK_NOTES.md`
- Main paper: `paper/main.pdf`
- Supplement: `paper/supplement.pdf`
- Checklist: `paper/ReproducibilityChecklist.pdf`
- Anonymous artifact: `paper/artifacts/KAFA_anonymous_artifact.zip`
