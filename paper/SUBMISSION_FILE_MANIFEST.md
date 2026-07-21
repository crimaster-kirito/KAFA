# Submission file manifest

This file is for the human submitter and is not included in the anonymous
artifact package.  Use it to verify that the files uploaded to OpenReview match
the locally verified outputs.

## Verified upload files

| Purpose | Local file | Pages | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| Main anonymous paper | `paper/main.pdf` | 7 | 342023 | `a34d5df1cb525ca099042a777c0edeb2c4fb265467e195d4a194c568229ee208` |
| Anonymous supplement | `paper/supplement.pdf` | 4 | 298045 | `96053096505e9158db2df9c2c5d4146ed267b0afe775f2ef0326fcb8c87881cc` |
| Reproducibility checklist | `paper/ReproducibilityChecklist.pdf` | 2 | 95788 | `45a4026e559a66b2dd3f47c9ad44ff454b230fe12b165f3cd5d26c8342e7373e` |
| Optional anonymous code/data artifact | `paper/artifacts/KAFA_anonymous_artifact.zip` | n/a | 10717452 | `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16` |

## Verified mechanical state

- `paper/verify_submission.py` verifies the main paper, supplement, checklist,
  PDF anonymity, embedded fonts, references, and named experiment matrices.
- `paper/verify_reported_numbers.py` verifies manuscript and supplement
  numbers against raw JSON summaries.
- `paper/final_submission_preflight.py` verifies the OpenReview abstract word
  count, upload-file hashes and page counts, artifact helper-file exclusion,
  manifest entry count, and artifact text scan.
- The most recent full submission build passed 147 focused research-related
  tests.
- The anonymous artifact contains 765 manifest files and 766 zip entries.
- `paper/OPENREVIEW_COPYPASTE_FIELDS.md` and
  `paper/REVIEWER_ATTACK_NOTES.md` are human helper files and are not inside
  the anonymous artifact.

## Upload placement

- Upload `paper/main.pdf` as the main anonymous paper.
- Upload `paper/supplement.pdf` as supplementary material.
- Upload `paper/ReproducibilityChecklist.pdf` in the designated separate
  reproducibility-checklist field.
- Upload `paper/artifacts/KAFA_anonymous_artifact.zip` only if the submission
  form has an appropriate code/data supplement field.

## One-command local preflight

Run this immediately before upload:

```bash
.venv/bin/python paper/final_submission_preflight.py
```

## Do not upload

- `paper/report.md`
- `RESEARCH_AUDIT.md`
- `paper/REVIEWER_ATTACK_NOTES.md`
- `paper/OPENREVIEW_COPYPASTE_FIELDS.md`
- shell histories, logs, local paths, private credentials, or non-anonymous
  metadata.
