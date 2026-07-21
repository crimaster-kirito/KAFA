# Submission handoff index

This is the one-page local entry point for the human submitter.  It is not
included in the anonymous artifact package.

## 0. Run the final local preflight

From the repository root:

```bash
.venv/bin/python paper/final_submission_preflight.py
```

Expected status: `final_submission_preflight_ok`.

This checks the OpenReview abstract, upload-file hashes/page counts, anonymous
artifact contents, and helper-file exclusion.

## 1. Fill human-only fields

Use:

- `paper/OPENREVIEW_HUMAN_FIELDS_WORKSHEET.md`

Do not infer authors, affiliations, conflicts, funding, or policy answers from
the repository.  These are human responsibility fields.

## 2. Copy OpenReview text fields

Use:

- `paper/OPENREVIEW_COPYPASTE_FIELDS.md`

It contains:

- title;
- 183-word abstract;
- keywords;
- AI-assistance disclosure text;
- upload guardrails.

## 3. Upload files and verify hashes

Use:

- `paper/SUBMISSION_FILE_MANIFEST.md`

Upload:

- `paper/main.pdf`
- `paper/supplement.pdf`
- `paper/ReproducibilityChecklist.pdf`
- `paper/artifacts/KAFA_anonymous_artifact.zip` if a code/data supplement field
  is available.

Current anonymous artifact SHA-256:

`948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`

## 4. Read the risk boundaries before clicking submit

Use:

- `paper/SUBMISSION_RISK_CHECKLIST.md`
- `paper/COMPLETION_AUDIT.md`

Do not claim:

- external preregistration;
- privacy preservation;
- new state-of-the-art optimizer;
- task-population generalization;
- whole-repository test-suite green status.

## 5. Do not upload these local helper files

- `paper/SUBMISSION_HANDOFF_INDEX.md`
- `paper/OPENREVIEW_COPYPASTE_FIELDS.md`
- `paper/OPENREVIEW_HUMAN_FIELDS_WORKSHEET.md`
- `paper/SUBMISSION_FILE_MANIFEST.md`
- `paper/COMPLETION_AUDIT.md`
- `paper/REVIEWER_ATTACK_NOTES.md`
- `paper/report.md`
- `RESEARCH_AUDIT.md`
- shell histories, logs, local paths, private credentials, or non-anonymous
  metadata.

## 6. After submission

Save the OpenReview paper ID, confirmation page, and any confirmation email.
Record exactly which files were uploaded and their hashes.

Use:

- `paper/OPENREVIEW_SUBMISSION_RECEIPT.md`
