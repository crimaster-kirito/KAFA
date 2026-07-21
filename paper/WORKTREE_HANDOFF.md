# Worktree handoff

This file is for repository maintenance after the submission package is
prepared.  It is not included in the anonymous artifact package.

## Current state

The worktree is intentionally dirty.  The changes are submission work products:
new experiment records, generated summaries, paper sources/PDFs, verification
scripts, anonymous artifact packaging, and human submission helper files.

Do not clean the tree with destructive commands unless the human owner has
first decided what to commit, archive, or discard.

## Upload-facing files

These are the files intended for OpenReview upload or optional upload:

- `paper/main.pdf`
- `paper/supplement.pdf`
- `paper/ReproducibilityChecklist.pdf`
- `paper/artifacts/KAFA_anonymous_artifact.zip`

Use `paper/SUBMISSION_FILE_MANIFEST.md` and
`.venv/bin/python paper/final_submission_preflight.py` to verify hashes before
upload.

## Human helper files, not for upload

These are local-only handoff and audit files.  They are not inside the
anonymous artifact and should not be uploaded to OpenReview:

- `paper/SUBMISSION_HANDOFF_INDEX.md`
- `paper/OPENREVIEW_COPYPASTE_FIELDS.md`
- `paper/OPENREVIEW_HUMAN_FIELDS_WORKSHEET.md`
- `paper/OPENREVIEW_SUBMISSION_RECEIPT.md`
- `paper/SUBMISSION_FILE_MANIFEST.md`
- `paper/COMPLETION_AUDIT.md`
- `paper/REVIEWER_ATTACK_NOTES.md`
- `paper/report.md`
- `RESEARCH_AUDIT.md`
- `paper/final_submission_preflight.py`

## Submission/package infrastructure

These should be retained if preserving the submission build:

- `paper/build_submission.sh`
- `paper/verify_submission.py`
- `paper/verify_reported_numbers.py`
- `paper/package_anonymous_artifact.py`
- `paper/README.md`
- `paper/SUBMISSION_RISK_CHECKLIST.md`
- `paper/OPENREVIEW_SUBMISSION.md`

## Paper source and generated assets

These are the manuscript and generated paper assets:

- `paper/main.tex`
- `paper/supplement.tex`
- `paper/references.bib`
- `paper/evidence_ledger.tex`
- `paper/figures/`
- `paper/generated/`

## Experiment/result additions to preserve

These support the current paper claims:

- `kafa_script/analysis_outputs/AGNEWS_SERVER_ABLATION_SECONDARY.md`
- `kafa_script/analysis_outputs/IMDB_MATCHED_CONTROL_OFFICIAL.md`
- `kafa_script/analysis_outputs/IMDB_SELECTOR_COMPONENT_ABLATION_OFFICIAL.md`
- `kafa_script/analysis_outputs/agnews_server_ablation/secondary_holdout/`
- `kafa_script/analysis_outputs/imdb_selector_component_ablation/`

The AG secondary directory includes JSON records and logs.  Logs are excluded
from the anonymous artifact but remain useful locally for audit/debugging.

## New/updated selector-component code and tests

- `kafa_script/imdb_selector_component_ablation.py`
- `kafa_script/imdb_selector_component_ablation_plan_v1.md`
- `kafa_script/imdb_selector_component_ablation_protocol_v1.json`
- `kafa_script/run_imdb_selector_component_ablation.py`
- `kafa_script/run_imdb_selector_component_ablation_matrix.sh`
- `kafa_script/summarize_imdb_selector_component_ablation.py`
- `kafa_script/verify_imdb_selector_component_ablation.py`
- `tests/test_imdb_selector_component_ablation.py`
- `tests/test_summarize_imdb_selector_component_ablation.py`
- `tests/test_verify_imdb_selector_component_ablation.py`

## Known verification commands

Full focused build:

```bash
bash paper/build_submission.sh
```

Final local preflight:

```bash
.venv/bin/python paper/final_submission_preflight.py
```

Lightweight manuscript verifiers:

```bash
.venv/bin/python paper/verify_submission.py
.venv/bin/python paper/verify_reported_numbers.py
```

## Current artifact identity

- File: `paper/artifacts/KAFA_anonymous_artifact.zip`
- Manifest files: 765
- Zip entries: 766
- SHA-256:
  `948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`

## Remaining human-only work

- Fill author metadata, conflicts, funding, data/code availability, simultaneous
  submission, and AI/policy answers.
- Submit through OpenReview.
- Fill `paper/OPENREVIEW_SUBMISSION_RECEIPT.md` after submission.
