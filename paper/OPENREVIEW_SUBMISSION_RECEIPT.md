# OpenReview submission receipt

This file is for the human submitter after submission.  It is intentionally not
included in the anonymous artifact package.  Fill it only after the OpenReview
entry has been created or finalized.

## Submission identity

- Venue: AAAI-27 Main Technical Track
- OpenReview paper ID:
- OpenReview URL:
- Submission creation time:
- Last modification time:
- Submitter name:
- Submitter email/account:

## Human fields entered

- Authors/order confirmed? yes/no:
- Affiliations confirmed? yes/no:
- Contact author confirmed? yes/no:
- Conflicts entered? yes/no:
- Funding answer entered? yes/no:
- Data/code availability answer entered? yes/no:
- Simultaneous-submission answer entered? yes/no:
- AI-assistance/policy answer entered? yes/no:

## Uploaded files

Use `paper/SUBMISSION_FILE_MANIFEST.md` to verify each hash.

| Upload field | Local file | Uploaded? | SHA-256 checked? | Notes |
|---|---|---|---|---|
| Main paper | `paper/main.pdf` | no | no |  |
| Supplementary material | `paper/supplement.pdf` | no | no |  |
| Reproducibility checklist | `paper/ReproducibilityChecklist.pdf` | no | no | Upload in the designated checklist field |
| Code/data artifact | `paper/artifacts/KAFA_anonymous_artifact.zip` | no | no | Upload only if the form has an appropriate field |

Current expected artifact SHA-256:

`948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16`

## Confirmation evidence

- Confirmation email saved? yes/no:
- Confirmation email path or mailbox label:
- OpenReview confirmation screenshot saved? yes/no:
- Screenshot/file path:
- Final preflight output saved? yes/no:
- Final preflight output path:

## Post-submit caution

- Do not add authors after submission.
- Do not substantially change the abstract between the abstract and full-paper
  deadlines.
- If modifying files before the full-paper deadline, rerun:

```bash
bash paper/build_submission.sh
.venv/bin/python paper/package_anonymous_artifact.py --dry-run
.venv/bin/python paper/package_anonymous_artifact.py --build
.venv/bin/python paper/final_submission_preflight.py
```

- If any file changes, update `paper/SUBMISSION_FILE_MANIFEST.md`, this receipt,
  and the artifact SHA recorded in the OpenReview notes or local records.
