"""Final local preflight for the human OpenReview submitter.

This script checks only local submission-facing artifacts.  It is intentionally
not included in the anonymous artifact package.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - compatibility fallback
    from PyPDF2 import PdfReader  # type: ignore


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_UPLOADS = {
    "paper/main.pdf": {
        "pages": 7,
        "bytes": 342023,
        "sha256": "a34d5df1cb525ca099042a777c0edeb2c4fb265467e195d4a194c568229ee208",
    },
    "paper/supplement.pdf": {
        "pages": 4,
        "bytes": 298045,
        "sha256": "96053096505e9158db2df9c2c5d4146ed267b0afe775f2ef0326fcb8c87881cc",
    },
    "paper/ReproducibilityChecklist.pdf": {
        "pages": 2,
        "bytes": 95788,
        "sha256": "45a4026e559a66b2dd3f47c9ad44ff454b230fe12b165f3cd5d26c8342e7373e",
    },
    "paper/artifacts/KAFA_anonymous_artifact.zip": {
        "pages": None,
        "bytes": 10717452,
        "sha256": "948ff960203025a4c9c44514be24610b59f9a50475e6994d1203d891d66a7d16",
    },
}

HELPERS_EXCLUDED_FROM_ARTIFACT = [
    "COMPLETION_AUDIT",
    "OPENREVIEW_COPYPASTE_FIELDS",
    "OPENREVIEW_HUMAN_FIELDS_WORKSHEET",
    "OPENREVIEW_SUBMISSION_RECEIPT",
    "REVIEWER_ATTACK_NOTES",
    "SUBMISSION_HANDOFF_INDEX",
    "SUBMISSION_FILE_MANIFEST",
    "WORKTREE_HANDOFF",
    "final_submission_preflight",
]

FORBIDDEN_ZIP_TEXT = [
    "fb2c02586",
    "86bba778",
    "b3f09647",
    "35509f08",
    "/root/projects",
    "git@github.com:crimaster-kirito",
    "crimaster",
    "kirito",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"preflight failed: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def extract_openreview_abstract(path: Path) -> str:
    text = path.read_text()
    if "## Abstract for registration" in text:
        pattern = r"## Abstract for registration\n\n(.*?)\n\n## Suggested keywords"
    else:
        pattern = r"## Abstract\n\n(.*?)\n\nLocal counter"
    match = re.search(pattern, text, re.S)
    require(match is not None, f"could not locate abstract in {path}")
    return match.group(1)


def check_upload_files() -> dict[str, object]:
    uploads: dict[str, object] = {}
    for rel, expected in EXPECTED_UPLOADS.items():
        path = ROOT / rel
        require(path.exists(), f"missing upload file {rel}")
        actual_bytes = path.stat().st_size
        actual_sha = sha256(path)
        require(actual_bytes == expected["bytes"],
                f"{rel} byte size changed: {actual_bytes} != {expected['bytes']}")
        require(actual_sha == expected["sha256"],
                f"{rel} sha changed: {actual_sha} != {expected['sha256']}")
        actual_pages = None
        if path.suffix.lower() == ".pdf":
            actual_pages = len(PdfReader(str(path)).pages)
            require(actual_pages == expected["pages"],
                    f"{rel} page count changed: {actual_pages} != {expected['pages']}")
        uploads[rel] = {
            "bytes": actual_bytes,
            "pages": actual_pages,
            "sha256": actual_sha,
        }
    return uploads


def check_abstracts() -> dict[str, int]:
    counts = {}
    for rel in [
        "paper/OPENREVIEW_SUBMISSION.md",
        "paper/OPENREVIEW_COPYPASTE_FIELDS.md",
    ]:
        count = word_count(extract_openreview_abstract(ROOT / rel))
        require(count == 183, f"{rel} abstract word count changed: {count}")
        counts[rel] = count
    require(
        extract_openreview_abstract(ROOT / "paper/OPENREVIEW_SUBMISSION.md")
        == extract_openreview_abstract(ROOT / "paper/OPENREVIEW_COPYPASTE_FIELDS.md"),
        "OpenReview and copy-paste abstracts differ",
    )
    return counts


def check_artifact_zip() -> dict[str, object]:
    path = ROOT / "paper/artifacts/KAFA_anonymous_artifact.zip"
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("ARTIFACT_MANIFEST.json"))
        manifest_files = manifest["files"]
        text_parts = []
        for name in names:
            if name.endswith((".md", ".tex", ".py", ".json", ".txt", ".sh", ".yml", ".yaml")):
                text_parts.append(zf.read(name).decode("utf-8", "ignore"))
    lower_text = "\n".join(text_parts).lower()
    for helper in HELPERS_EXCLUDED_FROM_ARTIFACT:
        require(not any(helper in name for name in names),
                f"non-artifact helper leaked into zip: {helper}")
    for needle in FORBIDDEN_ZIP_TEXT:
        require(needle.lower() not in lower_text,
                f"forbidden text found in artifact zip: {needle}")
    require(len(manifest_files) == 765, f"manifest file count changed: {len(manifest_files)}")
    require(len(names) == 766, f"zip entry count changed: {len(names)}")
    return {"manifest_files": len(manifest_files), "zip_entries": len(names)}


def main() -> None:
    result = {
        "status": "final_submission_preflight_ok",
        "uploads": check_upload_files(),
        "abstract_word_counts": check_abstracts(),
        "artifact": check_artifact_zip(),
        "human_fields_still_required": [
            "authors/order/affiliations/contact author",
            "conflicts",
            "funding",
            "data/code availability choice",
            "simultaneous-submission answer",
            "AI-assistance/policy answers",
            "actual OpenReview submission",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
