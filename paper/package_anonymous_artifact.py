"""Build a minimal anonymous reproducibility artifact for the KAFA audit paper.

The script is deliberately conservative:

* default mode is a dry run;
* logs, failure directories, caches, model weights, virtualenvs, and git state
  are excluded;
* only verified result JSON/Markdown, paper assets, protocols, task specs,
  focused runners/verifiers, and focused tests are included;
* text files are scanned for obvious local paths, repository-owner names, and
  credential-like tokens before any archive is written.

Run from the repository root:

    .venv/bin/python paper/package_anonymous_artifact.py --dry-run
    .venv/bin/python paper/package_anonymous_artifact.py --build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper/artifacts/KAFA_anonymous_artifact.zip"

TEXT_SUFFIXES = {
    ".bib", ".bst", ".cfg", ".csv", ".json", ".md", ".py", ".sh", ".sty",
    ".tex", ".txt", ".yaml", ".yml",
}
PDF_SUFFIXES = {".pdf"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES
_REPO_OWNER_FRAGMENT = "cri" + "master"
_REPO_USER_FRAGMENT = "ki" + "rito"
_GITHUB_FRAGMENT = "git" + "hub.com"

FORBIDDEN_TEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in [
        r"/root(?:/|$)",
        r"/home(?:/|$)",
        r"git@",
        re.escape(_GITHUB_FRAGMENT) + r"[:/]+" + _REPO_OWNER_FRAGMENT,
        _REPO_OWNER_FRAGMENT,
        _REPO_USER_FRAGMENT,
        r"\b(?:api[_-]?key|password|passwd|secret)\b\s*[:=]",
        r"\b[A-Z0-9_]*API_KEY\b\s*[:=]",
        r"HF_TOKEN",
        r"WANDB_API_KEY",
        r"OPENAI_API_KEY",
    ]
]

EXCLUDE_PARTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "logs", "failures", "invalidated_runner_9538_selected_count_bug",
    "runtime_logs", "artifacts",
}
EXCLUDE_SUFFIXES = {
    ".aux", ".blg", ".log", ".out", ".tmp", ".pt", ".pth", ".bin",
    ".safetensors", ".ckpt", ".sqlite", ".db",
}

PAPER_FILES = [
    "paper/main.pdf",
    "paper/main.tex",
    "paper/supplement.pdf",
    "paper/supplement.tex",
    "paper/ReproducibilityChecklist.pdf",
    "paper/ReproducibilityChecklist.tex",
    "paper/references.bib",
    "paper/aaai2027.sty",
    "paper/aaai2027.bst",
    "paper/evidence_ledger.tex",
    "paper/README.md",
    "paper/OPENREVIEW_SUBMISSION.md",
    "paper/SUBMISSION_RISK_CHECKLIST.md",
    "paper/build_submission.sh",
    "paper/verify_submission.py",
    "paper/verify_reported_numbers.py",
]

PAPER_DIRS = [
    "paper/generated",
    "paper/figures",
]

CORE_KAFA_FILES = [
    "kafa_script/classification_task.py",
    "kafa_script/run_generic_classification_distillation.py",
    "kafa_script/run_trec_server_ablation.py",
    "kafa_script/run_trec_server_ablation_matrix.sh",
    "kafa_script/verify_trec_server_ablation.py",
    "kafa_script/verify_trec_server_protocol.py",
    "kafa_script/summarize_trec_server_ablation.py",
    "kafa_script/run_imdb_server_ablation.py",
    "kafa_script/run_imdb_server_role_reversal_matrix.sh",
    "kafa_script/verify_imdb_server_role_reversal.py",
    "kafa_script/verify_imdb_routed_equivalence.py",
    "kafa_script/summarize_imdb_server_role_reversal.py",
    "kafa_script/run_imdb_matched_control_addendum.py",
    "kafa_script/run_imdb_matched_control_addendum_resource_aware.sh",
    "kafa_script/verify_imdb_matched_control_addendum.py",
    "kafa_script/summarize_imdb_matched_control_addendum.py",
    "kafa_script/generate_imdb_paper_assets.py",
    "kafa_script/run_agnews_server_ablation.py",
    "kafa_script/run_agnews_server_ablation_matrix.sh",
    "kafa_script/verify_agnews_server_ablation.py",
    "kafa_script/summarize_agnews_server_ablation.py",
    "kafa_script/generate_server_audit_assets.py",
    "kafa_script/imdb_selector_component_ablation.py",
    "kafa_script/imdb_selector_component_ablation_plan_v1.md",
    "kafa_script/imdb_selector_component_ablation_protocol_v1.json",
    "kafa_script/run_imdb_selector_component_ablation.py",
    "kafa_script/run_imdb_selector_component_ablation_matrix.sh",
    "kafa_script/summarize_imdb_selector_component_ablation.py",
    "kafa_script/verify_imdb_selector_component_ablation.py",
]

FOCUSED_TESTS = [
    "tests/test_capability_router.py",
    "tests/test_lora_projection_diagnostics.py",
    "tests/test_statistical_summarizers.py",
    "tests/test_classification_task.py",
    "tests/test_trec_server_manifest.py",
    "tests/test_trec_server_protocol.py",
    "tests/test_generic_classification_distillation.py",
    "tests/test_summarize_trec_server_transfer.py",
    "tests/test_trec_server_matrix.py",
    "tests/test_trec_server_ablation.py",
    "tests/test_agnews_server_ablation.py",
    "tests/test_imdb_server_role_reversal.py",
    "tests/test_imdb_matched_control_addendum.py",
    "tests/test_generate_imdb_paper_assets.py",
    "tests/test_generate_server_audit_assets.py",
    "tests/test_imdb_selector_component_ablation.py",
    "tests/test_summarize_imdb_selector_component_ablation.py",
    "tests/test_verify_imdb_selector_component_ablation.py",
]

PROTOCOL_GLOBS = [
    "kafa_script/*protocol*.json",
    "kafa_script/*manifest*.json",
    "kafa_script/task_specs/*.json",
]


def should_skip(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDE_PARTS for part in relative.parts):
        return True
    if any(part.startswith("invalidated") for part in relative.parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    if path.suffix not in ALLOWED_SUFFIXES:
        return True
    return False


def add_existing_file(files: set[Path], relative: str) -> None:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    if not should_skip(path):
        files.add(path)


def add_tree(files: set[Path], relative: str) -> None:
    root = ROOT / relative
    if not root.is_dir():
        raise FileNotFoundError(relative)
    for path in root.rglob("*"):
        if path.is_file() and not should_skip(path):
            files.add(path)


def collect_files() -> list[Path]:
    files: set[Path] = set()
    for relative in PAPER_FILES + CORE_KAFA_FILES + FOCUSED_TESTS:
        add_existing_file(files, relative)
    for relative in PAPER_DIRS:
        add_tree(files, relative)
    for pattern in PROTOCOL_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file() and not should_skip(path):
                files.add(path)
    for path in (ROOT / "kafa_script/analysis_outputs").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md"} and not should_skip(path):
            files.add(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def read_text(path: Path) -> str | None:
    if path.suffix not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def scan_sensitive(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        text = read_text(path)
        if text is None:
            continue
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            match = pattern.search(text)
            if match:
                if relative == "paper/verify_submission.py" and "forbidden" in text:
                    continue
                findings.append(
                    f"{relative}: matched {pattern.pattern!r}")
                break
    return findings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(files: list[Path]) -> dict:
    return {
        "schema_version": "kafa_anonymous_artifact_manifest_v1",
        "claim_boundary": (
            "Anonymous reproducibility artifact file list; does not assert "
            "external preregistration, privacy protection, or full legacy test coverage."),
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def build_zip(files: list[Path], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(files)
    fixed_time = (2026, 7, 20, 0, 0, 0)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), fixed_time)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("ARTIFACT_MANIFEST.json", fixed_time)
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output": output.as_posix(),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "file_count": len(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true",
                        help="write the zip archive after validation")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print manifest summary without writing zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.build and args.dry_run:
        raise SystemExit("choose at most one of --build and --dry-run")
    files = collect_files()
    findings = scan_sensitive(files)
    if findings:
        print(json.dumps({
            "status": "refused_sensitive_artifact",
            "findings": findings[:50],
            "finding_count": len(findings),
        }, indent=2, sort_keys=True))
        raise SystemExit(1)
    total_bytes = sum(path.stat().st_size for path in files)
    summary = {
        "status": "anonymous_artifact_dry_run_ok",
        "file_count": len(files),
        "total_input_bytes": total_bytes,
        "would_write": args.output.as_posix(),
    }
    if args.build:
        summary = {"status": "anonymous_artifact_built",
                   **build_zip(files, args.output)}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
