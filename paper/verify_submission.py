#!/usr/bin/env python3
"""Fail-fast structural verification for the generated anonymous submission."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.verify_trec_server_protocol import verify as verify_trec  # noqa: E402
from kafa_script.verify_trec_server_ablation import verify_matrix as verify_trec_ablation  # noqa: E402

PAPER = ROOT / "paper"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_embedded_fonts(reader: PdfReader, label: str) -> None:
    seen = 0
    for page in reader.pages:
        fonts = page.get("/Resources", {}).get("/Font", {})
        for reference in fonts.values():
            font = reference.get_object()
            descendants = font.get("/DescendantFonts")
            candidates = ([item.get_object() for item in descendants]
                          if descendants else [font])
            for candidate in candidates:
                descriptor = candidate.get("/FontDescriptor")
                require(descriptor is not None,
                        f"{label}: font lacks a descriptor")
                descriptor = descriptor.get_object()
                require(any(key in descriptor for key in
                            ("/FontFile", "/FontFile2", "/FontFile3")),
                        f"{label}: font is not embedded")
                seen += 1
    require(seen > 0, f"{label}: no fonts found")


def main() -> None:
    tex = (PAPER / "main.tex").read_text()
    log = (PAPER / "main.log").read_text(errors="replace")
    pdf = PdfReader(PAPER / "main.pdf")
    checklist = PdfReader(PAPER / "ReproducibilityChecklist.pdf")
    framework = PdfReader(PAPER / "figures/framework.pdf")
    evidence = PdfReader(PAPER / "figures/evidence.pdf")

    require("\\usepackage[submission]{aaai2027}" in tex,
            "official anonymous submission mode is missing")
    require(1 <= len(pdf.pages) <= 7, "main PDF exceeds seven total pages")
    for index, page in enumerate(pdf.pages, 1):
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        require(abs(width - 612) < 1 and abs(height - 792) < 1,
                f"page {index} is not US Letter")
    require(len(checklist.pages) >= 1, "reproducibility checklist is empty")
    require(not re.search(r"Warning|Overfull|Underfull \\hbox|undefined|multiply defined", log),
            "main LaTeX log contains a substantive layout/reference diagnostic")
    checklist_log = (PAPER / "ReproducibilityChecklist.log").read_text(errors="replace")
    require(not re.search(r"Warning|Overfull|undefined|multiply defined", checklist_log),
            "checklist log contains a substantive layout/reference diagnostic")
    for reader, label in ((pdf, "main PDF"), (checklist, "checklist PDF"),
                          (framework, "framework figure"),
                          (evidence, "evidence figure")):
        require_embedded_fonts(reader, label)
    metadata = " ".join(str(value) for value in (pdf.metadata or {}).values()).lower()
    extracted = "\n".join(page.extract_text() or "" for page in pdf.pages).lower()
    for forbidden in ("crimaster", "kirito", "github.com", "/root/projects"):
        require(forbidden not in metadata and forbidden not in extracted,
                f"anonymous PDF leaks forbidden identifier: {forbidden}")

    for section in ("Introduction", "Related Work", "Problem Setting",
                    "Experimental Protocol", "Results", "Discussion",
                    "Limitations and Ethics", "Reproducibility"):
        require(f"\\section{{{section}}}" in tex, f"missing section: {section}")
    require(tex.count("\\begin{table") == 4, "expected four paper tables")
    require(tex.count("\\begin{figure") == 2, "expected two paper figures")
    for asset in ("figures/framework.pdf", "figures/evidence.pdf"):
        require((PAPER / asset).stat().st_size > 10_000, f"missing figure: {asset}")
    require("AI-assistance disclosure" in tex, "AI disclosure is missing")
    require(not re.search(r"\b(TODO|TBD|FIXME)\b", tex), "draft marker remains")

    aux = (PAPER / "main.aux").read_text()
    cited = set()
    for group in re.findall(r"\\citation\{([^}]+)\}", aux):
        cited.update(group.split(","))
    bib = (PAPER / "references.bib").read_text()
    available = set(re.findall(r"@\w+\{([^,]+),", bib))
    require(cited == available, f"citation/bibliography mismatch: {cited ^ available}")
    require(len(cited) == 23, f"expected 23 resolved references, found {len(cited)}")

    natural_sources = {
        "hausa": ROOT / ".cache/data/natural_noise/hausa/COMPRISE_Data09_H-VOA-T opics_V1.0",
        "yoruba": ROOT / ".cache/data/natural_noise/yoruba/COMPRISE_Data13_Y-BBCTopics_V1.0",
    }
    for corpus, source in natural_sources.items():
        records = list((ROOT / "kafa_script/analysis_outputs").glob(
            f"{corpus}_natural_*_dev_seed*.json"))
        require(len(records) == 18, f"expected 18 {corpus} natural-noise records")
        dev_hash, test_hash = sha256(source / "dev.tsv"), sha256(source / "test.tsv")
        require(dev_hash != test_hash, f"{corpus} dev/test hashes unexpectedly match")
        for path in records:
            record = json.loads(path.read_text())
            require(record.get("evidence_label") == "benchmark_dev" and
                    record.get("evaluation_split") == "dev",
                    f"{path.name}: invalid natural-noise evidence scope")
            require(record.get("evaluation_file_sha256") == dev_hash,
                    f"{path.name}: evaluation hash is not source dev.tsv")
            require(record.get("evaluation_file_sha256") != test_hash,
                    f"{path.name}: clean test split was evaluated")
        require(not list((ROOT / "kafa_script/analysis_outputs").glob(
            f"{corpus}_natural_*_test_seed*.json")),
            f"{corpus} natural-noise test records exist")
    require((ROOT / "kafa_script/analysis_outputs/YORUBA_NATURAL_NOISE_RESULTS.md").exists(),
            "Yoruba report is missing")
    require((ROOT / "kafa_script/NATURAL_NOISE_PROTOCOL_CLARIFICATION.md").exists(),
            "natural-noise clean-label clarification is missing")
    replication_records = list((ROOT / "kafa_script/analysis_outputs").glob(
        "replication_*_seed*.json"))
    require(len(replication_records) == 10,
            "expected 10 AG News post-observation replication records")
    require((ROOT / "kafa_script/analysis_outputs/AGNEWS_POST_OBSERVATION_SEED_REPLICATION.md").exists(),
            "AG News post-observation replication report is missing")

    parent_path = ROOT / "kafa_script/gsm8k_benchmark_manifest_v1.json"
    secondary_path = ROOT / "kafa_script/gsm8k_secondary_holdout_manifest_v1.json"
    severity_protocol = json.loads((ROOT / "kafa_script/gsm8k_intermediate_severity_holdout_protocol_v1.json").read_text())
    parent = json.loads(parent_path.read_text())
    secondary = json.loads(secondary_path.read_text())
    require(sha256(parent_path) == severity_protocol["parent_manifest_sha256"],
            "severity protocol parent-manifest hash mismatch")
    require(sha256(secondary_path) == severity_protocol["evaluation_manifest_sha256"],
            "severity protocol evaluation-manifest hash mismatch")
    excluded = set(parent["partitions"]["anchor"]) | set(parent["partitions"]["development"])
    for pool in parent["partitions"]["client_pools"]:
        excluded.update(pool)
    secondary_indices = secondary["source_indices"]
    require(len(secondary_indices) == 1024 and len(set(secondary_indices)) == 1024,
            "secondary holdout is not 1,024 unique examples")
    require(not (set(secondary_indices) & excluded),
            "secondary holdout overlaps anchor/development/client pools")

    trec = verify_trec()
    ablation_root = ROOT / "kafa_script/analysis_outputs/trec_server_ablation"
    ablation_dev = verify_trec_ablation(
        "development", ablation_root / "development")
    ablation_official = verify_trec_ablation(
        "official_evaluation", ablation_root / "official_reused_test")
    invalidated = (ablation_root /
                   "invalidated_runner_9538_selected_count_bug")
    require(len(list(invalidated.rglob("*.json"))) == 13,
            "expected 12 development + 1 official invalidated ablation records")
    failures = ablation_root / "official_reused_test/failures"
    require(len(list(failures.glob("*.log"))) >= 4,
            "ablation infrastructure-failure history is incomplete")
    require(not list((ablation_root / "official_reused_test").glob("*.tmp*")),
            "ablation temporary artifact remains at top level")
    ablation_report = (ROOT /
        "kafa_script/analysis_outputs/TREC_SERVER_POST_PRIMARY_ABLATION.md")
    report_text = ablation_report.read_text(encoding="utf-8")
    summarizer = ROOT / "kafa_script/summarize_trec_server_ablation.py"
    verifier = ROOT / "kafa_script/verify_trec_server_ablation.py"
    require(f"Summarizer SHA-256: `{sha256(summarizer)}`" in report_text,
            "ablation report embeds a stale summarizer hash")
    require(f"Verifier SHA-256: `{sha256(verifier)}`" in report_text,
            "ablation report embeds a stale verifier hash")

    print(f"Verified main.pdf: {len(pdf.pages)} US-Letter pages, "
          f"4 tables, 2 vector figures, {len(cited)} resolved references")
    print(f"Verified checklist: {len(checklist.pages)} pages")
    print("Verified embedded fonts and anonymous PDF text/metadata")
    print("Verified natural-noise matrices: 18 Hausa + 18 Yoruba dev-only records")
    print("Verified secondary split: 1,024 unique examples with zero parent-partition overlap")
    print("Verified AG News post-observation replication: 10 records and derived report")
    print("Verified TREC frozen 3x2 development and 5x2 official matrices: "
          f"protocol {trec['protocol_sha256'][:12]}...")
    print("Verified TREC post-primary 3x4 development and 10x4 reused-test "
          f"ablation matrices: protocol {ablation_official['protocol_sha256'][:12]}...")


if __name__ == "__main__":
    main()
