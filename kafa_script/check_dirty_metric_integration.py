#!/usr/bin/env python
"""
Check whether dirty GSM8K experiments have expected metrics integrated.

Scope:
- Methods: rawfedavg, ditto
- Dataset path: exp/kafa_matrix/gsm8k
- Seeds: auto-discover seed_* folders

Outputs:
- Console summary
- CSV report at kafa_script/analysis_outputs/metric_integration_report.csv
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
GSM8K_DIR = ROOT / "exp" / "kafa_matrix" / "gsm8k"
OUTPUT_CSV = ROOT / "kafa_script" / "analysis_outputs" / \
    "metric_integration_report.csv"

METHODS = ("rawfedavg", "ditto")
EXPECTED_METRICS = (
    "loss",
    "perplexity",
    "token_acc",
    "gsm8k_exact_match_accuracy",
    "dirty_filter_recall",
    "dirty_filter_precision",
    "clean_retention_rate",
    "nan_rate",
    "stability_nan_rate",
)


@dataclass
class CheckRow:
    method: str
    seed: str
    config_path: str
    log_path: str
    config_has_all_metrics: bool
    log_has_all_metrics: bool
    missing_in_config: str
    missing_in_log: str


def parse_eval_metrics_from_config(cfg_path: Path) -> List[str]:
    if not cfg_path.exists():
        return []

    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    start_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if ln.strip() == "metrics:":
            start_idx = i
            break

    if start_idx is None:
        return []

    metrics: List[str] = []
    i = start_idx + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- "):
            metrics.append(stripped[2:].strip())
            i += 1
            continue
        break
    return metrics


def find_latest_eval_log(seed_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []

    root_eval = seed_dir / "eval_results.log"
    if root_eval.exists():
        candidates.append(root_eval)

    for p in seed_dir.glob("**/eval_results.log"):
        if p == root_eval:
            continue
        candidates.append(p)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_final_weighted_metrics(log_path: Path) -> Dict[str, object]:
    if not log_path.exists():
        return {}

    lines = [ln.strip() for ln in log_path.read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    for ln in reversed(lines):
        try:
            obj = ast.literal_eval(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("Role") != "Server #" or obj.get("Round") != "Final":
            continue
        raw = obj.get("Results_raw", {})
        if not isinstance(raw, dict):
            return {}
        weighted = raw.get("client_summarized_weighted_avg", {})
        return weighted if isinstance(weighted, dict) else {}
    return {}


def ensure_prefixed(metric_name: str, split: str = "test") -> str:
    return f"{split}_{metric_name}"


def run_check() -> List[CheckRow]:
    rows: List[CheckRow] = []
    for method in METHODS:
        method_dir = GSM8K_DIR / method
        if not method_dir.exists():
            continue

        for seed_dir in sorted(method_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            cfg = seed_dir / "config.yaml"
            metrics = parse_eval_metrics_from_config(cfg)
            missing_cfg = [m for m in EXPECTED_METRICS if m not in metrics]

            log = find_latest_eval_log(seed_dir)
            weighted_metrics = extract_final_weighted_metrics(
                log) if log else {}
            weighted_keys = set(weighted_metrics.keys())
            missing_log = [
                ensure_prefixed(m) for m in EXPECTED_METRICS
                if ensure_prefixed(m) not in weighted_keys
            ]

            rows.append(
                CheckRow(
                    method=method,
                    seed=seed_dir.name,
                    config_path=str(cfg),
                    log_path=str(log) if log else "",
                    config_has_all_metrics=not missing_cfg,
                    log_has_all_metrics=not missing_log,
                    missing_in_config=";".join(missing_cfg),
                    missing_in_log=";".join(missing_log),
                ))
    return rows


def write_csv(rows: Sequence[CheckRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "config_path",
                "log_path",
                "config_has_all_metrics",
                "log_has_all_metrics",
                "missing_in_config",
                "missing_in_log",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "method": r.method,
                    "seed": r.seed,
                    "config_path": r.config_path,
                    "log_path": r.log_path,
                    "config_has_all_metrics": r.config_has_all_metrics,
                    "log_has_all_metrics": r.log_has_all_metrics,
                    "missing_in_config": r.missing_in_config,
                    "missing_in_log": r.missing_in_log,
                })


def main() -> None:
    rows = run_check()
    write_csv(rows, OUTPUT_CSV)

    print("=== Dirty GSM8K Metric Integration Check ===")
    for r in rows:
        print(f"[{r.method}][{r.seed}] "
              f"config_ok={r.config_has_all_metrics} "
              f"log_ok={r.log_has_all_metrics}")
        if r.missing_in_config:
            print(f"  missing_in_config: {r.missing_in_config}")
        if r.missing_in_log:
            print(f"  missing_in_log: {r.missing_in_log}")
        if r.log_path:
            print(f"  log: {r.log_path}")
    print(f"\nCSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
