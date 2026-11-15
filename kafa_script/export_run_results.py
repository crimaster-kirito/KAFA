from __future__ import annotations

import ast
import csv
import gzip
import json
import re
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "exp" / "kafa_matrix" / "gsm8k" / "rawfedavg" / "seed_42" / "gsm8k_rawfedavg_seed42"
DEFAULT_OUT_DIR = ROOT / "kafa_script" / "analysis_outputs" / "rawfedavg_seed42_from_saved_run"


def parse_payload_lines(path: Path) -> list[dict]:
    payloads: list[dict] = []
    if not path.exists():
        return payloads
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"(\{.*\})\s*$", line)
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict) and "Role" in payload:
            payloads.append(payload)
    return payloads


def parse_raw_gz(path: Path) -> list[dict]:
    payloads: list[dict] = []
    if not path.exists():
        return payloads
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                payload = ast.literal_eval(line.strip())
            except (SyntaxError, ValueError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def parse_system_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    workers: list[dict] = []
    summaries: list[dict] = []
    if not path.exists():
        return workers, summaries
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("id"), int):
            workers.append(row)
        else:
            summaries.append(row)
    return workers, summaries


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric_rows(rows: list[dict], metric_key: str, method: str) -> list[dict]:
    flattened = []
    for row in rows:
        metrics = row.get(metric_key)
        if not isinstance(metrics, dict):
            continue
        flattened.append({"method": method, "round": row.get("Round"), **metrics})
    return flattened


def bar_svg(title: str, labels: list[str], values: list[float], ylabel: str, colors: list[str]) -> str:
    width, height = 920, 520
    left, right, top, bottom = 96, 34, 62, 104
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = max(values) * 1.12 if values else 1.0
    bar_gap = 24
    bar_w = (plot_w - bar_gap * (len(values) + 1)) / max(len(values), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>',
    ]
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        value = max_v * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dee9"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#111827"/>')
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + bar_gap + i * (bar_w + bar_gap)
        h = plot_h * value / max_v
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700">{value:.6f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def line_svg(title: str, rows: list[dict], y_key: str, ylabel: str) -> str:
    width, height = 920, 520
    left, right, top, bottom = 96, 34, 62, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [(int(row["round"]), float(row[y_key])) for row in rows if row.get("round") != "Final" and y_key in row]
    if not points:
        return ""
    min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
    min_y, max_y = min(y for _, y in points), max(y for _, y in points)
    y_span = max(max_y - min_y, 1e-9)
    coords = []
    for x, y in points:
        px = left + (x - min_x) / max(max_x - min_x, 1) * plot_w
        py = top + plot_h - (y - min_y) / y_span * plot_h
        coords.append((px, py, x, y))
    polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py, _, _ in coords)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>',
    ]
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        value = min_y + y_span * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dee9"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.3f}</text>')
    parts.append(f'<polyline fill="none" stroke="#4C78A8" stroke-width="3" points="{polyline}"/>')
    for px, py, x, y in coords:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#4C78A8"/>')
        parts.append(f'<text x="{px:.1f}" y="{py-10:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{y:.4f}</text>')
        parts.append(f'<text x="{px:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12">{x}</text>')
    parts.append(f'<text x="{left + plot_w/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="15">Communication Round</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    method = "rawfedavg"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_log = run_dir / "eval_results.log"
    exp_print = run_dir / "exp_print.log"
    system_log = run_dir / "system_metrics.log"
    raw_gz = run_dir / "eval_results.raw.gz"
    config = run_dir / "config.yaml"

    payloads = parse_payload_lines(eval_log)
    raw_payloads = parse_raw_gz(raw_gz)
    server_final = next(row for row in payloads if row.get("Role") == "Server #" and row.get("Round") == "Final")
    client_final_payloads = [row for row in payloads if str(row.get("Role", "")).startswith("Client #") and row.get("Round") == 30]

    final_rows = [{"method": method, "round": "Final", "summary_type": key, **metrics} for key, metrics in server_final["Results_raw"].items()]
    client_rows = [
        {"method": method, "round": row["Round"], "client_id": row["Role"].split("#", 1)[1], **row["Results_raw"]}
        for row in client_final_payloads
    ]
    eval_curve_weighted = flatten_metric_rows(payloads, "Results_weighted_avg", method)
    eval_curve_avg = flatten_metric_rows(payloads, "Results_avg", method)
    train_curve = [
        {"method": method, "round": row.get("Round"), "client_id": row["Role"].split("#", 1)[1], **row["Results_raw"]}
        for row in raw_payloads
        if str(row.get("Role", "")).startswith("Client #") and isinstance(row.get("Results_raw"), dict) and "train_avg_loss" in row["Results_raw"]
    ]
    workers, system_summaries = parse_system_metrics(system_log)
    worker_rows = [{"method": method, **row} for row in workers]
    system_summary_rows = [{"method": method, **row} for row in system_summaries]

    metric_fields = ["method", "round", "summary_type", "test_loss", "test_total", "test_avg_loss", "val_loss", "val_total", "val_avg_loss"]
    client_fields = ["method", "round", "client_id", "test_loss", "test_total", "test_avg_loss", "val_loss", "val_total", "val_avg_loss"]
    curve_fields = ["method", "round", "test_loss", "test_total", "test_avg_loss", "val_loss", "val_total", "val_avg_loss"]
    train_fields = ["method", "round", "client_id", "train_loss", "train_total", "train_avg_loss"]
    worker_fields = ["method", "id", "fl_end_time_minutes", "total_model_size", "total_flops", "total_upload_bytes", "total_download_bytes", "global_convergence_round", "local_convergence_round", "global_convergence_time_minutes", "local_convergence_time_minutes"]

    write_csv(out_dir / "final_summary.csv", final_rows, metric_fields)
    write_csv(out_dir / "main_result_for_paper.csv", [row for row in final_rows if row["summary_type"] == "client_summarized_weighted_avg"], metric_fields)
    write_csv(out_dir / "client_final.csv", client_rows, client_fields)
    write_csv(out_dir / "eval_curve_weighted_avg.csv", eval_curve_weighted, curve_fields)
    write_csv(out_dir / "eval_curve_avg.csv", eval_curve_avg, curve_fields)
    write_csv(out_dir / "train_curve_by_client.csv", train_curve, train_fields)
    write_csv(out_dir / "system_metrics_by_worker.csv", worker_rows, worker_fields)
    write_csv(out_dir / "system_metrics_summary.csv", system_summary_rows, sorted({key for row in system_summary_rows for key in row.keys()}))

    for src in (eval_log, exp_print, system_log, raw_gz, config):
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    svgs = {
        "final_test_avg_loss.svg": bar_svg(
            "RawFedAvg Final Test Avg Loss",
            [row["summary_type"] for row in final_rows],
            [float(row["test_avg_loss"]) for row in final_rows],
            "test_avg_loss",
            ["#4C78A8", "#59A14F", "#F28E2B"],
        ),
        "client_final_test_avg_loss.svg": bar_svg(
            "RawFedAvg Client Final Test Avg Loss",
            [f"Client {row['client_id']}" for row in client_rows],
            [float(row["test_avg_loss"]) for row in client_rows],
            "test_avg_loss",
            ["#4C78A8", "#E15759", "#76B7B2"],
        ),
        "eval_curve_weighted_test_avg_loss.svg": line_svg(
            "RawFedAvg Weighted Test Avg Loss by Round",
            eval_curve_weighted,
            "test_avg_loss",
            "test_avg_loss",
        ),
        "worker_time.svg": bar_svg(
            "RawFedAvg Worker End Time",
            [f"Worker {row['id']}" for row in worker_rows],
            [float(row["fl_end_time_minutes"]) for row in worker_rows],
            "minutes",
            ["#76B7B2", "#59A14F", "#F28E2B", "#E15759"],
        ),
    }
    for name, content in svgs.items():
        if content:
            (figure_dir / name).write_text(content, encoding="utf-8")

    report = "<!doctype html><meta charset='utf-8'><title>RawFedAvg Saved Run</title><body style='font-family:Arial;margin:32px'>"
    report += "<h1>RawFedAvg Saved Run Results</h1>"
    report += "<p>Main result: client_summarized_weighted_avg test_avg_loss = 0.4434730351170569.</p>"
    for content in svgs.values():
        if content:
            report += content
    report += "</body>"
    (out_dir / "report.html").write_text(report, encoding="utf-8")

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "source_files": [str(p) for p in (eval_log, exp_print, system_log, raw_gz, config) if p.exists()],
                "main_metric": "client_summarized_weighted_avg.test_avg_loss",
                "main_metric_value": 0.4434730351170569,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved: {out_dir}")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
