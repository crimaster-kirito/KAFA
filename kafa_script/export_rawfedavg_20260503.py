from __future__ import annotations

import ast
import csv
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kafa_script" / "analysis_outputs" / "rawfedavg_20260503"

CONSOLE_OUTPUT = r"""
2026-05-03 15:42:19,499 (server:420) INFO: Server: Final evaluation is finished! Starting merging results.
2026-05-03 15:42:19,544 (server:548) INFO: {'Role': 'Server #', 'Round': 'Final', 'Results_raw': {'client_best_individual': {'test_loss': 868.416016, 'test_total': 1992.0, 'test_avg_loss': 0.435952, 'val_loss': 1772.328125, 'val_total': 3984.0, 'val_avg_loss': 0.444861}, 'client_summarized_weighted_avg': {'test_loss': 883.999605, 'test_total': 1993.333333, 'test_avg_loss': 0.443473, 'val_loss': 1796.451865, 'val_total': 3985.333333, 'val_avg_loss': 0.450767}, 'client_summarized_avg': {'test_loss': 883.989583, 'test_total': 1993.333333, 'test_avg_loss': 0.443468, 'val_loss': 1796.455729, 'val_total': 3985.333333, 'val_avg_loss': 0.450768}}}
2026-05-03 15:42:19,544 (server:567) INFO: {'Role': 'Client #1', 'Round': 30, 'Results_raw': {'test_loss': 898.972656, 'test_total': 1996, 'test_avg_loss': 0.450387, 'val_loss': 1784.90625, 'val_total': 3988, 'val_avg_loss': 0.447569}}
2026-05-03 15:42:19,545 (server:567) INFO: {'Role': 'Client #2', 'Round': 30, 'Results_raw': {'test_loss': 868.416016, 'test_total': 1992, 'test_avg_loss': 0.435952, 'val_loss': 1772.328125, 'val_total': 3984, 'val_avg_loss': 0.444861}}
2026-05-03 15:42:19,545 (server:567) INFO: {'Role': 'Client #3', 'Round': 30, 'Results_raw': {'test_loss': 884.580078, 'test_total': 1992, 'test_avg_loss': 0.444066, 'val_loss': 1832.132812, 'val_total': 3984, 'val_avg_loss': 0.459873}}
2026-05-03 15:42:19,550 (monitor:173) INFO: In worker #0, the system-related metrics are: {'id': 0, 'fl_end_time_minutes': 91.70483, 'total_model_size': 0, 'total_flops': 0, 'total_upload_bytes': 0, 'total_download_bytes': 3487256, 'global_convergence_round': 0, 'local_convergence_round': 0, 'global_convergence_time_minutes': 0, 'local_convergence_time_minutes': 0}
2026-05-03 15:42:19,561 (monitor:173) INFO: In worker #1, the system-related metrics are: {'id': 1, 'fl_end_time_minutes': 91.635745, 'total_model_size': 6742618112, 'total_flops': 0, 'total_upload_bytes': 0, 'total_download_bytes': 1609056, 'global_convergence_round': 0, 'local_convergence_round': 0, 'global_convergence_time_minutes': 0, 'local_convergence_time_minutes': 0}
2026-05-03 15:42:19,571 (monitor:173) INFO: In worker #2, the system-related metrics are: {'id': 2, 'fl_end_time_minutes': 91.255574, 'total_model_size': 6742618112, 'total_flops': 0, 'total_upload_bytes': 0, 'total_download_bytes': 1609056, 'global_convergence_round': 0, 'local_convergence_round': 0, 'global_convergence_time_minutes': 0, 'local_convergence_time_minutes': 0}
2026-05-03 15:42:19,581 (monitor:173) INFO: In worker #3, the system-related metrics are: {'id': 3, 'fl_end_time_minutes': 90.876043, 'total_model_size': 6742618112, 'total_flops': 0, 'total_upload_bytes': 0, 'total_download_bytes': 1609056, 'global_convergence_round': 0, 'local_convergence_round': 0, 'global_convergence_time_minutes': 0, 'local_convergence_time_minutes': 0}
2026-05-03 15:42:19,583 (monitor:246) INFO: After merging the system metrics from all works, we got avg: defaultdict(None, {'id': 'sys_avg', 'sys_avg/fl_end_time_minutes': 91.368048, 'sys_avg/total_model_size': '4.71G', 'sys_avg/total_flops': '0.0', 'sys_avg/total_upload_bytes': '0.0', 'sys_avg/total_download_bytes': '1.98M', 'sys_avg/global_convergence_round': 0.0, 'sys_avg/local_convergence_round': 0.0, 'sys_avg/global_convergence_time_minutes': 0.0, 'sys_avg/local_convergence_time_minutes': 0.0})
2026-05-03 15:42:19,583 (monitor:249) INFO: After merging the system metrics from all works, we got std: defaultdict(None, {'id': 'sys_std', 'sys_std/fl_end_time_minutes': 0.331588, 'sys_std/total_model_size': '2.72G', 'sys_std/total_flops': '0.0', 'sys_std/total_upload_bytes': '0.0', 'sys_std/total_download_bytes': '794.22K', 'sys_std/global_convergence_round': 0.0, 'sys_std/local_convergence_round': 0.0, 'sys_std/global_convergence_time_minutes': 0.0, 'sys_std/local_convergence_time_minutes': 0.0})
""".strip()


def extract_payloads() -> list[dict]:
    payloads = []
    for line in CONSOLE_OUTPUT.splitlines():
        match = re.search(r"INFO:\s+(\{.*\})\s*$", line)
        if not match:
            continue
        payloads.append(ast.literal_eval(match.group(1)))
    return payloads


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(final_rows: list[dict], client_rows: list[dict], system_rows: list[dict]) -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def bar_svg(title: str, labels: list[str], values: list[float], ylabel: str, colors: list[str]) -> str:
        width, height = 900, 520
        left, right, top, bottom = 90, 30, 62, 92
        plot_w, plot_h = width - left - right, height - top - bottom
        max_v = max(values) * 1.12 if values else 1
        bar_gap = 24
        bar_w = (plot_w - bar_gap * (len(values) + 1)) / max(len(values), 1)
        ticks = 5
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
            f'<text x="22" y="{top + plot_h/2}" transform="rotate(-90 22 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>',
        ]
        for i in range(ticks + 1):
            y = top + plot_h - plot_h * i / ticks
            value = max_v * i / ticks
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8dee9" stroke-width="1"/>')
            parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.3f}</text>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827" stroke-width="1.4"/>')
        parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#111827" stroke-width="1.4"/>')
        for i, (label, value) in enumerate(zip(labels, values)):
            x = left + bar_gap + i * (bar_w + bar_gap)
            h = plot_h * value / max_v
            y = top + plot_h - h
            color = colors[i % len(colors)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700">{value:.6f}</text>')
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    final_svg = bar_svg(
        "RawFedAvg Final Test Avg Loss",
        [row["summary_type"] for row in final_rows],
        [float(row["test_avg_loss"]) for row in final_rows],
        "test_avg_loss",
        ["#4C78A8", "#59A14F", "#F28E2B"],
    )
    (figure_dir / "rawfedavg_final_test_avg_loss.svg").write_text(final_svg, encoding="utf-8")

    client_svg = bar_svg(
        "RawFedAvg Client Final Test Avg Loss",
        [f"Client {row['client_id']}" for row in client_rows],
        [float(row["test_avg_loss"]) for row in client_rows],
        "test_avg_loss",
        ["#4C78A8", "#E15759", "#76B7B2"],
    )
    (figure_dir / "rawfedavg_client_final_test_loss.svg").write_text(client_svg, encoding="utf-8")

    worker_svg = bar_svg(
        "RawFedAvg Worker End Time",
        [f"Worker {row['worker_id']}" for row in system_rows],
        [float(row["fl_end_time_minutes"]) for row in system_rows],
        "minutes",
        ["#76B7B2", "#59A14F", "#F28E2B", "#E15759"],
    )
    (figure_dir / "rawfedavg_worker_time.svg").write_text(worker_svg, encoding="utf-8")

    report_html = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>RawFedAvg 2026-05-03 Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
h1 {{ font-size: 24px; }}
section {{ margin: 28px 0; }}
img {{ max-width: 100%; border: 1px solid #d8dee9; }}
code {{ background: #f3f4f6; padding: 2px 5px; }}
</style>
<h1>RawFedAvg 2026-05-03 Results</h1>
<p>Main metric: <code>client_summarized_weighted_avg.test_avg_loss = 0.443473</code>.</p>
<section>{final_svg}</section>
<section>{client_svg}</section>
<section>{worker_svg}</section>
</html>
"""
    (OUT_DIR / "rawfedavg_report.html").write_text(report_html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rawfedavg_console_final_excerpt.log").write_text(CONSOLE_OUTPUT + "\n", encoding="utf-8")

    payloads = extract_payloads()
    server = next(p for p in payloads if p["Role"] == "Server #" and p["Round"] == "Final")
    clients = [p for p in payloads if p["Role"].startswith("Client #")]

    final_rows = []
    for summary_type, metrics in server["Results_raw"].items():
        final_rows.append(
            {
                "method": "rawfedavg",
                "run_time": "2026-05-03 15:42:19",
                "round": "Final",
                "summary_type": summary_type,
                **metrics,
            }
        )

    client_rows = []
    for payload in clients:
        client_rows.append(
            {
                "method": "rawfedavg",
                "run_time": "2026-05-03 15:42:19",
                "round": payload["Round"],
                "client_id": payload["Role"].split("#", 1)[1],
                **payload["Results_raw"],
            }
        )

    system_rows = []
    for line in CONSOLE_OUTPUT.splitlines():
        match = re.search(r"In worker #(\d+).*?(\{.*\})\s*$", line)
        if not match:
            continue
        metrics = ast.literal_eval(match.group(2))
        system_rows.append(
            {
                "method": "rawfedavg",
                "run_time": "2026-05-03 15:42:19",
                "worker_id": match.group(1),
                **metrics,
            }
        )

    system_summary_rows = [
        {
            "method": "rawfedavg",
            "summary": "sys_avg",
            "fl_end_time_minutes": 91.368048,
            "total_model_size": "4.71G",
            "total_flops": "0.0",
            "total_upload_bytes": "0.0",
            "total_download_bytes": "1.98M",
            "global_convergence_round": 0.0,
            "local_convergence_round": 0.0,
            "global_convergence_time_minutes": 0.0,
            "local_convergence_time_minutes": 0.0,
        },
        {
            "method": "rawfedavg",
            "summary": "sys_std",
            "fl_end_time_minutes": 0.331588,
            "total_model_size": "2.72G",
            "total_flops": "0.0",
            "total_upload_bytes": "0.0",
            "total_download_bytes": "794.22K",
            "global_convergence_round": 0.0,
            "local_convergence_round": 0.0,
            "global_convergence_time_minutes": 0.0,
            "local_convergence_time_minutes": 0.0,
        },
    ]

    final_fields = ["method", "run_time", "round", "summary_type", "test_loss", "test_total", "test_avg_loss", "val_loss", "val_total", "val_avg_loss"]
    client_fields = ["method", "run_time", "round", "client_id", "test_loss", "test_total", "test_avg_loss", "val_loss", "val_total", "val_avg_loss"]
    system_fields = ["method", "run_time", "worker_id", "id", "fl_end_time_minutes", "total_model_size", "total_flops", "total_upload_bytes", "total_download_bytes", "global_convergence_round", "local_convergence_round", "global_convergence_time_minutes", "local_convergence_time_minutes"]
    summary_fields = ["method", "summary", "fl_end_time_minutes", "total_model_size", "total_flops", "total_upload_bytes", "total_download_bytes", "global_convergence_round", "local_convergence_round", "global_convergence_time_minutes", "local_convergence_time_minutes"]

    write_csv(OUT_DIR / "rawfedavg_final_summary.csv", final_rows, final_fields)
    write_csv(OUT_DIR / "rawfedavg_client_final.csv", client_rows, client_fields)
    write_csv(OUT_DIR / "rawfedavg_system_metrics_by_worker.csv", system_rows, system_fields)
    write_csv(OUT_DIR / "rawfedavg_system_metrics_summary.csv", system_summary_rows, summary_fields)

    main_row = next(row for row in final_rows if row["summary_type"] == "client_summarized_weighted_avg")
    write_csv(OUT_DIR / "rawfedavg_main_result_for_paper.csv", [main_row], final_fields)

    (OUT_DIR / "rawfedavg_results.json").write_text(
        json.dumps(
            {
                "final_summary": final_rows,
                "client_final": client_rows,
                "system_metrics_by_worker": system_rows,
                "system_metrics_summary": system_summary_rows,
                "recommended_main_metric": "client_summarized_weighted_avg.test_avg_loss",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    source_config = ROOT / "exp" / "kafa_matrix" / "gsm8k" / "rawfedavg" / "seed_42" / "config.yaml"
    if source_config.exists():
        shutil.copy2(source_config, OUT_DIR / "config.yaml")

    plot_results(final_rows, client_rows, system_rows)

    print(f"saved: {OUT_DIR}")
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
