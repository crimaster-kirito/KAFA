import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kafa_script/run_trec_server_matrix.sh"
PROTOCOL = ROOT / "kafa_script/trec_server_transfer_protocol_v1.json"


def _run(env):
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=ROOT, env={**os.environ, **env},
        text=True, capture_output=True, timeout=30)


def _fake_runner(path):
    path.write_text(r'''#!/usr/bin/env python3
import argparse, hashlib, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--protocol", type=Path, required=True)
p.add_argument("--method", required=True)
p.add_argument("--evaluation-split", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
with open(os.environ["FAKE_CALL_LOG"], "a", encoding="utf-8") as h:
    h.write(f"{a.seed} {a.method}\n")
event_log = os.environ.get("FAKE_EVENT_LOG")
if event_log:
    with open(event_log, "a", encoding="utf-8") as h:
        h.write(f"START {a.seed} {a.method} {time.monotonic_ns()}\n")
    time.sleep(0.15)
if os.environ.get("FAKE_FAIL") == f"{a.seed}:{a.method}":
    raise SystemExit(9)
record = {
    "evidence_label": ("benchmark_development" if
                       a.evaluation_split == "development" else "benchmark_holdout"),
    "protocol_sha256": hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
    "runner_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "seed": a.seed, "method": a.method,
    "evaluation_split": a.evaluation_split,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(record), encoding="utf-8")
if event_log:
    with open(event_log, "a", encoding="utf-8") as h:
        h.write(f"END {a.seed} {a.method} {time.monotonic_ns()}\n")
''', encoding="utf-8")


def test_development_dry_run_plans_six_jobs_without_outputs(tmp_path):
    result = _run({"PHASE": "development", "DRY_RUN": "1",
                   "OUTPUT_ROOT": str(tmp_path / "outputs")})

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("DRY-RUN ") == 6
    assert not (tmp_path / "outputs").exists()


def test_lightweight_runner_resumes_only_exact_identity_matches(tmp_path):
    runner = tmp_path / "fake_runner.py"
    _fake_runner(runner)
    call_log = tmp_path / "calls.log"
    output_root = tmp_path / "outputs"
    env = {"PHASE": "development", "MAX_JOBS": "2", "RUNNER": str(runner),
           "OUTPUT_ROOT": str(output_root), "FAKE_CALL_LOG": str(call_log)}

    first = _run(env)
    assert first.returncode == 0, first.stderr
    assert len(call_log.read_text().splitlines()) == 6

    second = _run(env)
    assert second.returncode == 0, second.stderr
    assert len(call_log.read_text().splitlines()) == 6
    assert second.stdout.count("SKIP matched artifact") == 6

    stale = output_root / "development/seed241_public_anchor_only.json"
    record = json.loads(stale.read_text())
    record["runner_file_sha256"] = "0" * 64
    stale.write_text(json.dumps(record))
    third = _run(env)
    assert third.returncode == 0, third.stderr
    assert len(call_log.read_text().splitlines()) == 7


def test_official_lightweight_matrix_is_verified_and_failure_propagates(tmp_path):
    runner = tmp_path / "fake_runner.py"
    _fake_runner(runner)
    call_log = tmp_path / "calls.log"
    base = {"PHASE": "official", "MAX_JOBS": "2", "RUNNER": str(runner),
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FAKE_CALL_LOG": str(call_log)}

    failed = _run({**base, "FAKE_FAIL": "253:routed_distillation"})
    assert failed.returncode == 1
    assert "One or more TREC official jobs failed" in failed.stderr
    # All ten jobs were attempted even though one failed.
    assert len(call_log.read_text().splitlines()) == 10

    completed = _run(base)
    assert completed.returncode == 0, completed.stderr
    assert "Completed 10 verified TREC official matrix artifacts" in completed.stdout
    assert len(list((tmp_path / "outputs/official").glob("*.json"))) == 10


def test_method_aware_schedule_serializes_anchor_and_isolates_method_waves(tmp_path):
    runner = tmp_path / "fake_runner.py"
    _fake_runner(runner)
    call_log = tmp_path / "calls.log"
    event_log = tmp_path / "events.log"
    result = _run({
        "PHASE": "development", "ROUTED_MAX_JOBS": "2",
        "RUNNER": str(runner), "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "FAKE_CALL_LOG": str(call_log), "FAKE_EVENT_LOG": str(event_log),
    })
    assert result.returncode == 0, result.stderr

    intervals = {}
    for line in event_log.read_text().splitlines():
        event, seed, method, timestamp = line.split()
        intervals.setdefault((int(seed), method), {})[event] = int(timestamp)
    anchors = [value for (seed, method), value in intervals.items()
               if method == "public_anchor_only"]
    routed = [value for (seed, method), value in intervals.items()
              if method == "routed_distillation"]
    assert len(anchors) == len(routed) == 3

    def overlaps(left, right):
        return left["START"] < right["END"] and right["START"] < left["END"]

    assert not any(overlaps(left, right) for index, left in enumerate(anchors)
                   for right in anchors[index + 1:])
    assert not any(overlaps(anchor, route) for anchor in anchors for route in routed)
    assert any(overlaps(left, right) for index, left in enumerate(routed)
               for right in routed[index + 1:])
