import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kafa_script/run_imdb_matched_control_addendum_resource_aware.sh"


def _fake_runner(path: Path) -> None:
    path.write_text(r'''#!/usr/bin/env python3
import argparse, hashlib, json, os, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--protocol", type=Path, required=True)
p.add_argument("--method", required=True)
p.add_argument("--evaluation-split", required=True)
p.add_argument("--official-test", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()

def event(kind):
    with open(os.environ["FAKE_EVENT_LOG"], "a", encoding="utf-8") as h:
        h.write(f"{kind} {a.seed} {a.method} {time.monotonic_ns()}\n")

with open(os.environ["FAKE_CALL_LOG"], "a", encoding="utf-8") as h:
    h.write(f"{a.seed} {a.method}\n")
event("START")
time.sleep(float(os.environ.get("FAKE_SLEEP", "0.04")))
if os.environ.get("FAKE_FAIL") == f"{a.seed}:{a.method}":
    event("FAIL")
    raise SystemExit(9)
record = {
    "protocol_sha256": hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
    "runner_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "seed": a.seed, "method": a.method,
    "evaluation_split": a.evaluation_split,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(record), encoding="utf-8")
event("END")
''', encoding="utf-8")


def _run(tmp_path: Path, phase="development", **extra):
    runner = tmp_path / "fake_runner.py"
    if not runner.exists():
        _fake_runner(runner)
    env = {
        **os.environ,
        "ALLOW_TEST_RUNNER_OVERRIDE": "1",
        "RUNNER": str(runner),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "FAKE_CALL_LOG": str(tmp_path / "calls.log"),
        "FAKE_EVENT_LOG": str(tmp_path / "events.log"),
        **extra,
    }
    return subprocess.run(
        ["bash", str(SCRIPT), phase], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=30)


def _intervals(event_path: Path):
    result = {}
    for line in event_path.read_text().splitlines():
        event, seed, method, timestamp = line.split()
        result.setdefault((int(seed), method), {})[event] = int(timestamp)
    return result


def _overlaps(left, right):
    left_end = left.get("END", left.get("FAIL"))
    right_end = right.get("END", right.get("FAIL"))
    return left["START"] < right_end and right["START"] < left_end


@pytest.mark.parametrize(
    ("phase", "expected_records"),
    [("development", 9), ("official_evaluation", 30)],
)
def test_both_phases_cap_ordinary_at_two_and_keep_routing_exclusive(
        tmp_path, phase, expected_records):
    result = _run(tmp_path, phase=phase, MAX_JOBS="9", FAKE_SLEEP="0.025")
    assert result.returncode == 0, result.stderr
    assert f"Completed {expected_records} verified" in result.stdout
    assert "Capping ordinary matched-control concurrency at 2" in result.stdout

    intervals = _intervals(tmp_path / "events.log")
    ordinary = [value for (_, method), value in intervals.items()
                if method != "severe_routed_k"]
    routed = [value for (_, method), value in intervals.items()
              if method == "severe_routed_k"]
    assert len(ordinary) == expected_records * 2 // 3
    assert len(routed) == expected_records // 3
    assert max(value["END"] for value in ordinary) < \
        min(value["START"] for value in routed)
    assert not any(_overlaps(left, right) for index, left in enumerate(routed)
                   for right in routed[index + 1:])

    # No point in the ordinary wave has more than two live processes.
    points = sorted((time, delta)
                    for value in ordinary
                    for time, delta in ((value["START"], 1), (value["END"], -1)))
    live = peak = 0
    for _, delta in points:
        live += delta
        peak = max(peak, live)
    assert peak == 2


def test_ordinary_failure_is_fail_closed_before_any_routed_launch(tmp_path):
    result = _run(
        tmp_path, FAKE_FAIL="312:oracle_max_purity_k", MAX_JOBS="2")
    assert result.returncode == 1
    assert "routed arms were not started" in result.stderr
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert len(calls) == 6
    assert all("severe_routed_k" not in call for call in calls)
    failures = list((tmp_path / "outputs/development/failures").glob("*"))
    assert any("seed312_oracle_max_purity_k.failed" in path.name
               for path in failures)


def test_exact_identity_resume_skips_valid_and_replaces_stale_record(tmp_path):
    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr
    calls = tmp_path / "calls.log"
    assert len(calls.read_text().splitlines()) == 9

    second = _run(tmp_path)
    assert second.returncode == 0, second.stderr
    assert len(calls.read_text().splitlines()) == 9
    assert second.stdout.count("Skip verified") == 9

    stale = tmp_path / "outputs/development/seed311_severe_random_k.json"
    record = json.loads(stale.read_text())
    record["runner_file_sha256"] = "0" * 64
    stale.write_text(json.dumps(record))
    third = _run(tmp_path)
    assert third.returncode == 0, third.stderr
    assert len(calls.read_text().splitlines()) == 10
    quarantined = list(
        (tmp_path / "outputs/development/failures").glob(
            "seed311_severe_random_k.invalid.*.json"))
    assert len(quarantined) == 1


def test_production_runner_override_is_rejected(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "development"], cwd=ROOT,
        env={**os.environ, "RUNNER": str(tmp_path / "other.py")},
        text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert "requires ALLOW_TEST_RUNNER_OVERRIDE=1" in result.stderr


def test_live_coordinator_lock_rejects_duplicate_scheduler(tmp_path):
    lock = tmp_path / "outputs/.resource_aware_development.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(str(os.getpid()))
    result = _run(tmp_path)
    assert result.returncode == 73
    assert "Another matched-control coordinator holds" in result.stderr
    assert not (tmp_path / "calls.log").exists()


def test_frozen_artifacts_remain_at_protocol_recorded_hashes():
    protocol_path = ROOT / "kafa_script/imdb_matched_control_addendum_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == \
        "47175564a023ea59e0137c847bdf747e3ec70b436ae52bc51a0f92f9b2670d64"
    for relative, expected in protocol["implementation"]["file_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
