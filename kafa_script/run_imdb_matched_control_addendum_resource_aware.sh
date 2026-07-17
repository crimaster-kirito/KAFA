#!/usr/bin/env bash
set -euo pipefail

# Execution-only coordinator for the frozen matched-control addendum.  This
# file is intentionally outside the protocol's implementation hash set: the
# scientific runner, protocol, and original serial matrix remain immutable.
cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-.venv/bin/python}"
frozen_runner="kafa_script/run_imdb_matched_control_addendum.py"
frozen_matrix="kafa_script/run_imdb_matched_control_addendum_matrix.sh"
protocol="kafa_script/imdb_matched_control_addendum_protocol_v1.json"
expected_protocol_sha="47175564a023ea59e0137c847bdf747e3ec70b436ae52bc51a0f92f9b2670d64"
expected_matrix_sha="03356bec89161d48f1d42e509d837334d22cf8e0f73c61d21dd6a6676135a71d"
requested_max_jobs="${MAX_JOBS:-2}"
output_root="${OUTPUT_ROOT:-kafa_script/analysis_outputs/imdb_matched_control_addendum}"

if [[ ! "$requested_max_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer, got: $requested_max_jobs" >&2
  exit 2
fi
# Empirically, at most two ordinary IMDb clients fit safely on this GPU.
max_jobs=$(( requested_max_jobs > 2 ? 2 : requested_max_jobs ))
if (( requested_max_jobs > 2 )); then
  echo "Capping ordinary matched-control concurrency at 2 (requested $requested_max_jobs)"
fi

# Test doubles are opt-in; production cannot silently substitute a different
# scientific runner through an ambient environment variable.
runner="$frozen_runner"
if [[ "${ALLOW_TEST_RUNNER_OVERRIDE:-0}" == "1" ]]; then
  runner="${RUNNER:-$frozen_runner}"
elif [[ -n "${RUNNER:-}" && "${RUNNER}" != "$frozen_runner" ]]; then
  echo "RUNNER override requires ALLOW_TEST_RUNNER_OVERRIDE=1" >&2
  exit 2
fi

case "${1:-}" in
  development)
    split=development
    seeds=(311 312 313)
    output_dir="$output_root/development"
    official_test="$output_root/DO_NOT_CREATE_IMDB_TEST_SENTINEL"
    ;;
  official_evaluation)
    split=official_test
    seeds=(321 322 323 324 325 326 327 328 329 330)
    output_dir="$output_root/official_evaluation"
    official_test=".cache/datasets/imdb/test.jsonl"
    ;;
  *)
    echo "Usage: $0 {development|official_evaluation}" >&2
    exit 2
    ;;
esac

# Fail closed before scheduling if any part of the frozen research contract
# has drifted.  The original serial matrix is included because its hash is
# explicitly recorded in the frozen protocol.
"$python_bin" - "$protocol" "$expected_protocol_sha" \
    "$frozen_matrix" "$expected_matrix_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

protocol_path = pathlib.Path(sys.argv[1])
expected_protocol_sha, matrix_path, expected_matrix_sha = sys.argv[2:]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if sha256(protocol_path) != expected_protocol_sha:
    raise SystemExit("Frozen matched-control protocol hash mismatch")
if sha256(pathlib.Path(matrix_path)) != expected_matrix_sha:
    raise SystemExit("Frozen matched-control matrix hash mismatch")
protocol = json.loads(protocol_path.read_text())
for relative, expected in protocol["implementation"]["file_sha256"].items():
    actual = sha256(pathlib.Path(relative))
    if actual != expected:
        raise SystemExit(f"Frozen implementation hash mismatch: {relative}")
PY

# Prevent two coordinators from scheduling the same matrix concurrently.  The
# lock is execution metadata only and is removed on every normal/error exit.
mkdir -p "$output_root"
lock_dir="$output_root/.resource_aware_${1}.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Another matched-control coordinator holds $lock_dir" >&2
  exit 73
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

mkdir -p "$output_dir/logs" "$output_dir/failures"
protocol_sha="$($python_bin -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$protocol")"
runner_sha="$($python_bin -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$runner")"

valid_record() {
  local path="$1" seed="$2" method="$3"
  "$python_bin" - "$path" "$seed" "$method" "$split" "$protocol_sha" "$runner_sha" <<'PY'
import json, pathlib, sys
path, seed, method, split, protocol_sha, runner_sha = sys.argv[1:]
try:
    record = json.loads(pathlib.Path(path).read_text())
except Exception:
    raise SystemExit(1)
expected = {
    "seed": int(seed), "method": method, "evaluation_split": split,
    "protocol_sha256": protocol_sha, "runner_file_sha256": runner_sha,
}
raise SystemExit(0 if all(record.get(k) == v for k, v in expected.items()) else 1)
PY
}

active_pids=()
phase_failed=0

start_one() {
  local seed="$1" method="$2"
  local output="$output_dir/seed${seed}_${method}.json"
  local log="$output_dir/logs/seed${seed}_${method}.log"
  if [[ -s "$output" ]] && valid_record "$output" "$seed" "$method"; then
    echo "Skip verified $output"
    return
  fi
  if [[ -e "$output" ]]; then
    local stamp
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$output" "$output_dir/failures/seed${seed}_${method}.invalid.${stamp}.json"
  fi
  (
    local status=0 stamp
    "$python_bin" "$runner" --protocol "$protocol" --method "$method" \
      --evaluation-split "$split" --official-test "$official_test" \
      --seed "$seed" --output "$output" >"$log" 2>&1 || status=$?
    if (( status == 0 )) && ! valid_record "$output" "$seed" "$method"; then
      echo "Runner returned success without a valid identity-matched record" >>"$log"
      status=65
    fi
    if (( status != 0 )); then
      stamp="$(date -u +%Y%m%dT%H%M%SZ)"
      [[ ! -e "$output" ]] || mv "$output" \
        "$output_dir/failures/seed${seed}_${method}.failed.${stamp}.json"
      cp "$log" "$output_dir/failures/seed${seed}_${method}.failed.${stamp}.log"
      exit "$status"
    fi
  ) &
  active_pids+=("$!")
}

reap_one() {
  local done_pid="" status=0 remaining=() pid
  if wait -n -p done_pid "${active_pids[@]}"; then
    status=0
  else
    status=$?
    phase_failed=1
  fi
  for pid in "${active_pids[@]}"; do
    [[ "$pid" == "$done_pid" ]] || remaining+=("$pid")
  done
  active_pids=("${remaining[@]}")
  return "$status"
}

drain_active() {
  while (( ${#active_pids[@]} > 0 )); do
    reap_one || true
  done
}

# Phase 1: both lower-memory controls may run concurrently, with a hard cap of
# two total jobs.  Every ordinary arm across every seed finishes before phase 2.
ordinary_methods=(severe_random_k oracle_max_purity_k)
for seed in "${seeds[@]}"; do
  for method in "${ordinary_methods[@]}"; do
    start_one "$seed" "$method"
    while (( ${#active_pids[@]} >= max_jobs )); do
      reap_one || true
    done
  done
done
drain_active

if (( phase_failed != 0 )); then
  echo "One or more ordinary matched-control arms failed; routed arms were not started" >&2
  exit 1
fi

# Phase 2: per-example gradient routing is always exclusive.  Existing valid
# routed artifacts are skipped, so interrupted matrices resume safely.
for seed in "${seeds[@]}"; do
  start_one "$seed" severe_routed_k
  drain_active
done

if (( phase_failed != 0 )); then
  echo "One or more routed matched-control arms failed" >&2
  exit 1
fi

verified=0
for seed in "${seeds[@]}"; do
  for method in severe_random_k severe_routed_k oracle_max_purity_k; do
    output="$output_dir/seed${seed}_${method}.json"
    if [[ -s "$output" ]] && valid_record "$output" "$seed" "$method"; then
      verified=$((verified + 1))
    else
      echo "Missing or invalid final record: $output" >&2
      exit 1
    fi
  done
done
echo "Completed $verified verified IMDb matched-control $1 records"
