#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root" || exit 1

phase="${PHASE:-development}"
routed_max_jobs="${ROUTED_MAX_JOBS:-${MAX_JOBS:-2}}"
anchor_max_jobs=1
python_bin="${PYTHON_BIN:-.venv/bin/python}"
runner="${RUNNER:-kafa_script/run_generic_classification_distillation.py}"
protocol="${PROTOCOL:-kafa_script/trec_server_transfer_protocol_v1.json}"
output_root="${OUTPUT_ROOT:-kafa_script/analysis_outputs/trec_server}"
dry_run="${DRY_RUN:-0}"

if [[ "$phase" != "development" && "$phase" != "official" ]]; then
  echo "PHASE must be development or official, got: $phase" >&2
  exit 2
fi
if [[ ! "$routed_max_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROUTED_MAX_JOBS/MAX_JOBS must be a positive integer, got: $routed_max_jobs" >&2
  exit 2
fi
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1, got: $dry_run" >&2
  exit 2
fi
if [[ ! -f "$protocol" || ! -f "$runner" ]]; then
  echo "Missing protocol or runner: $protocol / $runner" >&2
  exit 2
fi

protocol_hash="$($python_bin - "$protocol" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)" || exit 2
runner_hash="$($python_bin - "$runner" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)" || exit 2

protocol_section="$phase"
evaluation_split="development"
if [[ "$phase" == "official" ]]; then
  protocol_section="official_evaluation"
  evaluation_split="official_test"
fi
output_dir="$output_root/$phase"
log_dir="${LOG_DIR:-/tmp/kafa_trec_server/$phase}"

mapfile -t matrix_specs < <("$python_bin" - "$protocol" "$protocol_section" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1], encoding="utf-8"))
section = protocol[sys.argv[2]]
# Method-major ordering is a memory-safety invariant: all anchor jobs finish
# before routed jobs begin, so the two model-allocation profiles never overlap.
for comparator in section["comparators"]:
    for seed in section["seeds"]:
        print(f"{seed}\t{comparator}")
PY
)
if (( ${#matrix_specs[@]} == 0 )); then
  echo "Protocol matrix is empty for $protocol_section" >&2
  exit 2
fi

# The official preflight deliberately points at a nonexistent temporary path:
# frozen-input verification must not mistake an incomplete output directory for
# a protocol failure before the ten jobs have had a chance to run.
preflight_root="$(mktemp -d)" || exit 2
trap 'rm -rf "$preflight_root"' EXIT
preflight_official="$preflight_root/not-created"
if ! "$python_bin" kafa_script/verify_trec_server_protocol.py \
    --protocol "$protocol" --development-output "$output_root/development" \
    --official-output "$preflight_official" >/dev/null; then
  echo "TREC frozen-input preflight failed" >&2
  exit 2
fi

if [[ "$dry_run" == "0" ]]; then
  mkdir -p "$output_dir" "$log_dir"
fi

artifact_matches() {
  local path="$1" seed="$2" method="$3"
  [[ -s "$path" ]] || return 1
  "$python_bin" - "$path" "$protocol_hash" "$runner_hash" "$seed" \
      "$method" "$evaluation_split" <<'PY'
import json, sys
try:
    record = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {
    "protocol_sha256": sys.argv[2],
    "runner_file_sha256": sys.argv[3],
    "seed": int(sys.argv[4]),
    "method": sys.argv[5],
    "evaluation_split": sys.argv[6],
}
raise SystemExit(0 if all(record.get(key) == value
                          for key, value in expected.items()) else 1)
PY
}

pids=()
labels=()
failed=0

wait_one() {
  local pid="${pids[0]}" label="${labels[0]}"
  if ! wait "$pid"; then
    echo "FAILED: $label" >&2
    failed=1
  fi
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
}

launch() {
  local seed="$1" method="$2"
  local method_max_jobs="$routed_max_jobs"
  if [[ "$method" == "public_anchor_only" ]]; then
    method_max_jobs="$anchor_max_jobs"
  fi
  local output="$output_dir/seed${seed}_${method}.json"
  local log="$log_dir/seed${seed}_${method}.log"
  if artifact_matches "$output" "$seed" "$method"; then
    echo "SKIP matched artifact: $output"
    return
  fi
  if [[ "$dry_run" == "1" ]]; then
    echo "DRY-RUN $python_bin $runner --protocol $protocol --method $method --evaluation-split $evaluation_split --seed $seed --output $output"
    return
  fi
  local temporary="${output}.tmp.$$.$RANDOM"
  rm -f "$temporary"
  (
    if "$python_bin" "$runner" --protocol "$protocol" --method "$method" \
        --evaluation-split "$evaluation_split" --seed "$seed" \
        --output "$temporary" >"$log" 2>&1; then
      mv -f "$temporary" "$output"
    else
      status=$?
      rm -f "$temporary"
      exit "$status"
    fi
  ) &
  pids+=("$!")
  labels+=("seed=$seed method=$method log=$log")
  if (( ${#pids[@]} >= method_max_jobs )); then
    wait_one
  fi
}

current_method=""
for spec in "${matrix_specs[@]}"; do
  IFS=$'\t' read -r seed method <<<"$spec"
  if [[ -n "$current_method" && "$method" != "$current_method" ]]; then
    while (( ${#pids[@]} > 0 )); do
      wait_one
    done
  fi
  current_method="$method"
  launch "$seed" "$method"
done
while (( ${#pids[@]} > 0 )); do
  wait_one
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run complete for ${#matrix_specs[@]} $phase matrix entries"
  exit 0
fi
if (( failed != 0 )); then
  echo "One or more TREC $phase jobs failed; see $log_dir" >&2
  exit 1
fi

for spec in "${matrix_specs[@]}"; do
  IFS=$'\t' read -r seed method <<<"$spec"
  output="$output_dir/seed${seed}_${method}.json"
  if ! artifact_matches "$output" "$seed" "$method"; then
    echo "Missing or stale completed artifact: $output" >&2
    exit 1
  fi
done

if [[ "$phase" == "official" ]]; then
  "$python_bin" kafa_script/verify_trec_server_protocol.py \
    --protocol "$protocol" --development-output "$output_root/development" \
    --official-output "$output_dir"
else
  "$python_bin" kafa_script/verify_trec_server_protocol.py \
    --protocol "$protocol" --development-output "$output_dir" \
    --official-output "$preflight_official"
fi
echo "Completed ${#matrix_specs[@]} verified TREC $phase matrix artifacts"
