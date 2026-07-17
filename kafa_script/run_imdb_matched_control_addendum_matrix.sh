#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-.venv/bin/python}"
runner="kafa_script/run_imdb_matched_control_addendum.py"
protocol="kafa_script/imdb_matched_control_addendum_protocol_v1.json"
methods=(severe_random_k severe_routed_k oracle_max_purity_k)

case "${1:-}" in
  development)
    split=development
    seeds=(311 312 313)
    output_dir="kafa_script/analysis_outputs/imdb_matched_control_addendum/development"
    official_test="kafa_script/analysis_outputs/imdb_matched_control_addendum/DO_NOT_CREATE_IMDB_TEST_SENTINEL"
    ;;
  official_evaluation)
    split=official_test
    seeds=(321 322 323 324 325 326 327 328 329 330)
    output_dir="kafa_script/analysis_outputs/imdb_matched_control_addendum/official_evaluation"
    official_test=".cache/datasets/imdb/test.jsonl"
    ;;
  *)
    echo "Usage: $0 {development|official_evaluation}" >&2
    exit 2
    ;;
esac

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
expected = {"seed": int(seed), "method": method,
            "evaluation_split": split, "protocol_sha256": protocol_sha,
            "runner_file_sha256": runner_sha}
raise SystemExit(0 if all(record.get(k) == v for k, v in expected.items()) else 1)
PY
}

run_one() {
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
  local status=0
  "$python_bin" "$runner" --protocol "$protocol" --method "$method" \
      --evaluation-split "$split" --official-test "$official_test" \
      --seed "$seed" --output "$output" >"$log" 2>&1 || status=$?
  if (( status != 0 )); then
    local stamp
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    [[ ! -e "$output" ]] || mv "$output" \
      "$output_dir/failures/seed${seed}_${method}.failed.${stamp}.json"
    cp "$log" "$output_dir/failures/seed${seed}_${method}.failed.${stamp}.log"
    return "$status"
  fi
}

# All three arms instantiate the client stack; routed-K additionally computes
# per-example gradients. Run the addendum serially to avoid the observed OOM
# boundary of concurrent IMDb client jobs and protect the active main matrix.
for seed in "${seeds[@]}"; do
  for method in "${methods[@]}"; do
    run_one "$seed" "$method"
  done
done

expected=$(( ${#seeds[@]} * ${#methods[@]} ))
observed=$(find "$output_dir" -maxdepth 1 -name 'seed*.json' | wc -l)
[[ "$observed" -eq "$expected" ]] || {
  echo "Expected $expected records, found $observed" >&2
  exit 1
}
echo "Completed $observed IMDb matched-control $1 records"
