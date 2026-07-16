#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-.venv/bin/python}"
runner="kafa_script/run_agnews_server_ablation.py"
protocol="kafa_script/agnews_server_post_primary_ablation_v1.json"
max_jobs="${MAX_JOBS:-2}"
methods=(public_anchor_only clean_unfiltered_distillation \
         severe_unfiltered_distillation severe_routed_distillation)

case "${1:-}" in
  development)
    split=development
    seeds=(271 272 273)
    output_dir="kafa_script/analysis_outputs/agnews_server_ablation/development"
    ;;
  secondary_holdout)
    split=secondary_holdout
    seeds=(281 282 283 284 285 286 287 288 289 290)
    output_dir="kafa_script/analysis_outputs/agnews_server_ablation/secondary_holdout"
    ;;
  *)
    echo "Usage: $0 {development|secondary_holdout}" >&2
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
expected = {
    "seed": int(seed), "method": method, "evaluation_split": split,
    "protocol_sha256": protocol_sha, "runner_file_sha256": runner_sha,
}
raise SystemExit(0 if all(record.get(k) == v for k, v in expected.items()) else 1)
PY
}

launch() {
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
    if "$python_bin" "$runner" --protocol "$protocol" --method "$method" \
      --evaluation-split "$split" --seed "$seed" --output "$output" \
      >"$log" 2>&1; then
      :
    else
      local status=$?
      local stamp
      stamp="$(date -u +%Y%m%dT%H%M%SZ)"
      [[ ! -e "$output" ]] || mv "$output" \
        "$output_dir/failures/seed${seed}_${method}.failed.${stamp}.json"
      cp "$log" "$output_dir/failures/seed${seed}_${method}.failed.${stamp}.log"
      exit "$status"
    fi
  ) &
  while (( $(jobs -pr | wc -l) >= max_jobs )); do
    wait -n
  done
}

for seed in "${seeds[@]}"; do
  for method in "${methods[@]}"; do
    launch "$seed" "$method"
  done
done
wait

expected=$(( ${#seeds[@]} * ${#methods[@]} ))
observed=$(find "$output_dir" -maxdepth 1 -name 'seed*.json' | wc -l)
[[ "$observed" -eq "$expected" ]] || {
  echo "Expected $expected records, found $observed" >&2
  exit 1
}
echo "Completed $observed verified-shape AG News $split records"
