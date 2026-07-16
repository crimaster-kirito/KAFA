#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# One process is the measured throughput optimum on the shared visible device:
# two concurrent runs increased per-pair completion time through GPU contention.
# MAX_JOBS remains available for genuinely independent accelerators.
max_jobs="${MAX_JOBS:-1}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
runner="kafa_script/run_heterogeneous_distillation_rounds_dev.py"
clients=(.cache/models/Qwen2.5-0.5B .cache/models/SmolLM2-360M .cache/models/TinyLlama-1.1B-Chat-v1.0)

launch() {
  local seed="$1" variant="$2"
  local output="kafa_script/analysis_outputs/replication_${variant}_seed${seed}.json"
  local log="/tmp/agnews_replication_${variant}_seed${seed}.log"
  if [[ -s "$output" ]]; then return; fi
  local method_args=()
  if [[ "$variant" == "anchor_only" ]]; then
    method_args=(--client-routing none --distillation-weight 0)
  else
    method_args=(--client-routing relative_balanced_rank
                 --routing-keep-fraction 0.5 --routing-gate-threshold 0.8
                 --distillation-weight 0.25)
  fi
  "$python_bin" "$runner" --client-models "${clients[@]}" --seed "$seed" \
    --rounds 2 --client-steps 40 --server-steps 40 \
    --batch-size 8 --server-batch-size 8 --anchor-score-batch-size 8 \
    --learning-rate 0.001 --server-learning-rate 0.0001 \
    --teacher-temperature 0.1 --anchor-supervision-weight 1.0 \
    --teacher-aggregation uniform --conflict-ratio 0.4 --off-task-ratio 0.2 \
    --anchor-per-class 32 --max-length 128 --evaluation-split official_test \
    "${method_args[@]}" --output "$output" >"$log" 2>&1 &
  while (( $(jobs -pr | wc -l) >= max_jobs )); do wait -n; done
}

for seed in 145 146 147 148 149; do
  launch "$seed" anchor_only
  launch "$seed" routed
done
wait

count=$(find kafa_script/analysis_outputs -maxdepth 1 \
  -name 'replication_*_seed*.json' | wc -l)
[[ "$count" -eq 10 ]] || { echo "Expected 10 replication artifacts, found $count" >&2; exit 1; }
echo "Completed $count AG News post-observation seed-replication artifacts"
