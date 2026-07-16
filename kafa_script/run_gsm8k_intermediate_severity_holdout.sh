#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
max_jobs="${MAX_JOBS:-4}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

launch() {
  local family="$1" model="$2" method="$3" level="$4" conflict="$5" off_task="$6" seed="$7"
  local output="kafa_script/analysis_outputs/gsm8k_severity_${family}_${method}_${level}_seed${seed}.json"
  local log="/tmp/gsm8k_severity_${family}_${method}_${level}_seed${seed}.log"
  if [[ -s "$output" ]]; then return; fi
  if [[ "$method" == naive ]]; then
    "$python_bin" kafa_script/run_gsm8k_mixture_dev.py \
      --model "$model" --seed "$seed" --evaluation-split secondary_holdout \
      --conflict-ratio "$conflict" --off-task-ratio "$off_task" \
      --eval-batch-size 16 --skip-initial-eval --output "$output" >"$log" 2>&1 &
  else
    "$python_bin" kafa_script/run_gsm8k_anchor_teacher_router_dev.py \
      --client-model "$model" --seed "$seed" --evaluation-split secondary_holdout \
      --conflict-ratio "$conflict" --off-task-ratio "$off_task" \
      --eval-batch-size 16 --skip-initial-eval --output "$output" >"$log" 2>&1 &
  fi
  while (( $(jobs -pr | wc -l) >= max_jobs )); do wait -n; done
}

for seed in 72 73 74; do
  for spec in 'clean 0 0' 'medium .2 .1' 'severe .4 .2'; do
    read -r level conflict off_task <<<"$spec"
    for method in naive teacher; do
      launch qwen .cache/models/Qwen2.5-0.5B "$method" "$level" "$conflict" "$off_task" "$seed"
      launch smol .cache/models/SmolLM2-360M "$method" "$level" "$conflict" "$off_task" "$seed"
      launch tiny .cache/models/TinyLlama-1.1B-Chat-v1.0 "$method" "$level" "$conflict" "$off_task" "$seed"
    done
  done
done
wait

count=$(find kafa_script/analysis_outputs -maxdepth 1 -name 'gsm8k_severity_*.json' | wc -l)
[[ "$count" -eq 54 ]] || { echo "Expected 54 artifacts, found $count" >&2; exit 1; }
echo "Completed $count intermediate-severity artifacts"
