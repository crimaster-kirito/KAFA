#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p kafa_script/analysis_outputs

max_jobs="${MAX_JOBS:-4}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

launch() {
  local family="$1" model="$2" method="$3" condition="$4" seed="$5"
  local conflict=0 off_task=0
  if [[ "$condition" == severe ]]; then
    conflict=.4
    off_task=.2
  fi
  local output="kafa_script/analysis_outputs/gsm8k_crossfamily_final_${family}_${method}_${condition}_seed${seed}.json"
  local log="/tmp/gsm8k_crossfamily_final_${family}_${method}_${condition}_seed${seed}.log"
  if [[ -s "$output" ]]; then
    return
  fi
  if [[ "$method" == naive ]]; then
    "$python_bin" kafa_script/run_gsm8k_mixture_dev.py \
      --model "$model" --seed "$seed" --evaluation-split official_test \
      --conflict-ratio "$conflict" --off-task-ratio "$off_task" \
      --output "$output" >"$log" 2>&1 &
  else
    "$python_bin" kafa_script/run_gsm8k_anchor_teacher_router_dev.py \
      --client-model "$model" --seed "$seed" --evaluation-split official_test \
      --conflict-ratio "$conflict" --off-task-ratio "$off_task" \
      --output "$output" >"$log" 2>&1 &
  fi
  while (( $(jobs -pr | wc -l) >= max_jobs )); do
    wait -n
  done
}

for seed in 164 165 166; do
  for condition in clean severe; do
    for method in naive teacher; do
      launch smol .cache/models/SmolLM2-360M "$method" "$condition" "$seed"
      launch tiny .cache/models/TinyLlama-1.1B-Chat-v1.0 "$method" "$condition" "$seed"
    done
  done
done
wait

count=$(find kafa_script/analysis_outputs -maxdepth 1 -name 'gsm8k_crossfamily_final_*.json' | wc -l)
if [[ "$count" -ne 24 ]]; then
  echo "Expected 24 final artifacts, found $count" >&2
  exit 1
fi
echo "Completed $count cross-family final artifacts"
