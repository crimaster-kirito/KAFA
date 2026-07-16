#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
max_jobs="${MAX_JOBS:-4}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

launch() {
  local family="$1" model="$2" selector="$3" condition="$4" seed="$5"
  local conflict=0 off_task=0
  if [[ "$condition" == severe ]]; then conflict=.4; off_task=.2; fi
  local output="kafa_script/analysis_outputs/gsm8k_matched_${family}_${selector}_${condition}_seed${seed}.json"
  local log="/tmp/gsm8k_matched_${family}_${selector}_${condition}_seed${seed}.log"
  if [[ -s "$output" ]]; then return; fi
  "$python_bin" kafa_script/run_gsm8k_selection_baseline_dev.py \
    --client-model "$model" --selector "$selector" --seed "$seed" \
    --eval-size 512 --conflict-ratio "$conflict" --off-task-ratio "$off_task" \
    --output "$output" >"$log" 2>&1 &
  while (( $(jobs -pr | wc -l) >= max_jobs )); do wait -n; done
}

for seed in 61 62 63; do
  for condition in clean severe; do
    for selector in numeric_random client_small_loss; do
      launch smol .cache/models/SmolLM2-360M "$selector" "$condition" "$seed"
      launch tiny .cache/models/TinyLlama-1.1B-Chat-v1.0 "$selector" "$condition" "$seed"
    done
  done
done
wait

count=$(find kafa_script/analysis_outputs -maxdepth 1 -name 'gsm8k_matched_*.json' | wc -l)
[[ "$count" -eq 24 ]] || { echo "Expected 24 artifacts, found $count" >&2; exit 1; }
echo "Completed $count matched baseline artifacts"
