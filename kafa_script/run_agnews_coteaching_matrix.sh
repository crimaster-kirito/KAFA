#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
max_jobs="${MAX_JOBS:-5}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

launch() {
  local family="$1" model="$2" method="$3" condition="$4" conflict="$5" off_task="$6" forget="$7" seed="$8"
  local output="kafa_script/analysis_outputs/agnews_coteach_${family}_${method}_${condition}_seed${seed}.json"
  local log="/tmp/agnews_coteach_${family}_${method}_${condition}_seed${seed}.log"
  if [[ -s "$output" ]]; then return; fi
  if [[ "$method" == naive ]]; then
    "$python_bin" kafa_script/run_agnews_mixture_dev.py \
      --model "$model" --seed "$seed" --conflict-ratio "$conflict" \
      --off-task-ratio "$off_task" --output "$output" >"$log" 2>&1 &
  else
    "$python_bin" kafa_script/run_agnews_coteaching_dev.py \
      --model "$model" --seed "$seed" --conflict-ratio "$conflict" \
      --off-task-ratio "$off_task" --forget-rate "$forget" \
      --output "$output" >"$log" 2>&1 &
  fi
  while (( $(jobs -pr | wc -l) >= max_jobs )); do wait -n; done
}

for seed in 75 76 77; do
  for spec in 'clean 0 0 0' 'conflict40 .4 0 .4' 'joint .4 .2 .6'; do
    read -r condition conflict off_task forget <<<"$spec"
    for method in naive coteach; do
      launch qwen .cache/models/Qwen2.5-0.5B "$method" "$condition" "$conflict" "$off_task" "$forget" "$seed"
      launch smol .cache/models/SmolLM2-360M "$method" "$condition" "$conflict" "$off_task" "$forget" "$seed"
      launch tiny .cache/models/TinyLlama-1.1B-Chat-v1.0 "$method" "$condition" "$conflict" "$off_task" "$forget" "$seed"
    done
  done
done
wait

count=$(find kafa_script/analysis_outputs -maxdepth 1 -name 'agnews_coteach_*.json' | wc -l)
[[ "$count" -eq 54 ]] || { echo "Expected 54 artifacts, found $count" >&2; exit 1; }
echo "Completed $count Co-Teaching matrix artifacts"
