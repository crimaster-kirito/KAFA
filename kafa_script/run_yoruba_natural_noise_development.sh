#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
max_jobs="${MAX_JOBS:-5}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
data_root='.cache/data/natural_noise/yoruba/COMPRISE_Data13_Y-BBCTopics_V1.0'
noise_rate='0.3328358208955224'

launch() {
  local family="$1" model="$2" method="$3" seed="$4"
  local output="kafa_script/analysis_outputs/yoruba_natural_${family}_${method}_dev_seed${seed}.json"
  local log="/tmp/yoruba_natural_${family}_${method}_dev_seed${seed}.log"
  if [[ -s "$output" ]]; then return; fi
  "$python_bin" kafa_script/run_natural_noise_causal_classification.py \
    --model "$model" --data-root "$data_root" --method "$method" \
    --dataset-name 'COMPRISE_Data13_Y-BBCTopics_V1.0' \
    --dataset-doi '10.5281/zenodo.5998515' --forget-rate "$noise_rate" \
    --seed "$seed" --output "$output" >"$log" 2>&1 &
  while (( $(jobs -pr | wc -l) >= max_jobs )); do wait -n; done
}

for seed in 84 85 86; do
  for method in naive coteach; do
    launch qwen .cache/models/Qwen2.5-0.5B "$method" "$seed"
    launch smol .cache/models/SmolLM2-360M "$method" "$seed"
    launch tiny .cache/models/TinyLlama-1.1B-Chat-v1.0 "$method" "$seed"
  done
done
wait

count=$(find kafa_script/analysis_outputs -maxdepth 1 -name 'yoruba_natural_*_dev_seed*.json' | wc -l)
[[ "$count" -eq 18 ]] || { echo "Expected 18 artifacts, found $count" >&2; exit 1; }
echo "Completed $count Yoruba natural-noise development artifacts"
