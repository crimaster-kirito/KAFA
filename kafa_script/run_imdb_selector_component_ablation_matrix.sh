#!/usr/bin/env bash
set -euo pipefail

phase="${1:-official_evaluation}"
if [[ "$phase" != "official_evaluation" ]]; then
  echo "Only official_evaluation is defined for the post-hoc selector ablation" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-.venv/bin/python}"
runner="kafa_script/run_imdb_selector_component_ablation.py"
protocol="kafa_script/imdb_selector_component_ablation_protocol_v1.json"
output_dir="kafa_script/analysis_outputs/imdb_selector_component_ablation/official_evaluation"
log_dir="$output_dir/logs"
mkdir -p "$log_dir"

seeds=(321 322 323 324 325 326 327 328 329 330)
methods=(gradient_only_k loss_only_k margin_only_k)

for seed in "${seeds[@]}"; do
  for method in "${methods[@]}"; do
    output="$output_dir/seed${seed}_${method}.json"
    log="$log_dir/seed${seed}_${method}.log"
    if [[ -s "$output" ]]; then
      echo "SKIP existing selector-component artifact: $output"
      continue
    fi
    echo "RUN seed=$seed method=$method"
    "$python_bin" "$runner" \
      --protocol "$protocol" \
      --method "$method" \
      --seed "$seed" \
      --output "$output" >"$log" 2>&1
  done
done

"$python_bin" kafa_script/verify_imdb_selector_component_ablation.py \
  --output-dir "$output_dir" \
  --protocol "$protocol"
