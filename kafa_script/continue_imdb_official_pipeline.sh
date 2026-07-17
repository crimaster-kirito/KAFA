#!/usr/bin/env bash
set -euo pipefail

# Execution-only, fail-closed handoff between the already running original
# IMDb matrix, the frozen matched-control matrix, and the AG secondary matrix.
# This script changes no scientific protocol or runner.
cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-.venv/bin/python}"
original_pid="${1:-}"
poll_seconds="${POLL_SECONDS:-30}"
state_root="kafa_script/analysis_outputs"
lock_dir="$state_root/.imdb_official_pipeline.lock"

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || (( poll_seconds > 60 )); then
  echo "POLL_SECONDS must be an integer from 1 to 60" >&2
  exit 2
fi

mkdir -p "$state_root"
if ! mkdir "$lock_dir" 2>/dev/null; then
  owner="$(cat "$lock_dir/pid" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
    echo "IMDb pipeline coordinator already running as PID $owner" >&2
    exit 73
  fi
  rm -f "$lock_dir/pid"
  rmdir "$lock_dir"
  mkdir "$lock_dir"
fi
printf '%s\n' "$$" >"$lock_dir/pid"
trap 'rm -f "$lock_dir/pid"; rmdir "$lock_dir" 2>/dev/null || true' EXIT

if [[ -n "$original_pid" ]] && kill -0 "$original_pid" 2>/dev/null; then
  original_command="$(ps -p "$original_pid" -o args=)"
  if [[ "$original_command" != *"run_imdb_server_role_reversal_matrix.sh official_evaluation"* ]]; then
    echo "PID $original_pid is not the expected original IMDb matrix" >&2
    exit 64
  fi
  echo "Waiting for original IMDb matrix PID $original_pid"
  while kill -0 "$original_pid" 2>/dev/null; do
    sleep "$poll_seconds"
  done
else
  while pgrep -f 'run_imdb_server_role_reversal_matrix[.]sh official_evaluation' >/dev/null; do
    echo "Waiting for active original IMDb matrix"
    sleep "$poll_seconds"
  done
fi

original_dir="$state_root/imdb_server_role_reversal/official_evaluation"
original_count="$(find "$original_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
if [[ "$original_count" -ne 40 ]]; then
  echo "Original IMDb matrix ended with $original_count/40 records; stopping pipeline" >&2
  exit 65
fi
"$python_bin" kafa_script/verify_imdb_server_role_reversal.py \
  --phase official_evaluation --output-dir "$original_dir"
"$python_bin" kafa_script/summarize_imdb_server_role_reversal.py \
  --phase official_evaluation --input-dir "$original_dir" \
  --output "$state_root/IMDB_SERVER_ROLE_REVERSAL_OFFICIAL.md" \
  >/tmp/kafa_pipeline_imdb_original_summary.log

matched_dir="$state_root/imdb_matched_control_addendum/official_evaluation"
if [[ -d "$matched_dir" ]]; then
  matched_count="$(find "$matched_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
else
  matched_count=0
fi
if [[ "$matched_count" -lt 30 ]]; then
  MAX_JOBS=2 bash kafa_script/run_imdb_matched_control_addendum_resource_aware.sh \
    official_evaluation
fi
"$python_bin" kafa_script/verify_imdb_matched_control_addendum.py \
  --phase official_evaluation --output-dir "$matched_dir"
"$python_bin" kafa_script/verify_imdb_routed_equivalence.py \
  --phase official_evaluation --original-dir "$original_dir" \
  --matched-dir "$matched_dir"
"$python_bin" kafa_script/summarize_imdb_matched_control_addendum.py \
  --phase official_evaluation --input-dir "$matched_dir" \
  --output "$state_root/IMDB_MATCHED_CONTROL_OFFICIAL.md" \
  >/tmp/kafa_pipeline_imdb_matched_summary.log
"$python_bin" kafa_script/generate_imdb_paper_assets.py

ag_dir="$state_root/agnews_server_ablation/secondary_holdout"
if [[ -d "$ag_dir" ]]; then
  ag_count="$(find "$ag_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
else
  ag_count=0
fi
if [[ "$ag_count" -lt 40 ]]; then
  MAX_JOBS=2 bash kafa_script/run_agnews_server_ablation_matrix.sh secondary_holdout
fi
"$python_bin" kafa_script/verify_agnews_server_ablation.py \
  --phase secondary_holdout --output-dir "$ag_dir"
"$python_bin" kafa_script/summarize_agnews_server_ablation.py \
  --phase secondary_holdout --input-dir "$ag_dir" \
  --output "$state_root/AGNEWS_SERVER_ABLATION_SECONDARY.md" \
  >/tmp/kafa_pipeline_ag_secondary_summary.log

echo "IMDb original, IMDb matched-control, and AG secondary matrices completed and verified"
