#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

phase="${PHASE:-development}"
protocol="kafa_script/trec_server_post_primary_ablation_v1.json"
runner="kafa_script/run_trec_server_ablation.py"
case "$phase" in
  development)
    split="development"
    output_dir="kafa_script/analysis_outputs/trec_server_ablation/development"
    seeds=(258 259 260)
    ;;
  official_evaluation)
    split="official_test"
    output_dir="kafa_script/analysis_outputs/trec_server_ablation/official_reused_test"
    seeds=(261 262 263 264 265 266 267 268 269 270)
    ;;
  *)
    echo "PHASE must be development or official_evaluation" >&2
    exit 2
    ;;
esac

methods=(
  public_anchor_only
  clean_unfiltered_distillation
  severe_unfiltered_distillation
  severe_routed_distillation
)
mkdir -p "$output_dir" "$output_dir/logs" "$output_dir/failures"

protocol_sha="$(sha256sum "$protocol" | awk '{print $1}')"
runner_sha="$(sha256sum "$runner" | awk '{print $1}')"
expected_runner_sha="$(.venv/bin/python -c \
  'import json; print(json.load(open("kafa_script/trec_server_post_primary_ablation_v1.json"))["implementation"]["ablation_runner_file_sha256"])')"
if [[ "$runner_sha" != "$expected_runner_sha" ]]; then
  echo "ablation runner differs from frozen protocol" >&2
  exit 1
fi

record_matches() {
  local path="$1" seed="$2" method="$3"
  [[ -f "$path" ]] || return 1
  .venv/bin/python - "$path" "$seed" "$method" "$split" \
      "$protocol_sha" "$runner_sha" <<'PY'
import json, sys
path, seed, method, split, protocol_sha, runner_sha = sys.argv[1:]
try:
    row = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
ok = (
    row.get("seed") == int(seed)
    and row.get("method") == method
    and row.get("evaluation_split") == split
    and row.get("protocol_sha256") == protocol_sha
    and row.get("runner_file_sha256") == runner_sha
)
raise SystemExit(0 if ok else 1)
PY
}

run_job() {
  local seed="$1" method="$2"
  local final="$output_dir/seed${seed}_${method}.json"
  local tmp="$final.tmp.$$"
  local attempted_at
  attempted_at="$(date -u +%Y%m%dT%H%M%SZ)"
  local log="$output_dir/logs/seed${seed}_${method}.attempt_${attempted_at}.log"
  if record_matches "$final" "$seed" "$method"; then
    echo "skip verified $phase seed=$seed method=$method"
    return 0
  fi
  rm -f "$tmp"
  if .venv/bin/python "$runner" \
      --protocol "$protocol" \
      --method "$method" \
      --evaluation-split "$split" \
      --seed "$seed" \
      --output "$tmp" >"$log" 2>&1; then
    mv "$tmp" "$final"
    echo "completed $phase seed=$seed method=$method"
  else
    local failed="$output_dir/failures/seed${seed}_${method}.failed_${attempted_at}.json"
    [[ -f "$tmp" ]] && mv "$tmp" "$failed"
    local failure_log="$output_dir/failures/$(basename "$log")"
    mv "$log" "$failure_log"
    echo "failed $phase seed=$seed method=$method; log=$failure_log" >&2
    return 1
  fi
}

# Anchor-only can transiently occupy nearly the whole device during its final
# 32x6-candidate evaluation, so it is strictly serial and isolated.
for seed in "${seeds[@]}"; do
  run_job "$seed" public_anchor_only
done

# Client-path training can overlap, but every arm shares the same high-memory
# 32x6 final server evaluation.  A formal clean-arm OOM showed that two full
# jobs are unsafe when one evaluation overlaps the peer process, so all full
# jobs are serial.  This preserves the frozen scientific configuration.
for method in clean_unfiltered_distillation \
              severe_unfiltered_distillation; do
  for seed in "${seeds[@]}"; do
    run_job "$seed" "$method"
  done
done

# Two routed jobs can overlap during client work, but a final 32x6 server
# evaluation can coincide with the peer's server update and exhaust 48 GiB.
# A development OOM established that the full routed job must be serial.
for seed in "${seeds[@]}"; do
  run_job "$seed" severe_routed_distillation
done

.venv/bin/python kafa_script/verify_trec_server_ablation.py \
  --phase "$phase" --output-dir "$output_dir"
