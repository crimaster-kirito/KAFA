#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

.venv/bin/python kafa_script/generate_paper_figures.py
.venv/bin/python kafa_script/summarize_hausa_natural_noise.py --corpus hausa \
  --output kafa_script/analysis_outputs/HAUSA_NATURAL_NOISE_RESULTS.md
.venv/bin/python kafa_script/summarize_hausa_natural_noise.py --corpus yoruba \
  --output kafa_script/analysis_outputs/YORUBA_NATURAL_NOISE_RESULTS.md
.venv/bin/python kafa_script/summarize_agnews_coteaching.py \
  --output kafa_script/analysis_outputs/AGNEWS_COTEACHING_RESULTS.md
.venv/bin/python kafa_script/summarize_final_heterogeneous.py \
  --output kafa_script/analysis_outputs/FINAL_HETEROGENEOUS_RESULTS.md
replication_count="$(find kafa_script/analysis_outputs -maxdepth 1 \
  -name 'replication_*_seed*.json' | wc -l)"
if [[ "$replication_count" -eq 10 ]]; then
  .venv/bin/python kafa_script/summarize_agnews_post_observation_replication.py \
    --output kafa_script/analysis_outputs/AGNEWS_POST_OBSERVATION_SEED_REPLICATION.md
fi
.venv/bin/python kafa_script/summarize_trec_server_transfer.py \
  --phase development \
  --input-dir kafa_script/analysis_outputs/trec_server/development \
  --output kafa_script/analysis_outputs/TREC_SERVER_DEVELOPMENT.md
.venv/bin/python kafa_script/summarize_trec_server_transfer.py \
  --phase official \
  --input-dir kafa_script/analysis_outputs/trec_server/official \
  --output kafa_script/analysis_outputs/TREC_SERVER_OFFICIAL_RESULTS.md
.venv/bin/python kafa_script/summarize_trec_server_ablation.py \
  --phase development \
  --input-dir kafa_script/analysis_outputs/trec_server_ablation/development \
  --output kafa_script/analysis_outputs/TREC_SERVER_ABLATION_DEVELOPMENT.md \
  >/tmp/kafa_trec_ablation_development_summary.log
.venv/bin/python kafa_script/summarize_trec_server_ablation.py \
  --phase official \
  --input-dir kafa_script/analysis_outputs/trec_server_ablation/official_reused_test \
  --output kafa_script/analysis_outputs/TREC_SERVER_POST_PRIMARY_ABLATION.md \
  >/tmp/kafa_trec_ablation_official_summary.log
.venv/bin/python kafa_script/summarize_imdb_server_role_reversal.py \
  --phase development \
  --input-dir kafa_script/analysis_outputs/imdb_server_role_reversal/development \
  --output kafa_script/analysis_outputs/IMDB_SERVER_ROLE_REVERSAL_DEVELOPMENT.md \
  >/tmp/kafa_imdb_development_summary.log
.venv/bin/python kafa_script/summarize_imdb_matched_control_addendum.py \
  --phase development \
  --input-dir kafa_script/analysis_outputs/imdb_matched_control_addendum/development \
  --output kafa_script/analysis_outputs/IMDB_MATCHED_CONTROL_DEVELOPMENT.md \
  >/tmp/kafa_imdb_matched_development_summary.log

imdb_official_dir="kafa_script/analysis_outputs/imdb_server_role_reversal/official_evaluation"
imdb_official_count="$(find "$imdb_official_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
if [[ "$imdb_official_count" -eq 40 ]]; then
  .venv/bin/python kafa_script/summarize_imdb_server_role_reversal.py \
    --phase official_evaluation \
    --input-dir "$imdb_official_dir" \
    --output kafa_script/analysis_outputs/IMDB_SERVER_ROLE_REVERSAL_OFFICIAL.md \
    >/tmp/kafa_imdb_official_summary.log
fi

imdb_matched_dir="kafa_script/analysis_outputs/imdb_matched_control_addendum/official_evaluation"
if [[ -d "$imdb_matched_dir" ]]; then
  imdb_matched_count="$(find "$imdb_matched_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
else
  imdb_matched_count=0
fi
if [[ "$imdb_matched_count" -eq 30 ]]; then
  .venv/bin/python kafa_script/summarize_imdb_matched_control_addendum.py \
    --phase official_evaluation \
    --input-dir "$imdb_matched_dir" \
    --output kafa_script/analysis_outputs/IMDB_MATCHED_CONTROL_OFFICIAL.md \
    >/tmp/kafa_imdb_matched_official_summary.log
fi
if [[ "$imdb_official_count" -eq 40 && "$imdb_matched_count" -eq 30 ]]; then
  .venv/bin/python kafa_script/generate_imdb_paper_assets.py
fi

imdb_selector_dir="kafa_script/analysis_outputs/imdb_selector_component_ablation/official_evaluation"
if [[ -d "$imdb_selector_dir" ]]; then
  imdb_selector_count="$(find "$imdb_selector_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
else
  imdb_selector_count=0
fi
if [[ "$imdb_selector_count" -eq 30 ]]; then
  .venv/bin/python kafa_script/verify_imdb_selector_component_ablation.py \
    --output-dir "$imdb_selector_dir" \
    --protocol kafa_script/imdb_selector_component_ablation_protocol_v1.json
  .venv/bin/python kafa_script/summarize_imdb_selector_component_ablation.py \
    --input-dir "$imdb_selector_dir" \
    --output kafa_script/analysis_outputs/IMDB_SELECTOR_COMPONENT_ABLATION_OFFICIAL.md \
    >/tmp/kafa_imdb_selector_component_summary.log
fi

trec_audit_dir="kafa_script/analysis_outputs/trec_server_ablation/official_reused_test"
ag_audit_dir="kafa_script/analysis_outputs/agnews_server_ablation/secondary_holdout"
trec_audit_count="$(find "$trec_audit_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
if [[ -d "$ag_audit_dir" ]]; then
  ag_audit_count="$(find "$ag_audit_dir" -maxdepth 1 -name 'seed*.json' | wc -l)"
else
  ag_audit_count=0
fi
if [[ "$trec_audit_count" -eq 40 && "$imdb_official_count" -eq 40 && \
      "$imdb_matched_count" -eq 30 && "$ag_audit_count" -eq 40 ]]; then
  .venv/bin/python kafa_script/generate_server_audit_assets.py
  .venv/bin/python kafa_script/generate_server_audit_assets.py --verify-only
elif compgen -G 'paper/generated/server_audit_*' >/dev/null; then
  echo "Refusing stale server_audit assets: the current 150-record input set is incomplete." >&2
  exit 1
fi

cd paper/figures
pdflatex -interaction=nonstopmode -halt-on-error framework_source.tex >/tmp/kafa_framework_build.log
mv framework_source.pdf framework.pdf
pdflatex -interaction=nonstopmode -halt-on-error evidence_source.tex >/tmp/kafa_evidence_build.log
mv evidence_source.pdf evidence.pdf

cd ..
rm -f main.aux main.bbl main.blg main.log main.out main.pdf
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/kafa_main_pass1.log
bibtex main >/tmp/kafa_bibtex.log
for pass in 2 3 4; do
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >"/tmp/kafa_main_pass${pass}.log"
done
pdflatex -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex \
  >/tmp/kafa_checklist_build.log
rm -f supplement.aux supplement.bbl supplement.blg supplement.log supplement.pdf
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex \
  >/tmp/kafa_supplement_pass1.log
bibtex supplement >/tmp/kafa_supplement_bibtex.log
for pass in 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error supplement.tex \
    >"/tmp/kafa_supplement_pass${pass}.log"
done

cd "$repo_root"
.venv/bin/python paper/verify_submission.py
.venv/bin/python paper/verify_reported_numbers.py
.venv/bin/python kafa_script/verify_trec_server_protocol.py
.venv/bin/python kafa_script/verify_trec_server_ablation.py \
  --phase development \
  --output-dir kafa_script/analysis_outputs/trec_server_ablation/development
.venv/bin/python kafa_script/verify_trec_server_ablation.py \
  --phase official_evaluation \
  --output-dir kafa_script/analysis_outputs/trec_server_ablation/official_reused_test
.venv/bin/python kafa_script/verify_imdb_server_role_reversal.py \
  --phase development \
  --output-dir kafa_script/analysis_outputs/imdb_server_role_reversal/development
.venv/bin/python kafa_script/verify_imdb_matched_control_addendum.py \
  --phase development \
  --output-dir kafa_script/analysis_outputs/imdb_matched_control_addendum/development
.venv/bin/python kafa_script/verify_imdb_routed_equivalence.py \
  --phase development \
  --original-dir kafa_script/analysis_outputs/imdb_server_role_reversal/development \
  --matched-dir kafa_script/analysis_outputs/imdb_matched_control_addendum/development
if [[ "$imdb_official_count" -eq 40 ]]; then
  .venv/bin/python kafa_script/verify_imdb_server_role_reversal.py \
    --phase official_evaluation --output-dir "$imdb_official_dir"
fi
if [[ "$imdb_matched_count" -eq 30 ]]; then
  .venv/bin/python kafa_script/verify_imdb_matched_control_addendum.py \
    --phase official_evaluation --output-dir "$imdb_matched_dir"
  .venv/bin/python kafa_script/verify_imdb_routed_equivalence.py \
    --phase official_evaluation \
    --original-dir "$imdb_official_dir" --matched-dir "$imdb_matched_dir"
fi
if [[ "$imdb_selector_count" -eq 30 ]]; then
  .venv/bin/python kafa_script/verify_imdb_selector_component_ablation.py \
    --output-dir "$imdb_selector_dir" \
    --protocol kafa_script/imdb_selector_component_ablation_protocol_v1.json
fi
.venv/bin/python -m pytest -q tests/test_capability_router.py \
  tests/test_lora_projection_diagnostics.py tests/test_statistical_summarizers.py \
  tests/test_classification_task.py tests/test_trec_server_manifest.py \
  tests/test_trec_server_protocol.py \
  tests/test_generic_classification_distillation.py \
  tests/test_summarize_trec_server_transfer.py \
  tests/test_trec_server_matrix.py \
  tests/test_trec_server_ablation.py \
  tests/test_agnews_server_ablation.py \
  tests/test_imdb_server_role_reversal.py \
  tests/test_imdb_matched_control_addendum.py \
  tests/test_imdb_matched_control_resource_scheduler.py \
  tests/test_imdb_selector_component_ablation.py \
  tests/test_verify_imdb_selector_component_ablation.py \
  tests/test_summarize_imdb_selector_component_ablation.py \
  tests/test_generate_imdb_paper_assets.py \
  tests/test_generate_server_audit_assets.py
