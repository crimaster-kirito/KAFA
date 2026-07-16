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
  --output kafa_script/analysis_outputs/TREC_SERVER_ABLATION_DEVELOPMENT.md
.venv/bin/python kafa_script/summarize_trec_server_ablation.py \
  --phase official \
  --input-dir kafa_script/analysis_outputs/trec_server_ablation/official_reused_test \
  --output kafa_script/analysis_outputs/TREC_SERVER_POST_PRIMARY_ABLATION.md

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
.venv/bin/python -m pytest -q tests/test_capability_router.py \
  tests/test_lora_projection_diagnostics.py tests/test_statistical_summarizers.py \
  tests/test_classification_task.py tests/test_trec_server_manifest.py \
  tests/test_trec_server_protocol.py \
  tests/test_generic_classification_distillation.py \
  tests/test_summarize_trec_server_transfer.py \
  tests/test_trec_server_matrix.py \
  tests/test_trec_server_ablation.py
