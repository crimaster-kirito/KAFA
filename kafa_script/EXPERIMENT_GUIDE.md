# KAFA Experiment Guide

## What this adds

- A single experiment matrix generator for `local-only`, `RawFedAvg`, `OffsiteTuning`, `KAFA-full`, and KAFA ablations.
- Stronger comparison baselines are included as well: `FedProx`, `FedOpt`, `pFedMe`, and `Ditto`.
- A report generator that reads real logs and writes a clearly labeled `synthetic` preview separately.
- LLM dataset loading for `squad`, `imdb`, and `agnews` in addition to the existing `gsm8k`, `alpaca`, and `dolly-15k` paths.

## Required paths

- Server model: defaults to `models/llama-2-7b-hf`.
- Client model: pass `KAFA_CLIENT_MODEL` if you want a compressed client model.
- Data root: defaults to `data`.
- Cache root: pass `KAFA_CACHE_DIR` if you want to separate Hugging Face cache files.
- Generated configs now keep repo-relative paths whenever the target lives under this workspace, so the emitted YAML can be moved to the server without hardcoded `F:\...` paths.
- `kafa_script/experiment_tooling.py` also writes `llm.paths.*` hints into the config for server-side inspection and debugging.
- Running the report generator now also creates plots under `kafa_script/analysis_outputs/plots/`.

## Typical usage

Dry run and write configs:

```bash
python kafa_script/run_quick_benchmark.py --profile paper --write-configs
```

Execute a smaller smoke test:

```bash
python kafa_script/run_quick_benchmark.py --profile smoke --write-configs --execute
```

Generate a report from existing logs:

```bash
python kafa_script/generate_quick_comparison.py
```

The generated report will include links to:

- `final_loss_by_task.png`
- `communication_tradeoff.png`
- `convergence_*.png` for each task

## Notes

- The `w/o projection` ablation expects a client model path that is architecture-compatible with the server-side LoRA/state dict handling. If it is missing, the matrix generator will still emit the config but mark it as needing a client model.
- Synthetic numbers are for trend checking only. Do not use them as paper results.
