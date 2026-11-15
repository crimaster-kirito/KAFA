# KAFA Experiment Report

This report combines real log summaries with a clearly labeled `synthetic` sanity-check preview. Synthetic rows are for preflight trend validation only and must not be treated as paper results.

## Real Runs
- Real runs found: 12 rows across 1 tasks and 12 methods.
- Evidence `log_only_unverified`: 12 runs.
- `gsm8k` best method so far: `KAFA-full` with mean final test loss `0.2180`

## Comparison Set
- `local-only`: checks the lower bound when each client keeps its own model.
- `RawFedAvg`: measures whether plain federated averaging can transfer useful knowledge without alignment.
- `FedProx`: adds proximal regularization to test whether a simple stability prior closes the gap.
- `FedOpt`: checks whether server-side adaptive optimization improves convergence over vanilla averaging.
- `OffsiteTuning`: isolates the effect of server-side emulator alignment and layer dropping.
- `pFedMe` and `Ditto`: personalized FL baselines that are often competitive on heterogeneous data.
- `KAFA-full`: combines projection, instruction anchoring, relevance filtering, and distillation.
- `w/o projection / instruction / distill / relevance`: ablations to identify which component drives gains.

## Synthetic Preview
- `alpaca`: expected best method is `KAFA-full` with final test loss around `0.3003` and std `0.0010`
- `gsm8k`: expected best method is `KAFA-full` with final test loss around `0.2575` and std `0.0017`

## Expected Ordering
- On `gsm8k`, the strongest expected gap is between `KAFA-full` and the ablations, especially `w/o instruction` and `w/o relevance`.
- On `alpaca` and `dolly-15k`, `KAFA-full` should still lead, but the main visible advantage is usually faster convergence and lower variance.
- On `squad`, `imdb`, and `agnews`, absolute gaps are often smaller, so communication cost and stability become part of the argument.
- If `w/o projection` beats `KAFA-full` consistently, treat that as a configuration or model-shape mismatch before trusting the numbers.

## Plots
- No plots were generated.

## Communication Estimate
- `gsm8k` / `Local-only`: 0.0 MB per round, 0.0 MB total
- `gsm8k` / `RawFedAvg`: 220.0 MB per round, 6600.0 MB total
- `gsm8k` / `FedProx`: 220.0 MB per round, 6600.0 MB total
- `gsm8k` / `FedOpt`: 220.0 MB per round, 6600.0 MB total
- `gsm8k` / `OffsiteTuning`: 175.0 MB per round, 5250.0 MB total
- `gsm8k` / `pFedMe`: 220.0 MB per round, 6600.0 MB total
- `gsm8k` / `Ditto`: 220.0 MB per round, 6600.0 MB total
- `gsm8k` / `KAFA-full`: 95.0 MB per round, 2850.0 MB total
- `gsm8k` / `w/o projection`: 220.0 MB per round, 6600.0 MB total
- `gsm8k` / `w/o instruction`: 95.0 MB per round, 2850.0 MB total
- `gsm8k` / `w/o distill`: 95.0 MB per round, 2850.0 MB total
- `gsm8k` / `w/o relevance`: 95.0 MB per round, 2850.0 MB total
- `alpaca` / `Local-only`: 0.0 MB per round, 0.0 MB total
- `alpaca` / `RawFedAvg`: 220.0 MB per round, 4400.0 MB total
- `alpaca` / `FedProx`: 220.0 MB per round, 4400.0 MB total
- `alpaca` / `FedOpt`: 220.0 MB per round, 4400.0 MB total
- `alpaca` / `OffsiteTuning`: 175.0 MB per round, 3500.0 MB total
- `alpaca` / `pFedMe`: 220.0 MB per round, 4400.0 MB total
- `alpaca` / `Ditto`: 220.0 MB per round, 4400.0 MB total
- `alpaca` / `KAFA-full`: 95.0 MB per round, 1900.0 MB total
- `alpaca` / `w/o projection`: 220.0 MB per round, 4400.0 MB total
- `alpaca` / `w/o instruction`: 95.0 MB per round, 1900.0 MB total
- `alpaca` / `w/o distill`: 95.0 MB per round, 1900.0 MB total
- `alpaca` / `w/o relevance`: 95.0 MB per round, 1900.0 MB total

## Interpretation
- Reasoning tasks should benefit most from the full KAFA stack; instruction augmentation and relevance filtering should dominate.
- Generation tasks should show a smoother gap: KAFA should still lead, but convergence stability is the main advantage.
- Extraction and classification tasks usually have smaller absolute gains, so communication efficiency and variance reduction matter more.
- If real logs reverse these trends, check model paths, dataset split logic, and the client projection / mapper settings first.