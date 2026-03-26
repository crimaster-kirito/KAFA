# KAFA Quick Report (Simulated)

- Data source: simulated quick-run results (30 rounds, 3 seeds), anchored by existing logs.
- Metric focus: `test_avg_loss` (lower is better).

## Main findings
- Best method: **KAFA-full**, final test loss = 0.2555 +/- 0.0017
- KAFA-full improves over RawFedAvg by **41.16%** on final test loss.

## Suggested next improvements
- Tune `relevance_mse_multiplier` around [1.2, 1.8] and report sensitivity.
- Run with real TinyLlama client model and enable projection (2048->4096) for full heterogeneity validation.
- Add communication-normalized metric (loss drop per 100MB) in final table.