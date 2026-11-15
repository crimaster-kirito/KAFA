# GSM8K KAFA Stable Ablation Plan

This plan evaluates the current stable KAFA design:

- heterogeneous LoRA projection with GQA-aware K/V mapping
- EMA-style large LoRA update
- client-space prototype filtering
- dirty supervision filtering

Run all commands from the repository root.

## 1. Stable Full

Purpose: current best stable backbone.

```bash
python federatedscope/main.py --cfg exp/kafa_matrix/gsm8k/kafa_stable_full/seed_42/config.yaml seed 42
```

Expected: test loss should stay stable through round 29; compare against the previous best around 1.20.

## 2. Without Client-Space Relevance

Purpose: measure whether client prototype relevance filtering helps beyond dirty-supervision filtering.

This keeps dirty filtering enabled but sets `relevance_filter_drop_ratio: 0.0`.

```bash
python federatedscope/main.py --cfg exp/kafa_matrix/gsm8k/kafa_wo_client_relevance/seed_42/config.yaml seed 42
```

Expected: if this is worse than Stable Full, client-space relevance filtering is useful.

## 3. Without Dirty Filter

Purpose: measure whether dirty-supervision filtering is the main contributor.

This keeps client-space relevance enabled but sets `enable_dirty_supervision_filter: false`.

```bash
python federatedscope/main.py --cfg exp/kafa_matrix/gsm8k/kafa_wo_dirty_filter/seed_42/config.yaml seed 42
```

Expected: if this is worse than Stable Full, dirty-supervision filtering is useful.

## Read Results

For each run, check:

- `eval_results.log`: round 4/9/14/19/24/29 and final best.
- `exp_print.log`: kept/filtered samples, dirty recall, clean samples filtered.
- no `nan`, no `RuntimeError`, no shape mismatch.

Primary metric:

```text
Final client_summarized_weighted_avg.test_avg_loss
```

Secondary metric:

```text
Best round test_avg_loss, usually from Round 24 in the current stable run.
```

