# AG News Server Post-Primary Four-Arm Ablation

Phase: **development**. Records: **12**. Protocol SHA-256: `9dd4d2696780eb67491dba440507639a9651dddbd991c8e94f892c09ee49b4ec`.
Summarizer SHA-256: `e3ab6aa488eacdfa563620bba26ef54841de77d2a5f38a2c0a4ab1fd42c133fa`. Verifier SHA-256: `ce160900de9cf24ac1f943eeea8caf1933af022289165d272123a3b1e7dbd69e`.

## Evidence boundary

The AG News primary and official-test seed replication were observed before this analysis was designed. The 4,000 balanced evaluation records were selected by a frozen content hash from the AG News training remainder and excluded from every client pool in this new runner. A reconstruction audit found that 303 of these IDs had been allocated to clients in at least one earlier server seed. This is therefore post-primary, run-disjoint same-task resampling evidence, not globally untouched data, external replication, or preregistration.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Label NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | 86.50 ± 1.27 | +86.32 [+82.82, +89.82] | +0.7518 [+0.6392, +0.8644] | +0.0356 [+0.0212, +0.0501] | +94.7 [+78.3, +111.2] | +16.08 [+16.08, +16.08] |
| Clean uniform | 86.29 ± 0.88 | +86.10 [+83.67, +88.52] | +0.7180 [+0.6395, +0.7965] | +0.0302 [+0.0184, +0.0420] | +129.0 [+115.6, +142.4] | +16.08 [+16.08, +16.08] |
| Severe uniform | 84.21 ± 0.92 | +83.97 [+81.39, +86.56] | +0.6950 [+0.6824, +0.7076] | +0.1063 [+0.0516, +0.1610] | +118.3 [+109.0, +127.7] | +16.08 [+16.08, +16.08] |
| Severe routed | 86.00 ± 1.15 | +85.83 [+82.65, +89.02] | +0.6964 [+0.6258, +0.7670] | +0.0336 [+0.0159, +0.0513] | +436.9 [+332.8, +541.0] | +16.08 [+16.08, +16.08] |

## Per-seed accuracy (%)

| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |
|---:|---:|---:|---:|---:|
| 271 | 85.37 | 85.37 | 83.50 | 84.75 |
| 272 | 87.88 | 87.12 | 85.25 | 87.00 |
| 273 | 86.25 | 86.37 | 83.88 | 86.25 |

## Complete per-seed scalar secondary values

| Seed | Arm | Macro-F1 | Label NLL | ECE | Tensor bytes | Wall s | Peak GiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 271 | Anchor only | 85.07 | 0.7241 | 0.0364 | 0 | 100.9 | 16.08 |
| 271 | Clean uniform | 85.08 | 0.6967 | 0.0333 | 21750 | 135.1 | 16.08 |
| 271 | Severe uniform | 83.19 | 0.6989 | 0.1126 | 21750 | 122.4 | 16.08 |
| 271 | Severe routed | 84.45 | 0.6826 | 0.0302 | 21750 | 467.2 | 16.08 |
| 272 | Anchor only | 87.84 | 0.7272 | 0.0295 | 0 | 95.4 | 16.08 |
| 272 | Clean uniform | 87.03 | 0.7030 | 0.0325 | 21750 | 126.9 | 16.08 |
| 272 | Severe uniform | 85.15 | 0.6968 | 0.1245 | 21750 | 117.4 | 16.08 |
| 272 | Severe routed | 86.97 | 0.6774 | 0.0418 | 21750 | 454.4 | 16.08 |
| 273 | Anchor only | 86.05 | 0.8041 | 0.0411 | 0 | 87.8 | 16.08 |
| 273 | Clean uniform | 86.17 | 0.7543 | 0.0247 | 21750 | 125.0 | 16.08 |
| 273 | Severe uniform | 83.58 | 0.6893 | 0.0819 | 21750 | 115.0 | 16.08 |
| 273 | Severe routed | 86.08 | 0.7290 | 0.0288 | 21750 | 389.1 | 16.08 |

## Mechanism and cost audit

- Severe routing triggered in 18/18 client-rounds.
- Aggregate routed selected-kind counts: `{'clean': 1689, 'conflict': 615}`.
- Routed selected clean fraction: 73.3% (audit only).
- Distillation-arm total serialized tensor bytes per run: `[21750]`.
- The anchor-only arm omits client execution and records zero client payload.
- Wall time is a scheduler-specific timed-region measurement, not deployment throughput.

## Raw artifact hashes

- `seed271_clean_unfiltered_distillation.json`: `c287a0e1f13f50765203aba096393e9914db6c13452858b7f0133f243332cb05`
- `seed271_public_anchor_only.json`: `e7d7d3d9576c7dd2559c4df8abe603df85b6aeba77c1dbcda074cecf5aaba952`
- `seed271_severe_routed_distillation.json`: `b3c033b402fadc7ea05f874a6c21dc9172f731d8c663d2aa7f2e6638b9a95fdf`
- `seed271_severe_unfiltered_distillation.json`: `6470bef7cd5ab4ec8a5e178cbc4a2c675d99a0d40dfee9b7e19b8a521a314d2c`
- `seed272_clean_unfiltered_distillation.json`: `6380a4fc7367a1763e8a647382dfe2da49f95003016f34ea1d9442038fb9dbad`
- `seed272_public_anchor_only.json`: `d5b4b3fcd530847220169393ea67ec8296774e670bb56feeb9cc47b5e358c0b1`
- `seed272_severe_routed_distillation.json`: `6acb401e634d9c42946e9b975724c94bfc809202f470471a27c364618e4da89d`
- `seed272_severe_unfiltered_distillation.json`: `686ce6ce9557a6c2e8a4ea0acee19041ada5a5fe48e64b07f4e5509bb8a06559`
- `seed273_clean_unfiltered_distillation.json`: `2eaf2da660479309a45fc9379393d388e499a76dcdbaf36386d1b70c8c3518a2`
- `seed273_public_anchor_only.json`: `03e618b50896a37867f16c49aac1a19230b29d848ff6bc866d80d08832c84a0a`
- `seed273_severe_routed_distillation.json`: `774325c1dd26af09ea630bad3be1d834d117121efeb7d242875c2bbce90d1e0d`
- `seed273_severe_unfiltered_distillation.json`: `ddc113220201ee71b05c373e7e19d2f85bc022717ed36fc90845ca13a3e6b513`

## Claim boundary

- The holdout records were selected after primary results were observed; this is post-primary evidence.
- The evaluation records are disjoint from anchors, development, and all clients in this runner, but originate from the AG News training distribution.
- A reconstruction found that 303 evaluation IDs had appeared in at least one prior server-client allocation; the split is not globally untouched.
- This is not a new task, external replication, preregistration, or proof of broad robustness.
- The audit-only clean/conflict/off-task composition is never used as a routing input.
- Off-task membership is identifiable through the explicit task interface; clean versus conflict identity is not provided to the router.
- The public-anchor-only arm omits the client path here; the older AG primary strict control instead executed shadow clients with zero distillation weight.
- No privacy, secure aggregation, or deployment-efficiency claim is authorized.
