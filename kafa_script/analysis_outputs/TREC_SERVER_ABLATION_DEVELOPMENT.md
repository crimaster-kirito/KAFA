# TREC Server Post-Primary Four-Arm Ablation

Phase: **development**. Records: **12**. Protocol SHA-256: `d86c06f0033b6521120edbd0d0cfc2c0b0089178dca6d678770bf08fd1def946`.
Summarizer SHA-256: `8d0934d665543ce48970e50ad58186b6ba2c50a42acaea69f41b98a226b7a7e0`. Verifier SHA-256: `5cc0144ee7c8e6a79a492c1e05d87a02db2110bbf1fa29b13be3ddf73a87019a`.

## Evidence boundary

The complete TREC official test and primary 5-by-2 result were observed before this ablation was designed. The official phase reuses that test with ten new jointly assigned seeds. It is a post-primary mechanistic analysis, not a new holdout, independent replication, preregistration, or repair of the primary inference.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Class NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | 83.33 ± 1.39 | +81.67 [+76.77, +86.58] | +0.5754 [+0.5307, +0.6202] | +0.1070 [+0.0417, +0.1722] | +16.0 [+15.9, +16.1] | +12.77 [+12.72, +12.82] |
| Clean uniform | 83.80 ± 1.60 | +82.58 [+78.10, +87.06] | +0.5024 [+0.4757, +0.5290] | +0.0555 [-0.0096, +0.1206] | +78.7 [+55.3, +102.0] | +12.77 [+12.73, +12.82] |
| Severe uniform | 82.87 ± 1.60 | +81.48 [+76.28, +86.67] | +0.4904 [+0.4502, +0.5306] | +0.1260 [+0.0855, +0.1665] | +82.1 [+55.1, +109.2] | +12.77 [+12.72, +12.82] |
| Severe routed | 82.87 ± 0.80 | +81.15 [+77.51, +84.80] | +0.4665 [+0.4169, +0.5162] | +0.0943 [+0.0733, +0.1152] | +175.8 [+165.6, +185.9] | +12.77 [+12.72, +12.82] |

## Per-seed accuracy (%)

| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |
|---:|---:|---:|---:|---:|
| 258 | 83.33 | 81.94 | 84.72 | 83.33 |
| 259 | 84.72 | 84.72 | 81.94 | 83.33 |
| 260 | 81.94 | 84.72 | 81.94 | 81.94 |

## Complete per-seed scalar secondary values

| Seed | Arm | Macro-F1 | Class NLL | ECE | Tensor bytes | Wall s | Peak GiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 258 | Anchor only | 81.72 | 0.5916 | 0.1141 | 0 | 15.9 | 12.76 |
| 258 | Clean uniform | 80.50 | 0.5092 | 0.0679 | 19830 | 83.5 | 12.76 |
| 258 | Severe uniform | 83.68 | 0.4944 | 0.1075 | 19830 | 88.2 | 12.76 |
| 258 | Severe routed | 82.35 | 0.4435 | 0.0997 | 19830 | 172.4 | 12.76 |
| 259 | Anchor only | 83.62 | 0.5786 | 0.1289 | 0 | 16.0 | 12.79 |
| 259 | Clean uniform | 83.62 | 0.5079 | 0.0732 | 19830 | 84.7 | 12.79 |
| 259 | Severe uniform | 81.23 | 0.4726 | 0.1380 | 19830 | 88.6 | 12.79 |
| 259 | Severe routed | 81.60 | 0.4764 | 0.0985 | 19830 | 174.6 | 12.79 |
| 260 | Anchor only | 79.68 | 0.5560 | 0.0779 | 0 | 16.0 | 12.76 |
| 260 | Clean uniform | 83.62 | 0.4900 | 0.0254 | 19830 | 67.8 | 12.76 |
| 260 | Severe uniform | 79.52 | 0.5042 | 0.1326 | 19830 | 69.6 | 12.76 |
| 260 | Severe routed | 79.52 | 0.4797 | 0.0846 | 19830 | 180.3 | 12.76 |

## Mechanism and cost audit

- Severe routing triggered in 16/18 client-rounds.
- Aggregate routed selected-kind counts: `{'clean': 723, 'conflict': 429, 'off_task': 48}`.
- Every distillation arm records 19,830 bytes of six serialized probability tensors per run; transport metadata is excluded.
- Anchor-only omits the client path. Severe uniform versus severe routed is the only comparison matching client models, mixture, optimizer steps, server work, and seed.
- Wall times are scheduler-specific timed-region measurements, not isolated deployment throughput.

### Routed selection composition by seed

| Seed | Triggered/client-rounds | Selected clean | Selected conflict | Selected off-task |
|---:|---:|---:|---:|---:|
| 258 | 5/6 | 237 | 159 | 24 |
| 259 | 6/6 | 237 | 123 | 0 |
| 260 | 5/6 | 249 | 147 | 24 |

## Complete per-class recall means and intervals

| Arm | ABBR | ENTY | DESC | HUM | LOC | NUM |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | +91.67 [+91.67, +91.67] | +33.33 [+12.63, +54.03] | +91.67 [+91.67, +91.67] | +91.67 [+91.67, +91.67] | +100.00 [+100.00, +100.00] | +91.67 [+91.67, +91.67] |
| Clean uniform | +91.67 [+91.67, +91.67] | +38.89 [+26.94, +50.84] | +88.89 [+76.94, +100.84] | +91.67 [+91.67, +91.67] | +100.00 [+100.00, +100.00] | +91.67 [+91.67, +91.67] |
| Severe uniform | +91.67 [+91.67, +91.67] | +36.11 [+12.21, +60.01] | +88.89 [+76.94, +100.84] | +91.67 [+91.67, +91.67] | +97.22 [+85.27, +109.17] | +91.67 [+91.67, +91.67] |
| Severe routed | +91.67 [+91.67, +91.67] | +33.33 [+12.63, +54.03] | +88.89 [+76.94, +100.84] | +91.67 [+91.67, +91.67] | +100.00 [+100.00, +100.00] | +91.67 [+91.67, +91.67] |

## Raw artifact hashes

- `seed258_clean_unfiltered_distillation.json`: `6cf367212f887bb81c267112b2ff40f21b17aba2297f62f452b7364e143eb634`
- `seed258_public_anchor_only.json`: `849d346ef3d3d9cbed0863b1ff5f79faba085922c9b08b2460b611a6cc1b1d76`
- `seed258_severe_routed_distillation.json`: `ab848b21050505ce01cdcd563f18b980710797f998bc5b0e31caf601aa94cffe`
- `seed258_severe_unfiltered_distillation.json`: `70649560ed61eecf2c4606845edef376d37866d633e0d0863d93df49d353504b`
- `seed259_clean_unfiltered_distillation.json`: `7d1ddf50ad0ac4e6dc07e4b4616a2ef4c21acf57465c027c1d52523ba6af913d`
- `seed259_public_anchor_only.json`: `f256691f268ce545bf4faefe8a50b8b39ec420dd6f0a56370efa44e35a8ac5eb`
- `seed259_severe_routed_distillation.json`: `912dc77dfdda2bd066b2e40db8d0a7b8427499f6cfe1a6238cdaac9630eaf4e8`
- `seed259_severe_unfiltered_distillation.json`: `f7c9815a1f482dc9aa83651d0e3fadaa102575742d2e0e39e0e9f4f83afc08bc`
- `seed260_clean_unfiltered_distillation.json`: `9646a971f39de49a3032a08b68bb1c4719823cb89cae3075ccb200787fa58c0c`
- `seed260_public_anchor_only.json`: `beeab7127f1dac0b68d27db7cf97735d370f18a9f12603fea9d9f9d90c26bf2e`
- `seed260_severe_routed_distillation.json`: `7dad53066e3477e33f3e63e7c0f1fc0e796d4c2a55f5d776ec74effe2f2bf71d`
- `seed260_severe_unfiltered_distillation.json`: `9f69ddc1f28c5810cf980b66bdde63ba6041a1ecb0a0f0ab721ae5e1af11f5c1`

## Claim boundary

- This protocol was designed after the TREC official test and primary results were observed.
- Results are post-primary mechanistic evidence on a reused official test, not an independent holdout or confirmation.
- The controlled severe mixture does not establish robustness to naturally occurring corruption.
- TREC's published data license is recorded as unknown and pretraining contamination is not ruled out.
- The 500-example official test and ten seeds do not define a population of tasks or models.
- No privacy, secure aggregation, deployment efficiency, or general cross-task claim is authorized.
- Wall time is scheduler-specific; only severe-unfiltered versus severe-routed matches client models, data mixture, and optimizer steps.
