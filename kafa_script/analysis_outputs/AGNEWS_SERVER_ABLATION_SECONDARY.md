# AG News Server Post-Primary Four-Arm Ablation

Phase: **secondary_holdout**. Records: **40**. Protocol SHA-256: `9dd4d2696780eb67491dba440507639a9651dddbd991c8e94f892c09ee49b4ec`.
Summarizer SHA-256: `e3ab6aa488eacdfa563620bba26ef54841de77d2a5f38a2c0a4ab1fd42c133fa`. Verifier SHA-256: `ce160900de9cf24ac1f943eeea8caf1933af022289165d272123a3b1e7dbd69e`.

## Evidence boundary

The AG News primary and official-test seed replication were observed before this analysis was designed. The 4,000 balanced evaluation records were selected by a frozen content hash from the AG News training remainder and excluded from every client pool in this new runner. A reconstruction audit found that 303 of these IDs had been allocated to clients in at least one earlier server seed. This is therefore post-primary, run-disjoint same-task resampling evidence, not globally untouched data, external replication, or preregistration.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Label NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | 84.84 ± 1.04 | +84.75 [+83.94, +85.57] | +0.7805 [+0.7396, +0.8215] | +0.0488 [+0.0398, +0.0578] | +268.4 [+242.3, +294.4] | +16.08 [+16.08, +16.08] |
| Clean uniform | 84.85 ± 0.90 | +84.78 [+84.06, +85.49] | +0.7249 [+0.6939, +0.7559] | +0.0335 [+0.0249, +0.0420] | +306.9 [+290.6, +323.3] | +16.08 [+16.08, +16.08] |
| Severe uniform | 82.81 ± 1.34 | +82.57 [+81.41, +83.74] | +0.6757 [+0.6609, +0.6906] | +0.0782 [+0.0627, +0.0938] | +185.4 [+180.6, +190.1] | +16.08 [+16.08, +16.08] |
| Severe routed | 84.27 ± 1.16 | +84.17 [+83.20, +85.15] | +0.7028 [+0.6764, +0.7292] | +0.0192 [+0.0135, +0.0250] | +467.5 [+461.5, +473.5] | +16.08 [+16.08, +16.08] |

## Per-seed accuracy (%)

| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |
|---:|---:|---:|---:|---:|
| 281 | 85.03 | 84.95 | 82.32 | 84.47 |
| 282 | 82.10 | 82.63 | 79.83 | 81.27 |
| 283 | 85.47 | 85.43 | 83.25 | 84.87 |
| 284 | 84.65 | 85.53 | 84.18 | 84.45 |
| 285 | 85.28 | 85.00 | 83.42 | 84.87 |
| 286 | 85.20 | 85.15 | 83.13 | 84.45 |
| 287 | 84.68 | 84.50 | 82.40 | 84.43 |
| 288 | 84.72 | 84.25 | 81.67 | 83.50 |
| 289 | 85.55 | 85.58 | 84.43 | 85.22 |
| 290 | 85.77 | 85.53 | 83.47 | 85.15 |

## Complete per-seed scalar secondary values

| Seed | Arm | Macro-F1 | Label NLL | ECE | Tensor bytes | Wall s | Peak GiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 281 | Anchor only | 84.94 | 0.7366 | 0.0504 | 0 | 164.8 | 16.08 |
| 281 | Clean uniform | 84.86 | 0.6963 | 0.0392 | 21750 | 242.0 | 16.08 |
| 281 | Severe uniform | 82.04 | 0.6646 | 0.0648 | 21750 | 203.9 | 16.08 |
| 281 | Severe routed | 84.38 | 0.6868 | 0.0229 | 21750 | 464.3 | 16.08 |
| 282 | Anchor only | 81.72 | 0.8894 | 0.0778 | 0 | 280.2 | 16.08 |
| 282 | Clean uniform | 82.33 | 0.7884 | 0.0558 | 21750 | 314.4 | 16.08 |
| 282 | Severe uniform | 78.91 | 0.7176 | 0.0442 | 21750 | 184.9 | 16.08 |
| 282 | Severe routed | 80.63 | 0.7697 | 0.0354 | 21750 | 481.3 | 16.08 |
| 283 | Anchor only | 85.48 | 0.7617 | 0.0408 | 0 | 279.0 | 16.08 |
| 283 | Clean uniform | 85.48 | 0.7282 | 0.0307 | 21750 | 313.8 | 16.08 |
| 283 | Severe uniform | 83.21 | 0.6625 | 0.0890 | 21750 | 183.3 | 16.08 |
| 283 | Severe routed | 84.92 | 0.6766 | 0.0097 | 21750 | 470.5 | 16.08 |
| 284 | Anchor only | 84.63 | 0.7485 | 0.0374 | 0 | 277.3 | 16.08 |
| 284 | Clean uniform | 85.57 | 0.6955 | 0.0179 | 21750 | 313.2 | 16.08 |
| 284 | Severe uniform | 84.27 | 0.6609 | 0.1137 | 21750 | 181.8 | 16.08 |
| 284 | Severe routed | 84.47 | 0.7194 | 0.0132 | 21750 | 462.5 | 16.08 |
| 285 | Anchor only | 85.16 | 0.7081 | 0.0338 | 0 | 280.6 | 16.08 |
| 285 | Clean uniform | 84.86 | 0.6628 | 0.0140 | 21750 | 314.0 | 16.08 |
| 285 | Severe uniform | 83.35 | 0.6561 | 0.0994 | 21750 | 184.0 | 16.08 |
| 285 | Severe routed | 84.80 | 0.6570 | 0.0210 | 21750 | 465.8 | 16.08 |
| 286 | Anchor only | 85.13 | 0.8020 | 0.0498 | 0 | 281.5 | 16.08 |
| 286 | Clean uniform | 85.07 | 0.7541 | 0.0362 | 21750 | 313.9 | 16.08 |
| 286 | Severe uniform | 83.00 | 0.6843 | 0.0759 | 21750 | 184.4 | 16.08 |
| 286 | Severe routed | 84.38 | 0.6972 | 0.0100 | 21750 | 460.7 | 16.08 |
| 287 | Anchor only | 84.59 | 0.8321 | 0.0559 | 0 | 278.8 | 16.08 |
| 287 | Clean uniform | 84.39 | 0.7978 | 0.0438 | 21750 | 313.3 | 16.08 |
| 287 | Severe uniform | 82.01 | 0.6951 | 0.0588 | 21750 | 181.4 | 16.08 |
| 287 | Severe routed | 84.40 | 0.7412 | 0.0218 | 21750 | 483.0 | 16.08 |
| 288 | Anchor only | 84.59 | 0.8255 | 0.0559 | 0 | 281.1 | 16.08 |
| 288 | Clean uniform | 84.11 | 0.7241 | 0.0356 | 21750 | 315.7 | 16.08 |
| 288 | Severe uniform | 81.18 | 0.6937 | 0.0583 | 21750 | 184.0 | 16.08 |
| 288 | Severe routed | 83.32 | 0.7265 | 0.0270 | 21750 | 464.8 | 16.08 |
| 289 | Anchor only | 85.54 | 0.7175 | 0.0410 | 0 | 280.1 | 16.08 |
| 289 | Clean uniform | 85.63 | 0.7056 | 0.0322 | 21750 | 314.3 | 16.08 |
| 289 | Severe uniform | 84.37 | 0.6570 | 0.0960 | 21750 | 182.9 | 16.08 |
| 289 | Severe routed | 85.28 | 0.6539 | 0.0143 | 21750 | 458.4 | 16.08 |
| 290 | Anchor only | 85.75 | 0.7841 | 0.0452 | 0 | 280.3 | 16.08 |
| 290 | Clean uniform | 85.49 | 0.6962 | 0.0290 | 21750 | 314.7 | 16.08 |
| 290 | Severe uniform | 83.38 | 0.6655 | 0.0825 | 21750 | 183.2 | 16.08 |
| 290 | Severe routed | 85.15 | 0.6999 | 0.0170 | 21750 | 463.3 | 16.08 |

## Frozen adjacent accuracy family

The three adjacent contrasts use paired two-sided Student-t tests over ten common seeds and one joint Holm correction. Intervals describe seed variation. Exact sign-flip p-values from integer correct-count differences are sensitivity analyses.

| Contrast (left − right) | Mean pp [95% CI] | Raw p | Holm p | Exact sign-flip p |
|---|---:|---:|---:|---:|
| clean_client_utility | +0.01 [-0.28, +0.29] | 0.954108 | 0.954108 | 0.970703 |
| joint_severe_mixture_effect | -2.04 [-2.44, -1.65] | 0.000001 | 0.000003 | 0.001953 |
| complete_routing_intervention | +1.46 [+1.06, +1.86] | 0.000018 | 0.000037 | 0.001953 |

The severe-routed minus anchor net accuracy is -0.58 [-0.80, -0.35] pp. It is descriptive and is not a fourth test.

## Secondary descriptive contrasts

| Metric | Clean−anchor | Severe−clean | Routed−severe |
|---|---:|---:|---:|
| macro_f1 | +0.02 [-0.29, +0.33] | -2.21 [-2.72, -1.69] | +1.60 [+1.12, +2.08] |
| label_nll | -0.0556 [-0.0776, -0.0337] | -0.0492 [-0.0692, -0.0291] | +0.0271 [+0.0119, +0.0423] |
| ece | -0.0153 [-0.0188, -0.0119] | +0.0448 [+0.0212, +0.0684] | -0.0590 [-0.0792, -0.0388] |

No secondary p-value or confirmatory secondary claim is used.

## Mechanism and cost audit

- Severe routing triggered in 60/60 client-rounds.
- Aggregate routed selected-kind counts: `{'clean': 5639, 'conflict': 2041}`.
- Routed selected clean fraction: 73.4% (audit only).
- Distillation-arm total serialized tensor bytes per run: `[21750]`.
- The anchor-only arm omits client execution and records zero client payload.
- Wall time is a scheduler-specific timed-region measurement, not deployment throughput.

## Raw artifact hashes

- `seed281_clean_unfiltered_distillation.json`: `3575e24960b13c0a9f4a998561d8ecf7d30e29636d0af04498d03ee9ea0ae48a`
- `seed281_public_anchor_only.json`: `549c743e828605b4bbf096ac2daeb5b90700a68a18769d1bc162319058f5abaa`
- `seed281_severe_routed_distillation.json`: `231f1d3ecc1d365e0fd1e1ca6c68e76860d365aaa6c63089c061a284faef5079`
- `seed281_severe_unfiltered_distillation.json`: `37e428fd16ca1a67bce002929a7687cbc0b48d5ec6802d638a06844bab5a1451`
- `seed282_clean_unfiltered_distillation.json`: `e2e732b12f6f0d9487980d82d60d65264cb0af53a1adf299b7b298ac9d584993`
- `seed282_public_anchor_only.json`: `4561992edadc38efe702821d9d5820500e319b7465653171cdc06d4452dcbc82`
- `seed282_severe_routed_distillation.json`: `b8501d6052f9715836cee913bde18d65735697a31e540e88406aed7dd822f623`
- `seed282_severe_unfiltered_distillation.json`: `7f202533549dba0de97f9b19cbb0b2af47478dc52cc7437e9b994a3253a0b6a9`
- `seed283_clean_unfiltered_distillation.json`: `fbae0737f31d697ffc4ba249fb3d4726295045dbc794cbb996c6595d3e33753f`
- `seed283_public_anchor_only.json`: `114b836de4191fd317021411f42e43c14b1c62a17531c06cd554a5bbb1b105d0`
- `seed283_severe_routed_distillation.json`: `4ebb18d4e0e876d2e0022b9b661546cf9a5f4268d5b499743f762e8b9b050e9b`
- `seed283_severe_unfiltered_distillation.json`: `af417c1a2c08608fed462d038fd8b6cfbd210546717ff632605bd7bbd7f31fb8`
- `seed284_clean_unfiltered_distillation.json`: `308f1f681a9be475b19e5e5561072b44754ff48764694376c029ad4d7abfe3be`
- `seed284_public_anchor_only.json`: `31831177dff3155d8390b8a955747eee994fbe017ff478a7dad904a16b1a0290`
- `seed284_severe_routed_distillation.json`: `2b3b5005ac1e9a25bfb8490908d48bebaf8829366b0e764a63c6da28b7388082`
- `seed284_severe_unfiltered_distillation.json`: `beaa237c033e6b53d2dd60d3f04f3a73629e4df38b9b4a0fc17d37ef9c919784`
- `seed285_clean_unfiltered_distillation.json`: `61351cd3328194bf29135f70151707df5b31002cc23bf4af298c9f856f5c48bb`
- `seed285_public_anchor_only.json`: `3abfa0bed1f81de4aa4f67f7ec8b5ce20bf1b899a8fc256fa48f92bb862f673f`
- `seed285_severe_routed_distillation.json`: `b65146bcc7e201de7fc2bbb0394f75a461b676f46345a5192dbecff77717b4eb`
- `seed285_severe_unfiltered_distillation.json`: `1c049cc3f2116d28e85801f10f911af33c8e38c20adbccbc856f91e39cce2a08`
- `seed286_clean_unfiltered_distillation.json`: `dec7ef9c075efc47a2f5955b2ed780853dfbe672d7cdc6d0a257805066d35895`
- `seed286_public_anchor_only.json`: `cbd97183509a3c92a2958785b2a4a9875df7a2c50f5471a64b3349ec90a42446`
- `seed286_severe_routed_distillation.json`: `f66b0b2851783729d8462faa521ef3696862594331d579a3714621fc844f3687`
- `seed286_severe_unfiltered_distillation.json`: `b6ee868cd80e6690fad86125117058e61142d31c33a79a1a5a7484180ead8ff0`
- `seed287_clean_unfiltered_distillation.json`: `db02c19971c091f65fcd19eb11390321ce1950e9c51e2dabff6a2525d84ebd07`
- `seed287_public_anchor_only.json`: `01f7b00917455ea1fc1a8e221f8d3cbdd543f8e6d9e7f973e7487d2514c2f934`
- `seed287_severe_routed_distillation.json`: `a4f2c0ca12d4e7a07af2ac373a72a1b821ce030f304d1567a14f3a1580aebf53`
- `seed287_severe_unfiltered_distillation.json`: `6bb3c31d574ddfa0c9fa82015d2d486e34fe6a09e8fcfbd8f77b941ffbb020b6`
- `seed288_clean_unfiltered_distillation.json`: `d972da98d86cd908102a9f1a5b47bc9c849e792384a12d44f0f3a86723e3bcc5`
- `seed288_public_anchor_only.json`: `408e138f21e00444b353cc303f871d10fcbbffbea43804667717480e3b6dffb6`
- `seed288_severe_routed_distillation.json`: `d4ab74263d921000f1b6fa3568f65b35240633a2c4a11bb6845297bb666c29d5`
- `seed288_severe_unfiltered_distillation.json`: `96ce1066a5277d2a15e73d64bba6dd8eaf10e89308a0969018930f2e6ca3b9bf`
- `seed289_clean_unfiltered_distillation.json`: `beb20f471feae0e7ad00abf354767b8a638afef19a5b3c25c58febe9b8e67b5a`
- `seed289_public_anchor_only.json`: `89a033fdb83113d914ab935b4423bbff114c3dc500bf2ba2d5e2cd5da088b7ba`
- `seed289_severe_routed_distillation.json`: `25b2965f6f9699b10b17aec99068aca1415ee6e1311bcb8c38bcdb289f5c6ba1`
- `seed289_severe_unfiltered_distillation.json`: `35a85092153bc24a323c1ae9e0cc2cc87a415fd0b75bb4e189f91551ae0f8527`
- `seed290_clean_unfiltered_distillation.json`: `7e37ecc9f960174a51518903c992fd3cdb9d84b9a2074ab0de20bcae23016f62`
- `seed290_public_anchor_only.json`: `6295c374247c11ef1c283f33a9459cefbcdb14718a035d82a9eea9131a6cfffa`
- `seed290_severe_routed_distillation.json`: `e75590969314f1a9c6b4f88b3a595d40a191e0a41d2d14828fec1876626f5523`
- `seed290_severe_unfiltered_distillation.json`: `ba7fed5c9ae986ad9f236725e716451e5ed5654c20f57262cd15a606ef4cc75c`

## Claim boundary

- The holdout records were selected after primary results were observed; this is post-primary evidence.
- The evaluation records are disjoint from anchors, development, and all clients in this runner, but originate from the AG News training distribution.
- A reconstruction found that 303 evaluation IDs had appeared in at least one prior server-client allocation; the split is not globally untouched.
- This is not a new task, external replication, preregistration, or proof of broad robustness.
- The audit-only clean/conflict/off-task composition is never used as a routing input.
- Off-task membership is identifiable through the explicit task interface; clean versus conflict identity is not provided to the router.
- The public-anchor-only arm omits the client path here; the older AG primary strict control instead executed shadow clients with zero distillation weight.
- No privacy, secure aggregation, or deployment-efficiency claim is authorized.
