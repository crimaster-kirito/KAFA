# IMDb Server Prospective Role-Reversal Experiment

Phase: **official_evaluation**. Records: **40**. Protocol SHA-256: `fbfc13064eca36163397017521474dde6aeda7c4304215b97491845c14ee9166`.
Summarizer SHA-256: `2115facef5488f5ad6ae36461c151dedeca682c0ee2dadd4af2a3985eabcddf5`. Verifier SHA-256: `b63008c9dc05d5fcfd54a56abe24270916ee3227da25c2aca00b363c4007b7f0`.

## Evidence boundary

The complete 25,000-row IMDb test had no model-evaluation record in the workspace audit before protocol freeze. Manifest construction hashed and counted raw lines without decoding JSON. IMDb train and the sentiment interface had previously served as off-task inputs, so this is a prospective untouched-test role-reversal extension, not dataset-naive or independently preregistered replication.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Class NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | 91.83 ± 0.10 | +91.83 [+91.76, +91.90] | +0.2470 [+0.2407, +0.2533] | +0.0436 [+0.0420, +0.0452] | +622.2 [+620.4, +623.9] | +9.82 [+9.82, +9.82] |
| Clean uniform | 91.79 ± 0.11 | +91.78 [+91.70, +91.87] | +0.2099 [+0.2081, +0.2117] | +0.0193 [+0.0165, +0.0221] | +683.1 [+681.0, +685.1] | +9.82 [+9.82, +9.82] |
| Severe uniform | 91.10 ± 0.18 | +91.10 [+90.97, +91.23] | +0.2578 [+0.2547, +0.2609] | +0.0707 [+0.0669, +0.0745] | +410.4 [+407.9, +412.8] | +9.82 [+9.82, +9.82] |
| Severe routed | 91.39 ± 0.26 | +91.39 [+91.20, +91.58] | +0.2139 [+0.2090, +0.2188] | +0.0115 [+0.0095, +0.0136] | +623.6 [+618.2, +629.0] | +27.09 [+27.09, +27.09] |

## Per-seed accuracy (%)

| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |
|---:|---:|---:|---:|---:|
| 321 | 91.87 | 91.86 | 91.31 | 91.61 |
| 322 | 91.92 | 91.84 | 91.22 | 91.61 |
| 323 | 91.82 | 91.51 | 91.19 | 91.57 |
| 324 | 91.81 | 91.79 | 91.19 | 91.45 |
| 325 | 91.81 | 91.82 | 91.06 | 90.98 |
| 326 | 91.61 | 91.76 | 90.96 | 91.27 |
| 327 | 91.88 | 91.88 | 90.74 | 91.56 |
| 328 | 91.75 | 91.72 | 90.95 | 91.56 |
| 329 | 91.86 | 91.75 | 91.14 | 90.90 |
| 330 | 91.95 | 91.92 | 91.26 | 91.40 |

## Single frozen primary contrast

| Contrast | Mean pp [95% CI] | Paired-t p | Exact sign-flip p |
|---|---:|---:|---:|
| routing_accuracy_role_reversal | +0.29 [+0.07, +0.51] | 0.014940 | 0.017578 |

## Frozen confirmation gate

- Routing triggered in 55/60 client-rounds.
- Routed audit clean fraction: 66.5% versus the severe baseline 40.2%: **pass**.
- Strictly negative primary interval with two-sided p<.05: **fail**.
- Overall narrow role-reversal confirmation gate: **fail**.

All other arm and metric comparisons are descriptive; no other p-value is used.

## Mechanism and cost audit

- Routing triggered in 55/60 client-rounds.
- Aggregate routed selected-kind counts: `{'clean': 5535, 'conflict': 2530, 'off_task': 255}`.
- Routed audit clean fraction: 66.5% versus the severe baseline 40.2%.
- The anchor arm omits the complete client path and records zero payload.
- Wall time is scheduler-specific, not isolated deployment throughput.

## Raw artifact hashes

- `seed321_clean_unfiltered_distillation.json`: `fcaecbe7c98e7397ef6b5ed66f71cba0159c43a2a8daabf280b0f16382e7ce6e`
- `seed321_public_anchor_only.json`: `bca08f3e1ba8becc17177196b6d1ccae4d56b9e94db6de08690038e3ee9959f6`
- `seed321_severe_routed_distillation.json`: `2294901e23ce75a564de735b198782ef96d62ce06f8fe9d6a93445957331c8c1`
- `seed321_severe_unfiltered_distillation.json`: `3157517529f7327892aae00d6ffac24242382533e2d0a4a7a12caf906429fb38`
- `seed322_clean_unfiltered_distillation.json`: `ec4bbc6266000063b537e52d99cfcbe18cdb2c47dda7aed22fffae1ec2600719`
- `seed322_public_anchor_only.json`: `0a8dabbb61642c990828db0cd037242a10606cc0a5fe11bca02e66f099add00b`
- `seed322_severe_routed_distillation.json`: `7fdf675b9b344d3bb5c0987c625c95819504f09742cf58538e2b442d1f2625d2`
- `seed322_severe_unfiltered_distillation.json`: `3e4255985e7cf86d82d82fb2d1c7496644356a4433a276962d3ee46864c6dbfe`
- `seed323_clean_unfiltered_distillation.json`: `cb4d0dd0a619b2a21971cdd24b684fd7338edae8209d522bb8d27d7345903432`
- `seed323_public_anchor_only.json`: `16d4fc6f6d25c9c290e6322a463279dbcb85942c411d1d7f17cb409484c7e136`
- `seed323_severe_routed_distillation.json`: `d7672fe5656544d653a0881d5eda248bdb96218088a21ab12dc4e9764586dfc6`
- `seed323_severe_unfiltered_distillation.json`: `2c79e1b95638ed1e4204016984db11598d7da5c31fcac46a98a0286213b61e24`
- `seed324_clean_unfiltered_distillation.json`: `eaebbf46eb7742c7b636501554747ee4316012c98acc6171345f4e586db29e53`
- `seed324_public_anchor_only.json`: `d1bee9215f2f1679f6fe5f0a7736aeb9fdd15faa8c6399a115d42a9c93dafbaf`
- `seed324_severe_routed_distillation.json`: `dd402eedaf1830d3570c76da8c35b68a7d9644ca1560a4bf75ff87e6d65aac7e`
- `seed324_severe_unfiltered_distillation.json`: `174d5f18b1f80b8945e4b74945b54fe53343455817c1a4719ae3e6b1bfe292df`
- `seed325_clean_unfiltered_distillation.json`: `c63a79cf9167534de08e4aea8156f88d13541cf307e67562dbfa56c1151e3e7e`
- `seed325_public_anchor_only.json`: `2af9a6084087c249cc6a7a13d8213398318580819d4b1ade6b70f74337dc01ae`
- `seed325_severe_routed_distillation.json`: `62c68c9ed2bce4b870b258be86b04b66144f6fc7f8dcfe3af377be53c829d120`
- `seed325_severe_unfiltered_distillation.json`: `61a8395ecef06cfe508740a97785380dac310aa31323e9c957373da80f9b0b42`
- `seed326_clean_unfiltered_distillation.json`: `ec012170e7e2d4b1ecb622d5849c67be8e22a8b78efdc98f700b1990ea623b29`
- `seed326_public_anchor_only.json`: `e92b61c5002f6e2ec8edcb985d906e716f552139f7d85f8beb6f5b73d6f9a25e`
- `seed326_severe_routed_distillation.json`: `fd7d57817a1a7fa1d5bc255931c370feb345ad1a37ec81803a89155bc699ca10`
- `seed326_severe_unfiltered_distillation.json`: `4c5fb4dfe16fe7e0c8bf8f64b4d98e06f3efd0f8345dbae371d6c537fac3c2a6`
- `seed327_clean_unfiltered_distillation.json`: `c70a208426d94d90010aa1a33c6a7f316f1c533953a59efb256da8a07a89565a`
- `seed327_public_anchor_only.json`: `fe9d5c6f271c862958f474eb1798da2d4e23545626f180754a48f4e9c9d59441`
- `seed327_severe_routed_distillation.json`: `fd8365bdfa29971bfe24fe568bfe5d4855dba7c45ad74977df8cd062ad3a5672`
- `seed327_severe_unfiltered_distillation.json`: `1b49c4eb66751e8a15658442a2c4dbda7eec6ac425dba303b296fbba0371ece9`
- `seed328_clean_unfiltered_distillation.json`: `87603e54c261692a1a9cec9c2f40a105520773be45c31c74793ba4043a63ace7`
- `seed328_public_anchor_only.json`: `9e3dc20161edb49dd26dba8090c98388cefdf5d420bfd0c9240008ba4092332b`
- `seed328_severe_routed_distillation.json`: `e5e9115042cbc3e4a37e82bca32743377e5fa764ca3d34d7f5cecdb7f1e3bbaa`
- `seed328_severe_unfiltered_distillation.json`: `297371f869c7651d9e30e67eb861cc7781e54877e49660c0ee1b82f581448db0`
- `seed329_clean_unfiltered_distillation.json`: `951f941099400b0df70680863125e9b25847073c6fdb724b46f43f8467520e40`
- `seed329_public_anchor_only.json`: `a5e275feb09004b44e2eceb8c50c46d98dfefd0d75073f12b0db9cf10fd52859`
- `seed329_severe_routed_distillation.json`: `daa9b170ec289f7a9ee6f622e85779fe36e8c398a9a4967e529810b40b4de48f`
- `seed329_severe_unfiltered_distillation.json`: `d7c36994994410a5344ec445c0aeb89dbe15af4ed1f97e5718f95a1c727893e3`
- `seed330_clean_unfiltered_distillation.json`: `cbc9c18aaa4db541b8e2ef2f16508b6156cee705247fc1825286f3d421eeac1b`
- `seed330_public_anchor_only.json`: `a00c7e7c24e7c9e6d340aa8506e235c97ffe75a7d1dac69f02213c6a77f8184b`
- `seed330_severe_routed_distillation.json`: `6e7df6029405283ec368aa8b7c823bee1a00610fe2aaf3007d45b67e34192d3f`
- `seed330_severe_unfiltered_distillation.json`: `b02ca1dc69b3c1d0cd114a319c5c80c9cad7802af74574d781e977eb8fe4babd`

## Claim boundary

- The complete IMDb test is prospectively untouched for model assessment according to the workspace audit, but raw bytes were hashed and counted.
- IMDb train and task semantics were previously used as off-task inputs; this is role reversal, not dataset-naive confirmation.
- The official phase decodes and evaluates all 25,000 rows only after development completes and the protocol remains unchanged.
- The audit-only clean/conflict/off-task labels do not enter routing scores; explicit off-task task membership is available through the interface.
- The complete routing intervention changes both selected-set size and composition.
- No privacy, security, licensing, deployment-efficiency, or population-of-tasks claim is authorized.
