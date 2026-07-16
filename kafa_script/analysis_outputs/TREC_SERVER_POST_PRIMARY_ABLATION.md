# TREC Server Post-Primary Four-Arm Ablation

Phase: **official**. Records: **40**. Protocol SHA-256: `d86c06f0033b6521120edbd0d0cfc2c0b0089178dca6d678770bf08fd1def946`.
Summarizer SHA-256: `8d0934d665543ce48970e50ad58186b6ba2c50a42acaea69f41b98a226b7a7e0`. Verifier SHA-256: `5cc0144ee7c8e6a79a492c1e05d87a02db2110bbf1fa29b13be3ddf73a87019a`.

## Evidence boundary

The complete TREC official test and primary 5-by-2 result were observed before this ablation was designed. The official phase reuses that test with ten new jointly assigned seeds. It is a post-primary mechanistic analysis, not a new holdout, independent replication, preregistration, or repair of the primary inference.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Class NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | 81.76 ± 2.01 | +78.39 [+77.33, +79.44] | +0.4609 [+0.4330, +0.4889] | +0.0605 [+0.0506, +0.0703] | +20.0 [+19.9, +20.1] | +12.76 [+12.76, +12.76] |
| Clean uniform | 81.56 ± 1.55 | +79.01 [+78.34, +79.69] | +0.4445 [+0.4242, +0.4648] | +0.0348 [+0.0259, +0.0437] | +73.7 [+69.8, +77.5] | +12.76 [+12.76, +12.76] |
| Severe uniform | 83.14 ± 0.77 | +80.80 [+80.17, +81.44] | +0.4833 [+0.4745, +0.4922] | +0.0985 [+0.0882, +0.1089] | +71.9 [+71.6, +72.1] | +12.76 [+12.76, +12.76] |
| Severe routed | 81.72 ± 1.38 | +79.43 [+78.32, +80.53] | +0.4519 [+0.4396, +0.4642] | +0.0389 [+0.0312, +0.0467] | +175.5 [+174.6, +176.4] | +12.76 [+12.76, +12.76] |

## Per-seed accuracy (%)

| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |
|---:|---:|---:|---:|---:|
| 261 | 82.20 | 81.20 | 82.80 | 82.60 |
| 262 | 81.40 | 81.00 | 83.20 | 81.20 |
| 263 | 84.40 | 83.60 | 82.60 | 81.80 |
| 264 | 85.80 | 84.60 | 84.00 | 84.40 |
| 265 | 79.40 | 80.00 | 81.60 | 79.60 |
| 266 | 81.00 | 81.40 | 82.60 | 81.80 |
| 267 | 79.60 | 79.40 | 83.20 | 80.00 |
| 268 | 82.20 | 81.80 | 84.20 | 82.80 |
| 269 | 80.80 | 80.80 | 83.60 | 81.20 |
| 270 | 80.80 | 81.80 | 83.60 | 81.80 |

## Complete per-seed scalar secondary values

| Seed | Arm | Macro-F1 | Class NLL | ECE | Tensor bytes | Wall s | Peak GiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 261 | Anchor only | 78.78 | 0.4589 | 0.0561 | 0 | 20.0 | 12.76 |
| 261 | Clean uniform | 78.78 | 0.4287 | 0.0361 | 19830 | 72.3 | 12.76 |
| 261 | Severe uniform | 81.72 | 0.4842 | 0.0974 | 19830 | 72.6 | 12.76 |
| 261 | Severe routed | 79.50 | 0.4292 | 0.0298 | 19830 | 174.8 | 12.76 |
| 262 | Anchor only | 78.23 | 0.4681 | 0.0622 | 0 | 20.0 | 12.76 |
| 262 | Clean uniform | 79.11 | 0.4526 | 0.0446 | 19830 | 88.9 | 12.76 |
| 262 | Severe uniform | 80.73 | 0.4746 | 0.0934 | 19830 | 71.6 | 12.76 |
| 262 | Severe routed | 78.96 | 0.4625 | 0.0352 | 19830 | 174.5 | 12.76 |
| 263 | Anchor only | 80.13 | 0.4286 | 0.0486 | 0 | 19.9 | 12.76 |
| 263 | Clean uniform | 79.71 | 0.4196 | 0.0266 | 19830 | 71.9 | 12.76 |
| 263 | Severe uniform | 80.43 | 0.4835 | 0.0856 | 19830 | 72.0 | 12.76 |
| 263 | Severe routed | 79.20 | 0.4550 | 0.0472 | 19830 | 175.5 | 12.76 |
| 264 | Anchor only | 81.69 | 0.3768 | 0.0338 | 0 | 19.9 | 12.76 |
| 264 | Clean uniform | 81.15 | 0.3895 | 0.0228 | 19830 | 71.5 | 12.76 |
| 264 | Severe uniform | 81.43 | 0.4597 | 0.0962 | 19830 | 71.7 | 12.76 |
| 264 | Severe routed | 83.04 | 0.4174 | 0.0532 | 19830 | 174.1 | 12.76 |
| 265 | Anchor only | 76.99 | 0.5001 | 0.0824 | 0 | 20.0 | 12.76 |
| 265 | Clean uniform | 78.86 | 0.4784 | 0.0456 | 19830 | 72.6 | 12.76 |
| 265 | Severe uniform | 80.79 | 0.4908 | 0.0918 | 19830 | 72.0 | 12.76 |
| 265 | Severe routed | 77.44 | 0.4575 | 0.0297 | 19830 | 174.6 | 12.76 |
| 266 | Anchor only | 77.33 | 0.4840 | 0.0669 | 0 | 19.8 | 12.76 |
| 266 | Clean uniform | 77.71 | 0.4622 | 0.0358 | 19830 | 71.6 | 12.76 |
| 266 | Severe uniform | 79.68 | 0.4783 | 0.0773 | 19830 | 71.7 | 12.76 |
| 266 | Severe routed | 78.38 | 0.4557 | 0.0218 | 19830 | 176.2 | 12.76 |
| 267 | Anchor only | 77.51 | 0.5102 | 0.0749 | 0 | 20.2 | 12.76 |
| 267 | Clean uniform | 77.93 | 0.4847 | 0.0607 | 19830 | 71.6 | 12.76 |
| 267 | Severe uniform | 79.65 | 0.4977 | 0.0951 | 19830 | 72.4 | 12.76 |
| 267 | Severe routed | 77.93 | 0.4789 | 0.0329 | 19830 | 178.5 | 12.76 |
| 268 | Anchor only | 78.03 | 0.4362 | 0.0521 | 0 | 20.3 | 12.76 |
| 268 | Clean uniform | 78.92 | 0.4345 | 0.0227 | 19830 | 72.0 | 12.76 |
| 268 | Severe uniform | 79.95 | 0.4979 | 0.1249 | 19830 | 71.8 | 12.76 |
| 268 | Severe routed | 80.05 | 0.4490 | 0.0548 | 19830 | 176.2 | 12.76 |
| 269 | Anchor only | 77.84 | 0.4808 | 0.0653 | 0 | 20.0 | 12.76 |
| 269 | Clean uniform | 78.87 | 0.4541 | 0.0283 | 19830 | 71.8 | 12.76 |
| 269 | Severe uniform | 81.40 | 0.4943 | 0.1197 | 19830 | 71.6 | 12.76 |
| 269 | Severe routed | 79.57 | 0.4577 | 0.0418 | 19830 | 175.3 | 12.76 |
| 270 | Anchor only | 77.34 | 0.4656 | 0.0624 | 0 | 19.9 | 12.76 |
| 270 | Clean uniform | 79.08 | 0.4404 | 0.0249 | 19830 | 72.8 | 12.76 |
| 270 | Severe uniform | 82.25 | 0.4722 | 0.1039 | 19830 | 71.5 | 12.76 |
| 270 | Severe routed | 80.17 | 0.4565 | 0.0428 | 19830 | 175.4 | 12.76 |

## Frozen primary accuracy family

All three adjacent contrasts use paired two-sided Student-t tests over the same ten seeds and one joint Holm correction. Intervals describe seed variation. Exact two-sided sign-flip p-values are sensitivity analyses, not the decision gate.

| Contrast (left − right) | Mean pp [95% CI] | Raw p | Holm p | Exact sign-flip p |
|---|---:|---:|---:|---:|
| clean_client_utility | -0.20 [-0.71, +0.31] | 0.397385 | 0.397385 | 0.437500 |
| severe_contamination_effect | +1.58 [+0.54, +2.62] | 0.007486 | 0.014972 | 0.013672 |
| routing_increment | -1.42 [-2.19, -0.65] | 0.002469 | 0.007407 | 0.005859 |

The descriptive net severe-routed minus anchor effect is -0.04 [-0.84, +0.76] pp. It is not a fourth primary test.

## Secondary descriptive contrasts

| Metric | Clean−anchor | Severe−clean | Routed−severe |
|---|---:|---:|---:|
| macro_f1 | +0.63 [+0.04, +1.21] | +1.79 [+1.12, +2.46] | -1.38 [-2.35, -0.40] |
| canonical_label_nll | -0.0165 [-0.0261, -0.0068] | +0.0389 [+0.0225, +0.0552] | -0.0314 [-0.0417, -0.0211] |
| ece | -0.0257 [-0.0328, -0.0185] | +0.0637 [+0.0478, +0.0797] | -0.0596 [-0.0681, -0.0511] |

No secondary p-value or confirmatory secondary claim is used.

## Mechanism and cost audit

- Severe routing triggered in 57/60 client-rounds.
- Aggregate routed selected-kind counts: `{'clean': 2443, 'conflict': 1265, 'off_task': 72}`.
- Every distillation arm records 19,830 bytes of six serialized probability tensors per run; transport metadata is excluded.
- Anchor-only omits the client path. Severe uniform versus severe routed is the only comparison matching client models, mixture, optimizer steps, server work, and seed.
- Wall times are scheduler-specific timed-region measurements, not isolated deployment throughput.

### Routed selection composition by seed

| Seed | Triggered/client-rounds | Selected clean | Selected conflict | Selected off-task |
|---:|---:|---:|---:|---:|
| 261 | 6/6 | 244 | 116 | 0 |
| 262 | 6/6 | 242 | 118 | 0 |
| 263 | 4/6 | 254 | 178 | 48 |
| 264 | 6/6 | 238 | 122 | 0 |
| 265 | 5/6 | 251 | 145 | 24 |
| 266 | 6/6 | 242 | 118 | 0 |
| 267 | 6/6 | 247 | 113 | 0 |
| 268 | 6/6 | 241 | 119 | 0 |
| 269 | 6/6 | 239 | 121 | 0 |
| 270 | 6/6 | 245 | 115 | 0 |

## Complete per-class recall means and intervals

| Arm | ABBR | ENTY | DESC | HUM | LOC | NUM |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | +90.00 [+87.49, +92.51] | +51.28 [+48.57, +53.98] | +74.64 [+68.57, +80.70] | +91.08 [+90.61, +91.54] | +98.40 [+97.80, +98.99] | +97.88 [+97.43, +98.32] |
| Clean uniform | +94.44 [+90.26, +98.63] | +53.62 [+50.61, +56.62] | +72.17 [+66.76, +77.59] | +90.77 [+90.77, +90.77] | +98.15 [+97.52, +98.77] | +98.05 [+97.79, +98.32] |
| Severe uniform | +95.56 [+91.45, +99.66] | +51.91 [+49.83, +54.00] | +80.22 [+76.99, +83.44] | +90.77 [+90.77, +90.77] | +95.68 [+95.21, +96.14] | +98.32 [+97.96, +98.68] |
| Severe routed | +95.56 [+91.45, +99.66] | +53.83 [+50.91, +56.75] | +72.97 [+68.77, +77.18] | +90.77 [+90.77, +90.77] | +97.04 [+96.29, +97.78] | +98.32 [+97.96, +98.68] |

## Raw artifact hashes

- `seed261_clean_unfiltered_distillation.json`: `e1a293c8752dc286f9a16b8480e541c487f3be4a2d9a1de02248a7ad6a45a949`
- `seed261_public_anchor_only.json`: `30e9377bc57ace13b00e36b63796ddc80d7b80489168910881ada07b7ee7bb4d`
- `seed261_severe_routed_distillation.json`: `59c92bdd6b861b15dc0b4365d778a8a0295cd1e6d43abb2abdca44e9db9e6037`
- `seed261_severe_unfiltered_distillation.json`: `51937c74893d9e17a1ea45b9bc519fbac0fc388c3e0d51b72daf83cc3b9cb8f9`
- `seed262_clean_unfiltered_distillation.json`: `f9bb81701f6066a452da4be89887a8dbd0598c67b5aa23c1a8c93f540e461192`
- `seed262_public_anchor_only.json`: `a6f3a9753473bfe5e3bafba18d06b678d3c67dc779a65af794c319e03a091fcb`
- `seed262_severe_routed_distillation.json`: `74be206c00d5af03942cf46ae438f737c290bbeda3cdf71061a861dffb34c08d`
- `seed262_severe_unfiltered_distillation.json`: `20da234e73baa81f7c9faee1b4539f4ec7206251e24e76b4732bab6439aa0d0d`
- `seed263_clean_unfiltered_distillation.json`: `9b3ed73f8058044a4055c91169efb70c065112a45cc841e1a7a827c9166a40b8`
- `seed263_public_anchor_only.json`: `9769657d3eb85102b004dfa6a579d05988a76981c1964881cedaa18181b4d481`
- `seed263_severe_routed_distillation.json`: `8115e8756e1dda8bd704f0a71c59a8f7df5eb99f3fc105e8d3eda41cba439839`
- `seed263_severe_unfiltered_distillation.json`: `8e1bf99ade1682b58602e77bc95de6f8265585883454072cea9d1834e9f29885`
- `seed264_clean_unfiltered_distillation.json`: `fa5b93daebe51529aec4021d64d23f03a04b1d3c4f2f40a589a7088a8ab529f1`
- `seed264_public_anchor_only.json`: `17df442b622b4fc4211b486af08959c03386800dba043c9ccdbf20d0e746ddde`
- `seed264_severe_routed_distillation.json`: `dd26209ef39b1126c9bc3150cea8629a599e7a0253e253d302f2129613cdbf1a`
- `seed264_severe_unfiltered_distillation.json`: `dfdd81a8c2759d4690e9dcf95eab426b0eed19adabb0e11cc38b1c804a084890`
- `seed265_clean_unfiltered_distillation.json`: `99317f3ac8da6d3567bcfe5f8bbdced381e116a0d3760fbd71ca0103cc94780b`
- `seed265_public_anchor_only.json`: `61604180a46235214af6435153782a92107dc21a4a0464f13c497ab37fb8ccca`
- `seed265_severe_routed_distillation.json`: `c4e8ffe19ee57e401e854b85b5d3488aeeb0c26d971b8a20ef3492e15f5edf29`
- `seed265_severe_unfiltered_distillation.json`: `9882c8c9e3c1a86a80ed22cbccc2be7d2dd9ffc9c05940eef8262b83efa76d01`
- `seed266_clean_unfiltered_distillation.json`: `bc211b80d51e6b603f66bd07917dd8d36461ad03040bddb505ec0efd068f8124`
- `seed266_public_anchor_only.json`: `03582b041ae6b8f1077fbc73989d555f298b33d144889d726a11dfedab5de44b`
- `seed266_severe_routed_distillation.json`: `f4ff43fc0247a9ceb6d7b34cf623284cf1fcfdf1bcfa7f820c01ed0ce824ccf5`
- `seed266_severe_unfiltered_distillation.json`: `26f59504eb7cb2b376668a82911f37df88e6c1f2c0fc126548279e8d659bedae`
- `seed267_clean_unfiltered_distillation.json`: `1c8407ee27a9ff0cc6a120998cd53b02a489b29a0543262bd0cd2fba4e677a0f`
- `seed267_public_anchor_only.json`: `2576b35b4965f93462db363c3cdfc7f06e22d62882167822e4bca7e689cab765`
- `seed267_severe_routed_distillation.json`: `31c1230d22865d51e87e89ab53f72911f4ecbd25cbc29e9cba3adb8bab2e47aa`
- `seed267_severe_unfiltered_distillation.json`: `afe2104bc520bdd96e72ddabda939aeb1c56da16fd06421085fc2e879f551a98`
- `seed268_clean_unfiltered_distillation.json`: `8380fb02ae1976868410decb1a7becea6d967a5a33c4d7a35a89b2a71b2a1040`
- `seed268_public_anchor_only.json`: `5d1bc71fb5564aa601193b0f9576d1e0f67df3d5c057d90cf409bc9a848550df`
- `seed268_severe_routed_distillation.json`: `5aa1f60669207f34c261fedebdb777db9f717a9db71da79eeedc65704be8deab`
- `seed268_severe_unfiltered_distillation.json`: `d351a1cef3220de47271851e9c7ea98ad32017a4c8ef6ebe7c412b5467bc628a`
- `seed269_clean_unfiltered_distillation.json`: `457877873cb25fd85d0f50e0c7e2420ae7c10c523afe1de118aea221aa9f3678`
- `seed269_public_anchor_only.json`: `67547def494bdcdfd8a7c6bd44a02db9eb3be35101520a24332bfcc1149e4714`
- `seed269_severe_routed_distillation.json`: `34da92d9e1e85321ceffdaa7a4b1662b46f398272374878b76ccc95ba0941ab1`
- `seed269_severe_unfiltered_distillation.json`: `7e628b677df43253bcc1997c8e9fffd8537c4b19b52055f9022581d7162f6b53`
- `seed270_clean_unfiltered_distillation.json`: `a02b162e05bedd6ab53c1a0779dffad2fa2c59fe936bb4362ee522cf233bac0c`
- `seed270_public_anchor_only.json`: `86a3631f676e9736dcaa34e61f16eb3905b94e4f9c1386c81e0cf73b8e1b2648`
- `seed270_severe_routed_distillation.json`: `1ca7ea660e7972965d64212fa23be809475934888423af20ba9c2be56c5295f3`
- `seed270_severe_unfiltered_distillation.json`: `f0a6e75b975fbcb4db66ef2ece1e57a347ff1d7838e73b9ad96d1c1be8db4d84`

## Claim boundary

- This protocol was designed after the TREC official test and primary results were observed.
- Results are post-primary mechanistic evidence on a reused official test, not an independent holdout or confirmation.
- The controlled severe mixture does not establish robustness to naturally occurring corruption.
- TREC's published data license is recorded as unknown and pretraining contamination is not ruled out.
- The 500-example official test and ten seeds do not define a population of tasks or models.
- No privacy, secure aggregation, deployment efficiency, or general cross-task claim is authorized.
- Wall time is scheduler-specific; only severe-unfiltered versus severe-routed matches client models, data mixture, and optimizer steps.
