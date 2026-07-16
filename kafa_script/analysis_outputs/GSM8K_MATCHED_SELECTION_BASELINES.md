# GSM8K matched selection baseline results

**Status: post-hoc frozen development comparison; no official-test labels used.**

Protocol SHA-256: `ff3e29f3351a3a890c002b9122b8eb9e2a0d846c6007c015ec60ce8bf56d9c22`. All methods retain 128 examples and use 40 client optimizer steps.

## Mean exact match

| Client | Condition | Naive all | Numeric random | Client small loss | Public-anchor teacher |
|---|---|---:|---:|---:|---:|
| smol | clean | 2.15% | 2.47% | 3.19% | 2.34% |
| smol | severe | 1.17% | 1.30% | 1.11% | 2.08% |
| tiny | clean | 1.82% | 1.95% | 2.47% | 3.65% |
| tiny | severe | 0.52% | 0.59% | 0.65% | 2.15% |

## Teacher-specific paired contrasts

| Client | Condition | Against | Teacher gain [descriptive 95% CI] | Per-seed gains |
|---|---|---|---:|---|
| smol | clean | numeric_random | -0.13 pp [-1.35, +1.09] | -0.59, -0.20, +0.39 |
| smol | clean | client_small_loss | -0.85 pp [-1.86, +0.16] | -1.17, -0.39, -0.98 |
| smol | severe | numeric_random | +0.78 pp [-0.50, +2.06] | +1.37, +0.59, +0.39 |
| smol | severe | client_small_loss | +0.98 pp [-1.14, +3.09] | +1.37, +1.56, +0.00 |
| tiny | clean | numeric_random | +1.69 pp [+0.95, +2.43] | +1.76, +1.95, +1.37 |
| tiny | clean | client_small_loss | +1.17 pp [-2.68, +5.02] | -0.59, +2.34, +1.76 |
| tiny | severe | numeric_random | +1.56 pp [+0.28, +2.85] | +1.95, +0.98, +1.76 |
| tiny | severe | client_small_loss | +1.50 pp [+0.28, +2.72] | +1.95, +1.56, +0.98 |

## Selection audit

- smol numeric_random: severe selected kinds by seed [{'clean': 67, 'conflict': 59, 'off_task': 2}, {'clean': 57, 'conflict': 67, 'off_task': 4}, {'clean': 65, 'conflict': 60, 'off_task': 3}]; mean selection time 0.00 s.
- smol client_small_loss: severe selected kinds by seed [{'clean': 71, 'conflict': 54, 'off_task': 3}, {'clean': 55, 'conflict': 69, 'off_task': 4}, {'clean': 74, 'conflict': 50, 'off_task': 4}]; mean selection time 7.26 s.
- tiny numeric_random: severe selected kinds by seed [{'clean': 67, 'conflict': 59, 'off_task': 2}, {'clean': 57, 'conflict': 67, 'off_task': 4}, {'clean': 65, 'conflict': 60, 'off_task': 3}]; mean selection time 0.00 s.
- tiny client_small_loss: severe selected kinds by seed [{'clean': 48, 'conflict': 77, 'off_task': 3}, {'clean': 50, 'conflict': 74, 'off_task': 4}, {'clean': 66, 'conflict': 58, 'off_task': 4}]; mean selection time 4.99 s.

## Interpretation

The teacher has higher mean severe EM than both matched selectors for both families. This supports a teacher-signal interpretation on development data, while uncertainty and the failed multiplicity-corrected holdout gate still preclude a general claim.

These controls are post-hoc development analyses. They do not repair the failed cross-family holdout claim and must not be reported as confirmatory tests.

## Source artifacts

- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_naive_clean_seed61.json` (SHA-256 `6716cc8e0c88c64f66cf3add9b5f802bf6d653b4c2e1b4851e0285ee251931b1`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_naive_clean_seed62.json` (SHA-256 `dfc7c10b942fd589cd22945429fc417be3757a333c7c9320ca1396ab468d0227`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_naive_clean_seed63.json` (SHA-256 `cb29f0a446dc2d9580ed14ea7c750c18f9682d539ee2fa1a90954cf928a2b94b`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_naive_severe_seed61.json` (SHA-256 `66c9fb346ee616bd6427e34bb0dbb36e4cd1b24c214e809014f044857c2493b0`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_naive_severe_seed62.json` (SHA-256 `f98be2006db32f0b01a6f9f5a29adfc647de679d3ac8c7b729e188353ebba0db`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_naive_severe_seed63.json` (SHA-256 `4876d70304030b0b4d68d85781618454a67180dc1d04468a4467b5d2be78b3d3`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_teacher_clean_seed61.json` (SHA-256 `668b9149267d435068413d28e9036fa4efd2cd8f715fb94692fd3195139466d3`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_teacher_clean_seed62.json` (SHA-256 `04f44bb4b3f829e9b3f0bb8265efde03cc8a14b176168ad3cd4cc249aac6b6dd`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_teacher_clean_seed63.json` (SHA-256 `604c9e64a27706df24ca54ad7655800a5ce8a83c1f1533bbd6993d08c1cff3b8`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_teacher_severe_seed61.json` (SHA-256 `4de153356ba97c9a36e0d2f3a8909b5b0e7097941d1dd124f255614bd7cd24cc`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_teacher_severe_seed62.json` (SHA-256 `4fa4ceba0d1aa01b6722a3382334cacbd64bf3791dfa2927985976f97db04134`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_smol_teacher_severe_seed63.json` (SHA-256 `7d3b5f851f6ad41d1566b36c87d4a4358b1f16cda70db1a1c78cf3017426fcab`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_naive_clean_seed61.json` (SHA-256 `e6faae843440d702424d270d0d9a7dfd725acb488f5997bbf374a458051f0332`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_naive_clean_seed62.json` (SHA-256 `b8602125f66231a036347ab7e4ca9f8912dbf4573ccc0aa383a60942361af152`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_naive_clean_seed63.json` (SHA-256 `2a6e8b5bf3428301edaec8f5ddf921ba12e6e52665c019c9d9b209d7f2e38e6a`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_naive_severe_seed61.json` (SHA-256 `b55f6f3097396686fdadd81b032b00b808c521823158f9f6f10f50b2545ba27b`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_naive_severe_seed62.json` (SHA-256 `69f6dfcd014dd1bcdbe74ce96a6ffb183988969b8d37445f6ac1997b964f099c`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_naive_severe_seed63.json` (SHA-256 `ea7351d8828f2ffcac4f298142d15ec37bfb198d87c7be788fe5a18e2dbd914b`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_teacher_clean_seed61.json` (SHA-256 `c6c1006d6f61998d0729694ae3511a38970f5d7a4b4c166043566723cf8561f2`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_teacher_clean_seed62.json` (SHA-256 `c36bd4960cd3c3984a29a48bc7da9ca2abe852a7e372b1eb777c53a9f3e40596`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_teacher_clean_seed63.json` (SHA-256 `b5177895704de8c85a53c70c3cfb8cd3f5e52abf11b3cacc55b14751cdef4cb6`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_teacher_severe_seed61.json` (SHA-256 `c20604faba1c71985da0a109b24569910318bc8fc5966d4fb1bfec337e523ba2`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_teacher_severe_seed62.json` (SHA-256 `fb65b8e997e70182b271411229eb1a1e4e6e1f884e0cbc20ffb412764b0e9296`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_tiny_teacher_severe_seed63.json` (SHA-256 `075e1c1d6b829137c0dc6f92074ffd1d8abded0afceba54b058947a00a90fe32`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_client_small_loss_clean_seed61.json` (SHA-256 `f2db87cf666dc4ef33d599d34a0defa3929e26a2ba46af207cbe180439d89fb6`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_client_small_loss_clean_seed62.json` (SHA-256 `e49e11fde7b0ddfc1ac6f9d7ae28985d5ecaceee21be3a13f7dd4432de834621`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_client_small_loss_clean_seed63.json` (SHA-256 `31d05db6d274090240d11115a1effe50a9b2a487c6a12b4494e477944c87e335`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_client_small_loss_severe_seed61.json` (SHA-256 `53267bfba59e31f53c3552949324ecc65e62816f27715b6c82dd7707fd2c94f6`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_client_small_loss_severe_seed62.json` (SHA-256 `27d8d5976abdaa65d5465ac0e97e323ff0127f271917c26bfbd174d56b757739`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_client_small_loss_severe_seed63.json` (SHA-256 `088ca985c10288350ab778f83e43aea0ad1693a75ccde9e48e16cd2f8dc0cffb`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_numeric_random_clean_seed61.json` (SHA-256 `88fe70adda1fa78d7d07cb062e1ffabf2713f368c7c2b122d5d09db8ababf4ad`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_numeric_random_clean_seed62.json` (SHA-256 `1843ce50331ae5d9f3a61639b8223e57a5851febf92171e7e329c1491c2d4d64`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_numeric_random_clean_seed63.json` (SHA-256 `1e164e5374de78a41be7ebf8ea1cfe9593ffddaa537eeba24e6ec5afb5705f19`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_numeric_random_severe_seed61.json` (SHA-256 `b9d700e6aa47f2e5e4f23c716db070ab17901333a87995100e27c1bacd9fa990`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_numeric_random_severe_seed62.json` (SHA-256 `993b2ecd9fd6033d9a9b9e06e3f99bf94a427b99756e8300365313c6efb01864`)
- `kafa_script/analysis_outputs/gsm8k_matched_smol_numeric_random_severe_seed63.json` (SHA-256 `32225790fbddb9d15e274812fbdd28e6a9879fe35ea395b335e4d8d5ec69cb2d`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_client_small_loss_clean_seed61.json` (SHA-256 `fee1c297c1abf1dc935dbad53477b538446b8bfebadb26f0f40a5a0cb7873917`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_client_small_loss_clean_seed62.json` (SHA-256 `991454147658c302f03b9dc2beb9df849028348bd4fdd3d17e0f0140d7053bd1`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_client_small_loss_clean_seed63.json` (SHA-256 `a213204847d09be4f50b9a293d0b8ce0088446c23a82fe8b4b4a03455a03ef83`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_client_small_loss_severe_seed61.json` (SHA-256 `95199052cf7101b54099b7087c173507f1907da21b7592ea106ca06a92b341d6`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_client_small_loss_severe_seed62.json` (SHA-256 `a9a4de4c3b41d17ae7a63c0a64066e6ec956a1d54fa69c3940212b5a05fc85a1`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_client_small_loss_severe_seed63.json` (SHA-256 `08376126cd393e4314be59e36432dae06501e67f1394ee3dc2029f204a50af4f`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_numeric_random_clean_seed61.json` (SHA-256 `299f57622ac9b8171eaf7249f60532cbca32dc75e65122d5ceb2d79140b99aea`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_numeric_random_clean_seed62.json` (SHA-256 `3ef7467f18b0c34cf878024a53f376cc88af8b346f4412bdebbd9008ac7322e9`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_numeric_random_clean_seed63.json` (SHA-256 `721f3f1e094ee2247082a5f07853fa4d68db21486417d5ce2ea6624247fa2c68`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_numeric_random_severe_seed61.json` (SHA-256 `ff3231c0be3edd3384120f4274ae0f9b22419a0f9c2fa849c884e8b3ee729701`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_numeric_random_severe_seed62.json` (SHA-256 `1e4e9589e3406901657309fca4fdd424ecf054149e18d421e23a3df1e8f42a6a`)
- `kafa_script/analysis_outputs/gsm8k_matched_tiny_numeric_random_severe_seed63.json` (SHA-256 `da55fb80cc065d344826894a15402c65a0a2ab2e2a123f9207a56be37cb458ab`)
