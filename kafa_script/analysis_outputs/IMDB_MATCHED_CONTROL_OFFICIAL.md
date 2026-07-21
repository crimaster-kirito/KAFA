# IMDb Matched-K Secondary Mechanistic Addendum

Phase: **official_evaluation**. Records: **30**. Protocol SHA-256: `47175564a023ea59e0137c847bdf747e3ec70b436ae52bc51a0f92f9b2670d64`.

This is a prospective secondary mechanistic addendum frozen after IMDb development results were observed but before any official IMDb model evaluation in the retained workspace. It does not alter or confirm the original role-reversal primary contrast.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [95% CI] | NLL mean [95% CI] | ECE mean [95% CI] |
|---|---:|---:|---:|---:|
| Severe random-K | 91.14 ± 0.13 | +91.13 [+91.04, +91.23] | +0.2558 [+0.2513, +0.2603] | +0.0690 [+0.0642, +0.0738] |
| Severe routed-K | 91.46 ± 0.23 | +91.46 [+91.29, +91.63] | +0.2118 [+0.2074, +0.2162] | +0.0101 [+0.0084, +0.0117] |
| Oracle maximum-purity-K | 91.51 ± 0.16 | +91.51 [+91.39, +91.62] | +0.2122 [+0.2093, +0.2152] | +0.0125 [+0.0097, +0.0153] |

## Per-seed accuracy (%)

| Seed | Random-K | Routed-K | Maximum-purity-K |
|---:|---:|---:|---:|
| 321 | 91.19 | 91.61 | 91.53 |
| 322 | 91.23 | 91.61 | 91.72 |
| 323 | 91.04 | 91.57 | 91.64 |
| 324 | 91.20 | 91.67 | 91.40 |
| 325 | 91.21 | 91.35 | 91.43 |
| 326 | 90.80 | 91.27 | 91.18 |
| 327 | 91.24 | 91.58 | 91.72 |
| 328 | 91.18 | 91.56 | 91.48 |
| 329 | 91.10 | 90.90 | 91.41 |
| 330 | 91.20 | 91.48 | 91.57 |

## Frozen secondary Holm family

| Contrast | Mean pp [95% CI] | Raw paired-t p | Holm p |
|---|---:|---:|---:|
| routed_k_minus_random_k | +0.32 [+0.17, +0.48] | 0.001022 | 0.002044 |
| oracle_max_purity_k_minus_routed_k | +0.05 [-0.10, +0.19] | 0.487230 | 0.487230 |

These two tests are secondary and are reported in parallel with, not substituted for, the original primary analysis.

## Mechanism audit

- Severe random-K selected-kind counts: `{'clean': 3814, 'conflict': 3866}`.
- Severe routed-K selected-kind counts: `{'clean': 5454, 'conflict': 2226}`.
- Oracle maximum-purity-K selected-kind counts: `{'clean': 6180, 'conflict': 1500}`.
- Every arm selects exactly 128 examples per client-round and uses the same 40 client steps.
- Random-K uses the known task interface to exclude off-task rows and balances claimed labels; it isolates ranking quality only within the target-eligible pool.
- Maximum-purity-K is not clean-only: it contains every 103 clean row and 25 deterministically chosen conflict rows per client-round.

## Raw artifact hashes

- `seed321_oracle_max_purity_k.json`: `61e32ccd9cb0d75ad69d92cead8e773e4e7c50a918d9b81ed86a3c8973a43b0e`
- `seed321_severe_random_k.json`: `75c16f5e505041be5ab05a7d49feac586199329882f52e57af9a8b95ef92a418`
- `seed321_severe_routed_k.json`: `64de7cd3936f3729965a6cde0e0d70272585793aa5f9db1684453de25bd6600d`
- `seed322_oracle_max_purity_k.json`: `3a64772f33476ba41b819ba1bc41b611e9584572e2c938c1c157f3526520730f`
- `seed322_severe_random_k.json`: `820e01a9458bf320b986325b40363fc61ebea516c0ef5ad8224c7bf956101330`
- `seed322_severe_routed_k.json`: `e05c6bdf3945860f83efe3b088ea95b32c50ba00e4cf8ea53d463d111736fb45`
- `seed323_oracle_max_purity_k.json`: `e8b34788ddf68e493b39908add6a2e66118b0ea70c21903c0f13fe5f7ac55e32`
- `seed323_severe_random_k.json`: `e1827dc59783331224eafe0008019b81a555129f5288da9cce36db545e3ab1d3`
- `seed323_severe_routed_k.json`: `d226911ebd9b4ee127c53338d56945df041d5a9e653be7d87281f72a4c773923`
- `seed324_oracle_max_purity_k.json`: `3d1de1b341143ac2771df854089cce2ccfcb19b30e131504fd83440ab24ade14`
- `seed324_severe_random_k.json`: `de906422dea3d8ff966595dd894207d78bc5496ed1885e760ddcf78408722ead`
- `seed324_severe_routed_k.json`: `3c6644c6df6d17bf326bd28b06fe6acbec222db2ac143b16788b8d4fc41ee181`
- `seed325_oracle_max_purity_k.json`: `16d57963a307521e0cf8d0a3afaad79c1a7c6b026731732e6fdbf1d98713f6c0`
- `seed325_severe_random_k.json`: `d1c44dd5c29d95accf463eb33b5ae39de0e1372d035877beb7c3db261772a145`
- `seed325_severe_routed_k.json`: `a2c48fcddb9a193ebfffa6920c0a14f4e249cf0bd1064d7a8d2d4f8deb57d77e`
- `seed326_oracle_max_purity_k.json`: `4f7c1d3b9eb065d46e3c4619cc56df2556694611ba4cac68b40c8a224eb7cf0f`
- `seed326_severe_random_k.json`: `6e616878b005f50d230e380c68fd59fc29abff193c72e76b12a63055114310a1`
- `seed326_severe_routed_k.json`: `6b2f451c5f2ba8d78467648d334ac7b7de0fc0918368c666c5feb1bf12cf74b7`
- `seed327_oracle_max_purity_k.json`: `a51bd9b07259dafdc52b4c5bf9237bef3bdda1d5e6981b8a163eb72157ac4817`
- `seed327_severe_random_k.json`: `f365a52691e9b955948967c3fd109b6e7ada8f1fdb2baa14ea0771578daaab47`
- `seed327_severe_routed_k.json`: `39e208e9eb47e59274ed2cc528940256d28fd431119d00256ce65fce35b18491`
- `seed328_oracle_max_purity_k.json`: `3ade0b09c4d20f95e3f1d7900fb0eb03db813628a2865ba387eb79e616b38bcb`
- `seed328_severe_random_k.json`: `991c64d23052a89bbaee99b3dde0b0077e0d3b21de1e310b6ad1dae7f52028d6`
- `seed328_severe_routed_k.json`: `08f099c4dc009141611831f0aeea4bfdf705df66200319e6d6d7614e3bb6a1f0`
- `seed329_oracle_max_purity_k.json`: `97017161df819900846d6f56d6b96787b0c75a82e19be779d4be651ef424f7e8`
- `seed329_severe_random_k.json`: `503469478836f20172d9d6589378d359da4dc56714f69eda5bc1a38ecc2e2f18`
- `seed329_severe_routed_k.json`: `8f3880914f05e1a7771abf93c80e806c59ad2b026435c0e6ea74a3c5703bb4fd`
- `seed330_oracle_max_purity_k.json`: `d83e68cf6fcf54ae057ead2b068b42e2032b87ac239407d842009c195e8a8bae`
- `seed330_severe_random_k.json`: `019129b40451aa7f4c42e2a8b2a9dd9106175e6c46f817e776690987f07698f2`
- `seed330_severe_routed_k.json`: `341aa8c3451cea2de1ffbcd7dbef42a46d34f43d0b04ebd436d9d01096b66879`

## Claim boundary

- The addendum was designed after IMDb development metrics were observed, so it is prospective only with respect to the still-unopened official model evaluation and is secondary.
- The official test hash and count are inherited from the earlier frozen manifest; addendum construction did not open or decode the test file.
- All arms share the exact severe mixture, K=128, 40 client steps per round, two rounds, models, anchors, server training, and evaluation.
- Random-K controls score-based ordering only within the target-eligible, claimed-label-balanced pool because the known task interface excludes off-task rows before randomization.
- Maximum-purity-K uses hidden audit metadata and is an unattainable diagnostic upper control, not a deployable method.
- Composition is the intended intervention; selected-set size, claimed-label quota, eligible target-task pool, and optimization budget are controlled.
- The same official IMDb test is used by the original analysis and this addendum, and both analyses must be reported together without upgrading secondary evidence to primary.
