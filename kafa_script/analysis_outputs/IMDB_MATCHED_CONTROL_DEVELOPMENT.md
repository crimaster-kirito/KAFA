# IMDb Matched-K Secondary Mechanistic Addendum

Phase: **development**. Records: **9**. Protocol SHA-256: `47175564a023ea59e0137c847bdf747e3ec70b436ae52bc51a0f92f9b2670d64`.

This is a prospective secondary mechanistic addendum frozen after IMDb development results were observed but before any official IMDb model evaluation in the retained workspace. It does not alter or confirm the original role-reversal primary contrast.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [95% CI] | NLL mean [95% CI] | ECE mean [95% CI] |
|---|---:|---:|---:|---:|
| Severe random-K | 89.70 ± 0.46 | +89.70 [+88.55, +90.84] | +0.2837 [+0.2737, +0.2937] | +0.0685 [+0.0500, +0.0870] |
| Severe routed-K | 91.33 ± 0.25 | +91.33 [+90.71, +91.96] | +0.2398 [+0.2325, +0.2471] | +0.0181 [+0.0050, +0.0311] |
| Oracle maximum-purity-K | 90.97 ± 0.57 | +90.97 [+89.55, +92.38] | +0.2443 [+0.2362, +0.2523] | +0.0212 [+0.0111, +0.0313] |

## Per-seed accuracy (%)

| Seed | Random-K | Routed-K | Maximum-purity-K |
|---:|---:|---:|---:|
| 311 | 90.20 | 91.60 | 91.60 |
| 312 | 89.30 | 91.10 | 90.80 |
| 313 | 89.60 | 91.30 | 90.50 |

## Mechanism audit

- Severe random-K selected-kind counts: `{'clean': 1178, 'conflict': 1126}`.
- Severe routed-K selected-kind counts: `{'clean': 1671, 'conflict': 633}`.
- Oracle maximum-purity-K selected-kind counts: `{'clean': 1854, 'conflict': 450}`.
- Every arm selects exactly 128 examples per client-round and uses the same 40 client steps.
- Random-K uses the known task interface to exclude off-task rows and balances claimed labels; it isolates ranking quality only within the target-eligible pool.
- Maximum-purity-K is not clean-only: it contains every 103 clean row and 25 deterministically chosen conflict rows per client-round.

## Raw artifact hashes

- `seed311_oracle_max_purity_k.json`: `6d09efbc32a3dbeb49dff2d566199e10c823b7820d5fa0499d23dfdfde02582b`
- `seed311_severe_random_k.json`: `b51c81791c2bf64fab18782a1dbfa12f4f1b1588053f3f537269c765740af10a`
- `seed311_severe_routed_k.json`: `9606f59045de7eb08693d48558bb07313dea87b29456c726dd24f2631d4dfa4d`
- `seed312_oracle_max_purity_k.json`: `bd5325bb7fef843336a754a35d950b633b32fdb24b36c3a8fa2e6a5a8a684317`
- `seed312_severe_random_k.json`: `baacf5b4f45c2bd296508d69a4f82db5545847cb9c9844cfd9d78cbf9adc2e9c`
- `seed312_severe_routed_k.json`: `f4dcb04a92bb3f70e44d4170555072e21f0142994c5a409c79a910b130761013`
- `seed313_oracle_max_purity_k.json`: `f26eec3fdb527966159afea9b7cb87745558117e46c05bd902b6ce4ffe352d56`
- `seed313_severe_random_k.json`: `a831cdedcd6acc1359b68bdb1c6358f0686006d781ebf01d0f4ca8fc38d7d5b1`
- `seed313_severe_routed_k.json`: `3f0ba73a4f045342e873d2a2713f4d1254bb795ce24efb862ebd0eaad067bc20`

## Claim boundary

- The addendum was designed after IMDb development metrics were observed, so it is prospective only with respect to the still-unopened official model evaluation and is secondary.
- The official test hash and count are inherited from the earlier frozen manifest; addendum construction did not open or decode the test file.
- All arms share the exact severe mixture, K=128, 40 client steps per round, two rounds, models, anchors, server training, and evaluation.
- Random-K controls score-based ordering only within the target-eligible, claimed-label-balanced pool because the known task interface excludes off-task rows before randomization.
- Maximum-purity-K uses hidden audit metadata and is an unattainable diagnostic upper control, not a deployable method.
- Composition is the intended intervention; selected-set size, claimed-label quota, eligible target-task pool, and optimization budget are controlled.
- The same official IMDb test is used by the original analysis and this addendum, and both analyses must be reported together without upgrading secondary evidence to primary.
