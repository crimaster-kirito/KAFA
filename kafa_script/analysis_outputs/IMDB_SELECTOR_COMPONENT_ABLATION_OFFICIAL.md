# IMDb Selector-Component Post-hoc Ablation

Records: **30** new component records plus **10** existing full-routed matched-K reference records.
Selector protocol SHA-256: `a904c814a7cff07641aa6c8d4f5e0aa464bd49bbb38f9239cce16bcda0491784`.
Matched-K verifier protocol SHA-256: `47175564a023ea59e0137c847bdf747e3ec70b436ae52bc51a0f92f9b2670d64`.
Summarizer SHA-256: `5f6fa4b96dadba183fcc2ff78839c8980649349329cf8e1df565ba9875965b5d`.

## Evidence boundary

This is a post-hoc reused-test secondary diagnostic designed after the official IMDb primary and matched-K results were observed. It can diagnose which score component carries ranking signal in this setting, but it cannot alter the original IMDb primary gate, replace the matched-K Holm family, or establish a general selector.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [95% CI] | Class NLL mean [95% CI] | ECE mean [95% CI] | Wall s mean [95% CI] | Peak GiB mean [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| Full routed-K | 91.46 ± 0.23 | +91.46 [+91.29, +91.63] | +0.2118 [+0.2074, +0.2162] | +0.0101 [+0.0084, +0.0117] | +614.4 [+611.4, +617.3] | +27.09 [+27.09, +27.09] |
| Gradient-only-K | 91.30 ± 0.11 | +91.30 [+91.22, +91.37] | +0.2251 [+0.2195, +0.2307] | +0.0299 [+0.0224, +0.0373] | +528.8 [+524.6, +532.9] | +27.09 [+27.09, +27.09] |
| Loss-only-K | 91.50 ± 0.19 | +91.50 [+91.36, +91.63] | +0.2129 [+0.2083, +0.2174] | +0.0101 [+0.0077, +0.0124] | +527.2 [+525.4, +529.0] | +27.09 [+27.09, +27.09] |
| Margin-only-K | 91.36 ± 0.21 | +91.35 [+91.20, +91.51] | +0.2141 [+0.2100, +0.2182] | +0.0090 [+0.0078, +0.0102] | +528.5 [+525.3, +531.6] | +27.09 [+27.09, +27.09] |

## Per-seed accuracy (%)

| Seed | Full routed-K | Gradient-only-K | Loss-only-K | Margin-only-K |
|---:|---:|---:|---:|---:|
| 321 | 91.61 | 91.25 | 91.64 | 91.53 |
| 322 | 91.61 | 91.44 | 91.34 | 91.61 |
| 323 | 91.57 | 91.32 | 91.51 | 91.41 |
| 324 | 91.67 | 91.16 | 91.66 | 91.26 |
| 325 | 91.35 | 91.28 | 91.51 | 91.38 |
| 326 | 91.27 | 91.20 | 91.13 | 91.04 |
| 327 | 91.58 | 91.36 | 91.66 | 91.38 |
| 328 | 91.56 | 91.30 | 91.65 | 91.52 |
| 329 | 90.90 | 91.20 | 91.28 | 90.96 |
| 330 | 91.48 | 91.48 | 91.62 | 91.48 |

## Post-hoc selector-component Holm family

| Contrast | Mean pp [95% CI] | Raw paired-t p | Holm p | Exact sign-flip p |
|---|---:|---:|---:|---:|
| full_routed_k_minus_gradient_only_k | +0.16 [+0.00, +0.32] | 0.044963 | 0.134889 | 0.048828 |
| full_routed_k_minus_loss_only_k | -0.04 [-0.17, +0.09] | 0.504867 | 0.504867 | 0.521484 |
| full_routed_k_minus_margin_only_k | +0.10 [-0.00, +0.21] | 0.050929 | 0.134889 | 0.054688 |

## Mechanism audit

- Full routed-K selected-kind counts: `{'clean': 5454, 'conflict': 2226}` (71.0% clean, 29.0% conflict, 0.0% off-task).
- Gradient-only-K selected-kind counts: `{'clean': 4559, 'conflict': 3121}` (59.4% clean, 40.6% conflict, 0.0% off-task).
- Loss-only-K selected-kind counts: `{'clean': 5171, 'conflict': 2509}` (67.3% clean, 32.7% conflict, 0.0% off-task).
- Margin-only-K selected-kind counts: `{'clean': 5496, 'conflict': 2184}` (71.6% clean, 28.4% conflict, 0.0% off-task).
- Each component arm selects exactly 128 target-eligible examples per client-round with 64 examples per claimed label.
- Clean/conflict identity is audit-only and is not used by any component selector.

## Raw artifact hashes

- `component/seed321_gradient_only_k.json`: `d0fc559c6513717537b8d74eba8199da7a172721495c8f19beddf51dd37bef0f`
- `component/seed321_loss_only_k.json`: `7bc9b62eee803fe423edd23d8b9a2071ca281278494de9296eb03abea4f33052`
- `component/seed321_margin_only_k.json`: `481c859d815a2156ba75b97dee672e5096b39b724adef9c2a69c3f1e0dc01406`
- `component/seed322_gradient_only_k.json`: `098c6b9a3f27de9633c80cf6409a5573cb0b6bbb20affd734b897b9d140ebfac`
- `component/seed322_loss_only_k.json`: `2f7107479eaf69e3d339a7e984881dfbe627410a3d3bf0caea29a67b602973b1`
- `component/seed322_margin_only_k.json`: `71a84d9813c76f0fb6e8893d78f10ff94c7cd055ef90bcc6fe81005cf4365904`
- `component/seed323_gradient_only_k.json`: `9f81d63835ad09755aec70e1df14d8c262e71da6872c9c3873a76cf01a986e22`
- `component/seed323_loss_only_k.json`: `183a9fce3e9241cdf3f63161e033330f5b24b4602dee2c3e60412d1f912441da`
- `component/seed323_margin_only_k.json`: `d9d02fa3f3092361e0325e9d29e7959a8b1f083ebfd1e7c8920856b19f192e34`
- `component/seed324_gradient_only_k.json`: `4e8b1d76c7058ab079a98e468a3bc25f03965a359c028704f00fc65f3f39ebb2`
- `component/seed324_loss_only_k.json`: `7acf4064779482e98594358a22cdb220ee61e632c9424628e5689d95ba9b86b0`
- `component/seed324_margin_only_k.json`: `749672ff6e3750b351a8aee45532b8b5408248ac37cd64ec182575beb509aaa2`
- `component/seed325_gradient_only_k.json`: `55f1385be8d026e54318151d05aa84a97e4a6bacaae0f959a87ebcbe1b4938a2`
- `component/seed325_loss_only_k.json`: `dd79fefe6a31929340e4c9e02a8ab958343548d4e78a9353b61f59b670224a08`
- `component/seed325_margin_only_k.json`: `1c970454c0cd92b8bc8b349b10d296abca7bb313327e144e6796ba324280bfbd`
- `component/seed326_gradient_only_k.json`: `e550c07ad5c915c376d88f4ba93e7f8ed66dc8c8914f135e90464bfdb2fc73c1`
- `component/seed326_loss_only_k.json`: `8c4f571adb676043762cd3fd43c550c2b1b8873241c3b1d53f910dd2a83f2121`
- `component/seed326_margin_only_k.json`: `3a475f308eca83f29a2de986f1dd17102984c30f7bbd76ae0b1ccef7d1ccd7ef`
- `component/seed327_gradient_only_k.json`: `dd171400881561d36777f8935a473dde127c61e1602efd205c3a124baadae5ab`
- `component/seed327_loss_only_k.json`: `8ce273a360db89915a86a4d5c48bf5fa9755ae82ea1f444baca052722a913f12`
- `component/seed327_margin_only_k.json`: `8e971b15eee9b83cf6aefb821b9f10abfcd3656c1c49938a5fcf61c1f1c2f27f`
- `component/seed328_gradient_only_k.json`: `b3e4159a1af7c71c04560c965d2c01a07a4700c09bf37e18ded77583d1e67d16`
- `component/seed328_loss_only_k.json`: `885cb0bc68c46f6b1960bb73010dff3f4f415066f14703f81dc82b1b4bb72ff1`
- `component/seed328_margin_only_k.json`: `4502191b3c20697b39fe2143dd9618e23d1498d4d37766f92c7a31f312780b6a`
- `component/seed329_gradient_only_k.json`: `bd86727bd4ec60c7d854c6e1b0d599359be46f4052cdb78b90c13ad731db0953`
- `component/seed329_loss_only_k.json`: `32a7135e9b7c4239003779abf529d56fbcc780dca816744c90faa2d0e1df0610`
- `component/seed329_margin_only_k.json`: `a397e71d20506289ca0cec5cc9ae79a17194a0685af36cb6441f81f10c2a10ae`
- `component/seed330_gradient_only_k.json`: `3e15121ec0f3400f4c954f171bb096ddb472af4b85e6d57615f242c8991c3953`
- `component/seed330_loss_only_k.json`: `6d9b86fc5224a664a9374271156638a9b698cc9fe9ea5a2e1d5cdb6177209f89`
- `component/seed330_margin_only_k.json`: `92daee4fa803c37c916ac537255ae56da86a17237c9826ad9a19428fc212b4d4`
- `matched_reference/seed321_severe_routed_k.json`: `64de7cd3936f3729965a6cde0e0d70272585793aa5f9db1684453de25bd6600d`
- `matched_reference/seed322_severe_routed_k.json`: `e05c6bdf3945860f83efe3b088ea95b32c50ba00e4cf8ea53d463d111736fb45`
- `matched_reference/seed323_severe_routed_k.json`: `d226911ebd9b4ee127c53338d56945df041d5a9e653be7d87281f72a4c773923`
- `matched_reference/seed324_severe_routed_k.json`: `3c6644c6df6d17bf326bd28b06fe6acbec222db2ac143b16788b8d4fc41ee181`
- `matched_reference/seed325_severe_routed_k.json`: `a2c48fcddb9a193ebfffa6920c0a14f4e249cf0bd1064d7a8d2d4f8deb57d77e`
- `matched_reference/seed326_severe_routed_k.json`: `6b2f451c5f2ba8d78467648d334ac7b7de0fc0918368c666c5feb1bf12cf74b7`
- `matched_reference/seed327_severe_routed_k.json`: `39e208e9eb47e59274ed2cc528940256d28fd431119d00256ce65fce35b18491`
- `matched_reference/seed328_severe_routed_k.json`: `08f099c4dc009141611831f0aeea4bfdf705df66200319e6d6d7614e3bb6a1f0`
- `matched_reference/seed329_severe_routed_k.json`: `8f3880914f05e1a7771abf93c80e806c59ad2b026435c0e6ea74a3c5703bb4fd`
- `matched_reference/seed330_severe_routed_k.json`: `341aa8c3451cea2de1ffbcd7dbef42a46d34f43d0b04ebd436d9d01096b66879`

## Claim boundary

- This ablation was designed after the IMDb official primary and matched-K results were observed.
- All records reuse the official IMDb test and are post-hoc secondary diagnostics.
- All arms share the severe mixture, K=128, 64-per-claimed-label quotas, optimizer settings, anchors, models, and server training budget.
- The known task interface excludes off-task rows before component selection; clean versus conflict identity is not used.
- The audit can diagnose score-component sensitivity within this setting but cannot confirm the TREC-derived direction, change the IMDb primary gate, or establish a general selector.
