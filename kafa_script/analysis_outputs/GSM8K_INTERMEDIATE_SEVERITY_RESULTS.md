# GSM8K intermediate-severity secondary holdout

**Status: completed frozen secondary-holdout study; the official GSM8K test split was not accessed.**

## Verification

- Protocol SHA-256: `652526eacb39736e159834ae8328f71fd8decb671dfc794d6c814b39ded99a5d`.
- Secondary holdout manifest SHA-256: `60e1c2ce19c6247546563fe0e8109c53bccfe425fdf66348483da35039c210d0`.
- All 54 records evaluate the same 1,024 disjoint examples, use seeds 72--74, and skip repeated initial evaluation.
- Nine one-sided paired contrasts are Holm-adjusted together; intervals are descriptive with three seeds.

## Severity curves

| Client | Severity | Naive EM | Teacher EM | Paired gain [95% CI] | Raw p | Holm p |
|---|---|---:|---:|---:|---:|---:|
| qwen | clean | 6.38% | 6.84% | +0.46 pp [-0.88, +1.79] | 0.1400 | 0.5598 |
| qwen | medium | 5.24% | 7.85% | +2.60 pp [-0.06, +5.27] | 0.0260 | 0.1821 |
| qwen | severe | 1.69% | 5.76% | +4.07 pp [+0.42, +7.72] | 0.0204 | 0.1633 |
| smol | clean | 3.68% | 2.80% | -0.88 pp [-2.56, +0.80] | 0.9233 | 1.0000 |
| smol | medium | 2.15% | 3.39% | +1.24 pp [+0.39, +2.09] | 0.0123 | 0.1111 |
| smol | severe | 1.46% | 2.47% | +1.01 pp [-1.48, +3.50] | 0.1116 | 0.5582 |
| tiny | clean | 2.99% | 2.93% | -0.07 pp [-3.03, +2.90] | 0.5333 | 1.0000 |
| tiny | medium | 2.02% | 2.44% | +0.42 pp [-3.61, +4.46] | 0.3479 | 1.0000 |
| tiny | severe | 1.07% | 1.73% | +0.65 pp [-0.44, +1.74] | 0.0623 | 0.3739 |

Per-seed paired gains:

- qwen clean: +0.49, +0.98, -0.10 pp.
- qwen medium: +1.76, +3.81, +2.25 pp.
- qwen severe: +3.32, +5.76, +3.12 pp.
- smol clean: -1.27, -0.10, -1.27 pp.
- smol medium: +1.27, +1.56, +0.88 pp.
- smol severe: +1.27, +1.86, -0.10 pp.
- tiny clean: +0.20, +0.98, -1.37 pp.
- tiny medium: +0.20, +2.15, -1.07 pp.
- tiny severe: +0.20, +0.68, +1.07 pp.

## Frozen curve decision

The predeclared descriptive robustness-curve gate passes: clean safety holds and every family has nonnegative mean teacher gain at medium and severe contamination. This does not imply that all multiplicity-corrected individual contrasts are significant.

This secondary holdout is disjoint from anchors, development examples, and client pools, but comes from the GSM8K training split. It is not a replacement for the official test and does not repair previously failed primary claims.

## Source artifacts

- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_clean_seed72.json` (SHA-256 `be821d377166757c380fd02ac3d13d1d5e74b25727080a5678c67794f182de2d`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_clean_seed73.json` (SHA-256 `1cba1e447449472fa57044410e5d1cbe9cb1c500f00b2628c224fbdd0482e180`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_clean_seed74.json` (SHA-256 `cf03c280ed8ee78f22502558c9ba1a9073674763db01919f857cf7fee00aad1d`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_medium_seed72.json` (SHA-256 `daad2bdb645d1444c915bc0f8874de8e13a76184847c13f5d0f1fcad0987dc65`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_medium_seed73.json` (SHA-256 `3fbadd5fac815e0f458661e63b15945eba02e8b24dffc30f0339c79e607d4864`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_medium_seed74.json` (SHA-256 `95fa62ffad4c68c7bd0ffb0bc86bb8ce2da6bf651718c0ee2406d61dc7c41c89`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_severe_seed72.json` (SHA-256 `3f2676f891b6a92ef42ac2640944382d33bcff4c542215e493676afdb798d18c`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_severe_seed73.json` (SHA-256 `47a333db699c1ae85aaac6aeace631af39df26f2a896f87eb4f1e31dffaea129`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_naive_severe_seed74.json` (SHA-256 `e37fd02b99df3270a9c7d5c59b1f38ae752da5ddaf0ea2e23dd9e47230d77752`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_clean_seed72.json` (SHA-256 `12840b2c0a3ae8df91f56d72f7c5a222a9c6a06405f0f8f2bc35af3a63a1676c`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_clean_seed73.json` (SHA-256 `641652dc26a8c71ad07cafe3938e81b64f8e31362a33beb7f760ca4e6eb0cd6e`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_clean_seed74.json` (SHA-256 `037002c03041495269fc6453c9d3856290851f8c64da359230aaa1f60e0b2c8d`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_medium_seed72.json` (SHA-256 `9a98fb1bdd6f6cdc84d843f9b6debf78b33823f197ac62d98da6167f42a41cc7`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_medium_seed73.json` (SHA-256 `ed1cc3bc7f147c04c55a63ce9b03027a7028736974a61e060a20675ba86f4d72`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_medium_seed74.json` (SHA-256 `72ee006e6f9ac1c0f62143f2168a15f939b3bf237db33a0b252ec91121825e97`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_severe_seed72.json` (SHA-256 `2a5e3f4cbd2d37a14a794fbbf0aa38b90f01c5c4eb981cdd3b0b015319c512ae`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_severe_seed73.json` (SHA-256 `15279669f3014cafc342dd025be79c5925b6704c7ae5f6f6b873e72621ce22ab`)
- `kafa_script/analysis_outputs/gsm8k_severity_qwen_teacher_severe_seed74.json` (SHA-256 `dfe7f56bab86fd54ad045360f6054e13366262e5c123bcb342413229c8d8a060`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_clean_seed72.json` (SHA-256 `7e2d5f95abcf1c4c67af3028bea26cbbd1cb8e79832f3b33d87fff5c42739aa6`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_clean_seed73.json` (SHA-256 `0fc9c54c10f3bf5b60d2411c82fdf8480b53a92a96d17445f4127e0108ddf357`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_clean_seed74.json` (SHA-256 `30cc9152dcbfcad156293436ce81dca76c594df154b998d131665b883fc7d514`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_medium_seed72.json` (SHA-256 `6a0f5812d09b11232c8dac6303f08267fed4f05f69790832a8567b49eeeced2c`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_medium_seed73.json` (SHA-256 `7cf0adf79ad0d8b7ac3153026a121d708e1290b0a52ef71d860849d50b71af92`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_medium_seed74.json` (SHA-256 `fcfaa46b5015a6d6ea4c1c1e8df374ff6ac2f203b22fe2ec1c214946e8742944`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_severe_seed72.json` (SHA-256 `4d0261124e5cb7d23eb5994092f779904985459881987d0536b14ac7e9dd6f34`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_severe_seed73.json` (SHA-256 `bfe04540b94028ab0e3a9093554aeed2580721b134206624d3fd8341dc84b6a8`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_naive_severe_seed74.json` (SHA-256 `4411fe35c49daa7db9167e235814b55ab91284743f5e36660961ec84ee034204`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_clean_seed72.json` (SHA-256 `530a4d12896cd854c3724f0ef54afa7d4dea2afb6d9bc1b2fc4700bfe2404917`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_clean_seed73.json` (SHA-256 `e6b97243cb96bc09ad541c492d745c3668e0c54335e807dda030a6ca69cbb588`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_clean_seed74.json` (SHA-256 `8f0e9fbf5dda01af466c7572f4d30595d9ccda9bf7cfa57dafda33d4d3b2d1cf`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_medium_seed72.json` (SHA-256 `1a8b4bee81edce6540d984f99eac54e1e786c94315e5cefee65ea21b167527ed`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_medium_seed73.json` (SHA-256 `1ac6ceb39d10521dc67fd4995b414a2fc77974f6de737f063e11dd66f286f48b`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_medium_seed74.json` (SHA-256 `29c9ef44c8f09794725b4da048dcd2cd3c8156923b4aeb7dc04cb92c52f0f754`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_severe_seed72.json` (SHA-256 `951148b48f847f981bc11769945d8edaa154006711ca6766126e0955b33a352b`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_severe_seed73.json` (SHA-256 `17f64dcb3b5ddad08cbdc5bebb1b0868b48e401a730fb7e479c0895cdf0ee12f`)
- `kafa_script/analysis_outputs/gsm8k_severity_smol_teacher_severe_seed74.json` (SHA-256 `5ec82f20a626a158d93efe3eca57c96fdab507dd7c30992c62661a2198f83bd7`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_clean_seed72.json` (SHA-256 `fbe58e4f86c94a4a79a3737fa15476324760efe4b1fcd1c0f772746ae5023ff4`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_clean_seed73.json` (SHA-256 `ee4d88a892d28abcbb99574f49f3ebdc29ba8d531eb91235e1e15ab9a7ea28d8`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_clean_seed74.json` (SHA-256 `03743156ecbe58a1c21078dfcfe2c234fbf0b04978c16380cc5cc0e19f45bb87`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_medium_seed72.json` (SHA-256 `0baece046f95e7453ed60dd47c6633ce34814e957b0b7c6da371eb009ca19718`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_medium_seed73.json` (SHA-256 `b8d256f90117c6ccc675119d732cc89d7868b60e15519f8f29879a6e1c801b1e`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_medium_seed74.json` (SHA-256 `da080d1854bf61d27e51285968c0bb1165c04387e7b3771c633af4db21553d69`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_severe_seed72.json` (SHA-256 `b23616d15f22b62d02d40f602225c85065cea4d3d4944b55947f46ac239cc4bf`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_severe_seed73.json` (SHA-256 `651be2fa0d412e86a5b1d199c5742208cc04ffb7fb8da3f7270d7b4b94b5e1ff`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_naive_severe_seed74.json` (SHA-256 `22c804df26aa1f530db529cec62d0502aef282ec34273d3b07e8f3a1e5c037bc`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_clean_seed72.json` (SHA-256 `289394eae4a5b753b80d9e6257856d650e3a177864994f299cad059e820fdf43`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_clean_seed73.json` (SHA-256 `8401a9c0dab24a8bdc570a5dad8d18a958ac4432844acd8e53a3fe19142ebc66`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_clean_seed74.json` (SHA-256 `9dd625f6d7375effddf105773834ac126c8b593c19b100efda9ad3a66bae6903`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_medium_seed72.json` (SHA-256 `87cd0be5221ccf20805d3ffc71bbf4ac71c76af3ec800a64ea031444a654bd4f`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_medium_seed73.json` (SHA-256 `f694ec140756bbf20d59367cf264f778aebaf393ae2f3de7130f04e895187da2`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_medium_seed74.json` (SHA-256 `498eccf4fdde57e3cd4ac7088bc5de6407fdc6a5e9edd48c9caf367eda4e88c2`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_severe_seed72.json` (SHA-256 `3ea1ccc0e2106ad3b6b9ed049bfd52fc900aa3fa9001f8140cd866a513cb37cf`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_severe_seed73.json` (SHA-256 `aded46eeb3b05992d074750ba067e8700a3fb8c1b90774e42697424043c5be46`)
- `kafa_script/analysis_outputs/gsm8k_severity_tiny_teacher_severe_seed74.json` (SHA-256 `e996ee9c50fc48eea7c1e41aab8c316fb1cd37f38adbca9011c41e051fa96539`)
