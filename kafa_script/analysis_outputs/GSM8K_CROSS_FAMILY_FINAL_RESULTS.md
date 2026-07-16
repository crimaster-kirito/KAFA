# Final GSM8K cross-family teacher-ranking result

**Status: completed frozen holdout evaluation.**

## Verification

- Frozen protocol SHA-256: `6781fcc51fc722006382a562f7eced787ba2a54ad3646df174e3afa0afa9719c`.
- Frozen data manifest SHA-256: `03ce4cac6dad028975adc1467d5c841e69a4f14dafb1427af71599b7174a7292`.
- All 24 records are labelled `benchmark_holdout`, evaluate all 1,319 official test questions, and use holdout seeds 164--166.
- Two primary severe-condition tests use Holm correction at alpha 0.05; clean safety requires every paired difference to be at least -2 pp.

## Results

| Client | Condition | Naive EM | Teacher EM | Paired gain [95% CI] | Raw one-sided p | Holm p | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| smol | clean | 4.55% | 4.90% | +0.35 pp [-0.04, +0.75] | -- | -- | safe |
| smol | severe | 2.58% | 4.17% | +1.59 pp [-0.39, +3.58] | 0.0373 | 0.0714 | fail |
| tiny | clean | 3.66% | 4.25% | +0.58 pp [-2.74, +3.90] | -- | -- | safe |
| tiny | severe | 1.42% | 3.44% | +2.02 pp [-0.44, +4.48] | 0.0357 | 0.0714 | fail |

Per-seed paired gains:

- smol clean: +0.53, +0.23, +0.30 pp.
- smol severe: +2.35, +1.67, +0.76 pp.
- tiny clean: +0.76, -0.83, +1.82 pp.
- tiny severe: +0.91, +2.81, +2.35 pp.

## Claim decision

At least one client family fails the frozen gate, so the study does not support a general cross-family holdout claim. Family-specific effects remain reportable with this failure explicit.

This result does not establish a complete federated aggregation method, privacy protection, or robustness to naturally occurring noise. The teacher-ranking endpoint is centralized and its per-example scoring adds substantial computation.

## Source records

- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_naive_clean_seed164.json` (SHA-256 `00cfc7d897e10e14e54fdef59f68f3769d6a7d016c8a8a48c2e63c55979b66d3`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_naive_clean_seed165.json` (SHA-256 `04713d8da61130727c5ce591092ad7348445d8134b6d8876beabeae040b24b28`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_naive_clean_seed166.json` (SHA-256 `c5211cc099903bd82047ba5edaf0f33592c5a8260a2f7017b2dc0e1cce00288d`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_naive_severe_seed164.json` (SHA-256 `65ed3eaac869674b26ae1a59a9ea2636028a44d854f39aaba9210eecd5b62613`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_naive_severe_seed165.json` (SHA-256 `39ffabf44df277d218f88eb7756f3461d60d78ff580eb8f4f6396e2b427eadb7`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_naive_severe_seed166.json` (SHA-256 `74d2165df0993af4095ba5aea5f2716616c8b7077ae8b3aadffba7072fead3e2`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_teacher_clean_seed164.json` (SHA-256 `2ae870010e5a439424ea08afb8c41ebf7498373eedade89adb07e7f3e38e27cf`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_teacher_clean_seed165.json` (SHA-256 `510caa24ba4bae60ef1c5b929bc79ab2062572eba6482ca10ab0e9edc4bee8c3`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_teacher_clean_seed166.json` (SHA-256 `479729fb355bbfc82b4f2ec91277f9d312f53104d78edcaaac301c5220fa49ea`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_teacher_severe_seed164.json` (SHA-256 `bc868cbd1d5663469c41cc4e8a87816423e116262d66b466248196426f9f3d65`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_teacher_severe_seed165.json` (SHA-256 `edd3be4ca1810f2456ffbbe2615693115937e5da81a10ce7b70c6387033192e9`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_smol_teacher_severe_seed166.json` (SHA-256 `9f0b6cc0c4ab195faaafd9793468442426c99eced001e3757ea0892b15301e47`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_naive_clean_seed164.json` (SHA-256 `920778ef258354114e8ae5d190c8823020e35a64f9cfbe45baae28c102797fd3`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_naive_clean_seed165.json` (SHA-256 `9ee8976d5459f78ebdbadd45301706b3a0ecff67cdd722e45f715bbb4cb89568`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_naive_clean_seed166.json` (SHA-256 `4a8fe657a9d24136a5569baad0c34a95946ae13db320b541b1c53cc1fe3d2898`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_naive_severe_seed164.json` (SHA-256 `e4e21e83383522ee9f2d983a35c4d2d98a07e34957acaff1ac146fa5ceb7cd22`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_naive_severe_seed165.json` (SHA-256 `973fee3ce6572de8b393c623ef0950d5b8cf2d861344b3ca3dba73d4b79b68b2`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_naive_severe_seed166.json` (SHA-256 `ee5b94b2ca5d7b7911fb5c7c2882178210507a1e057002770a71455c52989d85`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_teacher_clean_seed164.json` (SHA-256 `c2eca3a77aec06532ecff19021046c9867366ad2b451939dd9b37193a42b462d`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_teacher_clean_seed165.json` (SHA-256 `e223e5dc81bd192b0f2eaac71a81ddf53c9e56f94fa9f17cc6854881416c9d05`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_teacher_clean_seed166.json` (SHA-256 `a21ddff9552819002bf4e580677aaa1d744180709f7c4a6a54811d745dad0eb9`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_teacher_severe_seed164.json` (SHA-256 `04001c520cf38535cc515d9744b4e2cf53ac25743d21e671532b018173184edf`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_teacher_severe_seed165.json` (SHA-256 `7e5215b98c551887465e07053a00df03c99d45831ea33513d4dd43bc2ca3a4c5`)
- `kafa_script/analysis_outputs/gsm8k_crossfamily_final_tiny_teacher_severe_seed166.json` (SHA-256 `1d864f6bff47b3ac1dc9ded5f91a430b960c946349a898d2ed4e02ea7a18c562`)
