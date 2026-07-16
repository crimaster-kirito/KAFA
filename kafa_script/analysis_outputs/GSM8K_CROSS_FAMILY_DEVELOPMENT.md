# GSM8K cross-family development results

**Status: frozen development evaluation; not an official-test result.**

## Verification

- Protocol SHA-256: `89b29449bc15e486ace3c691fe538d63f550360c11684f79100deb909e8b0716`.
- Data manifest SHA-256: `03ce4cac6dad028975adc1467d5c841e69a4f14dafb1427af71599b7174a7292`.
- All 24 records use all 512 frozen development questions and seeds 61--63.

## Results

| Client family | Condition | Naive EM | Teacher EM | Paired gain | Descriptive 95% CI | Gate |
|---|---|---:|---:|---:|---:|---|
| smol | clean | 2.15% | 2.34% | +0.20 pp | [-1.09, +1.48] | clean-safe |
| smol | severe | 1.17% | 2.08% | +0.91 pp | [-2.32, +4.14] | positive |
| tiny | clean | 1.82% | 3.65% | +1.82 pp | [-0.62, +4.26] | clean-safe |
| tiny | severe | 0.52% | 2.15% | +1.63 pp | [+0.41, +2.85] | positive |

Per-seed audit:

- smol clean: per-seed gains +0.39, -0.39, +0.59 pp.
- smol severe: per-seed gains +2.34, +0.59, -0.20 pp.
- tiny clean: per-seed gains +0.78, +2.73, +1.95 pp.
- tiny severe: per-seed gains +1.56, +1.17, +2.15 pp.

## Decision

Both non-Qwen client families pass the predeclared development gate. This permits drafting a separate holdout protocol; it is not itself a cross-family final result.

This study transfers teacher scores across model families but remains centralized local adaptation, not a full federated server comparison. Confidence intervals are descriptive because each family has only three seeds.

## Source records

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
