# Yoruba natural-noise Co-Teaching development result

**Status: frozen development-only evidence; the clean test split was not evaluated.**

## Verification

- Protocol SHA-256: `faea4ad692e39ed5c48be7f52ab00da8feaf5d81acc6841a8e345174475fddb6`.
- All 18 records use all 1,340 weakly labelled training headlines and the same 189-example clean development split.
- The measured 33.28% natural error rate matches the source data and is given to Co-Teaching as oracle information.
- Every evaluation hash equals source `dev.tsv` (`2d4fe2cd7a0845b52d3e0ee2b6f272b0918df477d64aa301b7adca03cbb20a8d`), differs from `test.tsv` (`9934b609f76d850d2468fcb9e0421dcb9a33f59a78cea33bc67e5510dc034e0f`), and no natural-noise test record exists.
- Three family-wise accuracy tests are Holm-adjusted together; macro-F1 is secondary.

## Results

| Client | Naive accuracy | Co-Teaching accuracy | Gain [95% CI] | Naive macro-F1 | Co-Teaching macro-F1 | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 67.37% | 67.90% | +0.53 pp [-13.13, +14.19] | 60.31% | 59.49% | 0.4415 | 0.8830 |
| smol | 64.55% | 67.20% | +2.65 pp [-7.62, +12.91] | 55.45% | 57.28% | 0.1915 | 0.5745 |
| tiny | 74.60% | 70.37% | -4.23 pp [-23.32, +14.86] | 73.03% | 58.51% | 0.7796 | 0.8830 |

Per-seed accuracy gains:

- qwen: -2.65 pp, +6.88 pp, -2.65 pp
- smol: -2.12 pp, +4.76 pp, +5.29 pp
- tiny: -12.17 pp, -3.70 pp, +3.17 pp

## Selection and cost audit

- qwen: selected clean-label agreement 78.2%; train-wall ratio 1.64x; adapter ratio 2.00x.
- smol: selected clean-label agreement 78.2%; train-wall ratio 1.82x; adapter ratio 2.00x.
- tiny: selected clean-label agreement 80.6%; train-wall ratio 1.89x; adapter ratio 2.00x.

## Gate decision

The predeclared development gate fails. The clean test split must not be evaluated, and this result cannot support a natural-noise holdout claim.

This is a compatible Co-Teaching comparison, not Delora: Delora's published pipeline uses GPT-4o for positive-set construction and noisy-sample relabeling.

## Source artifacts

- `kafa_script/analysis_outputs/yoruba_natural_qwen_coteach_dev_seed84.json` (SHA-256 `f09bf88a690ab7a61172715f9db2d0d219d16bc0ed4c65cef8bdc7be0b960907`)
- `kafa_script/analysis_outputs/yoruba_natural_qwen_coteach_dev_seed85.json` (SHA-256 `afce6ce6a3c99cc199e518df6e02caa9521adb5971dd770df7fde419cbedba29`)
- `kafa_script/analysis_outputs/yoruba_natural_qwen_coteach_dev_seed86.json` (SHA-256 `d0b34a74c7dfbb5debebee04bc7db0ff4f5f984b72c09eb54dc494d2e1790926`)
- `kafa_script/analysis_outputs/yoruba_natural_qwen_naive_dev_seed84.json` (SHA-256 `967ec9c12411b888f201e6161796c313bcd560d4ed1365f42fd88ba2d588eaac`)
- `kafa_script/analysis_outputs/yoruba_natural_qwen_naive_dev_seed85.json` (SHA-256 `7d8e98f6eda2d642994532d56225293c2a73010b67eb21390548d621444a07aa`)
- `kafa_script/analysis_outputs/yoruba_natural_qwen_naive_dev_seed86.json` (SHA-256 `371c0a1dc1e37ac7c76b703eb8295758066a5d976638d96a307630d2a619d94e`)
- `kafa_script/analysis_outputs/yoruba_natural_smol_coteach_dev_seed84.json` (SHA-256 `7459ab834bd34aa897bf4fe2955776c342864890c038af8e3a61131ce700abe6`)
- `kafa_script/analysis_outputs/yoruba_natural_smol_coteach_dev_seed85.json` (SHA-256 `840a194f7a97cef8baaffa9759b24bd192fb421da2b78cf1be62df8b68b80650`)
- `kafa_script/analysis_outputs/yoruba_natural_smol_coteach_dev_seed86.json` (SHA-256 `3d3d13ca44f77cce367c24a4ab685e2a2d8cdaec91fb80d7d0347e0d6e63b535`)
- `kafa_script/analysis_outputs/yoruba_natural_smol_naive_dev_seed84.json` (SHA-256 `a227dec0b4e168dc4d82747400b3809f9fdef0c063a294d401e12da04b36b2f7`)
- `kafa_script/analysis_outputs/yoruba_natural_smol_naive_dev_seed85.json` (SHA-256 `26d5b2191128543eade2e6494b30f8c1b5d6e268c0d8504772186c8106192a0e`)
- `kafa_script/analysis_outputs/yoruba_natural_smol_naive_dev_seed86.json` (SHA-256 `e8df3d228c41f837175e97e36e93f69f51c39764fd1e7443043ebebe115ca02c`)
- `kafa_script/analysis_outputs/yoruba_natural_tiny_coteach_dev_seed84.json` (SHA-256 `c2321a2eb88c6157600658e93460b44eab15afbf42902f548f95f900f0aabb2f`)
- `kafa_script/analysis_outputs/yoruba_natural_tiny_coteach_dev_seed85.json` (SHA-256 `24b67a18119bcb809cb299e673d2f3136770a87eb415286f2cf676766bc288a2`)
- `kafa_script/analysis_outputs/yoruba_natural_tiny_coteach_dev_seed86.json` (SHA-256 `034b8c5e508bf7c00a4f197bd5a0ed88c0af01ef4c243a5dead93ec475e0e730`)
- `kafa_script/analysis_outputs/yoruba_natural_tiny_naive_dev_seed84.json` (SHA-256 `6aca40b6e51fc12b93027a31b21eafb3a5486d4b96368cf7c57d71a9e244e260`)
- `kafa_script/analysis_outputs/yoruba_natural_tiny_naive_dev_seed85.json` (SHA-256 `bc05b4fd97286103f0e161a69d1a9c21069ae2623eccc6e43218fbbf55fcbd90`)
- `kafa_script/analysis_outputs/yoruba_natural_tiny_naive_dev_seed86.json` (SHA-256 `eaab3a5bdccb8664e944d08bcdf73847376e860e614de298e8157f874afec00d`)
