# Hausa natural-noise Co-Teaching development result

**Status: frozen development-only evidence; the clean test split was not evaluated.**

## Verification

- Protocol SHA-256: `4667be5e74a0d43090bb3a9536855e180965a0829140507a572f703ce7e5efd8`.
- All 18 records use all 2,045 weakly labelled training headlines and the same 290-example clean development split.
- The measured 50.37% natural error rate matches the source data and is given to Co-Teaching as oracle information.
- Every evaluation hash equals source `dev.tsv` (`ca5629d297dec8c6f7878180ba2aa0a5852bafb57acc6b3ea4dc43cf600446e7`), differs from `test.tsv` (`a174e3071e4d1e4713fb73aa93de583909d3bf5231abd0d09ab6ae4087a8f79b`), and no natural-noise test record exists.
- Three family-wise accuracy tests are Holm-adjusted together; macro-F1 is secondary.

## Results

| Client | Naive accuracy | Co-Teaching accuracy | Gain [95% CI] | Naive macro-F1 | Co-Teaching macro-F1 | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 44.94% | 33.33% | -11.61 pp [-24.50, +1.28] | 40.90% | 18.58% | 0.9697 | 1.0000 |
| smol | 45.29% | 28.16% | -17.13 pp [-26.21, -8.05] | 40.40% | 9.68% | 0.9926 | 1.0000 |
| tiny | 47.82% | 28.74% | -19.08 pp [-26.06, -12.10] | 44.77% | 10.59% | 0.9964 | 1.0000 |

Per-seed accuracy gains:

- qwen: -8.28 pp, -8.97 pp, -17.59 pp
- smol: -16.55 pp, -13.79 pp, -21.03 pp
- tiny: -20.34 pp, -15.86 pp, -21.03 pp

## Selection and cost audit

- qwen: selected clean-label agreement 50.9%; train-wall ratio 1.74x; adapter ratio 2.00x.
- smol: selected clean-label agreement 45.3%; train-wall ratio 1.80x; adapter ratio 2.00x.
- tiny: selected clean-label agreement 46.6%; train-wall ratio 1.69x; adapter ratio 2.00x.

## Gate decision

The predeclared development gate fails. The clean test split must not be evaluated, and this result cannot support a natural-noise holdout claim.

This is a compatible Co-Teaching comparison, not Delora: Delora's published pipeline uses GPT-4o for positive-set construction and noisy-sample relabeling.

## Source artifacts

- `kafa_script/analysis_outputs/hausa_natural_qwen_coteach_dev_seed81.json` (SHA-256 `cb0a3219cb5c789159a5d1844d40d1fe2c27dc57cfc11161765b5302919aa598`)
- `kafa_script/analysis_outputs/hausa_natural_qwen_coteach_dev_seed82.json` (SHA-256 `e08717b9324d4e165cd25ed636a2ab25c95b65091644980c2948297cd0c64aed`)
- `kafa_script/analysis_outputs/hausa_natural_qwen_coteach_dev_seed83.json` (SHA-256 `9f2f899fb8b4f31f7089769e944140c316b06b682084e27c905da2a3e194ca20`)
- `kafa_script/analysis_outputs/hausa_natural_qwen_naive_dev_seed81.json` (SHA-256 `fc55894ae87d882f362edf7a8125a5131d6db581a5ce931b3e06d715e7c6cfbc`)
- `kafa_script/analysis_outputs/hausa_natural_qwen_naive_dev_seed82.json` (SHA-256 `738862c58ee4bffd82dfe4c561346f735cbf7dfc5f9f098aec7860bb0fb06ef1`)
- `kafa_script/analysis_outputs/hausa_natural_qwen_naive_dev_seed83.json` (SHA-256 `5f126dff9eea017427e45a7611745d81bbf0c4b434a6efcd7aaacb26a8e7778a`)
- `kafa_script/analysis_outputs/hausa_natural_smol_coteach_dev_seed81.json` (SHA-256 `398826eeeb56a25c92e4f6df129e9e89ae89266ef0de228f448e5a0c96e75a2f`)
- `kafa_script/analysis_outputs/hausa_natural_smol_coteach_dev_seed82.json` (SHA-256 `41fdf35a23ea75be46b92f291ffbcbb1020e275e577a2a747b9a3f10950d0950`)
- `kafa_script/analysis_outputs/hausa_natural_smol_coteach_dev_seed83.json` (SHA-256 `1616f98b4e8d3869be8bcfdd516db5de5899b3ee03748dd0dd06c6e2a0555fbd`)
- `kafa_script/analysis_outputs/hausa_natural_smol_naive_dev_seed81.json` (SHA-256 `83063ba0c93b8ba306e2453cb6eb1ef0ec76fd71bdb1fceb7178dc6c68d0c3ed`)
- `kafa_script/analysis_outputs/hausa_natural_smol_naive_dev_seed82.json` (SHA-256 `e1bb97020b0986753731f27a977348157b66bdc26fdbcdbefb96687f2cb9c313`)
- `kafa_script/analysis_outputs/hausa_natural_smol_naive_dev_seed83.json` (SHA-256 `4b307629958e4696f0e0fb2dc274d75c3423aabec8a314edbaf0516a7dba830e`)
- `kafa_script/analysis_outputs/hausa_natural_tiny_coteach_dev_seed81.json` (SHA-256 `5cea4284297b44857d5a45c4f9981e3462e6b1eeae9dcc45f9df16313b840cab`)
- `kafa_script/analysis_outputs/hausa_natural_tiny_coteach_dev_seed82.json` (SHA-256 `4663a0c6e178fa2a5927cb664859c2911ded66863d0cc1ea9fd7c31fd8f16a55`)
- `kafa_script/analysis_outputs/hausa_natural_tiny_coteach_dev_seed83.json` (SHA-256 `3192f886322d141e0d8015a313e87862548725e9cdc3ee7f4187eed20faafe1e`)
- `kafa_script/analysis_outputs/hausa_natural_tiny_naive_dev_seed81.json` (SHA-256 `bbeaee419209662de9328e3ca948841045debbed877dd5f9d78dbb8dc64ce9d4`)
- `kafa_script/analysis_outputs/hausa_natural_tiny_naive_dev_seed82.json` (SHA-256 `a1eeab361c50d233f96c6228f1b0fc1ca4737ff187555deac656a918fe57e649`)
- `kafa_script/analysis_outputs/hausa_natural_tiny_naive_dev_seed83.json` (SHA-256 `b9067eb2a7f10dde53432afc69bab0496a053e7208d223da32eaca5164d3bb97`)
