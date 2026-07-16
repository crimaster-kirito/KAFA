# TREC server-transfer result

**Status: official frozen-test paper evidence.**

All confidence intervals and paired t tests below describe variation over the frozen training seeds only; they do not quantify test-sample or task-population uncertainty.

## Per-seed results

| Seed | Anchor acc. | Routed acc. | Acc. delta | Macro-F1 delta | Class-NLL delta | ECE delta | Gates triggered | Wire bytes | Anchor / routed wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 251 | 81.80% | 81.00% | -0.80 pp | -1.06 pp | +0.0035 | -0.0367 | 5/6 | 19830 | 20.1 / 186.9 |
| 252 | 79.40% | 80.40% | +1.00 pp | +1.91 pp | -0.0523 | -0.0479 | 6/6 | 19830 | 20.0 / 196.8 |
| 253 | 82.60% | 80.80% | -1.80 pp | -0.91 pp | -0.0139 | -0.0227 | 5/6 | 19830 | 20.2 / 207.8 |
| 254 | 79.60% | 82.20% | +2.60 pp | +2.97 pp | -0.0602 | -0.0392 | 5/6 | 19830 | 20.1 / 197.2 |
| 255 | 80.20% | 81.20% | +1.00 pp | +0.73 pp | -0.0292 | -0.0398 | 6/6 | 19830 | 20.1 / 175.5 |

## Paired routed-minus-anchor inference

| Metric | Mean delta | Descriptive 95% t-CI | Two-sided paired t p |
|---|---:|---:|---:|
| Accuracy | +0.4000 pp | [-1.7362, +2.5362] pp | 0.6306 |
| Macro-F1 | +0.7290 pp | [-1.4467, +2.9048] pp | 0.4049 |
| Class NLL | -0.0304 | [-0.0632, +0.0024] | 0.06157 |
| ECE | -0.0373 | [-0.0487, -0.0259] | 0.0008125 |

## Routing and cost audit

- Routing gate triggers: 27/30 client-rounds.
- Routed wire payload: mean 19830 bytes (range 19830--19830). Anchor-only sends 0 bytes.
- Mean wall time: anchor 20.07 s; routed 192.84 s; ratio 9.61x.

## Raw records

- `kafa_script/analysis_outputs/trec_server/official/seed251_public_anchor_only.json` (SHA-256 `eb24f339af302ec27202b6950b32d6b948b19627dcb52477bc4e7d2b18dbbc6c`)
- `kafa_script/analysis_outputs/trec_server/official/seed251_routed_distillation.json` (SHA-256 `8ea4e3dc18cf18785899149ec34e53f2b3320433386224299a2742ab759087de`)
- `kafa_script/analysis_outputs/trec_server/official/seed252_public_anchor_only.json` (SHA-256 `2194f88d2e22ec0253c2d58b51edd17e81542cefbf680b21e144129a9604034b`)
- `kafa_script/analysis_outputs/trec_server/official/seed252_routed_distillation.json` (SHA-256 `637b2b5caf408ffcd2eb9d8a916775a7faaa372a3a7463876e4ca090cd1c4e2a`)
- `kafa_script/analysis_outputs/trec_server/official/seed253_public_anchor_only.json` (SHA-256 `3b3bc2714314668372166fb2421c0b8b3ef44532f47aa5fb149ea4bc5b4ba9fe`)
- `kafa_script/analysis_outputs/trec_server/official/seed253_routed_distillation.json` (SHA-256 `571b4b394c01b69e1b4ebce54bc6b306e2997f331b7826c6366f9610c983578a`)
- `kafa_script/analysis_outputs/trec_server/official/seed254_public_anchor_only.json` (SHA-256 `6f0959beee13b60c0d0e24eb8625ef3555016bd3b0a62505bf1449cca33f6564`)
- `kafa_script/analysis_outputs/trec_server/official/seed254_routed_distillation.json` (SHA-256 `40a3e06f5d1e8e21ffd37b6618bbe4f141b98cc8e0be75b9a610fcc82a7689ac`)
- `kafa_script/analysis_outputs/trec_server/official/seed255_public_anchor_only.json` (SHA-256 `f832a499122bbcd4a0c413cc0005749f0f3edb145b14b60de89b4626ab967708`)
- `kafa_script/analysis_outputs/trec_server/official/seed255_routed_distillation.json` (SHA-256 `f2aee9a45d19e8904279931572824b30779233276b8b3c70b9b9f595ebcd0cb2`)
