# TREC server-transfer result

**Status: development sanity evidence; non-paper result.**

All confidence intervals and paired t tests below describe variation over the frozen training seeds only; they do not quantify test-sample or task-population uncertainty.

## Per-seed results

| Seed | Anchor acc. | Routed acc. | Acc. delta | Macro-F1 delta | Class-NLL delta | ECE delta | Gates triggered | Wire bytes | Anchor / routed wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 241 | 83.33% | 84.72% | +1.39 pp | +1.36 pp | -0.1129 | -0.0268 | 5/6 | 19830 | 21.0 / 180.1 |
| 242 | 84.72% | 83.33% | -1.39 pp | -1.27 pp | -0.0709 | +0.0167 | 6/6 | 19830 | 22.1 / 180.4 |
| 243 | 84.72% | 81.94% | -2.78 pp | -2.47 pp | -0.0820 | -0.0071 | 5/6 | 19830 | 16.1 / 176.6 |

## Paired routed-minus-anchor inference

| Metric | Mean delta | Descriptive 95% t-CI | Two-sided paired t p |
|---|---:|---:|---:|
| Accuracy | -0.9259 pp | [-6.1962, +4.3443] pp | 0.5286 |
| Macro-F1 | -0.7950 pp | [-5.6659, +4.0760] pp | 0.5553 |
| Class NLL | -0.0886 | [-0.1427, -0.0345] | 0.01953 |
| ECE | -0.0058 | [-0.0598, +0.0483] | 0.6919 |

## Routing and cost audit

- Routing gate triggers: 16/18 client-rounds.
- Routed wire payload: mean 19830 bytes (range 19830--19830). Anchor-only sends 0 bytes.
- Mean wall time: anchor 19.73 s; routed 179.03 s; ratio 9.07x.

## Raw records

- `kafa_script/analysis_outputs/trec_server/development/seed241_public_anchor_only.json` (SHA-256 `d547e762fa3f27b1caf9c2a6fad0f60ac24a56338292124c7e695917ddcd6fbb`)
- `kafa_script/analysis_outputs/trec_server/development/seed241_routed_distillation.json` (SHA-256 `a1ad69a2187a6fe674b63c4f6969344861598a4b039b43b3c1bcb9a1e3867774`)
- `kafa_script/analysis_outputs/trec_server/development/seed242_public_anchor_only.json` (SHA-256 `e1e649e9b09abbf8ac1016c2867cd06b97e8a66c6f5d2daf14778b0a56c346cb`)
- `kafa_script/analysis_outputs/trec_server/development/seed242_routed_distillation.json` (SHA-256 `9e7e6af805df3d02986727e0b81e0354e99ff98c1f38e202fec0583e70299525`)
- `kafa_script/analysis_outputs/trec_server/development/seed243_public_anchor_only.json` (SHA-256 `63a05f3b896d410c03824cc81d964d3ae17b99a075d4fe3f8d90604371e860e8`)
- `kafa_script/analysis_outputs/trec_server/development/seed243_routed_distillation.json` (SHA-256 `328608db42e4a833c60e8b96dd1c2f3c03ea7477163227ba94a6d0e6d7fb292d`)
