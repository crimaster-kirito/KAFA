# Final heterogeneous-distillation holdout result

**Status: completed frozen holdout evaluation; negative primary result.** This derived report reads the immutable per-seed JSON records and does not alter them.  The raw records retain their conservative historical `paper_result: false` tag; their scope is verified here from the recorded official-test fields.

## Verification

- Frozen protocol: `kafa_script/heterogeneous_routed_final_protocol_v2.json` (SHA-256 `69ab272c7e7cbbf168c347b97d4d89f993c8a2d047127431541be8d617938df6`).
- Frozen data manifest: `073419bb7b444f358da469c330af26abdb005fbc8745b602f83b40b38e28e0e5`.
- All six records declare 2 rounds, 40 client and 40 server steps per round, 128 public anchors, the Qwen2.5-1.5B server, and all 7,600 official AG News test examples.
- The predeclared primary comparison is public-anchor-only versus relative-balanced-rank routed canonical-probability distillation (weight 0.25); seeds are 142, 143, and 144.

## Per-seed primary result

| Seed | Anchor-only accuracy | Routed accuracy | Routed − anchor (pp) | Anchor-only macro-F1 | Routed macro-F1 | Routed target-answer loss | Routed ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 142 | 84.50 | 83.92 | -0.58 | 84.50 | 83.86 | 0.741 | 0.015 |
| 143 | 83.66 | 83.18 | -0.47 | 83.44 | 82.97 | 0.708 | 0.021 |
| 144 | 84.87 | 84.24 | -0.63 | 84.90 | 84.19 | 0.691 | 0.011 |

## Paired summary

| Metric | Anchor-only mean ± SD | Routed mean ± SD | Routed − anchor mean (95% t CI) |
|---|---:|---:|---:|
| accuracy | 84.34 ± 0.62 | 83.78 ± 0.54 | -0.56 pp [-0.76, -0.36] |
| macro_f1 | 84.28 ± 0.75 | 83.67 ± 0.63 | -0.61 pp [-0.91, -0.31] |
| target_answer_loss | 0.834 ± 0.049 | 0.713 ± 0.025 | -0.121 [-0.241, +0.000] |
| ece | 0.054 ± 0.017 | 0.016 ± 0.005 | -0.039 [-0.068, -0.009] |

A descriptive directional paired test for **routed accuracy > anchor-only accuracy** has t(2) = -12.095, p = 0.9966; it provides no support for the intended direction. All three paired accuracy differences are negative.
The t intervals are descriptive with only three seeds and must not be read as broad generalization evidence.

## Cost and confidence

- Mean wall time: anchor-only 279.8 s; routed 561.3 s (2.01×).
- Recorded client-to-server payload: anchor-only 21750 bytes; routed 21750 bytes. The strict anchor-only control deliberately executes and records the same client loop, but its server objective assigns the teacher term weight zero; this is equal-budget control work, not useful communication.
- Routing lowers the token-normalized target-answer loss and 10-bin ECE while reducing both accuracy and macro-F1. The raw JSON key `label_nll` is a historical misnomer for target-answer loss in this AG News evaluator; it is not normalized class NLL. Lower ECE does not rescue the predeclared primary accuracy claim.

## Interpretation and boundary

Under this *single, frozen severe-mixture AG News configuration*, the tested routed heterogeneous distillation method reduces task accuracy against public-anchor-only server adaptation. The defensible finding is diagnostic: local routing can suppress harmful examples, yet its resulting teacher probabilities did not add enough useful signal to justify their cost or overcome the public-anchor baseline.

This neither proves that all cross-architecture federation fails nor supports a new general-purpose aggregation algorithm. The broader artifact adds GSM8K and two natural weak-label languages, but stronger compatible comparators, privacy analysis, and a complete positive federated method remain open submission risks.

## Source records

- `kafa_script/analysis_outputs/final_anchor_only_strict_seed142.json` (SHA-256 `c4895ac9c8087c174402fa39e2c4f30eb8273d750111f9475a16c9c21253e7eb`)
- `kafa_script/analysis_outputs/final_anchor_only_strict_seed143.json` (SHA-256 `72b3d863b6801ab70eea12a68679f84fc2f602900ebb041a700871ab59958108`)
- `kafa_script/analysis_outputs/final_anchor_only_strict_seed144.json` (SHA-256 `6e9a428a2040b1d340efd97e5c4efedf8e65c649a6eddf9a27c6dc508654b5a6`)
- `kafa_script/analysis_outputs/final_routed_seed142.json` (SHA-256 `e2e4bed9103cbceb51da6d06a92f645013bfc0da1fd3ad972c0dadcf41b9fd8a`)
- `kafa_script/analysis_outputs/final_routed_seed143.json` (SHA-256 `c98a4af1288bc26932801d2f91234cc32f359a7e5eb98df3170f90fad2be0da1`)
- `kafa_script/analysis_outputs/final_routed_seed144.json` (SHA-256 `8a212a35c1967d20e465c8a263f7658b5697007935826583d5b985e34095842f`)
