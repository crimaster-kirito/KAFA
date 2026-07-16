# GSM8K Stage-B development evidence

**Evidence status: development only (`paper_result: false`).** These results
use a 64-question prefix of the frozen GSM8K development partition and seeds
52--54. They must not be treated as a final GSM8K result, a cross-task claim,
or a submission table.

## Verified protocol

- Target: normalized exact match of a greedily generated final numeric answer.
- Model/client: Qwen2.5-0.5B with rank-4 LoRA; 40 local steps, batch size 4,
  learning rate `1e-3`, maximum prompt length 256, and 16 generated tokens.
- Clean condition: all 256 examples are GSM8K final-answer examples.
- Severe condition: 103 clean GSM8K examples, 102 deterministic in-pool
  final-answer conflicts, and 51 deterministic SQuAD extractive-QA examples.
- Development evaluation uses no official GSM8K test questions or labels.
- The fixed manifest SHA-256 is
  `03ce4cac6dad028975adc1467d5c841e69a4f14dafb1427af71599b7174a7292`;
  the frozen off-task SQuAD cache SHA-256 is
  `c1a980a88112a7630c07de593713d86e1414bc3039c7779b6506a622c0711620`.

## Local mixed-data mechanism check

| Seed | Clean EM | Severe EM | Severe − clean |
|---:|---:|---:|---:|
| 52 | 6.25 | 0.00 | -6.25 pp |
| 53 | 7.81 | 4.69 | -3.12 pp |
| 54 | 6.25 | 1.56 | -4.69 pp |
| Mean ± SD | 6.77 ± 0.90 | 2.08 ± 2.39 | -4.69 pp |

The paired difference is -4.69 pp (descriptive 95% t interval [-8.57,
-0.81]). A two-sided paired t-test yields t(2) = -5.196, uncorrected p =
0.0351. This p-value is reported for transparency only: the low-powered
three-seed development study was used to verify a mechanism and select no
final result, so it cannot support a paper claim.

All post-training generations parsed to a numeric answer in these runs; the
initial untuned model had a 42.2% invalid-output rate. A prior right-padded
generation smoke (`gsm8k_clean_dev_seed52.json`) is **invalidated and
excluded** because decoder-only batch generation requires left padding. The
left-padded records named `*_leftpad.json` are the only records used above.

## Interpretation and next gate

This confirms the intended Stage-B stressor: answer conflict plus unrelated
textual QA harms local numerical-answer adaptation under the same model and
optimizer budget. It does not validate the router, heterogeneous distillation,
or mathematical reasoning fidelity. The next development gate is a
task-appropriate router/teacher mechanism evaluated against this clean/severe
local baseline on the full 512-example development partition before any
official GSM8K test protocol is frozen.

## Relative-anchor-gradient router diagnostic

The development protocol v3 tests a task-appropriate selector: 16 frozen
numeric anchors form a mean LoRA gradient, and each local example is ranked by
the mean percentile rank of negative gradient cosine and supervised loss. The
selector does not read contamination kind. It gates when the client/reference
25th-percentile cosine ratio is below 0.8 and then retains 50% of examples.

| Seed | Naive severe EM | Routed severe EM | Difference | Gate ratio | Retained clean / conflict / off-task |
|---:|---:|---:|---:|---:|---:|
| 52 | 0.00 | 1.56 | +1.56 pp | 0.267 | 71 / 26 / 31 |
| 53 | 4.69 | 4.69 | +0.00 pp | 0.204 | 72 / 30 / 26 |
| 54 | 1.56 | 4.69 | +3.12 pp | 0.350 | 70 / 33 / 25 |
| Mean | 2.08 | 3.65 | +1.56 pp | — | — |

The directional paired test for routed EM greater than naive severe EM is
t(2)=1.732, p=0.113; the descriptive 95% interval for the improvement is
[-2.32, +5.44] pp. It therefore **fails the net-benefit selection gate** and
is not eligible for a final GSM8K protocol. The selector is clean-safe: for
all three clean seeds the gate does not trigger (ratios 0.837, 0.836, 0.825),
and routed outputs exactly equal naive clean outputs. This is useful mechanism
evidence, but not an algorithmic win. Notably, it retains 25--31 of 51
off-task examples, motivating stronger task-identification signals rather than
threshold tuning on this small development set.

## Numeric-interface router diagnostic

Protocol v4 adds an observable answer-interface check after the same gate: a
non-numeric supervised answer cannot train a numerical-final-answer task. It
does not inspect `kind`, source provenance, or the hidden correct answer of a
conflict example. Because some SQuAD answers are themselves numeric, this
filter is deliberately not treated as a perfect task detector.

| Seed | Naive severe EM | Numeric-interface routed EM | Difference | Retained clean / conflict / off-task |
|---:|---:|---:|---:|---:|
| 52 | 0.00 | 1.56 | +1.56 pp | 85 / 40 / 3 |
| 53 | 4.69 | 3.12 | -1.56 pp | 83 / 40 / 5 |
| 54 | 1.56 | 4.69 | +3.12 pp | 85 / 41 / 2 |
| Mean | 2.08 | 3.12 | +1.04 pp | — |

The candidate is exactly clean-safe: its gate does not trigger and its clean
outputs exactly match naive clean outputs for all three seeds. It substantially
reduces retained off-task examples, yet the severe paired improvement has a
descriptive 95% interval of [-4.89, +6.97] pp and one-sided p=0.264. Under
the frozen v4 rule, it is therefore **rejected** for final testing. This
isolates the remaining difficulty: numerical answer conflicts survive a
perfectly observable task-interface check.

## Loss-ranked numeric router diagnostic

An audit-only seed-52 analysis found per-example loss has conflict AUROC 0.756,
whereas gradient cosine mostly identifies off-task examples. Protocol v5
therefore retained numeric-interface-compatible examples by lowest loss only
after the same gate. This is a distinct frozen candidate, not a threshold
search.

| Seed | Naive severe EM | Loss-ranked EM | Difference | Retained clean / conflict / off-task |
|---:|---:|---:|---:|---:|
| 52 | 0.00 | 0.00 | +0.00 pp | 76 / 49 / 3 |
| 53 | 4.69 | 6.25 | +1.56 pp | 77 / 45 / 6 |
| 54 | 1.56 | 3.12 | +1.56 pp | 77 / 49 / 2 |
| Mean | 2.08 | 3.12 | +1.04 pp | — |

Clean gates again do not trigger, and all clean endpoints exactly equal their
naive references. However, the severe paired interval is [-1.20, +3.28] pp
with one-sided p=0.092. The candidate is **rejected** under the frozen v5
rule. Its 0.756 loss AUROC does not yield a selective enough top-half set:
45--49 of 102 conflicts remain. This is evidence against treating local loss
as a sufficient conflict-trust signal in this setting.
