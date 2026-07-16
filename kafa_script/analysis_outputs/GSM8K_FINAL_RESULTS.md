# GSM8K official holdout results

Frozen protocol: `gsm8k_anchor_teacher_final_protocol_v1.json`; official test split, 1319 questions, seeds 158/159/160. Raw records are retained with `paper_result:false` until the manuscript review is complete.

| condition | naive mean EM | teacher-router mean EM | paired gain |
|---|---:|---:|---:|
| clean | 6.62% | 7.91% | +1.29 pp |
| severe (.4 conflict, .2 off-task) | 3.49% | 6.82% | +3.34 pp |

Per-seed paired gains were +1.44/+1.82/+0.61 pp (clean) and +2.65/+3.71/+3.64 pp (severe). With only three seeds these are descriptive paired comparisons; the corresponding t statistics are 3.60 and 9.76 (df=2), and should not be treated as definitive significance claims. The result supports the frozen teacher-endpoint mechanism on this benchmark, but does not yet establish a federated or cross-family advantage.

Raw artifacts: `gsm8k_final_{naive,teacher}_{clean,severe}_seed{158,159,160}.json`.
