# GSM8K public-anchor teacher: full development replication

**Status: completed development replication; not a final test result.** The
method and hyperparameters were selected on the preceding 64-question
development check. This document evaluates them once on all 512 frozen
development questions using new seeds 55--57. Official GSM8K test labels were
not loaded.

## Protocol verification

- Client: Qwen2.5-0.5B rank-4 LoRA; teacher: Qwen2.5-1.5B rank-4 LoRA.
- Teacher trains only on the first 128 public anchors for 40 steps. It ranks
  numeric-interface-compatible client examples by teacher loss; the 128 lowest
  are used for 40 client steps.
- Naive and teacher-routed runs use the same client pool, seed, mixture,
  optimizer, prompt, and all 512 frozen development questions.
- Severe mix is 40% deterministic final-answer conflict and 20% frozen SQuAD
  off-task QA. Kinds below are post-selection audit labels only.

## Severe condition

| Seed | Naive EM | Teacher-routed EM | Difference | Retained clean / conflict / off-task |
|---:|---:|---:|---:|---:|
| 55 | 4.30 | 6.84 | +2.54 pp | 99 / 18 / 11 |
| 56 | 2.15 | 8.01 | +5.86 pp | 101 / 23 / 4 |
| 57 | 0.20 | 4.88 | +4.69 pp | 100 / 24 / 4 |
| Mean ± SD | 2.21 ± 2.05 | 6.58 ± 1.58 | +4.36 pp |

The paired severe improvement is +4.36 pp, descriptive 95% t interval
[+0.18,+8.55], t(2)=4.487, one-sided p=0.0231. This is statistically
descriptive development evidence with three seeds; it is not corrected across
the prior candidate search and cannot be a paper claim by itself.

## Clean safety

| Seed | Naive clean EM | Teacher-routed clean EM | Difference |
|---:|---:|---:|---:|
| 55 | 7.81 | 9.38 | +1.56 pp |
| 56 | 5.27 | 8.20 | +2.93 pp |
| 57 | 5.08 | 7.23 | +2.15 pp |

All paired clean differences exceed the frozen -2 pp safety bound. The
candidate therefore passes the full-development advancement rule. A separate
final protocol with disjoint seeds, fixed settings, and the 1,319-example
official test is required before any final claim.
