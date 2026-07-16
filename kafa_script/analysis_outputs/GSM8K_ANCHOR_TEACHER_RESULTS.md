# GSM8K public-anchor teacher: initial development result

**Status: positive development evidence only (`paper_result: false`).** This
report concerns 64 frozen GSM8K development questions, not the official test
set. It establishes a candidate for larger development replication, not a
paper claim or final protocol.

## Independent-signal diagnostic

Qwen2.5-1.5B is adapted only on 128 public GSM8K anchors and then scores the
provided answer of each local example. Across three predeclared severe mixes,
its conflict-loss AUROC is 0.989, 0.923, and 0.934 (AP 0.985, 0.883, 0.915),
versus the prior local-loss reference of 0.756 on seed 52. The teacher does
not identify textual off-task QA by loss; numeric-interface filtering handles
non-numeric answers before teacher ranking.

## Endpoint development comparison

The selector retains the 128 lowest independent-teacher-loss numeric examples
from a 256-example client pool, then trains Qwen2.5-0.5B for the same 40 steps
as naive adaptation. Kinds are shown only after selection for audit.

| Seed | Naive severe EM | Teacher-routed severe EM | Difference | Retained clean / conflict / off-task |
|---:|---:|---:|---:|---:|
| 52 | 0.00 | 3.12 | +3.12 pp | 103 / 22 / 3 |
| 53 | 4.69 | 7.81 | +3.12 pp | 99 / 23 / 6 |
| 54 | 1.56 | 6.25 | +4.69 pp | 97 / 29 / 2 |
| Mean ± SD | 2.08 ± 2.39 | 5.73 ± 2.39 | +3.65 pp |

The paired severe difference is +3.65 pp, descriptive 95% t interval
[+1.40,+5.89], with one-sided t(2)=7.000, p=0.0099. These statistics describe
only the selected 64-question development evaluation; no multiple-comparison
or final-test claim is made.

Clean safety also passes this initial gate: clean EM differences relative to
naive adaptation are +3.13, +0.00, and -1.56 pp, so none is below the frozen
-2 pp bound. Unlike the earlier local-only filters, the independent teacher
substantially reduces numeric conflicts while retaining most clean examples.

## Required next evidence

The candidate must be replicated on the full 512-question development split
using new development seeds and then compared against a public-anchor-only
server and heterogeneous teachers at matched useful communication. The
official 1,319-question GSM8K test remains unopened until a separate protocol
is frozen after those steps.
