# KAFA-Router Controlled Mechanism Pilot

Date: 2026-07-14

## Evidence status

These are **real computed results on synthetic data**, not synthetic/fabricated
metric previews. They test a mechanism under known ground truth, but they are
**not paper benchmark results** and provide no evidence about LLM performance.

The experiment uses eight federated clients, a binary target capability, 20%
off-task samples, and controlled label-conflict ratios. Each table entry is the
mean over five seeds (42--46); dispersion below is sample standard deviation.

Raw artifacts are preserved in sibling directories `router_pilot_v1`,
`router_pilot_v2`, and `router_pilot_v3`.

## Versions tested

1. **V1, per-example anchors + soft LCB routing.** Individual anchor gradients
   were treated as directions. This produced extreme variance and only about
   3% recall of known helpful samples.
2. **V2, micro-batch anchors + soft LCB routing.** Eight anchor micro-batch
   gradients reduced variance. It improved over hard filtering at 0% and 20%
   conflict, but failed badly at 40% and 60% conflict.
3. **V3, micro-batch anchors + conservative conflict gate.** Samples whose
   lower-confidence utility was negative were excluded from the shared update.
   This was robust at high conflict but over-filtered benign clients.

## V3 test accuracy

| Conflict | FedAvg | Hard filter | KAFA-Router V3 | Oracle |
|---:|---:|---:|---:|---:|
| 0% | 91.78 +/- 0.67 | 88.63 +/- 1.66 | 84.67 +/- 2.48 | 96.63 +/- 0.48 |
| 20% | 82.54 +/- 1.32 | 86.85 +/- 1.97 | 84.69 +/- 2.60 | 96.50 +/- 0.50 |
| 40% | 50.14 +/- 3.25 | 85.06 +/- 2.62 | 84.51 +/- 2.48 | 95.22 +/- 0.36 |
| 60% | 18.29 +/- 2.10 | 82.64 +/- 3.27 | 84.21 +/- 2.70 | 92.44 +/- 0.41 |

## Paired comparisons against hard filtering

Paired t-tests use only five seeds and are diagnostic, not confirmatory.

| Version | Conflict | Accuracy delta (percentage points) | Paired p-value |
|---|---:|---:|---:|
| V1 | 0% | +4.00 | 0.0054 |
| V1 | 20% | +0.21 | 0.7023 |
| V1 | 40% | -12.36 | 0.0003 |
| V1 | 60% | -40.24 | <0.0001 |
| V2 | 0% | +4.04 | 0.0016 |
| V2 | 20% | +1.64 | 0.0282 |
| V2 | 40% | -4.00 | 0.0036 |
| V2 | 60% | -20.51 | 0.0001 |
| V3 | 0% | -3.97 | 0.0007 |
| V3 | 20% | -2.16 | 0.0024 |
| V3 | 40% | -0.55 | 0.2250 |
| V3 | 60% | +1.57 | 0.1453 |

The apparent V3 gain at 60% conflict is not statistically significant with
five seeds. It must not be reported as an established improvement.

## What was falsified

- Per-example anchor-gradient uncertainty is not a useful routing signal in
  this setup.
- A single fixed soft router does not cover both mild and severe conflict.
- A single fixed safety gate sacrifices too much useful data in benign regimes.
- Current results do not establish superiority over a simple cosine hard
  filter.

## Next preregistered pilot

Before examining new evaluation seeds, define a risk-adaptive policy using the
development seeds already observed:

- benign estimated conflict: ordinary shared update;
- moderate estimated conflict: V2 soft routing;
- severe estimated conflict: V3 safety gate.

The thresholds and all hyperparameters must be written to metadata first. The
policy will then be evaluated on disjoint seeds and additional off-task ratios.
Success requires both:

1. no material degradation versus FedAvg in the benign regime; and
2. a statistically supported improvement over hard filtering in at least the
   severe-conflict regime.

If either condition fails, the current gradient-routing formulation should be
abandoned or substantially reformulated before any LLM-scale experiment.

