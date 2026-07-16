# Preregistered KAFA-Router Holdout Results

## Status

- Protocol: `kafa_script/router_adaptive_protocol.json`
- Development seeds: 42--46
- Disjoint holdout seeds: 100--119
- Runs: 800
- Evidence type: real computation on controlled synthetic data
- Paper benchmark evidence: **no**
- Preregistered decision: **fail**

The adaptive thresholds and hyperparameters were fixed before these holdout
seeds were evaluated. No holdout result was used to change the policy.

## Mean holdout accuracy

| Off-task | Label conflict | FedAvg | Hard filter | Fixed safe router | Adaptive router | Oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0% | 96.88 | 90.12 | 85.35 | 96.30 | 96.88 |
| 0% | 20% | 91.25 | 88.20 | 85.19 | 91.72 | 96.36 |
| 0% | 40% | 75.65 | 86.51 | 84.90 | 85.76 | 95.55 |
| 0% | 60% | 25.26 | 84.98 | 84.47 | 84.37 | 94.68 |
| 20% | 0% | 90.82 | 89.14 | 85.21 | 92.49 | 96.36 |
| 20% | 20% | 82.28 | 87.26 | 85.05 | 88.21 | 95.64 |
| 20% | 40% | 50.73 | 85.71 | 84.82 | 83.79 | 94.68 |
| 20% | 60% | 19.00 | 83.72 | 83.98 | 83.98 | 92.76 |

All values are percentages averaged over 20 holdout seeds.

## Preregistered gate

The protocol required:

1. adaptive routing no worse than FedAvg by more than 2 percentage points in
   the clean setting; and
2. a statistically significant improvement over hard filtering in at least
   one setting with total corruption at least 60%.

Condition 1 passed: adaptive routing was 0.58 pp below FedAvg in the clean
setting. Condition 2 failed:

- 60% conflict, 0% off-task: adaptive was **0.62 pp worse** (`p=0.0091`);
- 40% conflict, 20% off-task: adaptive was **1.92 pp worse** (`p<0.0001`);
- 60% conflict, 20% off-task: adaptive was 0.26 pp better, but not significant
  (`p=0.3177`).

The gradient-routing formulation therefore does not justify proceeding to an
LLM-scale performance claim.

## Exploratory observation (not preregistered)

Across all eight unknown-corruption regimes, adaptive routing had the best
non-oracle macro average:

| Method | Macro mean | Worst regime |
|---|---:|---:|
| Adaptive router | 88.33 | 83.79 |
| Hard filter | 86.95 | 83.72 |
| Fixed safe router | 84.87 | 83.98 |
| FedAvg | 66.49 | 19.00 |

The adaptive-vs-hard-filter macro difference was +1.37 pp when averaging each
seed across regimes (paired `p=1.70e-7`). This analysis was not the
preregistered primary test. It is hypothesis-generating only and must be
confirmed under a separately frozen protocol if used later.

## Research decision

- Do not present KAFA-Router as a superior severe-corruption defense.
- Retain it as a simple adaptive baseline for a possible mixed-heterogeneity
  evaluation benchmark.
- Do not spend LLM compute on this mechanism until the paper contribution is
  repositioned and competing methods are implemented.

