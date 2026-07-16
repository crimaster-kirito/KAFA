# AG News post-observation independent-seed replication

**Status: official test was already observed before this protocol; this is seed replication, not a new untouched holdout.**

## Verification

- Replication protocol SHA-256: `e952b40e07d830e0ea8ced6dafda31fd24e6549b661b87dc3941e278aedbd5a8` (matches the pre-run frozen digest).
- The frozen protocol and scheduler specify no method, hyperparameter, data, or evaluation changes from seeds 142--144; all fields present in both generations of raw records match. The legacy schema omits some CLI fields, so this is internal consistency evidence rather than complete independent provenance.
- New seeds 145--149 were frozen together before any new run.

## Per-seed accuracy

| Evidence phase | Seed | Anchor-only | Routed | Routed - anchor |
|---|---:|---:|---:|---:|
| original frozen holdout | 142 | 84.50% | 83.92% | -0.58 pp |
| original frozen holdout | 143 | 83.66% | 83.18% | -0.47 pp |
| original frozen holdout | 144 | 84.87% | 84.24% | -0.63 pp |
| post-observation replication | 145 | 85.05% | 84.28% | -0.78 pp |
| post-observation replication | 146 | 85.11% | 84.28% | -0.83 pp |
| post-observation replication | 147 | 85.12% | 84.45% | -0.67 pp |
| post-observation replication | 148 | 84.43% | 83.99% | -0.45 pp |
| post-observation replication | 149 | 85.12% | 83.84% | -1.28 pp |

## Separate phase summaries

| Phase | Seeds | Mean gain [descriptive 95% CI] | Nonpositive signs |
|---|---:|---:|---:|
| original | 3 | -0.56 pp [-0.76, -0.36] | 3/3 |
| replication | 5 | -0.80 pp [-1.18, -0.42] | 5/5 |
| combined descriptive | 8 | -0.71 pp [-0.93, -0.49] | 8/8 |

## Frozen directional seed-consistency heuristic

This 4-of-5 sign-and-mean rule is not an alpha-controlled significance test; it only checks whether the observed direction repeats across new seeds.

The predeclared seed-consistency heuristic passes: 5/5 new differences are nonpositive and the new-seed mean is -0.80 pp.

The combined eight-seed estimate is descriptive and must not replace the original predeclared three-seed result.

## Source records

- `kafa_script/analysis_outputs/final_anchor_only_strict_seed142.json` (SHA-256 `c4895ac9c8087c174402fa39e2c4f30eb8273d750111f9475a16c9c21253e7eb`)
- `kafa_script/analysis_outputs/final_routed_seed142.json` (SHA-256 `e2e4bed9103cbceb51da6d06a92f645013bfc0da1fd3ad972c0dadcf41b9fd8a`)
- `kafa_script/analysis_outputs/final_anchor_only_strict_seed143.json` (SHA-256 `72b3d863b6801ab70eea12a68679f84fc2f602900ebb041a700871ab59958108`)
- `kafa_script/analysis_outputs/final_routed_seed143.json` (SHA-256 `c98a4af1288bc26932801d2f91234cc32f359a7e5eb98df3170f90fad2be0da1`)
- `kafa_script/analysis_outputs/final_anchor_only_strict_seed144.json` (SHA-256 `6e9a428a2040b1d340efd97e5c4efedf8e65c649a6eddf9a27c6dc508654b5a6`)
- `kafa_script/analysis_outputs/final_routed_seed144.json` (SHA-256 `8a212a35c1967d20e465c8a263f7658b5697007935826583d5b985e34095842f`)
- `kafa_script/analysis_outputs/replication_anchor_only_seed145.json` (SHA-256 `74c96488f78504345b5091589123a3ab30949bb04da355af8e09d9d316c3dff1`)
- `kafa_script/analysis_outputs/replication_routed_seed145.json` (SHA-256 `d2bbe8d0ea788a538a587822c010fe8635f294d4eb763525963fe31ba0797682`)
- `kafa_script/analysis_outputs/replication_anchor_only_seed146.json` (SHA-256 `d03e2635a3ab63d3809fed25aa4d1adeae9d0a3c2f37c4fb81569b2ad50932f6`)
- `kafa_script/analysis_outputs/replication_routed_seed146.json` (SHA-256 `c61e5a6f63cc97d743bb8f9b33f0336488c642e073eb19bc3b9d3ecf19339250`)
- `kafa_script/analysis_outputs/replication_anchor_only_seed147.json` (SHA-256 `645f42ad45c5a27d1b0bb4f7a32c463e6b415525dd3bc651dbd9ece204c458a5`)
- `kafa_script/analysis_outputs/replication_routed_seed147.json` (SHA-256 `a380129c9ca42ec41bb39add10a7da67fd9c18f3886fdf4bb658d8a44ba59caa`)
- `kafa_script/analysis_outputs/replication_anchor_only_seed148.json` (SHA-256 `22e89941b359fbdd627a65148a9303b23882c69b362d260a8a39a731e9c5fad3`)
- `kafa_script/analysis_outputs/replication_routed_seed148.json` (SHA-256 `a8ca31a08edfe971335b363b4ee24ea74d638855578d8bb9d05d9c010e5b418e`)
- `kafa_script/analysis_outputs/replication_anchor_only_seed149.json` (SHA-256 `0d9398cb7fcc157c2e8b88a82688de6b5f40b22edbe233fa5ffdd5b6e4b1443a`)
- `kafa_script/analysis_outputs/replication_routed_seed149.json` (SHA-256 `65c44be38c7fbdb3407f3e5e8b1e2e633520c930563cf39ec8a62290be4e7720`)
