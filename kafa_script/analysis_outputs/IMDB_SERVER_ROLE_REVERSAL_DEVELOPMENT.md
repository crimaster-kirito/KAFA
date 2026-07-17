# IMDb Server Prospective Role-Reversal Experiment

Phase: **development**. Records: **12**. Protocol SHA-256: `fbfc13064eca36163397017521474dde6aeda7c4304215b97491845c14ee9166`.
Summarizer SHA-256: `2115facef5488f5ad6ae36461c151dedeca682c0ee2dadd4af2a3985eabcddf5`. Verifier SHA-256: `b63008c9dc05d5fcfd54a56abe24270916ee3227da25c2aca00b363c4007b7f0`.

## Evidence boundary

The complete 25,000-row IMDb test had no model-evaluation record in the workspace audit before protocol freeze. Manifest construction hashed and counted raw lines without decoding JSON. IMDb train and the sentiment interface had previously served as off-task inputs, so this is a prospective untouched-test role-reversal extension, not dataset-naive or independently preregistered replication.

## Arm summaries

| Arm | Accuracy mean ± SD | Macro-F1 mean [CI] | Class NLL mean [CI] | ECE mean [CI] | Wall s mean [CI] | Peak GiB mean [CI] |
|---|---:|---:|---:|---:|---:|---:|
| Anchor only | 91.47 ± 0.06 | +91.47 [+91.32, +91.61] | +0.2914 [+0.2739, +0.3089] | +0.0490 [+0.0434, +0.0546] | +51.7 [+24.0, +79.5] | +9.82 [+9.82, +9.82] |
| Clean uniform | 91.77 ± 0.32 | +91.77 [+90.97, +92.56] | +0.2462 [+0.2357, +0.2568] | +0.0292 [+0.0069, +0.0514] | +138.2 [+95.5, +180.8] | +9.82 [+9.82, +9.82] |
| Severe uniform | 90.03 ± 0.32 | +90.03 [+89.23, +90.83] | +0.2824 [+0.2740, +0.2908] | +0.0700 [+0.0618, +0.0782] | +124.6 [+68.0, +181.2] | +9.82 [+9.82, +9.82] |
| Severe routed | 91.33 ± 0.25 | +91.33 [+90.71, +91.96] | +0.2398 [+0.2325, +0.2471] | +0.0181 [+0.0050, +0.0311] | +314.9 [+298.2, +331.7] | +27.09 [+27.09, +27.09] |

## Per-seed accuracy (%)

| Seed | Anchor | Clean uniform | Severe uniform | Severe routed |
|---:|---:|---:|---:|---:|
| 311 | 91.50 | 92.00 | 89.90 | 91.60 |
| 312 | 91.50 | 91.90 | 90.40 | 91.10 |
| 313 | 91.40 | 91.40 | 89.80 | 91.30 |

## Mechanism and cost audit

- Routing triggered in 18/18 client-rounds.
- Aggregate routed selected-kind counts: `{'clean': 1671, 'conflict': 633}`.
- Routed audit clean fraction: 72.5% versus the severe baseline 40.2%.
- The anchor arm omits the complete client path and records zero payload.
- Wall time is scheduler-specific, not isolated deployment throughput.
- Frozen implementation gate (composition only): **pass**.
- No development accuracy direction is used for tuning or stopping.

## Raw artifact hashes

- `seed311_clean_unfiltered_distillation.json`: `ff480af04798a9bbd27fbe7e32721ab89fd437ef1c37a9253c74c1683e9462cd`
- `seed311_public_anchor_only.json`: `3edd17defa68841f307d4019939bac866df8005b38fb51842bd85eb62e977a0d`
- `seed311_severe_routed_distillation.json`: `f4f0d5102fe7075d53051436057712a199f997732655a767c5c1fa12c669c56a`
- `seed311_severe_unfiltered_distillation.json`: `d80ca2b3b81c6e0400b68fdfd3078a7a15c8dd67e6bb0c82f79c1fe6ca566314`
- `seed312_clean_unfiltered_distillation.json`: `49d71699a8e859ac3f848fb32939e5e8225c91599cc1685aa83069a00195f0c9`
- `seed312_public_anchor_only.json`: `92c8a6b24e1bb6df7b35659332fbfae604a7fe8429dcc521f7bd5b2d59131845`
- `seed312_severe_routed_distillation.json`: `72c0a91fb2d81144abcc6fac8c87aa261209bc440307cef604c86d66faba0116`
- `seed312_severe_unfiltered_distillation.json`: `d1cce3bf428b9649c9fe36d9ecab115a4686553b5bc6f5648e8ee757451191b6`
- `seed313_clean_unfiltered_distillation.json`: `2fd00a108bebc25c3637827139ad887b407b9ec22cd539c5def282c487a9e1e6`
- `seed313_public_anchor_only.json`: `ed5efbae985d23ef1c0cc4c2c3862d2348b8b22a4fcb437fdeae14fc8888154c`
- `seed313_severe_routed_distillation.json`: `59ef44c64debce96f056428e77781a63133159aa8c534764d6c15edc666e8843`
- `seed313_severe_unfiltered_distillation.json`: `09ca24eb1dbafd22c7755f2f58dab42f1017cf03b0f31ca74f90cbc1b9f2eab3`

## Claim boundary

- The complete IMDb test is prospectively untouched for model assessment according to the workspace audit, but raw bytes were hashed and counted.
- IMDb train and task semantics were previously used as off-task inputs; this is role reversal, not dataset-naive confirmation.
- The official phase decodes and evaluates all 25,000 rows only after development completes and the protocol remains unchanged.
- The audit-only clean/conflict/off-task labels do not enter routing scores; explicit off-task task membership is available through the interface.
- The complete routing intervention changes both selected-set size and composition.
- No privacy, security, licensing, deployment-efficiency, or population-of-tasks claim is authorized.
