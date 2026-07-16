# GSM8K teacher-selector efficiency profile

**Scope:** diagnostic seed 71, severe mixture, 64 frozen development questions, identical 40-step client adaptation. This is a runtime profile, not an accuracy claim.

| Client | Naive total | Teacher total | Ratio | Teacher anchor training | Teacher scoring | Client training | Teacher peak GPU allocation |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-360M | 22.28 s | 33.76 s | 1.52x | 5.71 s | 8.08 s | 3.91 s | 5.57 GiB |
| TinyLlama-1.1B | 18.74 s | 29.56 s | 1.58x | 5.59 s | 8.11 s | 3.04 s | 5.57 GiB |

The teacher endpoint adds roughly 52--58% end-to-end wall time in this concurrent diagnostic. Candidate scoring alone takes about 8.1 seconds and is serial over 256 mixture items in the current implementation. These numbers include initial and final generation evaluation, but concurrent GPU execution means they should be interpreted as deployment-specific measurements rather than isolated single-process throughput.

Source artifacts: `gsm8k_efficiency_{smol,tiny}_{naive,teacher}_seed71.json`.
