# Adaptive Retrieval Guarded 12-Seed Sweep

Stable 12-seed reference: mean delta `+0.033`, min delta `+0.005`, spread failures `0`.

| Config | Env | Mean Delta | Min Delta | Mean Spread | Min Spread | Neg Seeds | Spread Failures | Score | Retrieval Spread/TopK | Aniso Spread/TopK | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable | `none` | +0.033 | +0.005 | 1.02x | 1.01x | 0 | 0 | stable ref | 0/9000 | 188/4 | baseline |
| aggressive unguarded | `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=16 ADAPTIVE_K_NOISY=16 ADAPTIVE_TEMP_AMBIG=1.2 CLEAN_DISTANCE_THRESHOLD=0.20` | +0.039 | +0.011 | 1.01x | 1.00x | 0 | 2 | 55.13 | 0/9000 | 188/4 | reject: spread failures |
| conservative unguarded | `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=14 ADAPTIVE_K_NOISY=12 ADAPTIVE_TEMP_AMBIG=1.1 CLEAN_DISTANCE_THRESHOLD=0.20` | +0.036 | +0.008 | 1.01x | 1.00x | 0 | 2 | 50.15 | 0/9000 | 188/4 | reject: spread failures |
| aggressive guarded | `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=16 ADAPTIVE_K_NOISY=16 ADAPTIVE_TEMP_AMBIG=1.2 CLEAN_DETECT_MODE=distance CLEAN_DISTANCE_THRESHOLD=0.30` | +0.039 | +0.011 | 1.02x | 1.01x | 0 | 0 | 55.22 | 2/8998 | 192/0 | pass |

## Decision

`CLEAN_DISTANCE_THRESHOLD=0.30` fixes the anisotropy misrouting: all sampled anisotropy probes route to the spread branch, while only 2 of 9000 retrieval queries are rerouted away from top-k. The guarded aggressive adaptive retrieval config passes the 12-seed gate and was promoted to the 20-seed gate.
