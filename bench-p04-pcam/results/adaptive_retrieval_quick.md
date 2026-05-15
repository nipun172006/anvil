# Adaptive Retrieval Quick Sweep

Branch: `exp-adaptive-retrieval-topk`

Stable quick baseline: mean delta `+0.030`, min delta `+0.020`, mean spread `1.02x`, min spread `1.01x`, score `42.14 / 90`.

| Config | Env | Runtime | Mean Delta | Min Delta | Mean Spread | Min Spread | Score | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| stable | `none` | 10.6s | +0.030 | +0.020 | 1.02x | 1.01x | 42.14 / 90 |  |
| adaptive_default | `RETRIEVAL_POLICY=adaptive_margin` | 10.5s | +0.035 | +0.030 | 1.02x | 1.01x | 49.14 / 90 |  |
| adaptive_conservative | `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=14 ADAPTIVE_K_NOISY=12 ADAPTIVE_TEMP_AMBIG=1.1` | 10.4s | +0.035 | +0.030 | 1.02x | 1.01x | 49.14 / 90 |  |
| adaptive_aggressive | `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=16 ADAPTIVE_K_NOISY=16 ADAPTIVE_TEMP_AMBIG=1.2` | 10.5s | +0.045 | +0.030 | 1.02x | 1.01x | 63.14 / 90 |  |

## Best By Retrieval

`adaptive_aggressive` with env `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=16 ADAPTIVE_K_NOISY=16 ADAPTIVE_TEMP_AMBIG=1.2` reached mean delta `+0.045`, min delta `+0.030`, mean spread `1.02x`, min spread `1.01x`, score `63.14 / 90`.

## Raw Outputs

## stable

Env: `none`

```text
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time             10615.5 ms
  seeds                             2
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.790    0.810  +0.020   12.18   12.03   1.01×
   101     0.760    0.800  +0.040   12.33   12.09   1.02×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.030
  min  Δ accuracy (worst seed)    +0.020
  mean spread reduction             1.02×
  min  spread reduction             1.01×

  SCORE (automated, max 90)         POINTS
  ----------------------------------------------------------
  retrieval     (max 70)             42.00
  anisotropy    (max 20)              0.14
  code quality  (max 10)            (manual)
  TOTAL AUTOMATED                    42.14  / 90

  ✓  Solid retrieval gain on average.
  ▸  Spread reduction near baseline. Try Hessian-aware design (see README hints).
```

## adaptive_default

Env: `RETRIEVAL_POLICY=adaptive_margin`

```text
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time             10503.1 ms
  seeds                             2
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.790    0.820  +0.030   12.18   12.03   1.01×
   101     0.760    0.800  +0.040   12.33   12.09   1.02×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.035
  min  Δ accuracy (worst seed)    +0.030
  mean spread reduction             1.02×
  min  spread reduction             1.01×

  SCORE (automated, max 90)         POINTS
  ----------------------------------------------------------
  retrieval     (max 70)             49.00
  anisotropy    (max 20)              0.14
  code quality  (max 10)            (manual)
  TOTAL AUTOMATED                    49.14  / 90

  ✓  Solid retrieval gain on average.
  ▸  Spread reduction near baseline. Try Hessian-aware design (see README hints).
```

## adaptive_conservative

Env: `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=14 ADAPTIVE_K_NOISY=12 ADAPTIVE_TEMP_AMBIG=1.1`

```text
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time             10417.4 ms
  seeds                             2
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.790    0.820  +0.030   12.18   12.03   1.01×
   101     0.760    0.800  +0.040   12.33   12.09   1.02×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.035
  min  Δ accuracy (worst seed)    +0.030
  mean spread reduction             1.02×
  min  spread reduction             1.01×

  SCORE (automated, max 90)         POINTS
  ----------------------------------------------------------
  retrieval     (max 70)             49.00
  anisotropy    (max 20)              0.14
  code quality  (max 10)            (manual)
  TOTAL AUTOMATED                    49.14  / 90

  ✓  Solid retrieval gain on average.
  ▸  Spread reduction near baseline. Try Hessian-aware design (see README hints).
```

## adaptive_aggressive

Env: `RETRIEVAL_POLICY=adaptive_margin ADAPTIVE_K_AMBIG=16 ADAPTIVE_K_NOISY=16 ADAPTIVE_TEMP_AMBIG=1.2`

```text
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time             10529.9 ms
  seeds                             2
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.790    0.820  +0.030   12.18   12.03   1.01×
   101     0.760    0.820  +0.060   12.33   12.09   1.02×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.045
  min  Δ accuracy (worst seed)    +0.030
  mean spread reduction             1.02×
  min  spread reduction             1.01×

  SCORE (automated, max 90)         POINTS
  ----------------------------------------------------------
  retrieval     (max 70)             63.00
  anisotropy    (max 20)              0.14
  code quality  (max 10)            (manual)
  TOTAL AUTOMATED                    63.14  / 90

  ✓  Solid retrieval gain on average.
  ▸  Spread reduction near baseline. Try Hessian-aware design (see README hints).
```
