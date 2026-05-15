# Default Adaptive Self-Check

Date: 2026-05-15

Command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine
```

Output:

```text
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time            192653.2 ms
  seeds                             5
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.5, 0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.873    0.895  +0.021   12.17   12.01   1.01×
   101     0.788    0.836  +0.048   12.31   12.08   1.02×
   202     0.701    0.811  +0.109   12.27   12.03   1.02×
   303     0.795    0.816  +0.021   12.30   12.08   1.02×
   404     0.717    0.800  +0.083   12.40   12.15   1.02×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.057
  min  Δ accuracy (worst seed)    +0.021
  mean spread reduction             1.02×
  min  spread reduction             1.01×

  SCORE (automated, max 90)         POINTS
  ----------------------------------------------------------
  retrieval     (max 70)             70.00
  anisotropy    (max 20)              0.16
  code quality  (max 10)            (manual)
  TOTAL AUTOMATED                    70.16  / 90

  ✓  Solid retrieval gain on average.
  ▸  Spread reduction near baseline. Try Hessian-aware design (see README hints).
```

Summary:

- Mean delta accuracy: `+0.057`
- Min delta accuracy: `+0.021`
- Mean spread reduction: `1.02x`
- Min spread reduction: `1.01x`
- Automated score: `70.16 / 90`
- Negative-delta seeds: none
- Spread `<= 1.0x` seeds: none
