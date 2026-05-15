# Final Quick Sanity

Command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine --quick
```

Output:

```text
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time             10597.5 ms
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
