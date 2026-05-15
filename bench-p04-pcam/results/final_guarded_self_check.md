# Final Guarded Self-Check

Command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine
```

Output:

```text
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time            192290.5 ms
  seeds                             5
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.5, 0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.873    0.896  +0.023   12.17   12.01   1.01×
   101     0.788    0.848  +0.060   12.31   12.08   1.02×
   202     0.701    0.823  +0.121   12.27   12.03   1.02×
   303     0.795    0.821  +0.027   12.30   12.08   1.02×
   404     0.717    0.805  +0.088   12.40   12.15   1.02×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.064
  min  Δ accuracy (worst seed)    +0.023
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
