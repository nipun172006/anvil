
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time            189883.1 ms
  seeds                             5
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.5, 0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.873    0.936  +0.063   12.17   11.93   1.02×
   101     0.788    0.883  +0.095   12.31   12.00   1.03×
   202     0.701    0.875  +0.173   12.27   11.95   1.03×
   303     0.795    0.869  +0.075   12.30   11.98   1.03×
   404     0.717    0.861  +0.144   12.40   12.07   1.03×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.110
  min  Δ accuracy (worst seed)    +0.063
  mean spread reduction             1.03×
  min  spread reduction             1.02×

  SCORE (automated, max 90)         POINTS
  ----------------------------------------------------------
  retrieval     (max 70)             70.00
  anisotropy    (max 20)              0.22
  code quality  (max 10)            (manual)
  TOTAL AUTOMATED                    70.22  / 90

  ✓  Solid retrieval gain on average.
  ▸  Spread reduction near baseline. Try Hessian-aware design (see README hints).

