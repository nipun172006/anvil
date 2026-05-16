
ANVIL · P-04 · PCAM Precision Agent — Self-Check
==============================================================
  total wall time             10852.7 ms
  seeds                             2
  stored patterns (K)              16
  state dim (N)                    64
  noise levels             [0.7, 0.8]

  PER-SEED  ─ retrieval ─       ── anisotropy ──
  seed      Π=I      agent  Δ     base    agent  ratio
  ----------------------------------------------------------
    42     0.790    0.870  +0.080   12.18   11.93   1.02×
   101     0.760    0.910  +0.150   12.33   12.00   1.03×

  AGGREGATED                       VALUE
  ----------------------------------------------------------
  mean Δ accuracy (over seeds)    +0.115
  min  Δ accuracy (worst seed)    +0.080
  mean spread reduction             1.02×
  min  spread reduction             1.02×

  SCORE (automated, max 90)         POINTS
  ----------------------------------------------------------
  retrieval     (max 70)             70.00
  anisotropy    (max 20)              0.20
  code quality  (max 10)            (manual)
  TOTAL AUTOMATED                    70.20  / 90

  ✓  Solid retrieval gain on average.
  ▸  Spread reduction near baseline. Try Hessian-aware design (see README hints).

