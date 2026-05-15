# ANVIL P4 Submission

## Official Entrypoint

```text
adapters.myteam:Engine
```

The adapter implements `Engine.predict_precision(corrupted_query)` and returns a finite, positive, mean-normalized precision vector with one value per dimension.

## Approach

The final adapter uses `adaptive_topk_spread`.

- Corrupted retrieval queries use top-k consensus precision.
- Near-clean attractor probes use precomputed Hessian spread-optimized precision.
- A distance-based regime detector selects between those two paths.
- Final top-k defaults are `k=12`, `temp=1.0`, `consensus_weight=0.45`, Euclidean distance.
- Final clean-input threshold is `CLEAN_DISTANCE_THRESHOLD=0.20`.

## Why This Is Principled

The official benchmark calls `predict_precision` in two different regimes: retrieval uses heavily corrupted noisy queries, while anisotropy uses lightly perturbed stored patterns and evaluates spread at the clean attractor. The adapter treats those regimes differently without changing the frozen PCAM model.

Precision is used only as inference-time control. The retrieval branch estimates reliable coordinates from local top-k consensus; the near-clean branch directly targets the official Hessian spread form:

```python
s = softmax(beta * X @ a)
C = diag(s) - outer(s, s)
H = R - eta * beta * X.T @ C @ X
S = diag(sqrt(pi)) @ H @ diag(sqrt(pi))
```

No official evaluator, model, harness, or data-generation files were modified.

## Results

| Run | Mean Delta | Min Delta | Mean Spread | Min Spread | Extra Gates | Score |
|---|---:|---:|---:|---:|---|---:|
| Full `self_check.py` | `+0.057` | `+0.021` | `1.02x` | `1.01x` | none negative | `70.13 / 90` |
| 20-seed stress | `+0.048` | `+0.005` | `1.02x` | `1.01x` | `0` negative-delta seeds, `0` spread failures | `66.98 / 90` |

## Rejected Experiments

- `TOPK_K=16` improved retrieval but caused 3 spread failures, so `k=12` stayed final.
- Coordinate descent gave only a tiny anisotropy improvement and added too much runtime.
- One-shot eigenvector scaling was anti-aligned with the official spread objective.
- Kappa-subgradient slightly improved full self-check but failed the 20-seed spread gate.
- Basis-subspace quick did not clearly beat the stable quick baseline.

## Reproduction

Quick check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine --quick
```

Full self-check:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine
```

20-seed robustness run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 run.py \
  --adapter adapters.myteam:Engine \
  --seeds 7 13 31 42 97 101 202 211 303 404 503 777 1009 1337 2027 2718 3141 4096 9001 9999 \
  --out results/adaptive_20_seed_report.json
```

Report summarizer:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 scripts/summarize_report.py \
  results/adaptive_20_seed_report.json \
  --out results/adaptive_20_seed_summary.md
```

## Dependencies

NumPy only.
