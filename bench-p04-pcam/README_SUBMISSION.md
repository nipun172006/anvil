# ANVIL P4 Submission

## Official Entrypoint

```text
adapters.myteam:Engine
```

The adapter implements `Engine.predict_precision(corrupted_query)` and returns a finite, positive, mean-normalized precision vector with one value per dimension.

## Approach

The final adapter is a PCAM flow-margin precision controller.

- Corrupted retrieval queries use a reliability-weighted target predictor: coordinates with larger `|q_j|` are treated as more trustworthy under the mask-plus-noise corruption process.
- The adapter computes the frozen PCAM local gradient and shapes precision along dimensions that improve the target-vs-competitor flow margin.
- If the weighted classifier is too uncertain, the adapter safely returns identity precision rather than oversteering.
- Near-clean attractor probes use a guarded Hessian geometry branch for anisotropy.
- Geometry precision is accepted only when it reduces the local spread of `sqrt(Pi) H sqrt(Pi)`; otherwise it falls back to identity precision.

The implementation is deterministic by default, NumPy-only, and does not retrain or modify the frozen PCAM model.

## Why This Is Principled

The official benchmark calls `predict_precision` in two different regimes: retrieval uses heavily corrupted noisy queries, while anisotropy uses lightly perturbed stored patterns and evaluates spread at the clean attractor. The adapter treats those regimes differently without changing the frozen PCAM model.

Precision is used only as inference-time control. For retrieval, the adapter uses the PCAM descent direction:

```python
grad = R @ q - eta * X.T @ softmax(beta * X @ q)
descent = -grad
```

It increases precision where the descent improves the margin between the predicted target and nearby competitors. For near-clean probes, the geometry branch directly targets the official Hessian spread form:

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
| Quick `self_check.py --quick` | `+0.115` | `+0.080` | `1.02x` | `1.02x` | retrieval `70 / 70` | `70.20 / 90` |
| Full `self_check.py` | `+0.110` | `+0.063` | `1.03x` | `1.02x` | retrieval `70 / 70` | `70.22 / 90` |
| 20-seed stress | `+0.101` | `+0.053` | `1.03x` | `1.02x` | `0` negative-delta seeds, `0` spread failures | `70.22 / 90` |
| 40-seed stress | `+0.096` | `+0.053` | `1.03x` | `1.02x` | `0` negative-delta seeds, `0` spread failures | `70.22 / 90` |

## Earlier Experiments

- Top-k consensus precision was robust but left retrieval headroom.
- Guarded adaptive top-k improved retrieval while preserving anisotropy, but the flow-margin controller is stronger on the same official harness.
- Coordinate descent and kappa-subgradient geometry variants gave small anisotropy gains but were not worth the robustness/runtime tradeoff.
- One-shot eigenvector scaling was anti-aligned with the official spread objective.

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
  --out results/ported_friend_20_seed_report.json
```

Report summarizer:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 scripts/summarize_report.py \
  results/ported_friend_20_seed_report.json \
  --out results/ported_friend_20_seed_summary.md
```

40-seed stress run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 run.py \
  --adapter adapters.myteam:Engine \
  --seeds 7 13 31 42 97 101 202 211 303 404 503 777 1009 1337 2027 2718 3141 4096 9001 9999 123 456 789 1111 2222 3333 4444 5555 6666 7777 8888 1212 2323 3434 4545 5656 6767 7878 8989 9090 \
  --out results/ported_friend_40_seed_report.json
```

## Dependencies

NumPy only.
