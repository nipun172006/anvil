# Final Approach Notes

Current adapter default:

```bash
AGENT_MODE=adaptive_topk_spread
CLEAN_DETECT_MODE=distance
CLEAN_DISTANCE_THRESHOLD=0.20
SPREAD_OPT_BUDGET=80
SPREAD_OPT_SCALE=0.4
```

## Method

The agent uses two precision policies and switches based on the query regime.

Corrupted retrieval queries use top-k consensus precision:

- Find the nearest stored patterns.
- Use weighted local candidate agreement and candidate variance to infer reliable dimensions.
- Current retrieval configuration uses `k=12`, `temp=1`, `consensus_weight=0.45`, and Euclidean distance.

Near-clean attractor probes use spread-optimized precision:

- For each stored pattern, compute the official PCAM Hessian:

```python
s = softmax(beta * X @ a)
C = diag(s) - outer(s, s)
H = R - eta * beta * X.T @ C @ X
```

- Precompute a per-pattern precision vector by directly minimizing the condition number of:

```python
diag(sqrt(pi)) @ H @ diag(sqrt(pi))
```

- At prediction time, use the nearest stored pattern to select the precomputed precision.

The clean/corrupted split is based on squared Euclidean distance to the nearest stored pattern. Queries with distance `<= 0.20` use spread-optimized precision; the rest use top-k retrieval precision.

Precision is returned as a positive finite vector. The official harness also clips precision to `[0.1, 10.0]` and mean-normalizes before applying dynamics.

No official evaluator, harness, data, or PCAM model files are modified.

## Current Robustness Snapshot

Full default `self_check.py`:

- Mean delta accuracy: `+0.057`
- Min delta accuracy: `+0.021`
- Mean spread reduction: `1.02x`
- Min spread reduction: `1.01x`
- Automated score: `70.16 / 90`

Many-seed official `run.py` over 12 seeds:

- Mean delta accuracy: `+0.033`
- Min delta accuracy: `+0.005`
- Mean spread reduction: `1.02x`
- Min spread reduction: `1.01x`
- Negative-delta seeds: `0`
- Spread `<= 1.0x` seeds: `0`

20-seed official `run.py` stress test:

- Mean delta accuracy: `+0.048`
- Min delta accuracy: `+0.005`
- Mean spread reduction: `1.02x`
- Min spread reduction: `1.01x`
- Negative-delta seeds: `0`
- Spread `<= 1.0x` seeds: `0`

Top-k robustness sweep:

- A bounded one-factor sweep found `k=16` had slightly higher mean delta than `k=12` on a reduced-query 12-seed run.
- Default remains `k=12` because it is the configuration validated by the full 20-seed stress test, and the sweep gain was small.

## Runtime

- Engine initialization, `K=16`, `N=64`, default budget `80`: about `0.57s` per seed.
- Engine initialization with `SPREAD_OPT_BUDGET=500`: about `3.49s` per seed.
- Full 5-seed `self_check.py`: about `192.7s`.
- 12-seed official `run.py`: sum of per-seed measured durations `436.8s`, mean `36.4s` per seed.
- 20-seed official `run.py`: sum of per-seed measured durations `754.5s`, mean `37.7s` per seed, max `43.2s`.

Budget `500` is probably unnecessary for the current public benchmark: it is roughly six times slower at initialization and did not improve rounded quick metrics over much smaller budgets.
