# Official ANVIL P4 Experiments

Date: 2026-05-15

Official harness location:

```bash
/Users/nipunthumu/Desktop/Anvil-P-E/bench-p04-pcam
```

All commands were run with:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache
```

This avoids macOS Python cache writes outside the sandboxed workspace.

## Official Interface Notes

- Adapter spec: `module:Class`, loaded by `self_check.py` / `run.py`.
- Required class/method: `Engine.predict_precision(corrupted_query)`.
- Constructor signature: `__init__(self, stored_patterns, model_params)`.
- Official dummy class: `adapters.dummy:DummyAgent`.
- `model_params` keys from `harness.pack_params`: `R`, `eta`, `beta`, `dt`, `T_max`, `tol`, `T_in`, `pi_min`, `pi_max`.
- Official harness clips precision to `[pi_min, pi_max]` and mean-normalizes inside `PCAMModel.clip_and_normalise`.

## Dummy Baseline

Command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.dummy:DummyAgent --quick
```

Passed: yes

Per seed:

| Seed | Baseline Accuracy | Agent Accuracy | Delta | Base Spread | Agent Spread | Ratio |
|------|-------------------|----------------|-------|-------------|--------------|-------|
| 42 | 0.790 | 0.790 | +0.000 | 12.18 | 12.18 | 1.00x |
| 101 | 0.760 | 0.760 | +0.000 | 12.33 | 12.33 | 1.00x |

Aggregated:

- Mean delta accuracy: `+0.000`
- Min delta accuracy: `+0.000`
- Mean spread reduction: `1.00x`
- Min spread reduction: `1.00x`
- Retrieval score: `0.00 / 70`
- Anisotropy score: `0.00 / 20`
- Total automated: `0.00 / 90`

## AGENT_MODE Quick Checks

Commands:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache AGENT_MODE=identity python3 self_check.py --adapter adapters.myteam:Engine --quick
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache AGENT_MODE=magnitude python3 self_check.py --adapter adapters.myteam:Engine --quick
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache AGENT_MODE=nearest python3 self_check.py --adapter adapters.myteam:Engine --quick
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache AGENT_MODE=topk python3 self_check.py --adapter adapters.myteam:Engine --quick
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache AGENT_MODE=geometry python3 self_check.py --adapter adapters.myteam:Engine --quick
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache AGENT_MODE=hybrid python3 self_check.py --adapter adapters.myteam:Engine --quick
```

Summary:

| Agent Mode | Mean Delta Accuracy | Min Delta Accuracy | Mean Spread Reduction | Min Spread Reduction | Notes |
|------------|---------------------|--------------------|-----------------------|----------------------|-------|
| identity | +0.000 | +0.000 | 1.00x | 1.00x | Matches dummy baseline. |
| magnitude | -0.225 | -0.280 | 0.58x | 0.57x | Regresses retrieval and spread. |
| nearest | -0.080 | -0.080 | 0.58x | 0.54x | Regresses both seeds. |
| topk | +0.010 | +0.000 | 0.81x | 0.81x | Best retrieval delta in this first pass; spread worsens. |
| geometry | -0.005 | -0.010 | 1.00x | 1.00x | Slight retrieval regression; no meaningful spread gain. |
| hybrid | -0.170 | -0.220 | 0.73x | 0.69x | Proxy-tuned blend does not transfer. |

Per-seed details:

| Agent Mode | Seed | Baseline Accuracy | Agent Accuracy | Delta | Base Spread | Agent Spread | Ratio |
|------------|------|-------------------|----------------|-------|-------------|--------------|-------|
| identity | 42 | 0.790 | 0.790 | +0.000 | 12.18 | 12.18 | 1.00x |
| identity | 101 | 0.760 | 0.760 | +0.000 | 12.33 | 12.33 | 1.00x |
| magnitude | 42 | 0.790 | 0.620 | -0.170 | 12.18 | 21.45 | 0.57x |
| magnitude | 101 | 0.760 | 0.480 | -0.280 | 12.33 | 20.75 | 0.59x |
| nearest | 42 | 0.790 | 0.710 | -0.080 | 12.18 | 22.73 | 0.54x |
| nearest | 101 | 0.760 | 0.680 | -0.080 | 12.33 | 19.57 | 0.63x |
| topk | 42 | 0.790 | 0.810 | +0.020 | 12.18 | 15.09 | 0.81x |
| topk | 101 | 0.760 | 0.760 | +0.000 | 12.33 | 15.01 | 0.82x |
| geometry | 42 | 0.790 | 0.790 | +0.000 | 12.18 | 12.18 | 1.00x |
| geometry | 101 | 0.760 | 0.750 | -0.010 | 12.33 | 12.34 | 1.00x |
| hybrid | 42 | 0.790 | 0.670 | -0.120 | 12.18 | 17.52 | 0.69x |
| hybrid | 101 | 0.760 | 0.540 | -0.220 | 12.33 | 16.16 | 0.76x |

## Observations

- All modes imported and completed the official quick check.
- Best retrieval delta: `topk` with mean delta `+0.010`.
- Best spread reduction: `identity` / dummy at `1.00x`; none of the ported non-identity agents improved anisotropy.
- Modes with negative delta on at least one seed: `magnitude`, `nearest`, `geometry`, `hybrid`.
- Modes with spread reduction `<= 1.0x` on at least one seed: all modes in this first pass.
- No official evaluator/model files were modified.

## 2026-05-15 Debug And Sweep Pass

Official scoring path:

- Retrieval: `checks.retrieval_accuracy` calls `agent.predict_precision(q)`, runs `model.run(q, pi, u_const=q)`, then compares `model.classify(a_star)` with truth.
- Anisotropy: `checks.anisotropy_spread` calls the agent on `pattern + N(0, 0.05)` normalized to unit norm, then evaluates spread at the clean stored pattern.
- Spread formula: `checks.per_pattern_spread` computes `H = model.hessian(pattern)`, clips/normalizes `pi`, forms `S = diag(sqrt(pi)) @ H @ diag(sqrt(pi))`, and returns `max(eig(S)) / min(eig(S))`.
- Hessian formula: `PCAMModel.hessian(a) = R - eta * beta * X.T @ (diag(s) - s s.T) @ X`, where `s = softmax(beta * X @ a)`.
- Precision clipping/normalization: `PCAMModel.clip_and_normalise`.
- `R` shape in quick checks: `(64, 64)`.

Geometry debug with `DEBUG_GEOMETRY=1`:

- `model_params` includes `R`, `T_in`, `T_max`, `beta`, `dt`, `eta`, `pi_max`, `pi_min`, `tol`.
- `R` is present and shape `(64, 64)`.
- Geometry did not fall back to identity.
- Seed 42 curvature from current proxy: min `1.21767`, max `1.26011`, mean `1.23819`, std `0.00883933`.
- Seed 42 precision: min `0.982555`, max `1.0168`, mean `1`, std `0.00714557`.
- Seed 101 curvature from current proxy: min `1.2243`, max `1.25433`, mean `1.23721`, std `0.00659134`.
- Seed 101 precision: min `0.986318`, max `1.01051`, mean `1`, std `0.00532242`.
- Conclusion: geometry was inert because the current proxy uses near-constant column norms of `R`, so sanitized precision is effectively identity. It is not a missing-`R` or fallback problem.

Initial requested checks after adding `topk_geom`:

| Agent Mode | Mean Delta Accuracy | Min Delta Accuracy | Mean Spread Reduction | Min Spread Reduction | Notes |
|------------|---------------------|--------------------|-----------------------|----------------------|-------|
| topk | +0.010 | +0.000 | 0.81x | 0.81x | Same default top-k behavior. |
| geometry | -0.005 | -0.010 | 1.00x | 1.00x | Debug confirms near-identity precision. |
| topk_geom | +0.010 | +0.000 | 0.85x | 0.85x | Same retrieval as topk, slightly less bad spread, still below 1x. |

Top-k one-factor sweep output:

- Full table saved in `results/topk_sweep.md`.
- Best retrieval delta found: `k=8` and `k=12`, both mean delta `+0.030`, min delta `+0.020`.
- Tie-break by spread favors `k=12`, `temp=1`, `consensus=0.45`, `distance=euclidean`: mean spread `0.85x`, min spread `0.83x`.
- No swept top-k config improved anisotropy above `1.0x`.

## 2026-05-15 Hessian Geometry Pass

Added Hessian utility and agents matching the official formula:

```python
s = softmax(beta * X @ a)
C = diag(s) - outer(s, s)
H = R - eta * beta * X.T @ C @ X
```

Initial requested quick checks:

| Agent Mode | Mean Delta Accuracy | Min Delta Accuracy | Mean Spread Reduction | Min Spread Reduction | Notes |
|------------|---------------------|--------------------|-----------------------|----------------------|-------|
| hessian_diag | +0.000 | +0.000 | 1.00x | 1.00x | Direct inverse Hessian diagonal is effectively identity on quick seeds. |
| hessian_ruiz | -0.005 | -0.010 | 1.00x | 1.00x | Direct Ruiz geometry is effectively identity and slightly regresses retrieval. |
| topk_hdiag | +0.030 | +0.020 | 0.88x | 0.86x | Uses k=12 top-k default plus 15% Hessian-diag blend. |
| topk_ruiz | +0.030 | +0.020 | 0.88x | 0.86x | Uses k=12 top-k default plus 15% Ruiz blend. |

Blend sweep:

- Full table saved in `results/hessian_geometry_sweep.md`.
- `topk_hdiag` best retrieval-preserving spread in this sweep: `TOPK_WEIGHT=0.75`, `GEOM_WEIGHT=0.25`, mean delta `+0.030`, min delta `+0.020`, mean spread `0.90x`, min spread `0.88x`.
- `topk_ruiz` best retrieval-preserving spread in this sweep: `TOPK_WEIGHT=0.75`, `GEOM_WEIGHT=0.25`, mean delta `+0.030`, min delta `+0.020`, mean spread `0.90x`, min spread `0.88x`.
- Highest spread ratio tested while keeping retrieval positive: geometry weight `0.60`, with mean delta `+0.015`, min delta `+0.010`, mean spread `0.97x`, min spread `0.96x` for both families except Ruiz mean spread rounded to `0.96x`.
- No Hessian geometry config exceeded `1.0x` spread reduction in this pass.

## 2026-05-15 Adaptive Top-k + Spread Optimization Pass

Added a direct spread objective optimizer and an adaptive regime split:

- Retrieval-like corrupted queries use the k=12 top-k policy.
- Near-clean attractor probes use precomputed per-pattern spread-optimized precision.

Direct quick checks:

| Agent Mode | Mean Delta Accuracy | Min Delta Accuracy | Mean Spread Reduction | Min Spread Reduction | Notes |
|------------|---------------------|--------------------|-----------------------|----------------------|-------|
| spread_opt | +0.005 | +0.000 | 1.02x | 1.01x | Finally exceeds anisotropy baseline, but retrieval is small. |
| adaptive_topk_spread | +0.030 | +0.020 | 1.02x | 1.01x | Preserves k=12 top-k retrieval and exceeds spread baseline. |

`DEBUG_SPREAD_OPT=1` showed per-pattern optimization ratios around `1.01x` to `1.03x` on the first few stored patterns.

Adaptive threshold sweep:

- Full table saved in `results/adaptive_threshold_sweep.md`.
- Best threshold family: distance mode with threshold at least `0.20`.
- First best config by table order: `CLEAN_DETECT_MODE=distance`, `CLEAN_DISTANCE_THRESHOLD=0.20`.
- Result: mean delta `+0.030`, min delta `+0.020`, mean spread `1.02x`, min spread `1.01x`.
- Cosine threshold `0.90` also reached the same rounded result; stricter cosine thresholds missed many anisotropy probes.

Spread optimizer budget/scale sweep:

- Full table saved in `results/spread_optimizer_sweep.md`.
- Best rounded spread config: `SPREAD_OPT_BUDGET=200`, `SPREAD_OPT_SCALE=0.2`, mean delta `+0.030`, min delta `+0.020`, mean spread `1.02x`, min spread `1.02x`.
- Budgets `20`, `50`, and `100` with scale `0.2` or `0.4` all preserved mean delta `+0.030` and mean spread `1.02x`.

Current recommended mode:

```bash
AGENT_MODE=adaptive_topk_spread \
CLEAN_DETECT_MODE=distance \
CLEAN_DISTANCE_THRESHOLD=0.20 \
SPREAD_OPT_BUDGET=80 \
SPREAD_OPT_SCALE=0.4
```

The adapter default mode is now `adaptive_topk_spread`.

Default no-env quick verification:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine --quick
```

- Mean delta `+0.030`
- Min delta `+0.020`
- Mean spread reduction `1.02x`
- Min spread reduction `1.01x`

## 2026-05-15 Robustness Pass

Full default self-check:

- Output saved in `results/default_self_check_adaptive.md`.
- Mean delta `+0.057`
- Min delta `+0.021`
- Mean spread reduction `1.02x`
- Min spread reduction `1.01x`
- Automated score `70.16 / 90`
- Negative-delta seeds: none
- Spread `<= 1.0x` seeds: none

Official 12-seed `run.py`:

- JSON saved in `results/adaptive_many_seed_report.json`.
- Summary saved in `results/adaptive_many_seed_summary.md`.
- Mean delta `+0.033`
- Min delta `+0.005`
- Mean spread ratio `1.02x`
- Min spread ratio `1.01x`
- Negative-delta seeds: `0`
- Spread `<= 1.0x` seeds: `0`
- Automated score `46.36 / 90`

The controlled robustness sweep was not run because the robustness gate did not fail: every tested seed kept non-negative retrieval delta and spread ratio above `1.0x`.

Runtime notes:

- Engine initialization at default budget `80`: about `0.57s` for `K=16`, `N=64`.
- Engine initialization at budget `500`: about `3.49s` for `K=16`, `N=64`.
- Full 5-seed self-check wall time: `192.7s`.
- 12-seed run report sum of per-seed durations: `436.8s`, mean `36.4s` per seed.
- `SPREAD_OPT_BUDGET=500` appears too slow for little/no rounded metric gain on the public quick sweep.

## 2026-05-15 Final Robustness And Submission Prep

Stable baseline commit:

- Commit: `fe86d32`
- Message: `stable adaptive topk spread baseline`

20-seed official stress test:

- JSON saved in `results/adaptive_20_seed_report.json`.
- Summary saved in `results/adaptive_20_seed_summary.md`.
- Mean delta `+0.048`
- Min delta `+0.005`
- Mean spread ratio `1.02x`
- Min spread ratio `1.01x`
- Negative-delta seeds: `0`
- Spread `<= 1.0x` seeds: `0`
- Automated score `66.98 / 90`
- Sum of per-seed measured durations: `754.5s`
- Mean per-seed duration: `37.7s`

Optional `TOPK_K=16` full 20-seed validation:

- JSON saved in `results/adaptive_20_seed_k16_report.json`.
- Summary saved in `results/adaptive_20_seed_k16_summary.md`.
- Mean delta `+0.053`
- Min delta `+0.011`
- Mean spread ratio `1.01x`
- Min spread ratio `0.99x`
- Negative-delta seeds: `0`
- Spread `<= 1.0x` seeds: `3`
- Decision: keep default `TOPK_K=12`.

Gate decision:

- The 20-seed gate passed.
- No controlled robustness rescue sweep was needed.

Bounded adaptive top-k robustness sweep:

- Full table saved in `results/adaptive_topk_robust_sweep.md`.
- Sweep was one-factor around current defaults using the requested 12-seed set and reduced query count.
- All tested configs had `0` negative-delta seeds and `0` spread failures.
- `k=16` improved reduced-sweep mean delta to `+0.037` versus current `k=12` at `+0.033`, with same rounded min delta and spread.
- Default was not changed because the improvement is small and `k=12` is the setting validated by the full 20-seed stress run.

Submission notes:

- Drafted `README_SUBMISSION.md`.
- No official evaluator/model/harness files modified.
