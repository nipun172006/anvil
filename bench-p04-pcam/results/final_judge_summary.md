# Final Judge Summary

- Repository: https://github.com/nipun172006/anvil
- Official entrypoint: `adapters.myteam:Engine`
- Final command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine
```

## Results

| Run | Mean Delta | Min Delta | Mean Spread | Min Spread | Score / Gates |
|---|---:|---:|---:|---:|---|
| Full `self_check.py` | `+0.064` | `+0.023` | `1.02x` | `1.01x` | `70.16 / 90`, retrieval `70 / 70` |
| 20-seed robustness | `+0.055` | `+0.011` | `1.02x` | `1.01x` | `70.15 / 90`, `0` negative seeds, `0` spread failures |

## Method

- Use `adaptive_topk_spread` as the default precision controller.
- Route corrupted retrieval queries to guarded adaptive-margin top-k precision.
- Use `k=16` and `temp=1.2` for ambiguous or noisy retrieval queries.
- Route near-clean attractor probes to precomputed Hessian spread-optimized precision.
- Use `CLEAN_DISTANCE_THRESHOLD=0.30` so anisotropy probes stay on the spread branch.

No official evaluator, model, harness, or data-generation files were modified.
