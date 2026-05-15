# Final Judge Summary

- Repository: https://github.com/nipun172006/anvil
- Final code baseline commit: `676b372`
- Official entrypoint: `adapters.myteam:Engine`
- Final command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anvil_pycache python3 self_check.py --adapter adapters.myteam:Engine
```

## Results

| Run | Mean Delta | Min Delta | Mean Spread | Min Spread | Score / Gates |
|---|---:|---:|---:|---:|---|
| Full `self_check.py` | `+0.057` | `+0.021` | `1.02x` | `1.01x` | `70.13 / 90` |
| 20-seed robustness | `+0.048` | `+0.005` | `1.02x` | `1.01x` | `66.98 / 90`, `0` negative seeds, `0` spread failures |

## Method

- Use `adaptive_topk_spread` as the default precision controller.
- Route corrupted retrieval queries to top-k consensus precision.
- Route near-clean attractor probes to precomputed Hessian spread-optimized precision.
- Detect the regime with distance to the nearest stored pattern.
- Keep the frozen PCAM model unchanged and use precision only at inference time.

No official evaluator, model, harness, or data-generation files were modified.
