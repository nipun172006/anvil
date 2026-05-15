# Adaptive Top-k Robustness Sweep

Mode: `adaptive_topk_spread`

Seeds: `[7, 13, 31, 42, 97, 101, 211, 503, 1009, 2027, 3141, 9999]`

`n_per_level=100`, `n_anisotropy=16`

This is a one-factor sweep around the current default, not a full Cartesian grid.

| k | temp | consensus | mean delta | min delta | mean spread | min spread | neg delta | spread <= 1 | runtime s | ok | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 | 1 | 0.45 | +0.026 | +0.003 | 1.02x | 1.01x | 0 | 0 | 181.6 | yes |  |
| 10 | 1 | 0.45 | +0.029 | +0.007 | 1.02x | 1.01x | 0 | 0 | 183.1 | yes |  |
| 12 | 1 | 0.45 | +0.033 | +0.010 | 1.02x | 1.01x | 0 | 0 | 182.4 | yes |  |
| 14 | 1 | 0.45 | +0.034 | +0.010 | 1.02x | 1.01x | 0 | 0 | 182.2 | yes |  |
| 16 | 1 | 0.45 | +0.037 | +0.010 | 1.02x | 1.01x | 0 | 0 | 182.9 | yes |  |
| 12 | 0.8 | 0.45 | +0.031 | +0.010 | 1.02x | 1.01x | 0 | 0 | 180.4 | yes |  |
| 12 | 1.2 | 0.45 | +0.034 | +0.010 | 1.02x | 1.01x | 0 | 0 | 183.1 | yes |  |
| 12 | 1 | 0.35 | +0.033 | +0.010 | 1.02x | 1.01x | 0 | 0 | 183.6 | yes |  |
| 12 | 1 | 0.55 | +0.033 | +0.010 | 1.02x | 1.01x | 0 | 0 | 180.0 | yes |  |

## Best Gate-Passing Candidate

- k: `12`
- temp: `1.2`
- consensus weight: `0.45`
- mean delta: `+0.034`
- min delta: `+0.010`
- mean spread: `1.02x`
- min spread: `1.01x`
- runtime: `183.1s`
