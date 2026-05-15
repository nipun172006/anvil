# Adaptive Many-Seed Summary

| Seed | Baseline Acc | Agent Acc | Delta | Base Spread | Agent Spread | Spread Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.752 | 0.763 | +0.011 | 12.21 | 11.99 | 1.02x |
| 13 | 0.869 | 0.895 | +0.025 | 12.11 | 11.80 | 1.03x |
| 31 | 0.769 | 0.825 | +0.056 | 12.26 | 12.07 | 1.02x |
| 42 | 0.873 | 0.896 | +0.023 | 12.17 | 12.01 | 1.01x |
| 97 | 0.829 | 0.851 | +0.021 | 12.56 | 12.18 | 1.03x |
| 101 | 0.788 | 0.848 | +0.060 | 12.31 | 12.08 | 1.02x |
| 202 | 0.701 | 0.823 | +0.121 | 12.27 | 12.03 | 1.02x |
| 211 | 0.861 | 0.897 | +0.036 | 12.44 | 12.26 | 1.01x |
| 303 | 0.795 | 0.821 | +0.027 | 12.30 | 12.08 | 1.02x |
| 404 | 0.717 | 0.805 | +0.088 | 12.40 | 12.15 | 1.02x |
| 503 | 0.871 | 0.892 | +0.021 | 12.30 | 12.14 | 1.01x |
| 777 | 0.849 | 0.904 | +0.055 | 12.19 | 12.09 | 1.01x |
| 1009 | 0.820 | 0.857 | +0.037 | 12.49 | 12.21 | 1.02x |
| 1337 | 0.785 | 0.821 | +0.036 | 12.09 | 11.93 | 1.01x |
| 2027 | 0.784 | 0.872 | +0.088 | 12.29 | 12.07 | 1.02x |
| 2718 | 0.856 | 0.888 | +0.032 | 12.18 | 12.00 | 1.01x |
| 3141 | 0.779 | 0.821 | +0.043 | 12.45 | 12.20 | 1.02x |
| 4096 | 0.571 | 0.721 | +0.151 | 12.21 | 12.03 | 1.01x |
| 9001 | 0.660 | 0.772 | +0.112 | 12.33 | 11.98 | 1.03x |
| 9999 | 0.836 | 0.887 | +0.051 | 12.25 | 12.14 | 1.01x |

## Aggregated

- Mean delta: `+0.055`
- Min delta: `+0.011`
- Mean spread ratio: `1.02x`
- Min spread ratio: `1.01x`
- Negative-delta seeds: `0`
- Spread `<= 1.0x` seeds: `0`
- Retrieval points: `70.0`
- Anisotropy points: `0.15`
- Total automated: `70.15 / 90.0`


## Decision

Final main passes the guarded adaptive retrieval 20-seed gate with `0` negative-delta seeds and `0` spread failures. Compared with the prior stable 20-seed result (`+0.048` mean delta, `+0.005` min delta, `66.98 / 90`), guarded adaptive retrieval improves to `+0.055` mean delta, `+0.011` min delta, and `70.15 / 90` while preserving spread.
