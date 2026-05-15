# Adaptive Many-Seed Summary

| Seed | Baseline Acc | Agent Acc | Delta | Base Spread | Agent Spread | Spread Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.752 | 0.760 | +0.008 | 12.21 | 11.99 | 1.02x |
| 13 | 0.869 | 0.892 | +0.023 | 12.11 | 12.11 | 1.00x |
| 31 | 0.769 | 0.821 | +0.052 | 12.26 | 12.07 | 1.02x |
| 42 | 0.873 | 0.896 | +0.023 | 12.17 | 12.01 | 1.01x |
| 97 | 0.829 | 0.851 | +0.021 | 12.56 | 12.18 | 1.03x |
| 101 | 0.788 | 0.839 | +0.051 | 12.31 | 12.08 | 1.02x |
| 211 | 0.861 | 0.897 | +0.036 | 12.44 | 12.44 | 1.00x |
| 503 | 0.871 | 0.892 | +0.021 | 12.30 | 12.14 | 1.01x |
| 1009 | 0.820 | 0.855 | +0.035 | 12.49 | 12.39 | 1.01x |
| 2027 | 0.784 | 0.865 | +0.081 | 12.29 | 12.07 | 1.02x |
| 3141 | 0.779 | 0.812 | +0.033 | 12.45 | 12.20 | 1.02x |
| 9999 | 0.836 | 0.881 | +0.045 | 12.25 | 12.14 | 1.01x |

## Aggregated

- Mean delta: `+0.036`
- Min delta: `+0.008`
- Mean spread ratio: `1.01x`
- Min spread ratio: `1.00x`
- Negative-delta seeds: `0`
- Spread `<= 1.0x` seeds: `2`
- Retrieval points: `50.09`
- Anisotropy points: `0.06`
- Total automated: `50.15 / 90.0`


## Decision

Conservative adaptive retrieval improves 12-seed retrieval versus stable (`+0.036` mean delta, `+0.008` min delta), but also fails the spread gate with `2` seeds at spread `<= 1.0x`. Do not run 20-seed and do not adopt.
