# Adaptive Threshold Sweep

Mode: `adaptive_topk_spread`

| detect mode | distance threshold | cosine threshold | mean delta | min delta | mean spread | min spread | ok | note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| distance | 0.05 |  | +0.030 | +0.020 | 0.86x | 0.83x | yes |  |
| distance | 0.1 |  | +0.030 | +0.020 | 0.89x | 0.89x | yes |  |
| distance | 0.15 |  | +0.030 | +0.020 | 0.98x | 0.94x | yes |  |
| distance | 0.2 |  | +0.030 | +0.020 | 1.02x | 1.01x | yes |  |
| distance | 0.3 |  | +0.030 | +0.020 | 1.02x | 1.01x | yes |  |
| distance | 0.4 |  | +0.030 | +0.020 | 1.02x | 1.01x | yes |  |
| distance | 0.6 |  | +0.030 | +0.020 | 1.02x | 1.01x | yes |  |
| distance | 0.8 |  | +0.030 | +0.020 | 1.02x | 1.01x | yes |  |
| distance | 1 |  | +0.030 | +0.020 | 1.02x | 1.01x | yes |  |
| cosine |  | 0.9 | +0.030 | +0.020 | 1.02x | 1.01x | yes |  |
| cosine |  | 0.94 | +0.030 | +0.020 | 0.93x | 0.90x | yes |  |
| cosine |  | 0.96 | +0.030 | +0.020 | 0.86x | 0.83x | yes |  |
| cosine |  | 0.98 | +0.030 | +0.020 | 0.86x | 0.83x | yes |  |
| cosine |  | 0.99 | +0.030 | +0.020 | 0.86x | 0.83x | yes |  |

## Best By Mean Delta

- detect mode: `distance`
- distance threshold: `0.2`
- cosine threshold: `None`
- mean delta: `+0.030`
- min delta: `+0.020`
- mean spread reduction: `1.02x`
- min spread reduction: `1.01x`
