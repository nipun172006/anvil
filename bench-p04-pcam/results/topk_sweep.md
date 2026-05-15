# Top-k Official Quick Sweep

Mode: `topk`

Scope: one-factor sweep around current defaults

| k | temp | consensus | distance | mean delta | min delta | mean spread | min spread | ok | note |
|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 2 | 1 | 0.45 | euclidean | -0.050 | -0.060 | 0.89x | 0.89x | yes |  |
| 3 | 1 | 0.45 | euclidean | -0.020 | -0.020 | 0.84x | 0.82x | yes |  |
| 4 | 1 | 0.45 | euclidean | +0.005 | +0.000 | 0.83x | 0.81x | yes |  |
| 5 | 1 | 0.45 | euclidean | +0.010 | +0.000 | 0.81x | 0.81x | yes |  |
| 8 | 1 | 0.45 | euclidean | +0.030 | +0.020 | 0.83x | 0.82x | yes |  |
| 12 | 1 | 0.45 | euclidean | +0.030 | +0.020 | 0.85x | 0.83x | yes |  |
| 5 | 0.25 | 0.45 | euclidean | -0.045 | -0.060 | 0.91x | 0.90x | yes |  |
| 5 | 0.5 | 0.45 | euclidean | -0.005 | -0.010 | 0.86x | 0.84x | yes |  |
| 5 | 2 | 0.45 | euclidean | +0.020 | +0.020 | 0.80x | 0.80x | yes |  |
| 5 | 1 | 0 | euclidean | +0.015 | +0.010 | 0.88x | 0.86x | yes |  |
| 5 | 1 | 0.1 | euclidean | +0.015 | +0.010 | 0.87x | 0.85x | yes |  |
| 5 | 1 | 0.2 | euclidean | +0.015 | +0.010 | 0.85x | 0.84x | yes |  |
| 5 | 1 | 0.4 | euclidean | +0.010 | +0.000 | 0.82x | 0.81x | yes |  |
| 5 | 1 | 0.45 | cosine | +0.010 | +0.000 | 0.81x | 0.81x | yes |  |

## Best By Mean Delta

- k: `12`
- temp: `1`
- consensus weight: `0.45`
- distance: `euclidean`
- mean delta: `+0.030`
- min delta: `+0.020`
- mean spread reduction: `0.85x`
- min spread reduction: `0.83x`
