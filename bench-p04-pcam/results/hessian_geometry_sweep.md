# Hessian Geometry Blend Sweep

Modes: `topk_hdiag`, `topk_ruiz`

| mode | topk weight | geom weight | mean delta | min delta | mean spread | min spread | ok | note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| topk_hdiag | 0.95 | 0.05 | +0.030 | +0.020 | 0.86x | 0.84x | yes |  |
| topk_hdiag | 0.90 | 0.10 | +0.030 | +0.020 | 0.87x | 0.85x | yes |  |
| topk_hdiag | 0.85 | 0.15 | +0.030 | +0.020 | 0.88x | 0.86x | yes |  |
| topk_hdiag | 0.75 | 0.25 | +0.030 | +0.020 | 0.90x | 0.88x | yes |  |
| topk_hdiag | 0.60 | 0.40 | +0.025 | +0.020 | 0.93x | 0.92x | yes |  |
| topk_hdiag | 0.40 | 0.60 | +0.015 | +0.010 | 0.97x | 0.96x | yes |  |
| topk_ruiz | 0.95 | 0.05 | +0.030 | +0.020 | 0.86x | 0.84x | yes |  |
| topk_ruiz | 0.90 | 0.10 | +0.030 | +0.020 | 0.87x | 0.85x | yes |  |
| topk_ruiz | 0.85 | 0.15 | +0.030 | +0.020 | 0.88x | 0.86x | yes |  |
| topk_ruiz | 0.75 | 0.25 | +0.030 | +0.020 | 0.90x | 0.88x | yes |  |
| topk_ruiz | 0.60 | 0.40 | +0.025 | +0.020 | 0.93x | 0.92x | yes |  |
| topk_ruiz | 0.40 | 0.60 | +0.015 | +0.010 | 0.96x | 0.96x | yes |  |

## Best `topk_hdiag` By Mean Delta

- topk weight: `0.75`
- geom weight: `0.25`
- mean delta: `+0.030`
- min delta: `+0.020`
- mean spread reduction: `0.90x`
- min spread reduction: `0.88x`

## Best `topk_ruiz` By Mean Delta

- topk weight: `0.75`
- geom weight: `0.25`
- mean delta: `+0.030`
- min delta: `+0.020`
- mean spread reduction: `0.90x`
- min spread reduction: `0.88x`
