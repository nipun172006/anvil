# Retrieval Error Analysis

Seeds: `[42, 101]`
Noise levels: `[0.5, 0.7, 0.8]`
Queries per level: `250`
Runtime: `71.6s`
Policy: `fixed`

## Overall

- Baseline accuracy: `0.831`
- Agent accuracy: `0.865`
- Delta: `+0.035`
- Helped queries: `52`
- Hurt queries: `0`
- Same-wrong queries: `202`

## Weakest Seeds

- Seed `42`: delta `+0.021`, agent `0.895`, baseline `0.873`, helped `16`, hurt `0`
- Seed `101`: delta `+0.048`, agent `0.836`, baseline `0.788`, helped `36`, hurt `0`

## By Seed

| Group | N | Base Acc | Agent Acc | Delta | Helped | Hurt | Same Wrong | Mean d1 | Mean Margin | Mean Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 750 | 0.873 | 0.895 | +0.021 | 16 | 0 | 79 | 1.0769 | 0.4682 | 1.561 |
| 101 | 750 | 0.788 | 0.836 | +0.048 | 36 | 0 | 123 | 1.0699 | 0.4481 | 1.536 |

## By Noise Level

| Group | N | Base Acc | Agent Acc | Delta | Helped | Hurt | Same Wrong | Mean d1 | Mean Margin | Mean Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 500 | 0.960 | 0.976 | +0.016 | 8 | 0 | 12 | 0.7832 | 0.6996 | 2.002 |
| 0.7 | 500 | 0.874 | 0.900 | +0.026 | 13 | 0 | 50 | 1.1145 | 0.4212 | 1.427 |
| 0.8 | 500 | 0.658 | 0.720 | +0.062 | 31 | 0 | 140 | 1.3224 | 0.2536 | 1.216 |

## By Nearest-Neighbor Margin

| Group | N | Base Acc | Agent Acc | Delta | Helped | Hurt | Same Wrong | Mean d1 | Mean Margin | Mean Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <=0.01 | 23 | 0.130 | 0.348 | +0.217 | 5 | 0 | 15 | 1.4823 | 0.0060 | 1.004 |
| <=0.03 | 49 | 0.204 | 0.388 | +0.184 | 9 | 0 | 30 | 1.5121 | 0.0212 | 1.014 |
| >0.03 | 1428 | 0.863 | 0.890 | +0.027 | 38 | 0 | 157 | 1.0518 | 0.4804 | 1.575 |

## By Nearest Distance

| Group | N | Base Acc | Agent Acc | Delta | Helped | Hurt | Same Wrong | Mean d1 | Mean Margin | Mean Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <0.80 | 330 | 0.970 | 0.988 | +0.018 | 6 | 0 | 4 | 0.6669 | 0.8060 | 2.267 |
| >=0.80 | 213 | 0.962 | 0.972 | +0.009 | 2 | 0 | 6 | 0.8747 | 0.6168 | 1.708 |
| >=0.95 | 957 | 0.753 | 0.799 | +0.046 | 44 | 0 | 192 | 1.2578 | 0.3028 | 1.265 |

## Pattern Notes

- Small nearest-neighbor margin is a useful ambiguity signal when its hurt/help balance differs from the overall average.
- Large nearest distance is a useful noise signal when high-distance buckets have weaker delta.
- Hurt examples inspected: `0`; helped examples inspected: `52`.
