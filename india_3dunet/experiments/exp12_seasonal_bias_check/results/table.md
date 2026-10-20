## 1. ECMWF's bias through the year
Factor = (observed + 1) / (ECMWF + 1) on weekly totals, fitted on one issue month at a time (2004-2017). Median over Indian land cells; above 1 means ECMWF is too dry.

| issue month | wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|
| Jan | 0.79 | 0.78 | 0.80 | 0.89 |
| Feb | 0.91 | 0.99 | 1.19 | 1.21 |
| Mar | 1.04 | 1.17 | 1.23 | 1.31 |
| Apr | 1.15 | 1.33 | 1.22 | 1.14 |
| May | 1.10 | 1.04 | 0.95 | 0.91 |
| Jun | 0.94 | 0.93 | 0.91 | 0.90 |
| Jul | 0.95 | 0.97 | 0.99 | 0.97 |
| Aug | 0.94 | 1.03 | 0.98 | 1.04 |
| Sep | 0.95 | 0.92 | 0.91 | 0.85 |
| Oct | 0.83 | 0.65 | 0.69 | 0.72 |
| Nov | 0.73 | 0.66 | 0.51 | 0.51 |
| Dec | 0.72 | 0.72 | 0.67 | 0.70 |

## 2. Is the bias a seasonal pattern or noise?
Correlation over land of log factor maps: month m fitted on even years vs month m+k fitted on odd years (independent forecasts). Gap 0 = how alike the same month is in two independent samples (the ceiling). Mean over all 12 months; June-September in brackets.

| months apart | wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|
| 0 | 0.75 (0.83) | 0.70 (0.82) | 0.69 (0.78) | 0.68 (0.71) |
| 1 | 0.59 (0.64) | 0.56 (0.57) | 0.58 (0.58) | 0.53 (0.60) |
| 2 | 0.43 (0.48) | 0.37 (0.43) | 0.38 (0.38) | 0.34 (0.42) |
| 3 | 0.28 (0.29) | 0.23 (0.22) | 0.20 (0.15) | 0.20 (0.20) |
| 4 | 0.22 (0.18) | 0.19 (0.14) | 0.19 (0.20) | 0.19 (0.17) |
| 5 | 0.19 (0.12) | 0.17 (0.14) | 0.12 (0.15) | 0.14 (0.17) |
| 6 | 0.17 (0.16) | 0.14 (0.10) | 0.12 (0.09) | 0.13 (0.12) |

## 3. Dyn++ with different month windows
Weekly anomaly correlation (cell, then all-India). Choose on validation; test for reference.

**val** (155 forecasts, 2018-2020)

| window | cell wk1 | wk2 | wk3 | wk4 | all-India wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|---|---|---|---|
| same month only | 0.641 | 0.455 | 0.227 | 0.190 | 0.921 | 0.715 | 0.582 | 0.322 |
| +/-1 month (used) | 0.648 | 0.457 | 0.231 | 0.183 | 0.907 | 0.673 | 0.576 | 0.313 |
| +/-2 months | 0.636 | 0.439 | 0.219 | 0.169 | 0.892 | 0.637 | 0.547 | 0.272 |
| +/-3 months | 0.627 | 0.431 | 0.213 | 0.162 | 0.885 | 0.630 | 0.537 | 0.261 |
| all year | 0.619 | 0.423 | 0.200 | 0.150 | 0.878 | 0.633 | 0.536 | 0.270 |

**test** (153 forecasts, 2021-2023)

| window | cell wk1 | wk2 | wk3 | wk4 | all-India wk1 | wk2 | wk3 | wk4 |
|---|---|---|---|---|---|---|---|---|
| same month only | 0.667 | 0.509 | 0.335 | 0.249 | 0.916 | 0.838 | 0.578 | 0.381 |
| +/-1 month (used) | 0.668 | 0.512 | 0.330 | 0.239 | 0.910 | 0.837 | 0.590 | 0.389 |
| +/-2 months | 0.663 | 0.499 | 0.315 | 0.233 | 0.907 | 0.824 | 0.580 | 0.412 |
| +/-3 months | 0.656 | 0.492 | 0.307 | 0.221 | 0.902 | 0.817 | 0.576 | 0.400 |
| all year | 0.647 | 0.480 | 0.300 | 0.212 | 0.889 | 0.807 | 0.576 | 0.405 |

