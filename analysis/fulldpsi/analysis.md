# CD&R Simulation Results Analysis

## Experiment Overview

| Parameter | Values |
|-----------|--------|
| Result files | 28 JSON files across 6 directories |
| Uncertainty levels | 4 (CI_pos x CI_vel = {3, 10} m x {1, 3} m/s) |
| Recovery methods | Past-CPA, FTR, Probabilistic FTR |
| Gamma thresholds | 0.5, 0.75, 0.9, 0.99, 0.999 |
| Crossing angles | 49 values from 2 to 180 degrees |
| Protected zone radius (R_PZ) | 50 m |
| Ground speed | 20 kts (both aircraft) |
| Look-ahead time | 120 s |

## 1. Recovery Method Comparison

### IPR vs Crossing Angle

![IPR vs Crossing Angle](fig_crossing_angle_vs_ipr.png)

**Key observations:**

| Uncertainty | CPA Mean IPR | FTR Mean IPR | Prob. FTR Mean IPR | CPA Min IPR | FTR Min IPR | Prob. FTR Min IPR |
|-------------|-------------|-------------|-------------------|------------|------------|------------------|
| CI_pos=3m, CI_vel=1m/s | 0.9815 | 0.9561 | 0.9990 | 0.1331 | 0.7067 | 0.9949 |
| CI_pos=3m, CI_vel=3m/s | 0.9517 | 0.8103 | 0.9877 | 0.1290 | 0.4371 | 0.9008 |
| CI_pos=10m, CI_vel=1m/s | 0.9810 | 0.9609 | 0.9995 | 0.1170 | 0.7541 | 0.9971 |
| CI_pos=10m, CI_vel=3m/s | 0.9488 | 0.7887 | 0.9888 | 0.0967 | 0.4295 | 0.9218 |

- The probabilistic method (gamma=0.999) achieves the highest IPR across all conditions
- Both deterministic methods degrade sharply at small crossing angles (< 20 deg)
- FTR drops as low as 0.43 at 2 deg under the highest uncertainty level
- The probabilistic method remains above 0.92 even at 2 deg with highest uncertainty

### Median DCPA vs Crossing Angle

![Median DCPA](fig_crossing_angle_vs_dcpa_median.png)

**Key observations:**

- FTR yields the most efficient separation (median DCPA close to R_PZ = 50 m)
- CPA produces the largest separation (up to ~390 m at large crossing angles under high uncertainty)
- Probabilistic FTR sits between the two, trading efficiency for safety
- At large crossing angles, the probabilistic method converges toward FTR values

## 2. Effect of Confidence Threshold (gamma)

### IPR vs gamma

![Gamma IPR](fig_gamma_comparison_ipr.png)

**Key observations:**

- Increasing gamma monotonically improves IPR
- At gamma=0.5, performance is nearly identical to deterministic FTR
- At gamma>=0.99, IPR exceeds 0.99 at all crossing angles under low uncertainty
- The improvement from gamma=0.5 to gamma=0.75 is the largest single step

### Median DCPA vs gamma

![Gamma DCPA](fig_gamma_comparison_dcpa_median.png)

**Key observations:**

- Increasing gamma increases median DCPA (more conservative separation)
- The spread between gamma curves is wider under higher velocity uncertainty
- All gamma values keep median DCPA above R_PZ = 50 m
- gamma=0.999 reaches ~80 m (low unc.) to ~125 m (high unc.), well below CPA's ~200-390 m

## 3. Full Summary Statistics

| CI_pos [m] | CI_vel [m/s] | Method | Mean IPR | Min IPR | % angles with IPR >= 0.99 | Mean Median DCPA [m] |
|:---:|:---:|--------|:---:|:---:|:---:|:---:|
| 3 | 1 | Past-CPA | 0.9815 | 0.1331 | 93.3% | 121.0 |
| 3 | 1 | FTR | 0.9561 | 0.7067 | 0.0% | 53.5 |
| 3 | 1 | Prob. FTR (gamma=0.5) | 0.9580 | 0.7681 | 0.0% | 53.5 |
| 3 | 1 | Prob. FTR (gamma=0.75) | 0.9924 | 0.9407 | 91.1% | 57.3 |
| 3 | 1 | Prob. FTR (gamma=0.9) | 0.9974 | 0.9768 | 96.7% | 61.9 |
| 3 | 1 | Prob. FTR (gamma=0.99) | 0.9988 | 0.9925 | 100.0% | 71.6 |
| 3 | 1 | Prob. FTR (gamma=0.999) | 0.9990 | 0.9949 | 100.0% | 78.9 |
| 3 | 3 | Past-CPA | 0.9517 | 0.1290 | 44.4% | 241.7 |
| 3 | 3 | FTR | 0.8103 | 0.4371 | 0.0% | 52.9 |
| 3 | 3 | Prob. FTR (gamma=0.5) | 0.8043 | 0.4356 | 0.0% | 52.7 |
| 3 | 3 | Prob. FTR (gamma=0.75) | 0.9276 | 0.7653 | 0.0% | 57.0 |
| 3 | 3 | Prob. FTR (gamma=0.9) | 0.9699 | 0.8670 | 0.0% | 65.4 |
| 3 | 3 | Prob. FTR (gamma=0.99) | 0.9862 | 0.8981 | 63.3% | 95.1 |
| 3 | 3 | Prob. FTR (gamma=0.999) | 0.9877 | 0.9008 | 67.8% | 125.7 |
| 10 | 1 | Past-CPA | 0.9810 | 0.1170 | 93.3% | 123.4 |
| 10 | 1 | FTR | 0.9609 | 0.7541 | 0.0% | 56.4 |
| 10 | 1 | Prob. FTR (gamma=0.5) | 0.9637 | 0.7995 | 0.0% | 56.4 |
| 10 | 1 | Prob. FTR (gamma=0.75) | 0.9932 | 0.9533 | 92.2% | 60.1 |
| 10 | 1 | Prob. FTR (gamma=0.9) | 0.9981 | 0.9826 | 96.7% | 64.8 |
| 10 | 1 | Prob. FTR (gamma=0.99) | 0.9993 | 0.9955 | 100.0% | 75.1 |
| 10 | 1 | Prob. FTR (gamma=0.999) | 0.9995 | 0.9971 | 100.0% | 82.9 |
| 10 | 3 | Past-CPA | 0.9488 | 0.0967 | 52.2% | 243.2 |
| 10 | 3 | FTR | 0.7887 | 0.4295 | 0.0% | 54.2 |
| 10 | 3 | Prob. FTR (gamma=0.5) | 0.7827 | 0.4217 | 0.0% | 54.0 |
| 10 | 3 | Prob. FTR (gamma=0.75) | 0.9202 | 0.7482 | 0.0% | 59.0 |
| 10 | 3 | Prob. FTR (gamma=0.9) | 0.9669 | 0.8789 | 0.0% | 68.7 |
| 10 | 3 | Prob. FTR (gamma=0.99) | 0.9863 | 0.9177 | 60.0% | 99.0 |
| 10 | 3 | Prob. FTR (gamma=0.999) | 0.9888 | 0.9218 | 72.2% | 128.6 |
