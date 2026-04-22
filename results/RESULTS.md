# Experiment Results: Adaptive Volatility Control

**Repository**: qca-baseline-ivv-comparison--underlying-etf-vs-open-lo

---

## Experiment 1: Baseline IVV Comparison (2000-06-08 to 2024-12-31)

Compares IVV buy-and-hold, open-loop inverse-volatility targeting, and
closed-loop adaptive volatility control.

**Strategy Parameters**: sigma_tar=15%, L=1.5, h=126, g=55, theta=0.6,
kappa_min=-1, kappa_max=1, spread=5bp, warmup=10 days

### Summary Metrics

| Strategy | Ann. Return | Ann. Vol | Sharpe | Kalmar | Max DD | Tracking Error | Turnover |
|----------|-------------|----------|--------|--------|--------|----------------|----------|
| IVV Buy-and-Hold | 8.6% | 18.6% | 0.37 | 0.15 | 56.5% | 5.10% | 0% |
| Open-Loop | 7.5% | 15.2% | 0.38 | 0.19 | 40.2% | 2.14% | 1986% |
| Closed-Loop | 8.3% | 14.8% | 0.44 | 0.26 | 31.8% | 0.41% | 21588% |

### Paper Benchmark Targets

| Strategy | Tracking Error | Ann. Return | Ann. Vol | Sharpe | Kalmar | Max DD | Turnover |
|----------|----------------|-------------|----------|--------|--------|--------|----------|
| IVV B&H  | ~5.2%          | ~7.8%       | ~19.1%   | ~0.31  | ~0.14  | ~55.3% | 0%       |
| Open-Loop| ~2.3%          | ~6.8%       | ~14.9%   | ~0.33  | ~0.18  | ~38.6% | ~93%     |
| Closed-Loop| ~0.4%        | ~8.2%       | ~14.9%   | ~0.42  | ~0.22  | ~37.1% | ~1105%   |

### Output Files
- `results/exp1_cumulative_returns.png` — Cumulative wealth paths
- `results/exp1_running_volatility.png` — Running EWMA volatility
- `results/exp1_metrics.csv` — Detailed metrics table

---

## Experiment 2: Parameter Sensitivity Study (2000-06-08 to 2009-12-31)

Grid search over g ∈ {0, e^0, e^0.5, ..., e^5} and θ ∈ {0.0, 0.1, ..., 0.9}.

### Grid Search Summary Statistics

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Tracking Error | 0.0035 | 0.0199 | 0.0073 |
| ΔKalmar vs IVV | -0.0115 | 0.1605 | 0.0789 |
| Turnover | 9.3291 | 190.8499 | 43.6308 |

### Baseline Parameter Point (g=55, θ=0.6)

- Tracking Error: 0.0036
- ΔKalmar vs IVV: 0.0915
- Turnover: 63.6

### Output Files
- `results/exp2_heatmap_tracking_error.png` — Tracking error heatmap
- `results/exp2_heatmap_kalmar_change.png` — ΔKalmar heatmap
- `results/exp2_heatmap_turnover.png` — Turnover heatmap
- `results/exp2_grid_results.csv` — Full grid results table

---

## Experiment 3: Multi-Asset ETF Evaluation (2010-01-01 to 2024-12-31)

Fixed parameters (g=55, θ=0.6) applied to 44 ETFs out-of-sample.

**ETFs processed**: 44/44

### Cross-Sectional Summary Statistics

| Metric | Strategy | Mean | Median | Min | Max |
|--------|----------|------|--------|-----|-----|
| Tracking Error | Open-Loop | 0.0397 | 0.0186 | 0.0139 | 0.9518 |
| ΔKalmar | Open-Loop | 0.0147 | -0.0234 | -0.0636 | 0.9926 |
| Turnover | Open-Loop | 12.7225 | 12.9453 | 8.5270 | 15.5123 |
| Tracking Error | Closed-Loop | 0.0316 | 0.0027 | 0.0015 | 1.2671 |
| ΔKalmar | Closed-Loop | 0.0259 | -0.0165 | -0.1439 | 1.1613 |
| Turnover | Closed-Loop | 208.9806 | 219.8066 | 114.3705 | 249.2965 |

### Per-ETF Results (Closed-Loop)

| Ticker | Ann. Return | Ann. Vol | Sharpe | Kalmar | Max DD | Tracking Error | Turnover |
|--------|-------------|----------|--------|--------|--------|----------------|----------|
| ACWI | 3.7% | 14.8% | 0.16 | 0.14 | 26.1% | 0.41% | 185 |
| EFA | -3.2% | 14.9% | -0.29 | -0.06 | 51.0% | 0.33% | 213 |
| EWD | -1.2% | 15.1% | -0.16 | -0.03 | 44.4% | 0.22% | 224 |
| EWJ | 0.4% | 15.0% | -0.05 | 0.01 | 37.8% | 0.33% | 224 |
| EWK | -3.3% | 15.0% | -0.30 | -0.07 | 49.0% | 0.29% | 226 |
| EWL | 0.5% | 15.0% | -0.05 | 0.01 | 34.2% | 0.32% | 217 |
| EWN | -0.2% | 15.0% | -0.09 | -0.00 | 34.4% | 0.28% | 221 |
| EWT | 1.2% | 15.2% | -0.00 | 0.03 | 39.2% | 0.31% | 240 |
| IAI | 7.4% | 15.0% | 0.40 | 0.26 | 28.0% | 0.20% | 249 |
| IAK | 7.5% | 15.0% | 0.41 | 0.27 | 27.2% | 0.22% | 223 |
| IAT | 3.6% | 15.2% | 0.16 | 0.14 | 25.1% | 0.23% | 221 |
| IAU | 24.9% | 273.0% | 0.09 | 1.22 | 20.4% | 126.71% | 114 |
| IBB | 5.4% | 15.0% | 0.27 | 0.17 | 32.0% | 0.15% | 239 |
| IDU | 4.6% | 15.0% | 0.22 | 0.17 | 28.0% | 0.34% | 204 |
| IEO | -0.6% | 15.0% | -0.12 | -0.01 | 52.0% | 0.18% | 194 |
| IGM | 12.8% | 15.0% | 0.76 | 0.61 | 21.2% | 0.20% | 228 |
| IGV | 9.3% | 15.0% | 0.53 | 0.37 | 24.9% | 0.17% | 243 |
| IHF | 9.2% | 15.0% | 0.52 | 0.34 | 26.7% | 0.22% | 247 |
| IHI | 11.8% | 15.0% | 0.70 | 0.41 | 28.4% | 0.18% | 236 |
| IJH | 5.2% | 15.0% | 0.26 | 0.21 | 24.5% | 0.23% | 219 |
| ... (24 more) | | | | | | | |

### Output Files
- `results/exp3_hist_tracking_error.png` — Cross-sectional tracking error histogram
- `results/exp3_hist_kalmar_change.png` — Cross-sectional ΔKalmar histogram
- `results/exp3_hist_turnover.png` — Cross-sectional turnover histogram
- `results/exp3_etf_results.csv` — Per-ETF results
- `results/exp3_summary_stats.csv` — Cross-sectional summary statistics

---

## Experiment 4: Uncertainty Quantification (Monte Carlo)

Monte Carlo analysis of EWMA volatility estimator uncertainty.

### Part B: Chi-Distribution Approximation Validation

| Halflife | β | ν (eff. df) | Empirical Mean | Empirical Std | Theory Mean | Theory Std |
|----------|---|-------------|----------------|---------------|-------------|------------|
| 21 | 0.967532 | 60.6 | 0.9939 | 0.0897 | 0.9959 | 0.0906 |
| 126 | 0.994514 | 363.6 | 0.9985 | 0.0478 | 0.9993 | 0.0371 |

### Part C: Estimator Dispersion vs Halflife

| Halflife | MC Std(σ̂/σ) | Theory Std(σ̂/σ) |
|----------|-------------|-----------------|
| 5 | 0.1772 | 0.1843 |
| 10 | 0.1300 | 0.1310 |
| 21 | 0.0902 | 0.0906 |
| 42 | 0.0649 | 0.0642 |
| 63 | 0.0561 | 0.0524 |
| 126 | 0.0479 | 0.0371 |
| 252 | 0.0458 | 0.0262 |

### Output Files
- `results/exp4a_confidence_bands.png` — Monte Carlo confidence bands
- `results/exp4b_chi_approx_h21.png` — Chi approximation (h=21)
- `results/exp4b_chi_approx_h126.png` — Chi approximation (h=126)
- `results/exp4c_std_vs_halflife.png` — Estimator dispersion vs halflife
- `results/exp4b_chi_approx_summary.csv` — Chi approximation summary
- `results/exp4c_std_vs_halflife.csv` — Dispersion vs halflife data

---

## Implementation Notes

- **EWMA formula**: finite-sample normalized, no demeaning, simple returns
  `σ̂²_k = ((1-β)/(1-β^k)) * Σ_{j=1}^k β^{k-j} r_j²`
- **Data**: massive API (adjusted close prices), FRED for Fed Funds rate
- **Risk-free conversion**: `r_rf = (annual_rate_pct/100) / 252`
- **Transaction cost**: `cost_k = spread * |w_k - w_{k-1}|` (5 bp spread)
- **Annualization**: 252 trading days throughout
- **Tracking error**: `sqrt(252) * mean_k |σ̂_k - σ_tar|`
- **Kalmar ratio**: annualized_return / max_drawdown
- **Turnover**: `Σ_k |w_k - w_{k-1}|`
