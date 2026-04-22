# Repository: qca-baseline-ivv-comparison

## Purpose
Implements four quantitative finance experiments comparing buy-and-hold IVV,
open-loop inverse-volatility targeting, and closed-loop adaptive volatility control.

## Structure
- `data/` – data fetching (massive API for ETF prices, FRED for Fed Funds rate)
- `strategies/` – core strategy logic (EWMA estimator, open-loop, closed-loop controller)
- `experiments/` – experiment runners (exp1–exp4)
- `utils/` – metrics, plotting helpers
- `tests/` – pytest test suite
- `results/` – output plots and RESULTS.md
- `main.py` – orchestrates all experiments

## Key Implementation Notes
- EWMA formula: sigma_hat_k^2 = ((1-beta)/(1-beta^k)) * sum_{j=1}^k beta^(k-j) * r_j^2
  (finite-sample normalized, no demeaning, simple returns)
- beta = exp(-log(2)/h), h = 126 trading days
- sigma_tar = 0.15 / sqrt(252) (daily), L = 1.5 (leverage cap)
- Closed-loop: kappa_k = (1-theta)*clip(-g*e_k; [kappa_min, kappa_max]) + theta*kappa_{k-1}
  where e_k = log(sigma_hat_k_ind / sigma_tar)
- Baseline params: g=55, theta=0.6, kappa_min=-1, kappa_max=1
- Transaction cost: 5 bp spread on |w_k - w_{k-1}|
- 10-day warm-up for closed-loop (uses open-loop weights)
- Annualization: 252 trading days
- Data: massive API (adjusted close prices), FRED for Fed Funds rate

## Libraries Used
- massive: ETF price data
- pandas_datareader: FRED Federal Funds rate
- quantstats: performance metrics (Sharpe, Calmar, drawdown) — Context7 confirmed
- matplotlib/seaborn: visualization
- numpy/pandas: core computation
- scipy.stats: chi distribution for Exp 4

## Conventions
- Simple close-to-close returns: r_k = P_k/P_{k-1} - 1
- Fed Funds rate: annual rate / 252 for daily simple return
- Turnover: sum_k |w_k - w_{k-1}| (total over full period, not annualized)
- Tracking error: sqrt(252) * mean_k |sigma_hat_k_eval - sigma_tar|
- Kalmar ratio: annualized_return / max_drawdown (absolute value)

## Experiment Results (2026-04-22)
### Exp 1 (IVV 2000-2024)
- IVV B&H: 8.6% return, 18.6% vol, Sharpe 0.37, Kalmar 0.15, MDD 56.5%, TE 5.10%
- Open-Loop: 7.5% return, 15.2% vol, Sharpe 0.38, Kalmar 0.19, MDD 40.2%, TE 2.14%
- Closed-Loop: 8.3% return, 14.8% vol, Sharpe 0.44, Kalmar 0.26, MDD 31.8%, TE 0.41%
- Qualitative ordering matches paper: CL best TE, OL intermediate, IVV worst

### Exp 2 (Grid search 2000-2009)
- TE range: 0.35% to 1.99%; ΔKalmar range: -0.01 to +0.16; Turnover: 9.3 to 190.8
- Baseline (g=55, θ=0.6): TE=0.36%, ΔKalmar=+0.09, Turnover=63.6

### Exp 3 (44 ETFs 2010-2024)
- CL median TE: 0.27% (vs OL median 1.86%) — clear separation
- CL median turnover: 219.8 (vs OL 12.9) — much higher as expected
- IAU (gold ETF) is an outlier with very high TE due to low natural volatility

### Exp 4 (Monte Carlo)
- Chi approximation: h=21 empirical std=0.090 vs theory 0.091; h=126 empirical 0.048 vs theory 0.037
- Dispersion decreases with halflife as expected

## Running Experiments
```bash
export MASSIVE_TOKEN=<your_key>
python main.py --exp all   # run all experiments
python main.py --exp 1     # run single experiment
python -m pytest tests/    # run test suite (75 tests)
```
