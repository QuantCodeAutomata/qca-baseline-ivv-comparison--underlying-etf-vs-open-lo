"""
Experiment 2: IVV parameter sensitivity study.

Grid search over (g, theta) for the closed-loop controller on IVV
over the in-sample period 2000-06-08 to 2009-12-31.

Grid:
  theta in {0.0, 0.1, ..., 0.9}  (10 values)
  g in {0, exp(0), exp(0.5), ..., exp(5)}  (12 values)

Reports: heatmaps for tracking error, Kalmar change vs IVV, and turnover.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch import fetch_single_etf_prices, fetch_fed_funds_rate, build_aligned_returns
from strategies.volatility_control import (
    run_strategies,
    SIGMA_TAR,
    LEVERAGE_CAP,
    HALFLIFE,
    SPREAD,
    KAPPA_MIN,
    KAPPA_MAX,
    WARMUP_DAYS,
    TRADING_DAYS,
)
from strategies.ewma import compute_beta, ewma_variance_series
from utils.metrics import (
    annualized_return,
    max_drawdown,
    kalmar_ratio,
    volatility_tracking_error,
    total_turnover,
    annualized_volatility,
    sharpe_ratio,
)
from utils.plotting import plot_heatmap

logger = logging.getLogger(__name__)

# Experiment parameters
TICKER = "IVV"
START_DATE = "2000-06-08"
END_DATE = "2009-12-31"
RESULTS_DIR = Path("results")

# Parameter grid (exact as specified in paper)
THETA_GRID = np.arange(0.0, 1.0, 0.1)  # 0.0, 0.1, ..., 0.9
G_GRID = np.concatenate([[0.0], np.exp(np.arange(0.0, 5.5, 0.5))])  # 12 values


def run_single_closed_loop(
    risky_returns: np.ndarray,
    cash_returns: np.ndarray,
    g: float,
    theta: float,
    sigma_tar: float = SIGMA_TAR,
    leverage_cap: float = LEVERAGE_CAP,
    halflife: float = HALFLIFE,
    spread: float = SPREAD,
    kappa_min: float = KAPPA_MIN,
    kappa_max: float = KAPPA_MAX,
    warmup_days: int = WARMUP_DAYS,
) -> dict[str, float]:
    """
    Run a single closed-loop strategy and return performance metrics.

    Parameters
    ----------
    risky_returns : np.ndarray
        Daily simple risky returns.
    cash_returns : np.ndarray
        Daily simple cash returns.
    g : float
        Proportional gain.
    theta : float
        Smoothing parameter.
    (other params as in run_strategies)

    Returns
    -------
    dict with keys: tracking_error, kalmar, ann_return, ann_vol, sharpe,
                    max_dd, turnover.
    """
    result = run_strategies(
        risky_returns=risky_returns,
        cash_returns=cash_returns,
        sigma_tar=sigma_tar,
        leverage_cap=leverage_cap,
        halflife=halflife,
        spread=spread,
        g=g,
        theta=theta,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        warmup_days=warmup_days,
    )

    cl_returns_net = result["cl_returns_net"]
    cl_sigma_ind = result["cl_sigma_ind"]
    cl_turnover_abs = result["cl_turnover_abs"]

    te = volatility_tracking_error(cl_sigma_ind, sigma_tar)
    ann_ret = annualized_return(cl_returns_net)
    ann_vol = annualized_volatility(cl_returns_net)
    sr = sharpe_ratio(cl_returns_net, cash_returns)
    kal = kalmar_ratio(cl_returns_net)
    mdd = max_drawdown(cl_returns_net)
    to = total_turnover(cl_turnover_abs)

    return {
        "tracking_error": te,
        "kalmar": kal,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sr,
        "max_dd": mdd,
        "turnover": to,
    }


def run_exp2(results_dir: Path = RESULTS_DIR) -> dict:
    """
    Run Experiment 2: Parameter sensitivity grid search.

    Parameters
    ----------
    results_dir : Path
        Directory to save plots and results.

    Returns
    -------
    dict
        Grid results DataFrame and heatmap figures.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Experiment 2: Parameter Sensitivity Study ===")

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    logger.info("Fetching IVV prices %s to %s", START_DATE, END_DATE)
    prices = fetch_single_etf_prices(TICKER, START_DATE, END_DATE)
    ff_rate = fetch_fed_funds_rate(START_DATE, END_DATE)
    risky_returns, cash_returns = build_aligned_returns(prices, ff_rate)
    logger.info("Data loaded: %d trading days", len(risky_returns))

    # ------------------------------------------------------------------
    # 2. Compute IVV buy-and-hold Kalmar for reference
    # ------------------------------------------------------------------
    ivv_kalmar = kalmar_ratio(risky_returns.values)
    logger.info("IVV Kalmar ratio (in-sample): %.4f", ivv_kalmar)

    # ------------------------------------------------------------------
    # 3. Grid search
    # ------------------------------------------------------------------
    n_theta = len(THETA_GRID)
    n_g = len(G_GRID)

    # Result arrays: shape (n_theta, n_g)
    te_grid = np.full((n_theta, n_g), np.nan)
    kalmar_change_grid = np.full((n_theta, n_g), np.nan)
    turnover_grid = np.full((n_theta, n_g), np.nan)
    ann_ret_grid = np.full((n_theta, n_g), np.nan)
    ann_vol_grid = np.full((n_theta, n_g), np.nan)
    sharpe_grid = np.full((n_theta, n_g), np.nan)
    kalmar_grid = np.full((n_theta, n_g), np.nan)
    maxdd_grid = np.full((n_theta, n_g), np.nan)

    total_runs = n_theta * n_g
    run_count = 0

    for i, theta in enumerate(THETA_GRID):
        for j, g in enumerate(G_GRID):
            run_count += 1
            if run_count % 20 == 0:
                logger.info("Grid search: %d/%d (g=%.4f, theta=%.1f)",
                            run_count, total_runs, g, theta)

            metrics = run_single_closed_loop(
                risky_returns=risky_returns.values,
                cash_returns=cash_returns.values,
                g=g,
                theta=theta,
            )

            te_grid[i, j] = metrics["tracking_error"]
            kalmar_change_grid[i, j] = metrics["kalmar"] - ivv_kalmar
            turnover_grid[i, j] = metrics["turnover"]
            ann_ret_grid[i, j] = metrics["ann_return"]
            ann_vol_grid[i, j] = metrics["ann_vol"]
            sharpe_grid[i, j] = metrics["sharpe"]
            kalmar_grid[i, j] = metrics["kalmar"]
            maxdd_grid[i, j] = metrics["max_dd"]

    logger.info("Grid search complete.")

    # ------------------------------------------------------------------
    # 4. Build tidy results table
    # ------------------------------------------------------------------
    rows = []
    for i, theta in enumerate(THETA_GRID):
        for j, g in enumerate(G_GRID):
            rows.append({
                "g": g,
                "theta": theta,
                "tracking_error": te_grid[i, j],
                "kalmar_change_vs_ivv": kalmar_change_grid[i, j],
                "turnover": turnover_grid[i, j],
                "annualized_return": ann_ret_grid[i, j],
                "annualized_volatility": ann_vol_grid[i, j],
                "sharpe": sharpe_grid[i, j],
                "kalmar": kalmar_grid[i, j],
                "max_drawdown": maxdd_grid[i, j],
            })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_dir / "exp2_grid_results.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Heatmaps
    # ------------------------------------------------------------------
    g_labels = G_GRID.tolist()
    theta_labels = THETA_GRID.tolist()
    highlight = (55.0, 0.6)

    # Tracking error heatmap (lower is better → RdYlGn reversed)
    plot_heatmap(
        data=te_grid * 100,  # convert to percentage
        g_values=g_labels,
        theta_values=theta_labels,
        title="Exp 2: Tracking Error (%) — In-Sample IVV 2000–2009",
        fmt=".2f",
        cmap="RdYlGn_r",
        save_path=str(results_dir / "exp2_heatmap_tracking_error.png"),
        highlight=highlight,
    )

    # Kalmar change heatmap (higher is better → RdYlGn)
    plot_heatmap(
        data=kalmar_change_grid,
        g_values=g_labels,
        theta_values=theta_labels,
        title="Exp 2: ΔKalmar vs IVV — In-Sample IVV 2000–2009",
        fmt=".3f",
        cmap="RdYlGn",
        save_path=str(results_dir / "exp2_heatmap_kalmar_change.png"),
        highlight=highlight,
    )

    # Turnover heatmap (lower is better → RdYlGn reversed)
    plot_heatmap(
        data=turnover_grid,
        g_values=g_labels,
        theta_values=theta_labels,
        title="Exp 2: Turnover — In-Sample IVV 2000–2009",
        fmt=".0f",
        cmap="RdYlGn_r",
        save_path=str(results_dir / "exp2_heatmap_turnover.png"),
        highlight=highlight,
    )

    logger.info("Experiment 2 complete. Results saved to %s", results_dir)

    return {
        "results_df": results_df,
        "te_grid": te_grid,
        "kalmar_change_grid": kalmar_change_grid,
        "turnover_grid": turnover_grid,
        "ivv_kalmar": ivv_kalmar,
        "g_grid": G_GRID,
        "theta_grid": THETA_GRID,
    }
