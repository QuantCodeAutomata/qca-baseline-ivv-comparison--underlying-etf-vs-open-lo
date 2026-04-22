"""
Experiment 1: Baseline IVV comparison.

Compares three strategies on IVV from 2000-06-08 to 2024-12-31:
1. IVV buy-and-hold
2. Open-loop inverse-volatility targeting
3. Closed-loop adaptive volatility control

Reports: cumulative return plot, running volatility plot, summary metrics table.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch import fetch_single_etf_prices, fetch_fed_funds_rate, build_aligned_returns
from strategies.volatility_control import (
    run_strategies_pandas,
    SIGMA_TAR,
    LEVERAGE_CAP,
    HALFLIFE,
    SPREAD,
    G_DEFAULT,
    THETA_DEFAULT,
    KAPPA_MIN,
    KAPPA_MAX,
    WARMUP_DAYS,
    TRADING_DAYS,
)
from strategies.ewma import ewma_volatility_series_pandas
from utils.metrics import compute_all_metrics, volatility_tracking_error
from utils.plotting import plot_cumulative_returns, plot_running_volatility

logger = logging.getLogger(__name__)

# Experiment parameters
TICKER = "IVV"
START_DATE = "2000-06-08"
END_DATE = "2024-12-31"
RESULTS_DIR = Path("results")


def run_exp1(
    results_dir: Path = RESULTS_DIR,
    use_net_returns: bool = True,
) -> dict:
    """
    Run Experiment 1: Baseline IVV comparison.

    Parameters
    ----------
    results_dir : Path
        Directory to save plots and results.
    use_net_returns : bool
        If True, use net-of-cost returns for performance metrics.

    Returns
    -------
    dict
        Summary metrics for all three strategies.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Experiment 1: Baseline IVV Comparison ===")

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    logger.info("Fetching IVV prices %s to %s", START_DATE, END_DATE)
    prices = fetch_single_etf_prices(TICKER, START_DATE, END_DATE)

    logger.info("Fetching Federal Funds rate")
    ff_rate = fetch_fed_funds_rate(START_DATE, END_DATE)

    risky_returns, cash_returns = build_aligned_returns(prices, ff_rate)
    logger.info("Data loaded: %d trading days", len(risky_returns))

    # ------------------------------------------------------------------
    # 2. Run strategies
    # ------------------------------------------------------------------
    result = run_strategies_pandas(
        risky_returns=risky_returns,
        cash_returns=cash_returns,
        sigma_tar=SIGMA_TAR,
        leverage_cap=LEVERAGE_CAP,
        halflife=HALFLIFE,
        spread=SPREAD,
        g=G_DEFAULT,
        theta=THETA_DEFAULT,
        kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX,
        warmup_days=WARMUP_DAYS,
    )

    # Choose net or gross returns for reporting
    suffix = "net" if use_net_returns else "gross"
    ol_returns = result[f"ol_returns_{suffix}"]
    cl_returns = result[f"cl_returns_{suffix}"]
    bh_returns = result["bh_returns"]

    # ------------------------------------------------------------------
    # 3. Compute EWMA volatility for buy-and-hold (for tracking error)
    # ------------------------------------------------------------------
    bh_sigma = ewma_volatility_series_pandas(bh_returns, HALFLIFE)

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    rf_arr = cash_returns.values

    # Buy-and-hold: turnover = 0, sigma_hat = risky EWMA
    bh_turnover = np.zeros(len(bh_returns))
    bh_metrics = compute_all_metrics(
        returns=bh_returns.values,
        rf_returns=rf_arr,
        sigma_hat_series=bh_sigma.values,
        sigma_tar=SIGMA_TAR,
        turnover_abs=bh_turnover,
    )

    # Open-loop
    ol_metrics = compute_all_metrics(
        returns=ol_returns.values,
        rf_returns=rf_arr,
        sigma_hat_series=result["ol_sigma_ind"].values,
        sigma_tar=SIGMA_TAR,
        turnover_abs=result["ol_turnover_abs"].values,
    )

    # Closed-loop
    cl_metrics = compute_all_metrics(
        returns=cl_returns.values,
        rf_returns=rf_arr,
        sigma_hat_series=result["cl_sigma_ind"].values,
        sigma_tar=SIGMA_TAR,
        turnover_abs=result["cl_turnover_abs"].values,
    )

    # ------------------------------------------------------------------
    # 5. Build summary table
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(
        {
            "IVV Buy-and-Hold": bh_metrics,
            "Open-Loop": ol_metrics,
            "Closed-Loop": cl_metrics,
        }
    ).T

    # Format for display
    display_df = metrics_df.copy()
    display_df["ann_return"] = display_df["ann_return"].map("{:.1%}".format)
    display_df["ann_vol"] = display_df["ann_vol"].map("{:.1%}".format)
    display_df["sharpe"] = display_df["sharpe"].map("{:.2f}".format)
    display_df["kalmar"] = display_df["kalmar"].map("{:.2f}".format)
    display_df["max_dd"] = display_df["max_dd"].map("{:.1%}".format)
    display_df["tracking_error"] = display_df["tracking_error"].map("{:.2%}".format)
    display_df["turnover"] = display_df["turnover"].map("{:.0%}".format)

    logger.info("\nSummary Table:\n%s", display_df.to_string())

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    dates = risky_returns.index

    # Cumulative returns
    plot_cumulative_returns(
        returns_dict={
            "IVV Buy-and-Hold": bh_returns.values,
            "Open-Loop": ol_returns.values,
            "Closed-Loop": cl_returns.values,
        },
        dates=dates,
        title="Exp 1: Cumulative Returns — IVV vs Open-Loop vs Closed-Loop (2000–2024)",
        save_path=str(results_dir / "exp1_cumulative_returns.png"),
    )

    # Running volatility
    plot_running_volatility(
        sigma_dict={
            "IVV Buy-and-Hold": bh_sigma.values,
            "Open-Loop": result["ol_sigma_ind"].values,
            "Closed-Loop": result["cl_sigma_ind"].values,
        },
        dates=dates,
        sigma_tar_annual=0.15,
        title="Exp 1: Running Annualized EWMA Volatility (2000–2024)",
        save_path=str(results_dir / "exp1_running_volatility.png"),
    )

    # Save metrics CSV
    metrics_df.to_csv(results_dir / "exp1_metrics.csv")

    logger.info("Experiment 1 complete. Results saved to %s", results_dir)

    return {
        "metrics": metrics_df,
        "display": display_df,
        "result": result,
        "risky_returns": risky_returns,
        "cash_returns": cash_returns,
        "bh_sigma": bh_sigma,
    }
