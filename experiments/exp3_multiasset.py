"""
Experiment 3: Out-of-sample multi-asset ETF evaluation.

Applies fixed closed-loop controller (g=55, theta=0.6) to 44 ETFs
over 2010-01-01 to 2024-12-31.

Reports: cross-sectional histograms for tracking error, Kalmar change, turnover.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch import fetch_etf_prices, fetch_fed_funds_rate, build_aligned_returns
from strategies.volatility_control import (
    run_strategies,
    SIGMA_TAR,
    LEVERAGE_CAP,
    HALFLIFE,
    SPREAD,
    G_DEFAULT,
    THETA_DEFAULT,
    KAPPA_MIN,
    KAPPA_MAX,
    WARMUP_DAYS,
)
from strategies.ewma import ewma_volatility_series
from utils.metrics import (
    compute_all_metrics,
    kalmar_ratio,
    annualized_return,
    max_drawdown,
    annualized_volatility,
    sharpe_ratio,
    volatility_tracking_error,
    total_turnover,
)
from utils.plotting import plot_cross_sectional_histograms

logger = logging.getLogger(__name__)

# Experiment parameters
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
RESULTS_DIR = Path("results")

# 44-ETF universe (exact as specified in paper)
ETF_UNIVERSE = [
    "ACWI", "EFA", "EWD", "EWJ", "EWK", "EWL", "EWN", "EWT",
    "IAI", "IAK", "IAT", "IAU", "IBB", "IDU", "IEO", "IGM",
    "IGV", "IHF", "IHI", "IJH", "IJR", "IOO", "ITA", "ITB",
    "IVV", "IWB", "IWD", "IWF", "IWM", "IWN", "IWO", "IWV",
    "IXN", "IYC", "IYF", "IYH", "IYJ", "IYK", "IYM", "IYR",
    "IYW", "OEF", "SOXX", "THD",
]


def run_single_etf(
    risky_returns: np.ndarray,
    cash_returns: np.ndarray,
    ticker: str,
) -> dict[str, dict[str, float]]:
    """
    Run open-loop and closed-loop strategies for a single ETF.

    Parameters
    ----------
    risky_returns : np.ndarray
        Daily simple returns for the ETF.
    cash_returns : np.ndarray
        Daily simple cash returns.
    ticker : str
        ETF ticker (for logging).

    Returns
    -------
    dict with keys 'bh', 'ol', 'cl', each containing metric dicts.
    """
    from strategies.ewma import compute_beta
    beta = compute_beta(HALFLIFE)

    result = run_strategies(
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

    # Buy-and-hold sigma
    bh_sigma = ewma_volatility_series(risky_returns, beta)
    bh_turnover = np.zeros(len(risky_returns))

    bh_metrics = compute_all_metrics(
        returns=risky_returns,
        rf_returns=cash_returns,
        sigma_hat_series=bh_sigma,
        sigma_tar=SIGMA_TAR,
        turnover_abs=bh_turnover,
    )

    ol_metrics = compute_all_metrics(
        returns=result["ol_returns_net"],
        rf_returns=cash_returns,
        sigma_hat_series=result["ol_sigma_ind"],
        sigma_tar=SIGMA_TAR,
        turnover_abs=result["ol_turnover_abs"],
    )

    cl_metrics = compute_all_metrics(
        returns=result["cl_returns_net"],
        rf_returns=cash_returns,
        sigma_hat_series=result["cl_sigma_ind"],
        sigma_tar=SIGMA_TAR,
        turnover_abs=result["cl_turnover_abs"],
    )

    return {"bh": bh_metrics, "ol": ol_metrics, "cl": cl_metrics}


def run_exp3(results_dir: Path = RESULTS_DIR) -> dict:
    """
    Run Experiment 3: Out-of-sample multi-asset ETF evaluation.

    Parameters
    ----------
    results_dir : Path
        Directory to save plots and results.

    Returns
    -------
    dict
        Cross-sectional results DataFrame and summary statistics.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Experiment 3: Multi-Asset ETF Evaluation ===")

    # ------------------------------------------------------------------
    # 1. Fetch all ETF prices
    # ------------------------------------------------------------------
    logger.info("Fetching prices for %d ETFs", len(ETF_UNIVERSE))
    prices_df = fetch_etf_prices(ETF_UNIVERSE, START_DATE, END_DATE)
    ff_rate = fetch_fed_funds_rate(START_DATE, END_DATE)

    logger.info("Price data shape: %s", prices_df.shape)

    # ------------------------------------------------------------------
    # 2. Process each ETF
    # ------------------------------------------------------------------
    all_rows = []
    valid_tickers = []

    for ticker in ETF_UNIVERSE:
        if ticker not in prices_df.columns:
            logger.warning("Ticker %s not in price data, skipping.", ticker)
            continue

        prices = prices_df[ticker].dropna()
        if len(prices) < 252:
            logger.warning("Ticker %s has insufficient data (%d days), skipping.",
                           ticker, len(prices))
            continue

        # Check for non-positive prices
        if (prices <= 0).any():
            logger.warning("Ticker %s has non-positive prices, skipping.", ticker)
            continue

        try:
            risky_returns, cash_returns = build_aligned_returns(prices, ff_rate)
        except Exception as exc:
            logger.warning("Failed to build returns for %s: %s", ticker, exc)
            continue

        if len(risky_returns) < 252:
            logger.warning("Ticker %s has insufficient return data, skipping.", ticker)
            continue

        try:
            metrics = run_single_etf(
                risky_returns=risky_returns.values,
                cash_returns=cash_returns.values,
                ticker=ticker,
            )
        except Exception as exc:
            logger.warning("Strategy failed for %s: %s", ticker, exc)
            continue

        row = {"ticker": ticker}
        for strategy in ["bh", "ol", "cl"]:
            for metric_name, val in metrics[strategy].items():
                row[f"{strategy}_{metric_name}"] = val

        # Kalmar change vs underlying
        row["ol_kalmar_change"] = metrics["ol"]["kalmar"] - metrics["bh"]["kalmar"]
        row["cl_kalmar_change"] = metrics["cl"]["kalmar"] - metrics["bh"]["kalmar"]

        all_rows.append(row)
        valid_tickers.append(ticker)
        logger.info("Processed %s: CL tracking error=%.3f%%, OL tracking error=%.3f%%",
                    ticker,
                    metrics["cl"]["tracking_error"] * 100,
                    metrics["ol"]["tracking_error"] * 100)

    logger.info("Successfully processed %d/%d ETFs", len(valid_tickers), len(ETF_UNIVERSE))

    if not all_rows:
        raise RuntimeError("No ETFs were successfully processed.")

    results_df = pd.DataFrame(all_rows).set_index("ticker")
    results_df.to_csv(results_dir / "exp3_etf_results.csv")

    # ------------------------------------------------------------------
    # 3. Cross-sectional summary statistics
    # ------------------------------------------------------------------
    summary_stats = {}
    for strategy in ["ol", "cl"]:
        for metric in ["tracking_error", "kalmar_change", "turnover", "ann_return",
                       "ann_vol", "sharpe", "kalmar", "max_dd"]:
            col = f"{strategy}_{metric}"
            if col in results_df.columns:
                vals = results_df[col].dropna()
                summary_stats[col] = {
                    "mean": vals.mean(),
                    "median": vals.median(),
                    "min": vals.min(),
                    "max": vals.max(),
                    "q25": vals.quantile(0.25),
                    "q75": vals.quantile(0.75),
                }

    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv(results_dir / "exp3_summary_stats.csv")
    logger.info("\nCross-sectional summary:\n%s", summary_df.to_string())

    # ------------------------------------------------------------------
    # 4. Cross-sectional histograms
    # ------------------------------------------------------------------
    ol_te = results_df["ol_tracking_error"].dropna().values * 100  # percent
    cl_te = results_df["cl_tracking_error"].dropna().values * 100

    plot_cross_sectional_histograms(
        ol_values=ol_te,
        cl_values=cl_te,
        xlabel="Tracking Error (%)",
        title="Exp 3: Cross-Sectional Tracking Error — 44 ETFs (2010–2024)",
        save_path=str(results_dir / "exp3_hist_tracking_error.png"),
    )

    ol_kc = results_df["ol_kalmar_change"].dropna().values
    cl_kc = results_df["cl_kalmar_change"].dropna().values

    plot_cross_sectional_histograms(
        ol_values=ol_kc,
        cl_values=cl_kc,
        xlabel="ΔKalmar vs Underlying ETF",
        title="Exp 3: Cross-Sectional ΔKalmar — 44 ETFs (2010–2024)",
        save_path=str(results_dir / "exp3_hist_kalmar_change.png"),
    )

    ol_to = results_df["ol_turnover"].dropna().values
    cl_to = results_df["cl_turnover"].dropna().values

    plot_cross_sectional_histograms(
        ol_values=ol_to,
        cl_values=cl_to,
        xlabel="Total Turnover",
        title="Exp 3: Cross-Sectional Turnover — 44 ETFs (2010–2024)",
        save_path=str(results_dir / "exp3_hist_turnover.png"),
    )

    logger.info("Experiment 3 complete. Results saved to %s", results_dir)

    return {
        "results_df": results_df,
        "summary_df": summary_df,
        "valid_tickers": valid_tickers,
    }
