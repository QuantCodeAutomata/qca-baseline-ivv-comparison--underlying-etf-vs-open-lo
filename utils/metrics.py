"""
Performance metrics for strategy evaluation.

Implements paper-specific metrics:
- Volatility tracking error
- Annualized return, volatility, Sharpe ratio
- Kalmar ratio (annualized return / max drawdown)
- Maximum drawdown
- Turnover

# Using quantstats for max_drawdown — Context7 confirmed
# Custom implementations for paper-specific metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def annualized_return(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute annualized return from daily simple returns.

    Uses compound growth: (prod(1 + r_k))^(252/T) - 1.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns.
    periods_per_year : int
        Number of trading periods per year.

    Returns
    -------
    float
        Annualized return.
    """
    T = len(returns)
    if T == 0:
        return np.nan
    total_return = np.prod(1.0 + returns)
    return float(total_return ** (periods_per_year / T) - 1.0)


def annualized_volatility(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute annualized volatility (std dev of daily returns * sqrt(252)).

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns.
    periods_per_year : int
        Number of trading periods per year.

    Returns
    -------
    float
        Annualized volatility.
    """
    if len(returns) < 2:
        return np.nan
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    rf_returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Sharpe = annualized_excess_return / annualized_volatility_of_excess_returns.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple strategy returns.
    rf_returns : np.ndarray
        Daily simple risk-free returns (aligned).
    periods_per_year : int
        Number of trading periods per year.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    excess = returns - rf_returns
    ann_excess = annualized_return(excess, periods_per_year)
    ann_vol = annualized_volatility(excess, periods_per_year)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float(ann_excess / ann_vol)


def max_drawdown(returns: np.ndarray) -> float:
    """
    Compute maximum drawdown from daily simple returns.

    MDD = max over time of (peak - trough) / peak, expressed as a positive fraction.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns.

    Returns
    -------
    float
        Maximum drawdown (positive value, e.g., 0.55 for 55%).
    """
    if len(returns) == 0:
        return np.nan
    wealth = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (running_max - wealth) / running_max
    return float(np.max(drawdowns))


def kalmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Compute Kalmar ratio = annualized_return / max_drawdown.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple returns.
    periods_per_year : int
        Number of trading periods per year.

    Returns
    -------
    float
        Kalmar ratio.
    """
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / mdd)


def volatility_tracking_error(
    sigma_hat_series: np.ndarray,
    sigma_tar: float,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Compute the paper's volatility tracking error metric.

    TrackingError = sqrt(252) * mean_k |sigma_hat_k - sigma_tar|

    where sigma_hat_k is the running daily EWMA volatility estimate and
    sigma_tar is the daily target volatility.

    Parameters
    ----------
    sigma_hat_series : np.ndarray
        Running daily EWMA volatility estimates (not annualized).
    sigma_tar : float
        Daily target volatility.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
        Annualized tracking error (e.g., 0.004 for 0.4%).
    """
    if len(sigma_hat_series) == 0:
        return np.nan
    mae = np.mean(np.abs(sigma_hat_series - sigma_tar))
    return float(np.sqrt(periods_per_year) * mae)


def total_turnover(weight_changes_abs: np.ndarray) -> float:
    """
    Compute total turnover as sum of absolute weight changes.

    Parameters
    ----------
    weight_changes_abs : np.ndarray
        Array of |w_k - w_{k-1}| for each period.

    Returns
    -------
    float
        Total turnover (e.g., 9.3 for 930%).
    """
    return float(np.sum(weight_changes_abs))


def compute_all_metrics(
    returns: np.ndarray,
    rf_returns: np.ndarray,
    sigma_hat_series: np.ndarray,
    sigma_tar: float,
    turnover_abs: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """
    Compute all paper-reported metrics for a strategy.

    Parameters
    ----------
    returns : np.ndarray
        Daily simple strategy returns.
    rf_returns : np.ndarray
        Daily simple risk-free returns.
    sigma_hat_series : np.ndarray
        Running daily EWMA volatility estimates for the strategy.
    sigma_tar : float
        Daily target volatility.
    turnover_abs : np.ndarray
        Absolute weight changes |w_k - w_{k-1}|.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    dict with keys: ann_return, ann_vol, sharpe, kalmar, max_dd,
                    tracking_error, turnover.
    """
    return {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, rf_returns, periods_per_year),
        "kalmar": kalmar_ratio(returns, periods_per_year),
        "max_dd": max_drawdown(returns),
        "tracking_error": volatility_tracking_error(sigma_hat_series, sigma_tar, periods_per_year),
        "turnover": total_turnover(turnover_abs),
    }
