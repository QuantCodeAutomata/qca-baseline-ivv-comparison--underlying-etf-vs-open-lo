"""
Finite-sample normalized EWMA volatility estimator.

Implements the exact formula from the paper:
    sigma_hat_k^2 = ((1 - beta) / (1 - beta^k)) * sum_{j=1}^{k} beta^{k-j} * r_j^2

where beta = exp(-log(2) / h) and h is the halflife in trading days.
No demeaning is applied; simple returns are used directly.

# Custom — Context7 found no library equivalent (paper Eq. for finite-sample EWMA)
"""

import numpy as np
import pandas as pd


def compute_beta(halflife: float) -> float:
    """
    Compute EWMA decay factor from halflife.

    Parameters
    ----------
    halflife : float
        Halflife in trading days.

    Returns
    -------
    float
        Decay factor beta = exp(-log(2) / halflife).
    """
    return float(np.exp(-np.log(2.0) / halflife))


def ewma_variance_series(
    returns: np.ndarray,
    beta: float,
    floor: float = 1e-12,
) -> np.ndarray:
    """
    Compute the finite-sample normalized EWMA variance for each time step k.

    Formula (paper):
        sigma_hat_k^2 = ((1 - beta) / (1 - beta^k)) * sum_{j=1}^{k} beta^{k-j} * r_j^2

    This is computed recursively for efficiency:
        raw_k = beta * raw_{k-1} + (1 - beta) * r_k^2
        sigma_hat_k^2 = raw_k / (1 - beta^k)

    The recursive form is numerically equivalent to the closed-form sum.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of simple returns (length T).
    beta : float
        EWMA decay factor.
    floor : float
        Minimum variance floor to avoid division by zero. Applied only if
        the computed variance is below this threshold.

    Returns
    -------
    np.ndarray
        Array of length T with annualized-ready daily variance estimates.
        sigma_hat_k^2[i] corresponds to the variance estimate after observing
        returns[0..i].
    """
    T = len(returns)
    var_series = np.empty(T, dtype=np.float64)

    raw = 0.0  # unnormalized EWMA of r^2
    beta_pow = 1.0  # beta^k, starts at beta^0 = 1 before first update

    for k in range(T):
        r = returns[k]
        raw = beta * raw + (1.0 - beta) * r * r
        beta_pow = beta_pow * beta  # beta^(k+1)
        normalization = 1.0 - beta_pow  # 1 - beta^(k+1)
        var_k = raw / normalization if normalization > 1e-15 else raw
        var_series[k] = max(var_k, floor)

    return var_series


def ewma_volatility_series(
    returns: np.ndarray,
    beta: float,
    floor: float = 1e-12,
) -> np.ndarray:
    """
    Compute the finite-sample normalized EWMA volatility (std dev) for each step k.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of simple returns.
    beta : float
        EWMA decay factor.
    floor : float
        Minimum variance floor.

    Returns
    -------
    np.ndarray
        Array of daily volatility estimates sigma_hat_k (not annualized).
    """
    var_series = ewma_variance_series(returns, beta, floor)
    return np.sqrt(var_series)


def ewma_variance_series_pandas(
    returns: pd.Series,
    halflife: float,
    floor: float = 1e-12,
) -> pd.Series:
    """
    Pandas wrapper for ewma_variance_series.

    Parameters
    ----------
    returns : pd.Series
        Simple return series with DatetimeIndex.
    halflife : float
        Halflife in trading days.
    floor : float
        Minimum variance floor.

    Returns
    -------
    pd.Series
        Daily variance estimates indexed like returns.
    """
    beta = compute_beta(halflife)
    var_arr = ewma_variance_series(returns.values, beta, floor)
    return pd.Series(var_arr, index=returns.index, name="ewma_variance")


def ewma_volatility_series_pandas(
    returns: pd.Series,
    halflife: float,
    floor: float = 1e-12,
) -> pd.Series:
    """
    Pandas wrapper for ewma_volatility_series.

    Parameters
    ----------
    returns : pd.Series
        Simple return series with DatetimeIndex.
    halflife : float
        Halflife in trading days.
    floor : float
        Minimum variance floor.

    Returns
    -------
    pd.Series
        Daily volatility estimates indexed like returns.
    """
    beta = compute_beta(halflife)
    vol_arr = ewma_volatility_series(returns.values, beta, floor)
    return pd.Series(vol_arr, index=returns.index, name="ewma_volatility")
