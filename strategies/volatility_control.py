"""
Volatility control strategies.

Implements three strategies:
1. Buy-and-hold (pass-through of risky returns).
2. Open-loop inverse-volatility targeting.
3. Closed-loop adaptive volatility control (paper's main contribution).

All strategies share:
- sigma_tar = 0.15 / sqrt(252)  (daily target volatility)
- L = 1.5                        (leverage cap)
- h = 126                        (EWMA halflife in trading days)
- beta = exp(-log(2)/h)
- 5 bp bid-ask spread transaction cost

# Custom — Context7 found no library equivalent (paper closed-loop controller)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.ewma import compute_beta, ewma_variance_series

# ---------------------------------------------------------------------------
# Default strategy parameters
# ---------------------------------------------------------------------------
SIGMA_TAR_ANNUAL: float = 0.15
TRADING_DAYS: int = 252
SIGMA_TAR: float = SIGMA_TAR_ANNUAL / np.sqrt(TRADING_DAYS)  # daily target vol
LEVERAGE_CAP: float = 1.5
HALFLIFE: float = 126.0
SPREAD: float = 0.0005  # 5 bp bid-ask spread
WARMUP_DAYS: int = 10   # days before engaging closed-loop feedback

# Closed-loop baseline parameters
G_DEFAULT: float = 55.0
THETA_DEFAULT: float = 0.6
KAPPA_MIN: float = -1.0
KAPPA_MAX: float = 1.0


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategies(
    risky_returns: np.ndarray,
    cash_returns: np.ndarray,
    sigma_tar: float = SIGMA_TAR,
    leverage_cap: float = LEVERAGE_CAP,
    halflife: float = HALFLIFE,
    spread: float = SPREAD,
    g: float = G_DEFAULT,
    theta: float = THETA_DEFAULT,
    kappa_min: float = KAPPA_MIN,
    kappa_max: float = KAPPA_MAX,
    warmup_days: int = WARMUP_DAYS,
) -> dict[str, np.ndarray]:
    """
    Run all three strategies on a single risky asset.

    Timing convention (no look-ahead):
        For day k (0-indexed):
        1. Observe r_k (risky) and r_k_rf (cash).
        2. Compute strategy return using w_{k-1}: r_ind_k = r_k * w_{k-1} + r_rf_k * c_{k-1}.
        3. Update sigma_hat_k from risky returns[0..k].
        4. Update sigma_hat_k_ind from strategy returns[0..k].
        5. Compute w_k (applies to day k+1).

    Transaction cost: cost_k = spread * |w_k - w_{k-1}| deducted from
    portfolio value at rebalance. Implemented as a return adjustment:
        r_net_k = r_ind_k - spread * |w_k - w_{k-1}| (applied at end of period k).
    Note: cost is deducted at the end of period k when the new weight w_k is set.

    Parameters
    ----------
    risky_returns : np.ndarray
        Daily simple returns of the risky asset (length T).
    cash_returns : np.ndarray
        Daily simple cash returns aligned to risky_returns (length T).
    sigma_tar : float
        Daily target volatility.
    leverage_cap : float
        Maximum risky weight (L).
    halflife : float
        EWMA halflife in trading days.
    spread : float
        One-way bid-ask spread for transaction cost.
    g : float
        Proportional gain for closed-loop controller.
    theta : float
        Smoothing parameter for closed-loop controller (0 = no smoothing).
    kappa_min : float
        Lower clip bound for kappa update.
    kappa_max : float
        Upper clip bound for kappa update.
    warmup_days : int
        Number of days to use open-loop weights before engaging feedback.

    Returns
    -------
    dict with keys:
        'bh_returns'        : buy-and-hold daily returns (= risky_returns)
        'ol_returns_gross'  : open-loop gross strategy returns
        'ol_returns_net'    : open-loop net-of-cost strategy returns
        'cl_returns_gross'  : closed-loop gross strategy returns
        'cl_returns_net'    : closed-loop net-of-cost strategy returns
        'ol_weights'        : open-loop risky weights (length T, w_k set at end of k)
        'cl_weights'        : closed-loop risky weights
        'ol_sigma_risky'    : EWMA risky vol used by open-loop (length T)
        'cl_sigma_risky'    : EWMA risky vol used by closed-loop (same)
        'ol_sigma_ind'      : EWMA strategy vol for open-loop (length T)
        'cl_sigma_ind'      : EWMA strategy vol for closed-loop (length T)
        'cl_kappa'          : kappa values for closed-loop (length T)
        'ol_turnover_abs'   : |w_k - w_{k-1}| for open-loop (length T)
        'cl_turnover_abs'   : |w_k - w_{k-1}| for closed-loop (length T)
    """
    T = len(risky_returns)
    assert len(cash_returns) == T, "risky_returns and cash_returns must have same length"

    beta = compute_beta(halflife)

    # -----------------------------------------------------------------------
    # Pre-compute EWMA risky variance series (shared by both dynamic strategies)
    # -----------------------------------------------------------------------
    risky_var = ewma_variance_series(risky_returns, beta)
    risky_sigma = np.sqrt(risky_var)  # daily sigma_hat_k for risky asset

    # -----------------------------------------------------------------------
    # Arrays to fill
    # -----------------------------------------------------------------------
    ol_weights = np.zeros(T, dtype=np.float64)
    cl_weights = np.zeros(T, dtype=np.float64)
    ol_returns_gross = np.zeros(T, dtype=np.float64)
    cl_returns_gross = np.zeros(T, dtype=np.float64)
    ol_sigma_ind = np.zeros(T, dtype=np.float64)
    cl_sigma_ind = np.zeros(T, dtype=np.float64)
    cl_kappa = np.zeros(T, dtype=np.float64)
    ol_turnover_abs = np.zeros(T, dtype=np.float64)
    cl_turnover_abs = np.zeros(T, dtype=np.float64)

    # -----------------------------------------------------------------------
    # EWMA state for strategy returns (separate for OL and CL)
    # -----------------------------------------------------------------------
    ol_raw_var = 0.0   # unnormalized EWMA of r_ind^2 for open-loop
    cl_raw_var = 0.0   # unnormalized EWMA of r_ind^2 for closed-loop
    beta_pow = 1.0     # beta^(k+1) after k+1 observations

    # -----------------------------------------------------------------------
    # Initial weights (before any data): start at 0 (no position)
    # The first feasible weight is set after observing day 0.
    # -----------------------------------------------------------------------
    w_ol_prev = 0.0
    w_cl_prev = 0.0
    kappa_prev = 0.0

    for k in range(T):
        r_k = risky_returns[k]
        r_rf_k = cash_returns[k]

        # Step 1: Compute strategy returns using prior weights
        ol_r_ind = r_k * w_ol_prev + r_rf_k * (1.0 - w_ol_prev)
        cl_r_ind = r_k * w_cl_prev + r_rf_k * (1.0 - w_cl_prev)

        ol_returns_gross[k] = ol_r_ind
        cl_returns_gross[k] = cl_r_ind

        # Step 2: Update EWMA for strategy returns (finite-sample normalized)
        beta_pow_k = beta_pow * beta  # beta^(k+1)
        norm_k = 1.0 - beta_pow_k

        ol_raw_var = beta * ol_raw_var + (1.0 - beta) * ol_r_ind * ol_r_ind
        cl_raw_var = beta * cl_raw_var + (1.0 - beta) * cl_r_ind * cl_r_ind

        ol_var_ind = ol_raw_var / norm_k if norm_k > 1e-15 else ol_raw_var
        cl_var_ind = cl_raw_var / norm_k if norm_k > 1e-15 else cl_raw_var

        ol_sigma_ind[k] = max(np.sqrt(ol_var_ind), 1e-8)
        cl_sigma_ind[k] = max(np.sqrt(cl_var_ind), 1e-8)

        beta_pow = beta_pow_k  # advance for next iteration

        # Step 3: Compute new weights using information through day k
        sigma_k = risky_sigma[k]  # EWMA risky vol through day k

        # --- Open-loop weight ---
        w_ol_new = min(sigma_tar / sigma_k, leverage_cap)
        w_ol_new = max(w_ol_new, 0.0)

        # --- Closed-loop weight ---
        if k < warmup_days:
            # Warm-up: use open-loop weights, kappa stays at 0
            w_cl_new = w_ol_new
            kappa_new = 0.0
        else:
            # Feedback control
            sigma_ind_k = cl_sigma_ind[k]
            e_k = np.log(sigma_ind_k / sigma_tar)
            # Clip first, then smooth
            kappa_raw = np.clip(-g * e_k, kappa_min, kappa_max)
            kappa_new = (1.0 - theta) * kappa_raw + theta * kappa_prev
            w_cl_new = min(np.exp(kappa_new) * sigma_tar / sigma_k, leverage_cap)
            w_cl_new = max(w_cl_new, 0.0)

        # Record weights and turnover
        ol_weights[k] = w_ol_new
        cl_weights[k] = w_cl_new
        cl_kappa[k] = kappa_new
        ol_turnover_abs[k] = abs(w_ol_new - w_ol_prev)
        cl_turnover_abs[k] = abs(w_cl_new - w_cl_prev)

        # Advance state
        w_ol_prev = w_ol_new
        w_cl_prev = w_cl_new
        kappa_prev = kappa_new

    # -----------------------------------------------------------------------
    # Transaction costs: deduct spread * |w_k - w_{k-1}| from gross return
    # at the end of period k (when the new weight w_k is set).
    # -----------------------------------------------------------------------
    ol_returns_net = ol_returns_gross - spread * ol_turnover_abs
    cl_returns_net = cl_returns_gross - spread * cl_turnover_abs

    return {
        "bh_returns": risky_returns.copy(),
        "ol_returns_gross": ol_returns_gross,
        "ol_returns_net": ol_returns_net,
        "cl_returns_gross": cl_returns_gross,
        "cl_returns_net": cl_returns_net,
        "ol_weights": ol_weights,
        "cl_weights": cl_weights,
        "ol_sigma_risky": risky_sigma,
        "cl_sigma_risky": risky_sigma,
        "ol_sigma_ind": ol_sigma_ind,
        "cl_sigma_ind": cl_sigma_ind,
        "cl_kappa": cl_kappa,
        "ol_turnover_abs": ol_turnover_abs,
        "cl_turnover_abs": cl_turnover_abs,
    }


def run_strategies_pandas(
    risky_returns: pd.Series,
    cash_returns: pd.Series,
    sigma_tar: float = SIGMA_TAR,
    leverage_cap: float = LEVERAGE_CAP,
    halflife: float = HALFLIFE,
    spread: float = SPREAD,
    g: float = G_DEFAULT,
    theta: float = THETA_DEFAULT,
    kappa_min: float = KAPPA_MIN,
    kappa_max: float = KAPPA_MAX,
    warmup_days: int = WARMUP_DAYS,
) -> dict[str, pd.Series]:
    """
    Pandas wrapper for run_strategies.

    Parameters
    ----------
    risky_returns : pd.Series
        Daily simple returns with DatetimeIndex.
    cash_returns : pd.Series
        Daily cash returns aligned to risky_returns.
    (other params same as run_strategies)

    Returns
    -------
    dict of pd.Series with same keys as run_strategies, all indexed like risky_returns.
    """
    idx = risky_returns.index
    assert len(risky_returns) == len(cash_returns), "Series must have same length"

    result_np = run_strategies(
        risky_returns=risky_returns.values,
        cash_returns=cash_returns.values,
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

    return {k: pd.Series(v, index=idx, name=k) for k, v in result_np.items()}
