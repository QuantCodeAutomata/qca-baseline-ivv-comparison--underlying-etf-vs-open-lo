"""
Experiment 4: Uncertainty quantification for EWMA volatility estimates.

Three parts:
A. Monte Carlo confidence bands for running EWMA volatility estimates.
B. Chi-distribution approximation validation after burn-in (h=21, h=126).
C. Std dev of EWMA estimator vs halflife.

# Custom — Context7 found no library equivalent (paper Appendix B chi-approximation)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from strategies.ewma import compute_beta, ewma_variance_series
from utils.plotting import (
    plot_chi_approximation,
    plot_std_vs_halflife,
    plot_running_volatility,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
TRADING_DAYS = 252

# Monte Carlo parameters
N_PATHS = 10000       # number of simulation paths
BURN_IN = 252         # burn-in observations for steady-state analysis
SIGMA_TRUE = 0.15 / np.sqrt(TRADING_DAYS)  # daily true volatility
RANDOM_SEED = 42


def simulate_ewma_paths(
    n_paths: int,
    T: int,
    beta: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate N_PATHS independent return paths and compute EWMA volatility at each step.

    Uses vectorized computation across all paths simultaneously for efficiency.

    Parameters
    ----------
    n_paths : int
        Number of independent simulation paths.
    T : int
        Length of each path (trading days).
    beta : float
        EWMA decay factor.
    sigma : float
        True daily volatility.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Shape (n_paths, T) array of annualized EWMA volatility estimates.
    """
    # Generate all returns at once: shape (n_paths, T)
    returns_sq = (sigma * rng.standard_normal((n_paths, T))) ** 2  # r^2

    # Vectorized EWMA: process all paths simultaneously
    # raw[k] = beta * raw[k-1] + (1-beta) * r[k]^2
    raw = np.zeros(n_paths, dtype=np.float64)
    var_matrix = np.empty((n_paths, T), dtype=np.float64)
    beta_pow = 1.0  # beta^(k+1) after k+1 observations

    for k in range(T):
        raw = beta * raw + (1.0 - beta) * returns_sq[:, k]
        beta_pow_k = beta_pow * beta
        norm_k = 1.0 - beta_pow_k
        var_matrix[:, k] = raw / norm_k if norm_k > 1e-15 else raw
        beta_pow = beta_pow_k

    sigma_hat_matrix = np.sqrt(np.maximum(var_matrix, 1e-24))
    return sigma_hat_matrix * np.sqrt(TRADING_DAYS)  # annualize


def compute_confidence_bands(
    sigma_hat_matrix: np.ndarray,
    lower_pct: float = 10.0,
    upper_pct: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pointwise percentile bands across simulation paths.

    Parameters
    ----------
    sigma_hat_matrix : np.ndarray
        Shape (n_paths, T) annualized EWMA volatility estimates.
    lower_pct : float
        Lower percentile (e.g., 10 for 10th percentile).
    upper_pct : float
        Upper percentile (e.g., 90 for 90th percentile).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lower_band, upper_band) each of length T.
    """
    lower = np.percentile(sigma_hat_matrix, lower_pct, axis=0)
    upper = np.percentile(sigma_hat_matrix, upper_pct, axis=0)
    return lower, upper


def chi_approximation_pdf(
    x: np.ndarray,
    nu: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute the theoretical PDF for the chi-distribution approximation of
    the normalized EWMA volatility estimator.

    The paper claims sigma_hat / sigma ~ (1/sqrt(nu)) * chi(nu), where
    nu = (1 + beta) / (1 - beta) is the effective degrees of freedom.

    For a chi(nu) distribution with scale factor 1/sqrt(nu):
        If Y ~ chi(nu), then X = Y / sqrt(nu) has:
        E[X] = sqrt(2) * Gamma((nu+1)/2) / (sqrt(nu) * Gamma(nu/2))
        Var[X] = 1 - E[X]^2

    The PDF of X = Y/sqrt(nu) where Y ~ chi(nu) is:
        f_X(x) = f_chi(x * sqrt(nu)) * sqrt(nu)

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the PDF (normalized: sigma_hat / sigma).
    nu : float
        Effective degrees of freedom.
    sigma : float
        Scale (1.0 for normalized comparison).

    Returns
    -------
    np.ndarray
        PDF values at x.
    """
    # Scale: sigma_hat/sigma ~ chi(nu) / sqrt(nu)
    # So sigma_hat/sigma * sqrt(nu) ~ chi(nu)
    scale = 1.0 / np.sqrt(nu)
    # chi distribution: scipy.stats.chi(df=nu, scale=scale)
    return stats.chi.pdf(x, df=nu, scale=scale)


def run_part_a(
    halflife: float = 126.0,
    T: int = 2520,  # ~10 years of trading days
    n_paths: int = N_PATHS,
    rng: np.random.Generator | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Part A: Monte Carlo confidence bands for running EWMA volatility.

    Parameters
    ----------
    halflife : float
        EWMA halflife in trading days.
    T : int
        Length of simulation paths.
    n_paths : int
        Number of simulation paths.
    rng : np.random.Generator or None
        Random number generator (created if None).
    results_dir : Path
        Output directory.

    Returns
    -------
    dict with keys: lower_band, upper_band, sigma_hat_matrix.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    beta = compute_beta(halflife)
    logger.info("Part A: Simulating %d paths of length %d (h=%d)", n_paths, T, int(halflife))

    sigma_hat_matrix = simulate_ewma_paths(n_paths, T, beta, SIGMA_TRUE, rng)
    lower_band, upper_band = compute_confidence_bands(sigma_hat_matrix, 10.0, 90.0)

    # Plot confidence bands
    dates = pd.date_range("2000-01-01", periods=T, freq="B")[:T]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dates, lower_band, upper_band, alpha=0.3, color="#1f77b4",
                    label="10th–90th percentile band")
    ax.axhline(0.15, color="black", linestyle="--", linewidth=1.5, label="Target 15%")
    ax.set_title(f"Exp 4A: Monte Carlo EWMA Volatility Confidence Bands (h={int(halflife)})",
                 fontsize=13)
    ax.set_xlabel("Simulated Date")
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(str(results_dir / "exp4a_confidence_bands.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Part A complete.")
    return {
        "lower_band": lower_band,
        "upper_band": upper_band,
        "sigma_hat_matrix": sigma_hat_matrix,
    }


def run_part_b(
    halflives: list[int] = [21, 126],
    n_paths: int = N_PATHS,
    burn_in: int = BURN_IN,
    rng: np.random.Generator | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Part B: Chi-distribution approximation validation after burn-in.

    For each halflife, generates N_PATHS independent paths of length burn_in + 1,
    extracts the steady-state EWMA volatility estimate (after burn-in), and
    compares the empirical distribution to the chi approximation.

    Parameters
    ----------
    halflives : list of int
        Halflives to analyze.
    n_paths : int
        Number of Monte Carlo paths.
    burn_in : int
        Number of burn-in observations.
    rng : np.random.Generator or None
        Random number generator.
    results_dir : Path
        Output directory.

    Returns
    -------
    dict with keys: results per halflife.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED + 1)

    results = {}

    for h in halflives:
        beta = compute_beta(float(h))
        nu = (1.0 + beta) / (1.0 - beta)  # effective degrees of freedom
        logger.info("Part B: h=%d, beta=%.6f, nu=%.2f", h, beta, nu)

        # Generate paths of length burn_in + 1 to get steady-state estimate
        T_total = burn_in + 1
        returns = SIGMA_TRUE * rng.standard_normal((n_paths, T_total))

        # Compute EWMA volatility at the last step (after burn-in)
        sigma_hat_final = np.empty(n_paths, dtype=np.float64)
        for path_idx in range(n_paths):
            var_arr = ewma_variance_series(returns[path_idx], beta)
            sigma_hat_final[path_idx] = np.sqrt(var_arr[-1])

        # Normalize by true volatility
        normalized = sigma_hat_final / SIGMA_TRUE

        # Theoretical chi approximation: sigma_hat/sigma ~ chi(nu)/sqrt(nu)
        x_grid = np.linspace(0.01, 3.0, 500)
        theoretical_pdf = chi_approximation_pdf(x_grid, nu)

        # Plot
        plot_chi_approximation(
            empirical_samples=normalized,
            theoretical_x=x_grid,
            theoretical_pdf=theoretical_pdf,
            halflife=h,
            nu=nu,
            save_path=str(results_dir / f"exp4b_chi_approx_h{h}.png"),
        )

        # Compute fit statistics
        empirical_mean = np.mean(normalized)
        empirical_std = np.std(normalized)
        theoretical_mean = stats.chi.mean(df=nu, scale=1.0 / np.sqrt(nu))
        theoretical_std = stats.chi.std(df=nu, scale=1.0 / np.sqrt(nu))

        logger.info(
            "h=%d: empirical mean=%.4f, std=%.4f | theoretical mean=%.4f, std=%.4f",
            h, empirical_mean, empirical_std, theoretical_mean, theoretical_std
        )

        results[h] = {
            "beta": beta,
            "nu": nu,
            "normalized_samples": normalized,
            "empirical_mean": empirical_mean,
            "empirical_std": empirical_std,
            "theoretical_mean": theoretical_mean,
            "theoretical_std": theoretical_std,
        }

    logger.info("Part B complete.")
    return results


def run_part_c(
    halflife_grid: np.ndarray | None = None,
    n_paths: int = 5000,
    burn_in: int = BURN_IN,
    rng: np.random.Generator | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Part C: Std dev of EWMA estimator vs halflife.

    Computes both Monte Carlo and theoretical (chi-approximation) std dev
    of sigma_hat/sigma as a function of halflife.

    Parameters
    ----------
    halflife_grid : np.ndarray or None
        Array of halflives to evaluate. Defaults to [5, 10, 21, 42, 63, 126, 252].
    n_paths : int
        Number of Monte Carlo paths per halflife.
    burn_in : int
        Burn-in observations.
    rng : np.random.Generator or None
        Random number generator.
    results_dir : Path
        Output directory.

    Returns
    -------
    dict with keys: halflives, std_mc, std_theory.
    """
    if halflife_grid is None:
        halflife_grid = np.array([5, 10, 21, 42, 63, 126, 252], dtype=float)

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED + 2)

    std_mc = np.empty(len(halflife_grid))
    std_theory = np.empty(len(halflife_grid))

    for idx, h in enumerate(halflife_grid):
        beta = compute_beta(float(h))
        nu = (1.0 + beta) / (1.0 - beta)

        # Monte Carlo
        T_total = burn_in + 1
        returns = SIGMA_TRUE * rng.standard_normal((n_paths, T_total))
        sigma_hat_final = np.empty(n_paths)
        for path_idx in range(n_paths):
            var_arr = ewma_variance_series(returns[path_idx], beta)
            sigma_hat_final[path_idx] = np.sqrt(var_arr[-1])

        normalized = sigma_hat_final / SIGMA_TRUE
        std_mc[idx] = np.std(normalized)

        # Theoretical: std of chi(nu)/sqrt(nu)
        std_theory[idx] = stats.chi.std(df=nu, scale=1.0 / np.sqrt(nu))

        logger.info("h=%.0f: MC std=%.4f, theory std=%.4f", h, std_mc[idx], std_theory[idx])

    # Plot
    plot_std_vs_halflife(
        halflives=halflife_grid,
        std_mc=std_mc,
        std_theory=std_theory,
        save_path=str(results_dir / "exp4c_std_vs_halflife.png"),
    )

    # Save results
    df = pd.DataFrame({
        "halflife": halflife_grid,
        "std_mc": std_mc,
        "std_theory": std_theory,
    })
    df.to_csv(results_dir / "exp4c_std_vs_halflife.csv", index=False)

    logger.info("Part C complete.")
    return {
        "halflives": halflife_grid,
        "std_mc": std_mc,
        "std_theory": std_theory,
    }


def run_exp4(results_dir: Path = RESULTS_DIR) -> dict:
    """
    Run Experiment 4: Uncertainty quantification for EWMA volatility estimates.

    Parameters
    ----------
    results_dir : Path
        Directory to save plots and results.

    Returns
    -------
    dict
        Results from Parts A, B, and C.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Experiment 4: Uncertainty Quantification ===")

    rng = np.random.default_rng(RANDOM_SEED)

    part_a = run_part_a(halflife=126.0, T=2520, n_paths=N_PATHS, rng=rng,
                        results_dir=results_dir)
    part_b = run_part_b(halflives=[21, 126], n_paths=N_PATHS, burn_in=BURN_IN,
                        rng=rng, results_dir=results_dir)
    part_c = run_part_c(rng=rng, results_dir=results_dir)

    # Save Part B summary
    b_rows = []
    for h, res in part_b.items():
        b_rows.append({
            "halflife": h,
            "beta": res["beta"],
            "nu": res["nu"],
            "empirical_mean": res["empirical_mean"],
            "empirical_std": res["empirical_std"],
            "theoretical_mean": res["theoretical_mean"],
            "theoretical_std": res["theoretical_std"],
        })
    pd.DataFrame(b_rows).to_csv(results_dir / "exp4b_chi_approx_summary.csv", index=False)

    logger.info("Experiment 4 complete. Results saved to %s", results_dir)

    return {"part_a": part_a, "part_b": part_b, "part_c": part_c}
