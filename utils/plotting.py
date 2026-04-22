"""
Plotting utilities for volatility control experiments.

Uses matplotlib and seaborn for all visualizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Style
sns.set_theme(style="whitegrid", palette="colorblind")
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (8, 6)
FIGSIZE_TALL = (10, 8)


def plot_cumulative_returns(
    returns_dict: dict[str, np.ndarray | pd.Series],
    dates: pd.DatetimeIndex,
    title: str = "Cumulative Returns",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot cumulative wealth paths for multiple strategies.

    Parameters
    ----------
    returns_dict : dict
        Mapping of strategy name -> daily simple returns array/series.
    dates : pd.DatetimeIndex
        Date index for x-axis.
    title : str
        Plot title.
    save_path : str or None
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, rets) in enumerate(returns_dict.items()):
        r = np.asarray(rets)
        wealth = np.cumprod(1.0 + r)
        ax.plot(dates, wealth, label=name, color=colors[i % len(colors)], linewidth=1.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Wealth (starting at 1)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_running_volatility(
    sigma_dict: dict[str, np.ndarray | pd.Series],
    dates: pd.DatetimeIndex,
    sigma_tar_annual: float = 0.15,
    title: str = "Running Annualized EWMA Volatility",
    percentile_bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot running annualized EWMA volatility for multiple strategies.

    Parameters
    ----------
    sigma_dict : dict
        Mapping of strategy name -> daily EWMA volatility array.
    dates : pd.DatetimeIndex
        Date index.
    sigma_tar_annual : float
        Annualized target volatility (horizontal reference line).
    title : str
        Plot title.
    percentile_bands : dict or None
        Optional mapping of band_name -> (lower_band, upper_band) in annualized units.
    save_path : str or None
        Save path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, sigma) in enumerate(sigma_dict.items()):
        s = np.asarray(sigma) * np.sqrt(252)  # annualize
        ax.plot(dates, s, label=name, color=colors[i % len(colors)], linewidth=1.2)

    # Target line
    ax.axhline(sigma_tar_annual, color="black", linestyle="--", linewidth=1.5,
               label=f"Target {sigma_tar_annual:.0%}")

    # Optional percentile bands
    if percentile_bands:
        band_colors = ["#aec7e8", "#ffbb78"]
        for j, (band_name, (lo, hi)) in enumerate(percentile_bands.items()):
            ax.fill_between(dates, lo, hi, alpha=0.2,
                            color=band_colors[j % len(band_colors)],
                            label=band_name)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_heatmap(
    data: np.ndarray,
    g_values: list[float],
    theta_values: list[float],
    title: str,
    fmt: str = ".3f",
    cmap: str = "RdYlGn",
    save_path: str | None = None,
    highlight: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot a heatmap over the (g, theta) parameter grid.

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape (len(theta_values), len(g_values)).
    g_values : list of float
        Gain values (x-axis).
    theta_values : list of float
        Smoothing values (y-axis).
    title : str
        Plot title.
    fmt : str
        Number format for annotations.
    cmap : str
        Colormap.
    save_path : str or None
        Save path.
    highlight : tuple (g, theta) or None
        Parameter pair to highlight with a rectangle.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    g_labels = [f"{g:.1f}" for g in g_values]
    theta_labels = [f"{t:.1f}" for t in theta_values]

    df_heat = pd.DataFrame(data, index=theta_labels, columns=g_labels)
    sns.heatmap(
        df_heat,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    if highlight is not None:
        g_h, theta_h = highlight
        # Find closest indices
        g_idx = int(np.argmin(np.abs(np.array(g_values) - g_h)))
        theta_idx = int(np.argmin(np.abs(np.array(theta_values) - theta_h)))
        ax.add_patch(plt.Rectangle(
            (g_idx, theta_idx), 1, 1,
            fill=False, edgecolor="red", lw=3, label=f"Baseline (g={g_h}, θ={theta_h})"
        ))

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Gain g")
    ax.set_ylabel("Smoothing θ")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cross_sectional_histograms(
    ol_values: np.ndarray,
    cl_values: np.ndarray,
    xlabel: str,
    title: str,
    save_path: str | None = None,
    bins: int = 20,
) -> plt.Figure:
    """
    Plot overlapping histograms for open-loop and closed-loop cross-sectional distributions.

    Parameters
    ----------
    ol_values : np.ndarray
        Open-loop metric values across ETFs.
    cl_values : np.ndarray
        Closed-loop metric values across ETFs.
    xlabel : str
        X-axis label.
    title : str
        Plot title.
    save_path : str or None
        Save path.
    bins : int
        Number of histogram bins.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    ax.hist(ol_values, bins=bins, alpha=0.6, label="Open-loop", color="#ff7f0e", edgecolor="white")
    ax.hist(cl_values, bins=bins, alpha=0.6, label="Closed-loop", color="#2ca02c", edgecolor="white")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_chi_approximation(
    empirical_samples: np.ndarray,
    theoretical_x: np.ndarray,
    theoretical_pdf: np.ndarray,
    halflife: int,
    nu: float,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot empirical histogram vs theoretical chi-distribution approximation.

    Parameters
    ----------
    empirical_samples : np.ndarray
        Normalized EWMA volatility samples (sigma_hat / sigma_true).
    theoretical_x : np.ndarray
        x-values for theoretical PDF.
    theoretical_pdf : np.ndarray
        Theoretical PDF values.
    halflife : int
        EWMA halflife used.
    nu : float
        Effective degrees of freedom.
    save_path : str or None
        Save path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    ax.hist(empirical_samples, bins=60, density=True, alpha=0.6,
            color="#1f77b4", edgecolor="white", label="Monte Carlo empirical")
    ax.plot(theoretical_x, theoretical_pdf, "r-", linewidth=2,
            label=f"Chi approx (ν={nu:.1f})")

    ax.set_xlabel("σ̂ / σ (normalized EWMA volatility)")
    ax.set_ylabel("Density")
    ax.set_title(f"EWMA Volatility Distribution: h={halflife}, ν={nu:.1f}", fontsize=13)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_std_vs_halflife(
    halflives: np.ndarray,
    std_mc: np.ndarray,
    std_theory: np.ndarray | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot standard deviation of normalized EWMA estimator vs halflife.

    Parameters
    ----------
    halflives : np.ndarray
        Array of halflife values.
    std_mc : np.ndarray
        Monte Carlo std dev of sigma_hat/sigma for each halflife.
    std_theory : np.ndarray or None
        Theoretical std dev from chi approximation.
    save_path : str or None
        Save path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    ax.plot(halflives, std_mc, "o-", color="#1f77b4", label="Monte Carlo", linewidth=2)
    if std_theory is not None:
        ax.plot(halflives, std_theory, "s--", color="#d62728",
                label="Chi approximation", linewidth=2)

    ax.set_xlabel("EWMA Halflife (trading days)")
    ax.set_ylabel("Std Dev of σ̂ / σ")
    ax.set_title("EWMA Estimator Dispersion vs Halflife", fontsize=13)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
