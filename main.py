"""
Main experiment runner.

Orchestrates all four experiments and saves results to results/RESULTS.md.

Usage:
    python main.py [--exp {1,2,3,4,all}]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def format_metric(val: float, fmt: str = ".4f") -> str:
    """Format a metric value, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}"


def run_experiment_1() -> dict:
    """Run Experiment 1: Baseline IVV comparison."""
    from experiments.exp1_baseline import run_exp1
    logger.info("Starting Experiment 1...")
    t0 = time.time()
    result = run_exp1(results_dir=RESULTS_DIR)
    elapsed = time.time() - t0
    logger.info("Experiment 1 completed in %.1f seconds.", elapsed)
    return result


def run_experiment_2() -> dict:
    """Run Experiment 2: Parameter sensitivity study."""
    from experiments.exp2_sensitivity import run_exp2
    logger.info("Starting Experiment 2...")
    t0 = time.time()
    result = run_exp2(results_dir=RESULTS_DIR)
    elapsed = time.time() - t0
    logger.info("Experiment 2 completed in %.1f seconds.", elapsed)
    return result


def run_experiment_3() -> dict:
    """Run Experiment 3: Multi-asset ETF evaluation."""
    from experiments.exp3_multiasset import run_exp3
    logger.info("Starting Experiment 3...")
    t0 = time.time()
    result = run_exp3(results_dir=RESULTS_DIR)
    elapsed = time.time() - t0
    logger.info("Experiment 3 completed in %.1f seconds.", elapsed)
    return result


def run_experiment_4() -> dict:
    """Run Experiment 4: Uncertainty quantification."""
    from experiments.exp4_uncertainty import run_exp4
    logger.info("Starting Experiment 4...")
    t0 = time.time()
    result = run_exp4(results_dir=RESULTS_DIR)
    elapsed = time.time() - t0
    logger.info("Experiment 4 completed in %.1f seconds.", elapsed)
    return result


def write_results_md(
    exp1_result: dict | None = None,
    exp2_result: dict | None = None,
    exp3_result: dict | None = None,
    exp4_result: dict | None = None,
) -> None:
    """
    Write all experiment results to results/RESULTS.md.

    Parameters
    ----------
    exp1_result : dict or None
        Results from Experiment 1.
    exp2_result : dict or None
        Results from Experiment 2.
    exp3_result : dict or None
        Results from Experiment 3.
    exp4_result : dict or None
        Results from Experiment 4.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / "RESULTS.md"

    lines = [
        "# Experiment Results: Adaptive Volatility Control",
        "",
        "**Repository**: qca-baseline-ivv-comparison--underlying-etf-vs-open-lo",
        "",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # Experiment 1
    # -----------------------------------------------------------------------
    lines += [
        "## Experiment 1: Baseline IVV Comparison (2000-06-08 to 2024-12-31)",
        "",
        "Compares IVV buy-and-hold, open-loop inverse-volatility targeting, and",
        "closed-loop adaptive volatility control.",
        "",
        "**Strategy Parameters**: sigma_tar=15%, L=1.5, h=126, g=55, theta=0.6,",
        "kappa_min=-1, kappa_max=1, spread=5bp, warmup=10 days",
        "",
    ]

    if exp1_result is not None:
        metrics_df = exp1_result.get("metrics")
        if metrics_df is not None:
            lines.append("### Summary Metrics")
            lines.append("")
            lines.append("| Strategy | Ann. Return | Ann. Vol | Sharpe | Kalmar | Max DD | Tracking Error | Turnover |")
            lines.append("|----------|-------------|----------|--------|--------|--------|----------------|----------|")

            strategy_names = {
                "IVV Buy-and-Hold": "IVV Buy-and-Hold",
                "Open-Loop": "Open-Loop",
                "Closed-Loop": "Closed-Loop",
            }

            for strat_name in ["IVV Buy-and-Hold", "Open-Loop", "Closed-Loop"]:
                if strat_name in metrics_df.index:
                    row = metrics_df.loc[strat_name]
                    lines.append(
                        f"| {strat_name} "
                        f"| {row['ann_return']:.1%} "
                        f"| {row['ann_vol']:.1%} "
                        f"| {row['sharpe']:.2f} "
                        f"| {row['kalmar']:.2f} "
                        f"| {row['max_dd']:.1%} "
                        f"| {row['tracking_error']:.2%} "
                        f"| {row['turnover']:.0%} |"
                    )

            lines.append("")
            lines.append("### Paper Benchmark Targets")
            lines.append("")
            lines.append("| Strategy | Tracking Error | Ann. Return | Ann. Vol | Sharpe | Kalmar | Max DD | Turnover |")
            lines.append("|----------|----------------|-------------|----------|--------|--------|--------|----------|")
            lines.append("| IVV B&H  | ~5.2%          | ~7.8%       | ~19.1%   | ~0.31  | ~0.14  | ~55.3% | 0%       |")
            lines.append("| Open-Loop| ~2.3%          | ~6.8%       | ~14.9%   | ~0.33  | ~0.18  | ~38.6% | ~93%     |")
            lines.append("| Closed-Loop| ~0.4%        | ~8.2%       | ~14.9%   | ~0.42  | ~0.22  | ~37.1% | ~1105%   |")
            lines.append("")
    else:
        lines.append("*Experiment 1 not run.*")
        lines.append("")

    lines += [
        "### Output Files",
        "- `results/exp1_cumulative_returns.png` — Cumulative wealth paths",
        "- `results/exp1_running_volatility.png` — Running EWMA volatility",
        "- `results/exp1_metrics.csv` — Detailed metrics table",
        "",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # Experiment 2
    # -----------------------------------------------------------------------
    lines += [
        "## Experiment 2: Parameter Sensitivity Study (2000-06-08 to 2009-12-31)",
        "",
        "Grid search over g ∈ {0, e^0, e^0.5, ..., e^5} and θ ∈ {0.0, 0.1, ..., 0.9}.",
        "",
    ]

    if exp2_result is not None:
        results_df = exp2_result.get("results_df")
        if results_df is not None:
            lines.append("### Grid Search Summary Statistics")
            lines.append("")
            lines.append("| Metric | Min | Max | Mean |")
            lines.append("|--------|-----|-----|------|")

            for col, label in [
                ("tracking_error", "Tracking Error"),
                ("kalmar_change_vs_ivv", "ΔKalmar vs IVV"),
                ("turnover", "Turnover"),
            ]:
                if col in results_df.columns:
                    vals = results_df[col].dropna()
                    lines.append(
                        f"| {label} | {vals.min():.4f} | {vals.max():.4f} | {vals.mean():.4f} |"
                    )

            lines.append("")
            lines.append("### Baseline Parameter Point (g=55, θ=0.6)")
            lines.append("")
            baseline_row = results_df[
                (results_df["g"].between(50, 60)) &
                (results_df["theta"].between(0.55, 0.65))
            ]
            if not baseline_row.empty:
                r = baseline_row.iloc[0]
                lines.append(f"- Tracking Error: {r['tracking_error']:.4f}")
                lines.append(f"- ΔKalmar vs IVV: {r['kalmar_change_vs_ivv']:.4f}")
                lines.append(f"- Turnover: {r['turnover']:.1f}")
            lines.append("")
    else:
        lines.append("*Experiment 2 not run.*")
        lines.append("")

    lines += [
        "### Output Files",
        "- `results/exp2_heatmap_tracking_error.png` — Tracking error heatmap",
        "- `results/exp2_heatmap_kalmar_change.png` — ΔKalmar heatmap",
        "- `results/exp2_heatmap_turnover.png` — Turnover heatmap",
        "- `results/exp2_grid_results.csv` — Full grid results table",
        "",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # Experiment 3
    # -----------------------------------------------------------------------
    lines += [
        "## Experiment 3: Multi-Asset ETF Evaluation (2010-01-01 to 2024-12-31)",
        "",
        "Fixed parameters (g=55, θ=0.6) applied to 44 ETFs out-of-sample.",
        "",
    ]

    if exp3_result is not None:
        results_df = exp3_result.get("results_df")
        summary_df = exp3_result.get("summary_df")
        valid_tickers = exp3_result.get("valid_tickers", [])

        lines.append(f"**ETFs processed**: {len(valid_tickers)}/44")
        lines.append("")

        if summary_df is not None and not summary_df.empty:
            lines.append("### Cross-Sectional Summary Statistics")
            lines.append("")
            lines.append("| Metric | Strategy | Mean | Median | Min | Max |")
            lines.append("|--------|----------|------|--------|-----|-----|")

            for strategy, label in [("ol", "Open-Loop"), ("cl", "Closed-Loop")]:
                for metric, m_label in [
                    ("tracking_error", "Tracking Error"),
                    ("kalmar_change", "ΔKalmar"),
                    ("turnover", "Turnover"),
                ]:
                    col = f"{strategy}_{metric}"
                    if col in summary_df.index:
                        row = summary_df.loc[col]
                        lines.append(
                            f"| {m_label} | {label} "
                            f"| {row['mean']:.4f} | {row['median']:.4f} "
                            f"| {row['min']:.4f} | {row['max']:.4f} |"
                        )

            lines.append("")

        if results_df is not None and not results_df.empty:
            lines.append("### Per-ETF Results (Closed-Loop)")
            lines.append("")
            lines.append("| Ticker | Ann. Return | Ann. Vol | Sharpe | Kalmar | Max DD | Tracking Error | Turnover |")
            lines.append("|--------|-------------|----------|--------|--------|--------|----------------|----------|")
            for ticker in results_df.index[:20]:  # Show first 20
                row = results_df.loc[ticker]
                lines.append(
                    f"| {ticker} "
                    f"| {row.get('cl_ann_return', float('nan')):.1%} "
                    f"| {row.get('cl_ann_vol', float('nan')):.1%} "
                    f"| {row.get('cl_sharpe', float('nan')):.2f} "
                    f"| {row.get('cl_kalmar', float('nan')):.2f} "
                    f"| {row.get('cl_max_dd', float('nan')):.1%} "
                    f"| {row.get('cl_tracking_error', float('nan')):.2%} "
                    f"| {row.get('cl_turnover', float('nan')):.0f} |"
                )
            if len(results_df) > 20:
                lines.append(f"| ... ({len(results_df) - 20} more) | | | | | | | |")
            lines.append("")
    else:
        lines.append("*Experiment 3 not run.*")
        lines.append("")

    lines += [
        "### Output Files",
        "- `results/exp3_hist_tracking_error.png` — Cross-sectional tracking error histogram",
        "- `results/exp3_hist_kalmar_change.png` — Cross-sectional ΔKalmar histogram",
        "- `results/exp3_hist_turnover.png` — Cross-sectional turnover histogram",
        "- `results/exp3_etf_results.csv` — Per-ETF results",
        "- `results/exp3_summary_stats.csv` — Cross-sectional summary statistics",
        "",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # Experiment 4
    # -----------------------------------------------------------------------
    lines += [
        "## Experiment 4: Uncertainty Quantification (Monte Carlo)",
        "",
        "Monte Carlo analysis of EWMA volatility estimator uncertainty.",
        "",
    ]

    if exp4_result is not None:
        part_b = exp4_result.get("part_b", {})
        part_c = exp4_result.get("part_c", {})

        if part_b:
            lines.append("### Part B: Chi-Distribution Approximation Validation")
            lines.append("")
            lines.append("| Halflife | β | ν (eff. df) | Empirical Mean | Empirical Std | Theory Mean | Theory Std |")
            lines.append("|----------|---|-------------|----------------|---------------|-------------|------------|")
            for h, res in part_b.items():
                lines.append(
                    f"| {h} "
                    f"| {res['beta']:.6f} "
                    f"| {res['nu']:.1f} "
                    f"| {res['empirical_mean']:.4f} "
                    f"| {res['empirical_std']:.4f} "
                    f"| {res['theoretical_mean']:.4f} "
                    f"| {res['theoretical_std']:.4f} |"
                )
            lines.append("")

        if part_c:
            lines.append("### Part C: Estimator Dispersion vs Halflife")
            lines.append("")
            lines.append("| Halflife | MC Std(σ̂/σ) | Theory Std(σ̂/σ) |")
            lines.append("|----------|-------------|-----------------|")
            for h, std_mc, std_th in zip(
                part_c["halflives"], part_c["std_mc"], part_c["std_theory"]
            ):
                lines.append(f"| {h:.0f} | {std_mc:.4f} | {std_th:.4f} |")
            lines.append("")
    else:
        lines.append("*Experiment 4 not run.*")
        lines.append("")

    lines += [
        "### Output Files",
        "- `results/exp4a_confidence_bands.png` — Monte Carlo confidence bands",
        "- `results/exp4b_chi_approx_h21.png` — Chi approximation (h=21)",
        "- `results/exp4b_chi_approx_h126.png` — Chi approximation (h=126)",
        "- `results/exp4c_std_vs_halflife.png` — Estimator dispersion vs halflife",
        "- `results/exp4b_chi_approx_summary.csv` — Chi approximation summary",
        "- `results/exp4c_std_vs_halflife.csv` — Dispersion vs halflife data",
        "",
        "---",
        "",
        "## Implementation Notes",
        "",
        "- **EWMA formula**: finite-sample normalized, no demeaning, simple returns",
        "  `σ̂²_k = ((1-β)/(1-β^k)) * Σ_{j=1}^k β^{k-j} r_j²`",
        "- **Data**: massive API (adjusted close prices), FRED for Fed Funds rate",
        "- **Risk-free conversion**: `r_rf = (annual_rate_pct/100) / 252`",
        "- **Transaction cost**: `cost_k = spread * |w_k - w_{k-1}|` (5 bp spread)",
        "- **Annualization**: 252 trading days throughout",
        "- **Tracking error**: `sqrt(252) * mean_k |σ̂_k - σ_tar|`",
        "- **Kalmar ratio**: annualized_return / max_drawdown",
        "- **Turnover**: `Σ_k |w_k - w_{k-1}|`",
        "",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Results written to %s", md_path)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run volatility control experiments.")
    parser.add_argument(
        "--exp",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Which experiment(s) to run (default: all)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exp1_result = None
    exp2_result = None
    exp3_result = None
    exp4_result = None

    run_all = args.exp == "all"

    if run_all or args.exp == "1":
        try:
            exp1_result = run_experiment_1()
        except Exception as exc:
            logger.error("Experiment 1 failed: %s", exc, exc_info=True)

    if run_all or args.exp == "2":
        try:
            exp2_result = run_experiment_2()
        except Exception as exc:
            logger.error("Experiment 2 failed: %s", exc, exc_info=True)

    if run_all or args.exp == "3":
        try:
            exp3_result = run_experiment_3()
        except Exception as exc:
            logger.error("Experiment 3 failed: %s", exc, exc_info=True)

    if run_all or args.exp == "4":
        try:
            exp4_result = run_experiment_4()
        except Exception as exc:
            logger.error("Experiment 4 failed: %s", exc, exc_info=True)

    # Write results
    write_results_md(exp1_result, exp2_result, exp3_result, exp4_result)
    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
