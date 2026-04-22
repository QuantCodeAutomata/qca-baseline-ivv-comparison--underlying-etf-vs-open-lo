"""
Tests for Experiment 4: Monte Carlo uncertainty quantification.

Validates:
- Chi-distribution approximation properties
- Confidence band computation
- Effective degrees of freedom formula
- Estimator dispersion decreases with halflife
"""

import numpy as np
import pytest
from scipy import stats

from strategies.ewma import compute_beta, ewma_variance_series
from experiments.exp4_uncertainty import (
    simulate_ewma_paths,
    compute_confidence_bands,
    chi_approximation_pdf,
    SIGMA_TRUE,
    TRADING_DAYS,
)


class TestEffectiveDegreesOfFreedom:
    """Tests for nu = (1 + beta) / (1 - beta)."""

    def test_nu_increases_with_halflife(self):
        """Longer halflife should give larger nu (more effective observations)."""
        beta_short = compute_beta(21.0)
        beta_long = compute_beta(126.0)
        nu_short = (1 + beta_short) / (1 - beta_short)
        nu_long = (1 + beta_long) / (1 - beta_long)
        assert nu_long > nu_short

    def test_nu_positive(self):
        """nu must be positive for any valid halflife."""
        for h in [5, 21, 63, 126, 252]:
            beta = compute_beta(float(h))
            nu = (1 + beta) / (1 - beta)
            assert nu > 0

    def test_nu_formula_h21(self):
        """Verify nu for h=21."""
        beta = compute_beta(21.0)
        nu = (1 + beta) / (1 - beta)
        # For h=21, beta ≈ 0.9672, nu ≈ 59.5
        assert 50 < nu < 70

    def test_nu_formula_h126(self):
        """Verify nu for h=126."""
        beta = compute_beta(126.0)
        nu = (1 + beta) / (1 - beta)
        # For h=126, beta ≈ 0.9945, nu ≈ 362
        assert 300 < nu < 450


class TestChiApproximation:
    """Tests for chi-distribution approximation."""

    def test_pdf_integrates_to_one(self):
        """Chi approximation PDF should integrate to approximately 1."""
        beta = compute_beta(126.0)
        nu = (1 + beta) / (1 - beta)
        x = np.linspace(0.001, 5.0, 10000)
        dx = x[1] - x[0]
        pdf = chi_approximation_pdf(x, nu)
        integral = np.sum(pdf) * dx
        assert abs(integral - 1.0) < 0.01

    def test_pdf_non_negative(self):
        """PDF values must be non-negative."""
        beta = compute_beta(21.0)
        nu = (1 + beta) / (1 - beta)
        x = np.linspace(0.01, 3.0, 100)
        pdf = chi_approximation_pdf(x, nu)
        assert np.all(pdf >= 0.0)

    def test_chi_mean_near_one(self):
        """Mean of chi(nu)/sqrt(nu) should be close to 1 for large nu."""
        beta = compute_beta(126.0)
        nu = (1 + beta) / (1 - beta)
        # E[chi(nu)/sqrt(nu)] = sqrt(2) * Gamma((nu+1)/2) / (sqrt(nu) * Gamma(nu/2))
        # For large nu, this approaches 1
        theoretical_mean = stats.chi.mean(df=nu, scale=1.0 / np.sqrt(nu))
        assert abs(theoretical_mean - 1.0) < 0.05  # within 5% of 1


class TestSimulateEWMAPaths:
    """Tests for simulate_ewma_paths."""

    def test_output_shape(self):
        """Output shape should be (n_paths, T)."""
        rng = np.random.default_rng(42)
        beta = compute_beta(126.0)
        result = simulate_ewma_paths(100, 50, beta, SIGMA_TRUE, rng)
        assert result.shape == (100, 50)

    def test_output_positive(self):
        """All volatility estimates should be positive."""
        rng = np.random.default_rng(42)
        beta = compute_beta(126.0)
        result = simulate_ewma_paths(100, 50, beta, SIGMA_TRUE, rng)
        assert np.all(result > 0)

    def test_mean_converges_to_target(self):
        """Mean of annualized EWMA vol should converge to true annualized vol."""
        rng = np.random.default_rng(42)
        beta = compute_beta(126.0)
        n_paths = 5000
        T = 500
        result = simulate_ewma_paths(n_paths, T, beta, SIGMA_TRUE, rng)
        # Mean at last step should be close to 0.15 (annualized)
        mean_last = np.mean(result[:, -1])
        assert abs(mean_last - 0.15) / 0.15 < 0.05  # within 5%


class TestConfidenceBands:
    """Tests for compute_confidence_bands."""

    def test_lower_leq_upper(self):
        """Lower band must be <= upper band at every time step."""
        rng = np.random.default_rng(42)
        beta = compute_beta(126.0)
        matrix = simulate_ewma_paths(200, 100, beta, SIGMA_TRUE, rng)
        lower, upper = compute_confidence_bands(matrix, 10.0, 90.0)
        assert np.all(lower <= upper + 1e-10)

    def test_bands_contain_target(self):
        """The 10th-90th percentile band should contain the target most of the time."""
        rng = np.random.default_rng(42)
        beta = compute_beta(126.0)
        matrix = simulate_ewma_paths(2000, 500, beta, SIGMA_TRUE, rng)
        lower, upper = compute_confidence_bands(matrix, 10.0, 90.0)
        # Target is 0.15 annualized
        target = 0.15
        # After burn-in, target should be within bands
        in_band = (lower[252:] <= target) & (target <= upper[252:])
        assert np.mean(in_band) > 0.5

    def test_longer_halflife_narrower_bands(self):
        """Longer halflife should produce narrower confidence bands."""
        rng = np.random.default_rng(42)
        T = 500

        beta_short = compute_beta(21.0)
        matrix_short = simulate_ewma_paths(2000, T, beta_short, SIGMA_TRUE, rng)
        lower_s, upper_s = compute_confidence_bands(matrix_short, 10.0, 90.0)
        width_short = np.mean(upper_s[252:] - lower_s[252:])

        rng2 = np.random.default_rng(42)
        beta_long = compute_beta(126.0)
        matrix_long = simulate_ewma_paths(2000, T, beta_long, SIGMA_TRUE, rng2)
        lower_l, upper_l = compute_confidence_bands(matrix_long, 10.0, 90.0)
        width_long = np.mean(upper_l[252:] - lower_l[252:])

        assert width_long < width_short


class TestEstimatorDispersion:
    """Tests for estimator dispersion vs halflife."""

    def test_dispersion_decreases_with_halflife(self):
        """Std dev of sigma_hat/sigma should decrease as halflife increases."""
        rng = np.random.default_rng(42)
        burn_in = 252
        n_paths = 2000
        halflives = [21, 63, 126]
        stds = []

        for h in halflives:
            beta = compute_beta(float(h))
            T_total = burn_in + 1
            returns = SIGMA_TRUE * rng.standard_normal((n_paths, T_total))
            sigma_hat_final = np.empty(n_paths)
            for i in range(n_paths):
                var_arr = ewma_variance_series(returns[i], beta)
                sigma_hat_final[i] = np.sqrt(var_arr[-1])
            normalized = sigma_hat_final / SIGMA_TRUE
            stds.append(np.std(normalized))

        # Std should decrease as halflife increases
        assert stds[0] > stds[1] > stds[2], \
            f"Dispersion not decreasing: {stds}"
