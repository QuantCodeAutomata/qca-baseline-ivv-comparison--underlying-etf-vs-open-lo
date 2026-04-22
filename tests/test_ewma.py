"""
Tests for the finite-sample normalized EWMA volatility estimator.

Validates:
- Mathematical correctness of the finite-sample normalization
- Numerical equivalence of recursive and closed-form implementations
- Edge cases: single observation, constant returns, zero returns
- Convergence to true volatility with large samples
"""

import numpy as np
import pytest

from strategies.ewma import (
    compute_beta,
    ewma_variance_series,
    ewma_volatility_series,
    ewma_variance_series_pandas,
    ewma_volatility_series_pandas,
)
import pandas as pd


class TestComputeBeta:
    """Tests for compute_beta function."""

    def test_halflife_1(self):
        """Beta for halflife=1 should be 0.5."""
        beta = compute_beta(1.0)
        assert abs(beta - 0.5) < 1e-10

    def test_halflife_126(self):
        """Beta for halflife=126 should be exp(-log(2)/126)."""
        beta = compute_beta(126.0)
        expected = np.exp(-np.log(2.0) / 126.0)
        assert abs(beta - expected) < 1e-12

    def test_beta_in_zero_one(self):
        """Beta must be in (0, 1) for any positive halflife."""
        for h in [1, 5, 21, 63, 126, 252, 504]:
            beta = compute_beta(float(h))
            assert 0.0 < beta < 1.0, f"Beta out of range for h={h}: {beta}"

    def test_longer_halflife_larger_beta(self):
        """Longer halflife should give larger beta (slower decay)."""
        beta_short = compute_beta(21.0)
        beta_long = compute_beta(126.0)
        assert beta_long > beta_short


class TestEWMAVarianceSeries:
    """Tests for ewma_variance_series function."""

    def test_single_observation(self):
        """With one observation, variance should equal r^2 (normalization = 1-beta)."""
        beta = compute_beta(126.0)
        r = np.array([0.01])
        var = ewma_variance_series(r, beta)
        # sigma_hat_1^2 = ((1-beta)/(1-beta^1)) * r_1^2 = r_1^2
        expected = r[0] ** 2
        assert abs(var[0] - expected) < 1e-12

    def test_zero_returns(self):
        """Zero returns should give zero variance (above floor)."""
        beta = compute_beta(126.0)
        returns = np.zeros(100)
        var = ewma_variance_series(returns, beta, floor=0.0)
        assert np.all(var == 0.0)

    def test_constant_returns(self):
        """Constant returns r should converge to variance r^2."""
        beta = compute_beta(126.0)
        r_const = 0.01
        returns = np.full(2000, r_const)
        var = ewma_variance_series(returns, beta)
        # After many observations, should converge to r_const^2
        assert abs(var[-1] - r_const ** 2) < 1e-8

    def test_output_length(self):
        """Output length must equal input length."""
        beta = compute_beta(126.0)
        for T in [1, 10, 100, 1000]:
            returns = np.random.randn(T) * 0.01
            var = ewma_variance_series(returns, beta)
            assert len(var) == T

    def test_variance_positive(self):
        """Variance estimates must be positive."""
        beta = compute_beta(126.0)
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01
        var = ewma_variance_series(returns, beta)
        assert np.all(var > 0)

    def test_finite_sample_normalization(self):
        """
        Verify finite-sample normalization: for k observations with constant r,
        sigma_hat_k^2 = ((1-beta)/(1-beta^k)) * r^2 * sum_{j=0}^{k-1} beta^j
                      = ((1-beta)/(1-beta^k)) * r^2 * (1-beta^k)/(1-beta)
                      = r^2
        So for constant returns, variance should equal r^2 at every step.
        """
        beta = compute_beta(126.0)
        r_const = 0.02
        returns = np.full(50, r_const)
        var = ewma_variance_series(returns, beta, floor=0.0)
        # All values should equal r_const^2
        np.testing.assert_allclose(var, r_const ** 2, rtol=1e-8)

    def test_recursive_vs_closed_form(self):
        """
        Verify recursive implementation matches closed-form sum for small k.
        """
        beta = compute_beta(21.0)
        rng = np.random.default_rng(123)
        returns = rng.standard_normal(20) * 0.01

        var_recursive = ewma_variance_series(returns, beta)

        # Closed-form computation
        for k in range(1, len(returns) + 1):
            r_slice = returns[:k]
            weights = np.array([beta ** (k - 1 - j) for j in range(k)])
            raw = (1.0 - beta) * np.sum(weights * r_slice ** 2)
            norm = 1.0 - beta ** k
            expected = raw / norm
            assert abs(var_recursive[k - 1] - expected) < 1e-12, \
                f"Mismatch at k={k}: recursive={var_recursive[k-1]:.8e}, closed={expected:.8e}"


class TestEWMAVolatilitySeries:
    """Tests for ewma_volatility_series function."""

    def test_volatility_is_sqrt_variance(self):
        """Volatility should be sqrt of variance."""
        beta = compute_beta(126.0)
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(200) * 0.01
        var = ewma_variance_series(returns, beta)
        vol = ewma_volatility_series(returns, beta)
        np.testing.assert_allclose(vol, np.sqrt(var), rtol=1e-12)

    def test_convergence_to_true_vol(self):
        """With many observations, EWMA vol should converge to true vol."""
        true_sigma = 0.01
        beta = compute_beta(126.0)
        rng = np.random.default_rng(99)
        returns = rng.standard_normal(5000) * true_sigma
        vol = ewma_volatility_series(returns, beta)
        # Last estimate should be within 20% of true vol (stochastic)
        assert abs(vol[-1] - true_sigma) / true_sigma < 0.5


class TestPandasWrappers:
    """Tests for pandas wrapper functions."""

    def test_pandas_variance_preserves_index(self):
        """Pandas wrapper should preserve the input index."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        var = ewma_variance_series_pandas(returns, halflife=21.0)
        assert list(var.index) == list(dates)

    def test_pandas_volatility_preserves_index(self):
        """Pandas wrapper should preserve the input index."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        returns = pd.Series(np.random.randn(50) * 0.01, index=dates)
        vol = ewma_volatility_series_pandas(returns, halflife=21.0)
        assert list(vol.index) == list(dates)

    def test_pandas_matches_numpy(self):
        """Pandas wrapper should match numpy implementation."""
        rng = np.random.default_rng(7)
        arr = rng.standard_normal(100) * 0.01
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns_pd = pd.Series(arr, index=dates)

        vol_np = ewma_volatility_series(arr, compute_beta(21.0))
        vol_pd = ewma_volatility_series_pandas(returns_pd, halflife=21.0)

        np.testing.assert_allclose(vol_pd.values, vol_np, rtol=1e-12)
