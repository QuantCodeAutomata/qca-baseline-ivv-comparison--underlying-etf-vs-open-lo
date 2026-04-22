"""
Tests for performance metrics.

Validates:
- Annualized return formula
- Annualized volatility formula
- Sharpe ratio computation
- Kalmar ratio computation
- Maximum drawdown computation
- Volatility tracking error
- Turnover computation
"""

import numpy as np
import pytest

from utils.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    kalmar_ratio,
    volatility_tracking_error,
    total_turnover,
    compute_all_metrics,
    TRADING_DAYS,
)


class TestAnnualizedReturn:
    """Tests for annualized_return."""

    def test_zero_returns(self):
        """Zero returns should give zero annualized return."""
        returns = np.zeros(252)
        assert abs(annualized_return(returns)) < 1e-10

    def test_known_value(self):
        """Test with known compound return."""
        # 1% daily return for 252 days: (1.01)^252 - 1
        returns = np.full(252, 0.01)
        expected = 1.01 ** 252 - 1.0
        assert abs(annualized_return(returns) - expected) < 1e-8

    def test_empty_returns(self):
        """Empty returns should return NaN."""
        assert np.isnan(annualized_return(np.array([])))

    def test_single_return(self):
        """Single return annualized over 252 days."""
        r = np.array([0.01])
        expected = (1.01) ** 252 - 1.0
        assert abs(annualized_return(r) - expected) < 1e-8

    def test_negative_returns(self):
        """Negative returns should give negative annualized return."""
        returns = np.full(252, -0.001)
        assert annualized_return(returns) < 0


class TestAnnualizedVolatility:
    """Tests for annualized_volatility."""

    def test_constant_returns(self):
        """Constant returns have zero volatility."""
        returns = np.full(252, 0.01)
        assert abs(annualized_volatility(returns)) < 1e-10

    def test_known_value(self):
        """Test with known daily std dev."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(10000) * 0.01
        vol = annualized_volatility(returns)
        # Should be close to 0.01 * sqrt(252)
        expected = 0.01 * np.sqrt(252)
        assert abs(vol - expected) / expected < 0.05  # within 5%

    def test_single_observation(self):
        """Single observation should return NaN."""
        assert np.isnan(annualized_volatility(np.array([0.01])))


class TestSharpeRatio:
    """Tests for sharpe_ratio."""

    def test_zero_excess_return(self):
        """Zero excess return gives zero Sharpe."""
        returns = np.full(252, 0.001)
        rf = np.full(252, 0.001)
        # Excess returns are zero, so Sharpe is 0/vol = 0 or NaN
        # (std of zeros is 0, so NaN)
        result = sharpe_ratio(returns, rf)
        assert np.isnan(result) or abs(result) < 1e-8

    def test_positive_excess_return(self):
        """Positive excess return with positive vol gives positive Sharpe."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(1000) * 0.01 + 0.001
        rf = np.full(1000, 0.0001)
        sr = sharpe_ratio(returns, rf)
        assert sr > 0

    def test_sharpe_uses_excess_returns(self):
        """Sharpe should use excess returns, not raw returns."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01 + 0.002
        rf_high = np.full(500, 0.002)
        rf_low = np.full(500, 0.0001)
        sr_high_rf = sharpe_ratio(returns, rf_high)
        sr_low_rf = sharpe_ratio(returns, rf_low)
        # Higher risk-free rate reduces Sharpe
        assert sr_high_rf < sr_low_rf


class TestMaxDrawdown:
    """Tests for max_drawdown."""

    def test_no_drawdown(self):
        """Monotonically increasing wealth has zero drawdown."""
        returns = np.full(100, 0.01)
        assert abs(max_drawdown(returns)) < 1e-10

    def test_known_drawdown(self):
        """Test with known drawdown scenario."""
        # Wealth goes 1 -> 2 -> 1 -> drawdown = (2-1)/2 = 0.5
        returns = np.array([1.0, -0.5])  # 100% up, 50% down
        mdd = max_drawdown(returns)
        assert abs(mdd - 0.5) < 1e-10

    def test_drawdown_positive(self):
        """Max drawdown should always be non-negative."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01
        assert max_drawdown(returns) >= 0.0

    def test_drawdown_at_most_one(self):
        """Max drawdown cannot exceed 100%."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.05
        assert max_drawdown(returns) <= 1.0 + 1e-10

    def test_empty_returns(self):
        """Empty returns should return NaN."""
        assert np.isnan(max_drawdown(np.array([])))


class TestKalmarRatio:
    """Tests for kalmar_ratio."""

    def test_positive_return_positive_drawdown(self):
        """Positive return with positive drawdown gives positive Kalmar."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(1000) * 0.01 + 0.0005
        kal = kalmar_ratio(returns)
        assert kal > 0

    def test_kalmar_formula(self):
        """Kalmar = annualized_return / max_drawdown."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01 + 0.0003
        kal = kalmar_ratio(returns)
        ann_ret = annualized_return(returns)
        mdd = max_drawdown(returns)
        expected = ann_ret / mdd
        assert abs(kal - expected) < 1e-10


class TestVolatilityTrackingError:
    """Tests for volatility_tracking_error."""

    def test_perfect_tracking(self):
        """If sigma_hat always equals sigma_tar, tracking error is zero."""
        sigma_tar = 0.15 / np.sqrt(252)
        sigma_hat = np.full(500, sigma_tar)
        te = volatility_tracking_error(sigma_hat, sigma_tar)
        assert abs(te) < 1e-12

    def test_tracking_error_positive(self):
        """Tracking error should be non-negative."""
        sigma_tar = 0.15 / np.sqrt(252)
        sigma_hat = np.random.uniform(0.005, 0.02, 500)
        te = volatility_tracking_error(sigma_hat, sigma_tar)
        assert te >= 0.0

    def test_tracking_error_annualized(self):
        """Tracking error should be annualized by sqrt(252)."""
        sigma_tar = 0.15 / np.sqrt(252)
        # Constant deviation of 0.001 daily
        sigma_hat = np.full(500, sigma_tar + 0.001)
        te = volatility_tracking_error(sigma_hat, sigma_tar)
        expected = np.sqrt(252) * 0.001
        assert abs(te - expected) < 1e-10

    def test_empty_series(self):
        """Empty series should return NaN."""
        assert np.isnan(volatility_tracking_error(np.array([]), 0.01))


class TestTotalTurnover:
    """Tests for total_turnover."""

    def test_zero_turnover(self):
        """No weight changes gives zero turnover."""
        assert abs(total_turnover(np.zeros(100))) < 1e-12

    def test_known_turnover(self):
        """Sum of absolute weight changes."""
        changes = np.array([0.1, 0.2, 0.3])
        assert abs(total_turnover(changes) - 0.6) < 1e-10

    def test_turnover_non_negative(self):
        """Turnover should be non-negative."""
        changes = np.abs(np.random.randn(100) * 0.1)
        assert total_turnover(changes) >= 0.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_returns_all_keys(self):
        """Should return all expected metric keys."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01 + 0.0003
        rf = np.full(500, 0.0001)
        sigma_hat = np.full(500, 0.01)
        sigma_tar = 0.15 / np.sqrt(252)
        turnover = np.abs(rng.standard_normal(500) * 0.05)

        metrics = compute_all_metrics(returns, rf, sigma_hat, sigma_tar, turnover)

        expected_keys = ["ann_return", "ann_vol", "sharpe", "kalmar", "max_dd",
                         "tracking_error", "turnover"]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_are_finite(self):
        """All metrics should be finite for reasonable inputs."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01 + 0.0003
        rf = np.full(500, 0.0001)
        sigma_hat = np.full(500, 0.01)
        sigma_tar = 0.15 / np.sqrt(252)
        turnover = np.abs(rng.standard_normal(500) * 0.05)

        metrics = compute_all_metrics(returns, rf, sigma_hat, sigma_tar, turnover)

        for key, val in metrics.items():
            assert np.isfinite(val), f"Metric {key} is not finite: {val}"
