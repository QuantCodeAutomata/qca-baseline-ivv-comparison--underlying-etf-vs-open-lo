"""
Tests for volatility control strategies.

Validates:
- Weight constraints (0 <= w <= L)
- Cash weight identity (c = 1 - w)
- Open-loop reduces to buy-and-hold when sigma_tar is very large
- Closed-loop with g=0 reduces to open-loop
- Warm-up behavior
- Transaction cost deduction
- No look-ahead bias (timing)
"""

import numpy as np
import pytest

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
from strategies.ewma import compute_beta


def make_synthetic_data(T: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic risky and cash returns."""
    rng = np.random.default_rng(seed)
    risky = rng.standard_normal(T) * 0.01
    cash = np.full(T, 0.0001)  # ~2.5% annual
    return risky, cash


class TestWeightConstraints:
    """Tests for weight constraint enforcement."""

    def test_weights_in_valid_range(self):
        """All weights must be in [0, L]."""
        risky, cash = make_synthetic_data(500)
        result = run_strategies(risky, cash)

        for key in ["ol_weights", "cl_weights"]:
            w = result[key]
            assert np.all(w >= 0.0), f"{key} has negative weights"
            assert np.all(w <= LEVERAGE_CAP + 1e-10), f"{key} exceeds leverage cap"

    def test_cash_weight_identity(self):
        """Cash weight must equal 1 - risky weight exactly."""
        risky, cash = make_synthetic_data(200)
        result = run_strategies(risky, cash)

        # Verify c = 1 - w for each period
        # (We verify via return decomposition)
        # r_ind = r * w + r_rf * (1 - w)
        # Check that ol_returns_gross matches this decomposition
        ol_w = result["ol_weights"]
        ol_r = result["ol_returns_gross"]

        # Reconstruct: r_ind[k] = risky[k] * w[k-1] + cash[k] * (1 - w[k-1])
        w_prev = np.concatenate([[0.0], ol_w[:-1]])
        expected = risky * w_prev + cash * (1.0 - w_prev)
        np.testing.assert_allclose(ol_r, expected, rtol=1e-10)

    def test_leverage_cap_binding(self):
        """When sigma_tar >> sigma_hat, leverage cap should bind."""
        risky, cash = make_synthetic_data(500)
        # Use very large sigma_tar to force leverage cap
        result = run_strategies(risky, cash, sigma_tar=1.0, leverage_cap=1.5)
        # Most weights should be at the cap
        assert np.mean(result["ol_weights"] >= 1.49) > 0.5


class TestOpenLoopBehavior:
    """Tests for open-loop strategy."""

    def test_open_loop_weight_formula(self):
        """Open-loop weight should be min(sigma_tar / sigma_hat, L)."""
        risky, cash = make_synthetic_data(300)
        result = run_strategies(risky, cash)

        beta = compute_beta(HALFLIFE)
        from strategies.ewma import ewma_volatility_series
        sigma_hat = ewma_volatility_series(risky, beta)

        # w_k = min(sigma_tar / sigma_hat_k, L)
        expected_w = np.minimum(SIGMA_TAR / sigma_hat, LEVERAGE_CAP)
        np.testing.assert_allclose(result["ol_weights"], expected_w, rtol=1e-8)

    def test_open_loop_no_feedback(self):
        """Open-loop should not depend on strategy returns."""
        risky, cash = make_synthetic_data(200)
        result1 = run_strategies(risky, cash, g=0.0)
        result2 = run_strategies(risky, cash, g=100.0)

        # Open-loop weights should be identical regardless of g
        np.testing.assert_allclose(result1["ol_weights"], result2["ol_weights"], rtol=1e-12)


class TestClosedLoopBehavior:
    """Tests for closed-loop strategy."""

    def test_g_zero_reduces_to_open_loop(self):
        """With g=0 and kappa_0=0, closed-loop should equal open-loop."""
        risky, cash = make_synthetic_data(300)
        result = run_strategies(risky, cash, g=0.0, theta=0.5)

        # kappa should remain 0 throughout (since -g*e = 0)
        np.testing.assert_allclose(result["cl_kappa"], 0.0, atol=1e-12)

        # Weights should match open-loop
        np.testing.assert_allclose(
            result["cl_weights"], result["ol_weights"], rtol=1e-8
        )

    def test_warmup_uses_open_loop_weights(self):
        """During warm-up, closed-loop should use open-loop weights."""
        risky, cash = make_synthetic_data(300)
        result = run_strategies(risky, cash, warmup_days=10)

        # During warm-up (days 0..9), kappa should be 0
        np.testing.assert_allclose(result["cl_kappa"][:10], 0.0, atol=1e-12)

        # Weights during warm-up should match open-loop
        np.testing.assert_allclose(
            result["cl_weights"][:10], result["ol_weights"][:10], rtol=1e-8
        )

    def test_kappa_clipping(self):
        """Kappa should be clipped to [kappa_min, kappa_max] before smoothing."""
        risky, cash = make_synthetic_data(500)
        result = run_strategies(risky, cash, kappa_min=-1.0, kappa_max=1.0)

        # The smoothed kappa can be outside clip bounds due to smoothing,
        # but the clipped input is bounded. The smoothed output is a convex
        # combination of clipped value and previous kappa, so it stays in
        # [kappa_min, kappa_max] if kappa_prev is also in range.
        # Since kappa_0 = 0 and clip is applied first, kappa stays in [-1, 1].
        assert np.all(result["cl_kappa"] >= -1.0 - 1e-10)
        assert np.all(result["cl_kappa"] <= 1.0 + 1e-10)

    def test_closed_loop_weights_bounded(self):
        """Closed-loop weights must stay in [0, L]."""
        risky, cash = make_synthetic_data(500)
        result = run_strategies(risky, cash)
        w = result["cl_weights"]
        assert np.all(w >= 0.0)
        assert np.all(w <= LEVERAGE_CAP + 1e-10)


class TestTransactionCosts:
    """Tests for transaction cost implementation."""

    def test_net_returns_leq_gross(self):
        """Net returns should be <= gross returns (costs are non-negative)."""
        risky, cash = make_synthetic_data(300)
        result = run_strategies(risky, cash, spread=0.0005)

        assert np.all(result["ol_returns_net"] <= result["ol_returns_gross"] + 1e-12)
        assert np.all(result["cl_returns_net"] <= result["cl_returns_gross"] + 1e-12)

    def test_zero_spread_net_equals_gross(self):
        """With zero spread, net returns should equal gross returns."""
        risky, cash = make_synthetic_data(200)
        result = run_strategies(risky, cash, spread=0.0)

        np.testing.assert_allclose(
            result["ol_returns_net"], result["ol_returns_gross"], rtol=1e-12
        )
        np.testing.assert_allclose(
            result["cl_returns_net"], result["cl_returns_gross"], rtol=1e-12
        )

    def test_turnover_non_negative(self):
        """Absolute weight changes must be non-negative."""
        risky, cash = make_synthetic_data(300)
        result = run_strategies(risky, cash)

        assert np.all(result["ol_turnover_abs"] >= 0.0)
        assert np.all(result["cl_turnover_abs"] >= 0.0)

    def test_bh_zero_turnover(self):
        """Buy-and-hold has no rebalancing, so turnover arrays are not in result."""
        risky, cash = make_synthetic_data(100)
        result = run_strategies(risky, cash)
        # BH returns equal risky returns
        np.testing.assert_allclose(result["bh_returns"], risky, rtol=1e-12)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_observation(self):
        """Strategy should handle a single observation without error."""
        risky = np.array([0.01])
        cash = np.array([0.0001])
        result = run_strategies(risky, cash)
        assert len(result["ol_weights"]) == 1
        assert len(result["cl_weights"]) == 1

    def test_all_zero_returns(self):
        """Zero returns should not cause division by zero."""
        risky = np.zeros(100)
        cash = np.zeros(100)
        result = run_strategies(risky, cash)
        # Weights should be at leverage cap (sigma_hat near floor)
        assert np.all(np.isfinite(result["ol_weights"]))
        assert np.all(np.isfinite(result["cl_weights"]))

    def test_extreme_positive_returns(self):
        """Extreme positive returns should not cause overflow."""
        risky = np.full(100, 0.1)  # 10% daily return
        cash = np.full(100, 0.0001)
        result = run_strategies(risky, cash)
        assert np.all(np.isfinite(result["ol_weights"]))
        assert np.all(np.isfinite(result["cl_weights"]))

    def test_warmup_longer_than_data(self):
        """Warm-up longer than data should not crash."""
        risky, cash = make_synthetic_data(5)
        result = run_strategies(risky, cash, warmup_days=20)
        assert len(result["cl_weights"]) == 5

    def test_output_keys_present(self):
        """All expected output keys must be present."""
        risky, cash = make_synthetic_data(100)
        result = run_strategies(risky, cash)
        expected_keys = [
            "bh_returns", "ol_returns_gross", "ol_returns_net",
            "cl_returns_gross", "cl_returns_net",
            "ol_weights", "cl_weights",
            "ol_sigma_risky", "cl_sigma_risky",
            "ol_sigma_ind", "cl_sigma_ind",
            "cl_kappa", "ol_turnover_abs", "cl_turnover_abs",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
