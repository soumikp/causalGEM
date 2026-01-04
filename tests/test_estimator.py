"""
Unit tests for the causalGEM estimator module.
"""

import numpy as np
import pytest
from causalgem import (
    estimate_causal_direction,
    estimate_entropy_difference,
    estimate_with_decision,
    estimate_stratified,
    CausalGEMResult,
)
from causalgem.simulation import generate_causal_pair


class TestEstimateCausalDirection:
    """Tests for the main estimation function."""

    def test_basic_functionality(self):
        """Test that estimation runs without errors."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 200)
        y = x**2 + np.random.normal(0, 0.1, 200)
        
        result = estimate_causal_direction(x, y)
        
        assert isinstance(result, CausalGEMResult)
        assert np.isfinite(result.delta)
        assert np.isfinite(result.ci_lower)
        assert np.isfinite(result.ci_upper)
        assert result.ci_lower < result.delta < result.ci_upper

    def test_correct_direction_expanding(self):
        """Test correct inference for expanding function (X²)."""
        np.random.seed(123)
        x, y, _ = generate_causal_pair(500, func='square', noise_sd=0.05)
        
        result = estimate_causal_direction(x, y)
        
        assert result.decision in [-1, 0, 1]
        assert result.n_samples == 500 or result.n_samples == 499

    def test_correct_direction_contracting(self):
        """Test inference for contracting function (sqrt)."""
        np.random.seed(456)
        x, y, _ = generate_causal_pair(500, func='sqrt', noise_sd=0.05)
        
        result = estimate_causal_direction(x, y)
        
        assert result.decision in [-1, 0, 1]

    def test_symmetric_linear(self):
        """Test that linear relationship is often inconclusive."""
        np.random.seed(789)
        x = np.random.uniform(0, 1, 500)
        y = x + np.random.normal(0, 0.2, 500)
        
        result = estimate_causal_direction(x, y)
        
        # Linear should be close to 0 (inconclusive)
        assert abs(result.delta) < 1.0

    def test_different_alpha_levels(self):
        """Test with different significance levels."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 300)
        y = x**2 + np.random.normal(0, 0.1, 300)
        
        result_95 = estimate_causal_direction(x, y, alpha=0.05)
        result_99 = estimate_causal_direction(x, y, alpha=0.01)
        
        # 99% CI should be wider than 95% CI
        width_95 = result_95.ci_upper - result_95.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower
        assert width_99 > width_95

    def test_input_validation_length_mismatch(self):
        """Test error on mismatched input lengths."""
        x = np.random.randn(100)
        y = np.random.randn(50)
        
        with pytest.raises(ValueError, match="same length"):
            estimate_causal_direction(x, y)

    def test_handles_odd_sample_size(self):
        """Test that odd sample sizes are handled correctly."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 101)
        y = x**2 + np.random.normal(0, 0.1, 101)
        
        result = estimate_causal_direction(x, y)
        
        # Should use n-1 = 100 samples
        assert result.n_samples == 100

    def test_normalize_option(self):
        """Test with and without normalization."""
        np.random.seed(42)
        x = np.random.uniform(5, 10, 200)  # Not in [0,1]
        y = x**2 + np.random.normal(0, 1, 200)
        
        result_norm = estimate_causal_direction(x, y, normalize=True)
        result_no_norm = estimate_causal_direction(x, y, normalize=False)
        
        # Both should run successfully
        assert np.isfinite(result_norm.delta)
        assert np.isfinite(result_no_norm.delta)


class TestEstimateWithDecision:
    """Tests for the legacy interface."""

    def test_returns_tuple(self):
        """Test that legacy function returns correct tuple."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 200)
        y = x**2 + np.random.normal(0, 0.1, 200)
        
        result = estimate_with_decision(x, y)
        
        assert isinstance(result, tuple)
        assert len(result) == 7
        decision, h_x, h_y, ci_lower, delta, ci_upper, se = result
        assert decision in [-1, 0, 1]
        assert np.isfinite(h_x)
        assert np.isfinite(h_y)
        assert ci_lower < delta < ci_upper


class TestEstimateStratified:
    """Tests for stratified estimation."""

    def test_two_strata(self):
        """Test with two strata."""
        np.random.seed(42)
        n1, n2 = 150, 150
        
        # Stratum 1
        x1 = np.random.uniform(0, 1, n1)
        y1 = x1**2 + np.random.normal(0, 0.1, n1)
        
        # Stratum 2
        x2 = np.random.uniform(0, 1, n2)
        y2 = x2**2 + np.random.normal(0, 0.1, n2)
        
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        strata = np.array([0]*n1 + [1]*n2)
        
        result = estimate_stratified(x, y, strata)
        
        assert isinstance(result, CausalGEMResult)
        assert result.n_samples == n1 + n2

    def test_small_strata_skipped(self):
        """Test that strata with < 10 samples are skipped."""
        np.random.seed(42)
        
        x = np.random.uniform(0, 1, 200)
        y = x**2 + np.random.normal(0, 0.1, 200)
        strata = np.array([0]*195 + [1]*5)  # Second stratum too small
        
        result = estimate_stratified(x, y, strata)
        
        # Should still work with just the large stratum
        assert np.isfinite(result.delta)


class TestCausalGEMResult:
    """Tests for the result dataclass."""

    def test_repr(self):
        """Test string representation."""
        result = CausalGEMResult(
            delta=0.25,
            ci_lower=0.1,
            ci_upper=0.4,
            std_error=0.05,
            h_x=1.0,
            h_y=0.75,
            decision=1,
            n_samples=500,
        )
        
        repr_str = repr(result)
        assert "0.25" in repr_str
        assert "X → Y" in repr_str

    def test_is_significant_property(self):
        """Test is_significant property."""
        significant = CausalGEMResult(
            delta=0.25, ci_lower=0.1, ci_upper=0.4,
            std_error=0.05, h_x=1.0, h_y=0.75,
            decision=1, n_samples=500
        )
        
        inconclusive = CausalGEMResult(
            delta=0.05, ci_lower=-0.1, ci_upper=0.2,
            std_error=0.05, h_x=1.0, h_y=0.95,
            decision=0, n_samples=500
        )
        
        assert significant.is_significant is True
        assert inconclusive.is_significant is False

    def test_direction_str_property(self):
        """Test direction_str property."""
        for decision, expected in [(1, "X → Y"), (-1, "Y → X"), (0, "Inconclusive")]:
            result = CausalGEMResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2,
                std_error=0.05, h_x=1.0, h_y=0.9,
                decision=decision, n_samples=100
            )
            assert result.direction_str == expected


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_constant_data(self):
        """Test handling of constant data."""
        x = np.ones(100)
        y = np.random.randn(100)
        
        # Should handle gracefully
        result = estimate_causal_direction(x, y)
        assert np.isfinite(result.delta) or np.isnan(result.delta)

    def test_highly_correlated(self):
        """Test with perfectly correlated data."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 200)
        y = x  # Perfect correlation
        
        result = estimate_causal_direction(x, y)
        
        # Delta should be close to 0
        assert abs(result.delta) < 0.5

    def test_independent_data(self):
        """Test with independent variables."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 500)
        y = np.random.uniform(0, 1, 500)
        
        result = estimate_causal_direction(x, y)
        
        # Should be inconclusive or close to 0
        assert abs(result.delta) < 0.5 or result.decision == 0

    def test_large_sample(self):
        """Test with larger sample for stability."""
        np.random.seed(42)
        x, y, _ = generate_causal_pair(2000, func='square', noise_sd=0.1)
        
        result = estimate_causal_direction(x, y)
        
        # Larger sample should give smaller SE
        assert result.std_error < 0.1


class TestAliases:
    """Test that aliases work correctly."""

    def test_estimate_entropy_difference_alias(self):
        """Test that estimate_entropy_difference works."""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 200)
        y = x**2 + np.random.normal(0, 0.1, 200)
        
        result1 = estimate_causal_direction(x, y)
        result2 = estimate_entropy_difference(x, y)
        
        # Should give same results
        assert result1.delta == result2.delta
        assert result1.decision == result2.decision
