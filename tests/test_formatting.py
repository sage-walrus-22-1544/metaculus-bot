"""Unit tests for the pure formatting helpers."""

import math

import pytest

from src.formatting import (
    build_cdf,
    clamp_binary_probability,
    ensure_monotonic,
    normalize_option_probabilities,
)


class TestClampBinaryProbability:
    def test_inside_range_unchanged(self):
        assert clamp_binary_probability(0.5) == 0.5
        assert clamp_binary_probability(0.37) == pytest.approx(0.37)

    def test_bounds(self):
        assert clamp_binary_probability(0.0) == 0.01
        assert clamp_binary_probability(1.0) == 0.99
        assert clamp_binary_probability(-3.0) == 0.01
        assert clamp_binary_probability(7.5) == 0.99
        assert clamp_binary_probability(0.001) == 0.01
        assert clamp_binary_probability(0.999) == 0.99

    def test_nan_and_inf_rejected(self):
        with pytest.raises(ValueError):
            clamp_binary_probability(float("nan"))
        with pytest.raises(ValueError):
            clamp_binary_probability(float("inf"))


class TestEnsureMonotonic:
    def test_already_monotonic(self):
        assert ensure_monotonic([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_inversions_fixed_by_running_max(self):
        assert ensure_monotonic([3.0, 1.0, 2.0, 0.5]) == [3.0, 3.0, 3.0, 3.0]
        assert ensure_monotonic([1.0, 5.0, 4.0, 6.0]) == [1.0, 5.0, 5.0, 6.0]

    def test_empty(self):
        assert ensure_monotonic([]) == []


class TestNormalizeOptionProbabilities:
    def test_sums_to_one(self):
        result = normalize_option_probabilities({"A": 0.6, "B": 0.3, "C": 0.1})
        assert sum(result.values()) == pytest.approx(1.0)
        assert set(result) == {"A", "B", "C"}

    def test_order_preserved(self):
        result = normalize_option_probabilities({"A": 0.9, "B": 0.1})
        assert result["A"] > result["B"]

    def test_floor_applied(self):
        result = normalize_option_probabilities(
            {"A": 1.0, "B": 0.0, "C": 0.0}, floor=0.02
        )
        assert result["B"] >= 0.02
        assert result["C"] >= 0.02
        assert sum(result.values()) == pytest.approx(1.0)

    def test_all_zero_falls_back_to_uniform(self):
        result = normalize_option_probabilities({"A": 0.0, "B": 0.0})
        assert result == {"A": 0.5, "B": 0.5}

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            normalize_option_probabilities({"A": -0.5, "B": 1.5})


class TestBuildCdf:
    PERCENTILES = [(0.10, 20.0), (0.25, 35.0), (0.50, 50.0), (0.75, 65.0), (0.90, 80.0)]

    def test_shape_and_endpoints(self):
        cdf = build_cdf(self.PERCENTILES, lower_bound=0.0, upper_bound=100.0)
        assert len(cdf) == 201
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0

    def test_monotonic_and_bounded(self):
        cdf = build_cdf(self.PERCENTILES, lower_bound=0.0, upper_bound=100.0)
        assert all(0.0 <= p <= 1.0 for p in cdf)
        assert all(b >= a for a, b in zip(cdf, cdf[1:]))

    def test_passes_through_declared_percentiles(self):
        cdf = build_cdf(self.PERCENTILES, lower_bound=0.0, upper_bound=100.0)
        # x=50 is grid index 100; declared median is 50 -> CDF ~= 0.5.
        assert cdf[100] == pytest.approx(0.5, abs=0.02)
        # x=20 is grid index 40; declared P10=20 -> CDF ~= 0.1.
        assert cdf[40] == pytest.approx(0.10, abs=0.02)

    def test_nonmonotonic_input_is_repaired(self):
        messy = [(0.10, 60.0), (0.50, 40.0), (0.90, 90.0)]
        cdf = build_cdf(messy, lower_bound=0.0, upper_bound=100.0)
        assert all(b >= a for a, b in zip(cdf, cdf[1:]))
        assert not any(math.isnan(p) for p in cdf)

    def test_invalid_bounds_rejected(self):
        with pytest.raises(ValueError):
            build_cdf(self.PERCENTILES, lower_bound=100.0, upper_bound=0.0)
