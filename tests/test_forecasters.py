"""Unit tests for the forecasters: fallback determinism, LLM output parsing,
and the resilient wrapper."""

import pytest

from src.config import LlmConfig
from src.forecasters import (
    FallbackForecaster,
    LlmError,
    LlmForecaster,
    QuestionContext,
    ResilientForecaster,
)

BINARY_CTX = QuestionContext(
    question_text="Will it rain in Tokyo tomorrow?",
    background_info="Fixture background.",
    resolution_criteria="Resolves Yes if measurable rain is recorded.",
    research="No research.",
)
NUMERIC_CTX = QuestionContext(
    question_text="What will the price be?",
    lower_bound=2.0,
    upper_bound=6.0,
    unit_of_measure="USD",
)
MC_CTX = QuestionContext(
    question_text="Which option wins?",
    options=["Alpha", "Beta", "Gamma"],
)
DATE_CTX = QuestionContext(
    question_text="When will it happen?",
    lower_bound=1767225600.0,  # 2026-01-01 UTC
    upper_bound=1798761600.0,  # 2027-01-01 UTC
)


class TestFallbackForecaster:
    def setup_method(self):
        self.forecaster = FallbackForecaster()

    def test_binary_is_base_rate_and_deterministic(self):
        first = self.forecaster.forecast_binary(BINARY_CTX)
        second = self.forecaster.forecast_binary(BINARY_CTX)
        assert first.probability == second.probability == 0.5
        assert "50%" in first.rationale  # rationale states the forecast value

    def test_numeric_percentiles_within_bounds_and_monotonic(self):
        result = self.forecaster.forecast_numeric(NUMERIC_CTX)
        values = [v for _, v in result.percentiles]
        assert all(2.0 <= v <= 6.0 for v in values)
        assert values == sorted(values)
        assert "Forecast percentiles" in result.rationale

    def test_numeric_log_spread_for_wide_ranges(self):
        ctx = QuestionContext(
            question_text="Big range?", lower_bound=1.0, upper_bound=100_000.0
        )
        result = self.forecaster.forecast_numeric(ctx)
        median = dict(result.percentiles)[0.5]
        # Log-space median should be near sqrt(1 * 100000) ~= 316, far below
        # the linear midpoint of 50000.
        assert median < 1000

    def test_multiple_choice_uniform_and_sums_to_one(self):
        result = self.forecaster.forecast_multiple_choice(MC_CTX)
        assert sum(result.probabilities.values()) == pytest.approx(1.0)
        assert all(
            p == pytest.approx(1 / 3) for p in result.probabilities.values()
        )

    def test_date_percentiles_within_range_and_monotonic(self):
        result = self.forecaster.forecast_date(DATE_CTX)
        values = [v for _, v in result.percentiles]
        assert all(DATE_CTX.lower_bound <= v <= DATE_CTX.upper_bound for v in values)
        assert values == sorted(values)


class StubbedLlmForecaster(LlmForecaster):
    """LlmForecaster with the network call replaced by canned text."""

    def __init__(self, canned: str):
        super().__init__(
            LlmConfig(model="test-model", api_key="test", base_url="http://localhost")
        )
        self._canned = canned

    def _invoke(self, prompt: str) -> str:
        return self._canned


class TestLlmForecasterParsing:
    def test_binary_parse(self):
        forecaster = StubbedLlmForecaster(
            "Some reasoning...\nProbability: 37%"
        )
        result = forecaster.forecast_binary(BINARY_CTX)
        assert result.probability == pytest.approx(0.37)
        assert "37" in result.rationale

    def test_binary_parse_uses_last_probability_line(self):
        forecaster = StubbedLlmForecaster(
            "First I thought Probability: 80%, but on reflection\nProbability: 25%"
        )
        assert forecaster.forecast_binary(BINARY_CTX).probability == pytest.approx(0.25)

    def test_binary_parse_failure_raises(self):
        forecaster = StubbedLlmForecaster("No final answer here.")
        with pytest.raises(LlmError):
            forecaster.forecast_binary(BINARY_CTX)

    def test_binary_extremes_clamped(self):
        forecaster = StubbedLlmForecaster("Probability: 100%")
        assert forecaster.forecast_binary(BINARY_CTX).probability == 0.99

    def test_numeric_parse(self):
        canned = (
            "Reasoning...\n"
            "Percentile 10: 2.8\nPercentile 25: 3.1\nPercentile 50: 3.4\n"
            "Percentile 75: 3.9\nPercentile 90: 4.5"
        )
        result = StubbedLlmForecaster(canned).forecast_numeric(NUMERIC_CTX)
        values = dict(result.percentiles)
        assert values[0.5] == pytest.approx(3.4)
        assert list(values) == sorted(values)

    def test_numeric_out_of_bounds_values_clamped(self):
        canned = (
            "Percentile 10: 0.5\nPercentile 25: 3.0\nPercentile 50: 3.4\n"
            "Percentile 75: 3.9\nPercentile 90: 99.0"
        )
        result = StubbedLlmForecaster(canned).forecast_numeric(NUMERIC_CTX)
        values = [v for _, v in result.percentiles]
        assert all(2.0 <= v <= 6.0 for v in values)

    def test_multiple_choice_parse(self):
        canned = "Thinking...\nAlpha: 0.5\nBeta: 0.3\nGamma: 0.2"
        result = StubbedLlmForecaster(canned).forecast_multiple_choice(MC_CTX)
        assert sum(result.probabilities.values()) == pytest.approx(1.0)
        assert result.probabilities["Alpha"] > result.probabilities["Gamma"]

    def test_date_parse(self):
        canned = (
            "Percentile 10: 2026-03-01\nPercentile 25: 2026-05-01\n"
            "Percentile 50: 2026-07-01\nPercentile 75: 2026-09-01\n"
            "Percentile 90: 2026-11-01"
        )
        result = StubbedLlmForecaster(canned).forecast_date(DATE_CTX)
        values = [v for _, v in result.percentiles]
        assert values == sorted(values)
        assert all(DATE_CTX.lower_bound <= v <= DATE_CTX.upper_bound for v in values)


class TestResilientForecaster:
    def test_falls_back_on_llm_error(self):
        broken = StubbedLlmForecaster("unparseable output")
        resilient = ResilientForecaster(broken, FallbackForecaster())
        result = resilient.forecast_binary(BINARY_CTX)
        assert result.probability == 0.5  # fallback base rate

    def test_uses_primary_when_it_works(self):
        working = StubbedLlmForecaster("Probability: 63%")
        resilient = ResilientForecaster(working, FallbackForecaster())
        assert resilient.forecast_binary(BINARY_CTX).probability == pytest.approx(0.63)
