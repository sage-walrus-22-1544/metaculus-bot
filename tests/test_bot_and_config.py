"""Tests for config loading (missing-key behaviour) and for the bot's
per-question pipeline on fixture questions (offline, nothing posted)."""

import asyncio

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

import src.config as config_module
from src.bot import FutureEvalBot
from src.config import Config, ConfigError, load_config


def _empty_config() -> Config:
    return Config(
        metaculus_api_key=None,
        llm=None,
        asknews_client_id=None,
        asknews_secret=None,
    )


class TestLoadConfig:
    def test_missing_key_raises_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / "does-not-exist.env")
        for var in (
            "METACULUS_API_KEY",
            "METACULUS_TOKEN",
            "LLM_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ASKNEWS_CLIENT_ID",
            "ASKNEWS_SECRET",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ConfigError) as excinfo:
            load_config(require_metaculus_key=True)
        assert "METACULUS_API_KEY is not set" in str(excinfo.value)

    def test_missing_key_allowed_when_not_required(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / "does-not-exist.env")
        for var in (
            "METACULUS_API_KEY",
            "METACULUS_TOKEN",
            "LLM_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        config = load_config(require_metaculus_key=False)
        assert config.metaculus_api_key is None
        assert not config.llm_configured

    def test_key_from_env_is_picked_up(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config_module, "ENV_PATH", tmp_path / "does-not-exist.env")
        monkeypatch.setenv("METACULUS_API_KEY", "dummy-token-for-test")
        config = load_config(require_metaculus_key=True)
        assert config.metaculus_api_key == "dummy-token-for-test"


class TestBotPipelineOffline:
    """Exercise the bot's forecast methods directly on fixture questions.

    publish=False and no API calls are made: run_research returns the stub
    (no AskNews keys) and the fallback forecaster is deterministic.
    """

    def setup_method(self):
        self.bot = FutureEvalBot(_empty_config(), publish=False)

    def test_binary_pipeline(self):
        question = BinaryQuestion(
            question_text="Will the fixture event happen?",
            background_info="Fixture.",
            resolution_criteria="Resolves Yes if it happens.",
        )

        async def run():
            research = await self.bot.run_research(question)
            return await self.bot._run_forecast_on_binary(question, research)

        report = asyncio.run(run())
        assert 0.01 <= report.prediction_value <= 0.99
        assert report.reasoning  # rationale comment text is present
        assert "%" in report.reasoning  # and states the forecast value

    def test_numeric_pipeline_produces_valid_distribution(self):
        question = NumericQuestion(
            question_text="What will the fixture value be?",
            lower_bound=0.0,
            upper_bound=100.0,
            open_lower_bound=False,
            open_upper_bound=False,
            unit_of_measure="units",
        )

        async def run():
            research = await self.bot.run_research(question)
            return await self.bot._run_forecast_on_numeric(question, research)

        report = asyncio.run(run())
        distribution = report.prediction_value
        declared = [(p.percentile, p.value) for p in distribution.declared_percentiles]
        values = [v for _, v in declared]
        assert values == sorted(values)  # CDF percentiles are monotonic
        assert all(0.0 <= v <= 100.0 for v in values)
        cdf = distribution.cdf
        assert len(cdf) == 201
        cdf_values = [point.percentile for point in cdf]
        assert all(b >= a for a, b in zip(cdf_values, cdf_values[1:]))
        assert cdf_values[0] == pytest.approx(0.0)
        assert cdf_values[-1] == pytest.approx(1.0)

    def test_multiple_choice_pipeline(self):
        question = MultipleChoiceQuestion(
            question_text="Which fixture option wins?",
            options=["One", "Two", "Three"],
        )

        async def run():
            research = await self.bot.run_research(question)
            return await self.bot._run_forecast_on_multiple_choice(question, research)

        report = asyncio.run(run())
        probabilities = {
            opt.option_name: opt.probability
            for opt in report.prediction_value.predicted_options
        }
        assert set(probabilities) == {"One", "Two", "Three"}
        assert sum(probabilities.values()) == pytest.approx(1.0)
