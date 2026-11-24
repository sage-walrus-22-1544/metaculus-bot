"""Forecasters: turn a question (+ research) into a forecast and a rationale.

Two implementations share one interface:

- ``LlmForecaster`` calls an OpenAI-compatible chat endpoint (configured via
  environment variables; see ``config.py``) and parses the model's final
  answer out of its reasoning text.
- ``FallbackForecaster`` is a deterministic base-rate forecaster used when no
  LLM key is configured, and as a safety net if the LLM call or its parsing
  fails. It is deliberately simple and its rationales say so.

Every rationale includes the forecast value itself, as the tournament rules
require the posted comment to state the forecast.
"""

from __future__ import annotations

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from .config import LlmConfig
from .formatting import (
    clamp_binary_probability,
    ensure_monotonic,
    normalize_option_probabilities,
)

logger = logging.getLogger(__name__)

# Percentiles requested for numeric / date questions (fractions, 0-1).
NUMERIC_PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


@dataclass
class BinaryForecast:
    probability: float  # 0-1, already clamped to [0.01, 0.99]
    rationale: str


@dataclass
class NumericForecast:
    # (percentile as a fraction 0-1, value) pairs, values non-decreasing.
    percentiles: list[tuple[float, float]]
    rationale: str


@dataclass
class MultipleChoiceForecast:
    probabilities: dict[str, float]  # sums to 1
    rationale: str


@dataclass
class QuestionContext:
    """Everything a forecaster needs about one question."""

    question_text: str
    background_info: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    research: str = ""
    # Numeric / date questions:
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit_of_measure: str = ""
    # Multiple-choice questions:
    options: list[str] = field(default_factory=list)


class Forecaster(ABC):
    name = "abstract"

    @abstractmethod
    def forecast_binary(self, ctx: QuestionContext) -> BinaryForecast: ...

    @abstractmethod
    def forecast_numeric(self, ctx: QuestionContext) -> NumericForecast: ...

    @abstractmethod
    def forecast_multiple_choice(
        self, ctx: QuestionContext
    ) -> MultipleChoiceForecast: ...

    @abstractmethod
    def forecast_date(self, ctx: QuestionContext) -> NumericForecast:
        """Percentile values are POSIX timestamps (seconds)."""


class LlmError(RuntimeError):
    """Raised when the LLM call or the parsing of its output fails."""


class LlmForecaster(Forecaster):
    """Forecaster backed by an OpenAI-compatible chat-completions endpoint."""

    name = "llm"

    def __init__(self, llm_config: LlmConfig, timeout_seconds: int = 120):
        self._config = llm_config
        self._timeout = timeout_seconds

    # -- transport ---------------------------------------------------------
    def _invoke(self, prompt: str) -> str:
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        try:
            response = requests.post(
                self._config.chat_completions_url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise LlmError(f"LLM request failed: {exc}") from exc
        if response.status_code != 200:
            raise LlmError(
                f"LLM endpoint returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LlmError(f"Unexpected LLM response shape: {exc}") from exc

    # -- prompts (shared scaffolding) --------------------------------------
    @staticmethod
    def _question_block(ctx: QuestionContext) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return (
            f"Question: {ctx.question_text}\n\n"
            f"Background: {ctx.background_info}\n\n"
            f"Resolution criteria: {ctx.resolution_criteria}\n\n"
            f"{ctx.fine_print}\n\n"
            f"Research notes from my assistant:\n{ctx.research}\n\n"
            f"Today's date: {today}\n"
        )

    # -- binary -------------------------------------------------------------
    def forecast_binary(self, ctx: QuestionContext) -> BinaryForecast:
        prompt = (
            "You are an expert superforecaster. Reason carefully, weigh the "
            "base rate and the status quo, consider a scenario for Yes and a "
            "scenario for No, then commit to a single probability.\n\n"
            + self._question_block(ctx)
            + "\nEnd your reply with exactly one line of the form:\n"
            "Probability: ZZ%\n"
            "where ZZ is a number between 0 and 100."
        )
        reasoning = self._invoke(prompt)
        matches = re.findall(r"Probability:\s*(\d+(?:\.\d+)?)\s*%", reasoning)
        if not matches:
            raise LlmError("Could not parse a 'Probability: ZZ%' line from LLM output.")
        probability = clamp_binary_probability(float(matches[-1]) / 100.0)
        rationale = reasoning.strip()
        if f"{probability * 100:.0f}%" not in rationale and "Probability:" not in rationale:
            rationale += f"\n\nForecast: {probability * 100:.1f}%"
        return BinaryForecast(probability=probability, rationale=rationale)

    # -- numeric ------------------------------------------------------------
    def forecast_numeric(self, ctx: QuestionContext) -> NumericForecast:
        prompt = (
            "You are an expert superforecaster. Reason about the status quo, "
            "the current trend, and expert/market expectations, and keep your "
            "intervals wide enough to cover surprises.\n\n"
            + self._question_block(ctx)
            + f"\nUnits: {ctx.unit_of_measure or 'as stated in the question'}. "
            f"The value lies between {ctx.lower_bound} and {ctx.upper_bound}.\n"
            "End your reply with exactly these lines (plain numbers, no "
            "scientific notation, increasing values):\n"
            "Percentile 10: <value>\nPercentile 25: <value>\n"
            "Percentile 50: <value>\nPercentile 75: <value>\n"
            "Percentile 90: <value>"
        )
        reasoning = self._invoke(prompt)
        percentiles = _parse_percentile_lines(reasoning)
        if percentiles is None:
            raise LlmError("Could not parse percentile lines from LLM output.")
        values = ensure_monotonic(
            [_clamp_to_bounds(v, ctx.lower_bound, ctx.upper_bound) for _, v in percentiles]
        )
        fixed = [(p, v) for (p, _), v in zip(percentiles, values)]
        return NumericForecast(percentiles=fixed, rationale=reasoning.strip())

    # -- multiple choice ------------------------------------------------------
    def forecast_multiple_choice(self, ctx: QuestionContext) -> MultipleChoiceForecast:
        options_block = "\n".join(f"- {opt}" for opt in ctx.options)
        prompt = (
            "You are an expert superforecaster. Reason about each option, "
            "weight the status quo, and leave some probability on unexpected "
            "outcomes.\n\n"
            + self._question_block(ctx)
            + f"\nThe options are:\n{options_block}\n\n"
            "End your reply with one line per option, in the order listed, "
            "of the form:\n<option name>: <probability between 0 and 1>"
        )
        reasoning = self._invoke(prompt)
        probabilities: dict[str, float] = {}
        for option in ctx.options:
            pattern = re.compile(
                r"^\s*(?:[-*]\s*)?" + re.escape(option) + r"\s*:\s*(\d+(?:\.\d+)?)\s*(%?)\s*$",
                re.MULTILINE,
            )
            match = pattern.search(reasoning)
            if not match:
                raise LlmError(f"Could not parse a probability for option {option!r}.")
            value = float(match.group(1))
            if match.group(2) == "%" or value > 1.0:
                value /= 100.0
            probabilities[option] = value
        probabilities = normalize_option_probabilities(probabilities)
        return MultipleChoiceForecast(
            probabilities=probabilities, rationale=reasoning.strip()
        )

    # -- date -----------------------------------------------------------------
    def forecast_date(self, ctx: QuestionContext) -> NumericForecast:
        lower_dt = datetime.fromtimestamp(ctx.lower_bound, tz=timezone.utc)
        upper_dt = datetime.fromtimestamp(ctx.upper_bound, tz=timezone.utc)
        prompt = (
            "You are an expert superforecaster answering a date question. "
            "Reason about the status quo and current trend, and keep your "
            "intervals wide.\n\n"
            + self._question_block(ctx)
            + f"\nThe date lies between {lower_dt.date().isoformat()} and "
            f"{upper_dt.date().isoformat()}.\n"
            "End your reply with exactly these lines (dates as YYYY-MM-DD, "
            "chronologically increasing):\n"
            "Percentile 10: <date>\nPercentile 25: <date>\n"
            "Percentile 50: <date>\nPercentile 75: <date>\n"
            "Percentile 90: <date>"
        )
        reasoning = self._invoke(prompt)
        pattern = re.compile(
            r"Percentile\s*(\d+)\s*:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE
        )
        found = {}
        for pct, date_str in pattern.findall(reasoning):
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            found[int(pct) / 100.0] = dt.timestamp()
        if len(found) < 3:
            raise LlmError("Could not parse date percentile lines from LLM output.")
        ordered = sorted(found.items())
        values = ensure_monotonic(
            [_clamp_to_bounds(v, ctx.lower_bound, ctx.upper_bound) for _, v in ordered]
        )
        fixed = [(p, v) for (p, _), v in zip(ordered, values)]
        return NumericForecast(percentiles=fixed, rationale=reasoning.strip())


def _clamp_to_bounds(
    value: float, lower: float | None, upper: float | None
) -> float:
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return value


def _parse_percentile_lines(text: str) -> list[tuple[float, float]] | None:
    pattern = re.compile(
        r"Percentile\s*(\d+)\s*:\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE
    )
    found: dict[float, float] = {}
    for pct, raw in pattern.findall(text):
        found[int(pct) / 100.0] = float(raw.replace(",", ""))
    if len(found) < 3:
        return None
    return sorted(found.items())


class FallbackForecaster(Forecaster):
    """Deterministic base-rate forecaster (no LLM, no network).

    Used when no LLM key is configured and as a per-question safety net.
    The heuristics are intentionally transparent:

    - Binary: 50% (the uninformative base rate), rationale says so.
    - Numeric: percentiles at fixed fractions of the [lower, upper] range
      (log-space when the range spans more than two orders of magnitude
      with a positive lower bound).
    - Multiple choice: uniform over the options.
    - Date: percentiles at fixed fractions of the date range.
    """

    name = "fallback"

    _DISCLAIMER = (
        "Produced by the deterministic base-rate fallback forecaster "
        "(no LLM was configured / available for this question)."
    )

    def forecast_binary(self, ctx: QuestionContext) -> BinaryForecast:
        probability = clamp_binary_probability(0.5)
        rationale = (
            f"{self._DISCLAIMER}\n"
            "With no question-specific signal available, the forecast is the "
            "uninformative base rate.\n"
            f"Forecast: {probability * 100:.0f}%"
        )
        return BinaryForecast(probability=probability, rationale=rationale)

    def _spread_values(self, lower: float, upper: float) -> list[tuple[float, float]]:
        fractions = [0.20, 0.35, 0.50, 0.65, 0.80]
        use_log = lower > 0 and upper / lower > 100
        values = []
        for fraction in fractions:
            if use_log:
                value = math.exp(
                    math.log(lower) + fraction * (math.log(upper) - math.log(lower))
                )
            else:
                value = lower + fraction * (upper - lower)
            values.append(value)
        return list(zip(NUMERIC_PERCENTILES, ensure_monotonic(values)))

    def forecast_numeric(self, ctx: QuestionContext) -> NumericForecast:
        if ctx.lower_bound is None or ctx.upper_bound is None:
            raise ValueError("Numeric fallback needs lower and upper bounds.")
        percentiles = self._spread_values(ctx.lower_bound, ctx.upper_bound)
        lines = ", ".join(
            f"P{int(p * 100)}={v:,.4g}" for p, v in percentiles
        )
        rationale = (
            f"{self._DISCLAIMER}\n"
            f"Percentiles are spread at fixed fractions of the allowed range "
            f"[{ctx.lower_bound}, {ctx.upper_bound}] {ctx.unit_of_measure}.\n"
            f"Forecast percentiles: {lines}"
        )
        return NumericForecast(percentiles=percentiles, rationale=rationale)

    def forecast_multiple_choice(self, ctx: QuestionContext) -> MultipleChoiceForecast:
        if not ctx.options:
            raise ValueError("Multiple-choice fallback needs options.")
        probabilities = normalize_option_probabilities(
            {option: 1.0 for option in ctx.options}, floor=0.0
        )
        lines = ", ".join(
            f"{name}: {prob * 100:.1f}%" for name, prob in probabilities.items()
        )
        rationale = (
            f"{self._DISCLAIMER}\n"
            "With no option-specific signal available, probability is spread "
            "uniformly.\n"
            f"Forecast: {lines}"
        )
        return MultipleChoiceForecast(probabilities=probabilities, rationale=rationale)

    def forecast_date(self, ctx: QuestionContext) -> NumericForecast:
        if ctx.lower_bound is None or ctx.upper_bound is None:
            raise ValueError("Date fallback needs lower and upper bounds.")
        percentiles = self._spread_values(ctx.lower_bound, ctx.upper_bound)
        lines = ", ".join(
            f"P{int(p * 100)}="
            + datetime.fromtimestamp(v, tz=timezone.utc).date().isoformat()
            for p, v in percentiles
        )
        rationale = (
            f"{self._DISCLAIMER}\n"
            "Percentile dates are spread at fixed fractions of the allowed "
            "date range.\n"
            f"Forecast percentiles: {lines}"
        )
        return NumericForecast(percentiles=percentiles, rationale=rationale)


class ResilientForecaster(Forecaster):
    """Try the primary forecaster; on any failure use the fallback."""

    def __init__(self, primary: Forecaster, fallback: Forecaster):
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}+{fallback.name}-safety-net"

    def _run(self, method: str, ctx: QuestionContext):
        try:
            return getattr(self._primary, method)(ctx)
        except Exception as exc:  # noqa: BLE001 - safety net by design
            logger.warning(
                "Primary forecaster failed on %s (%s); using fallback.", method, exc
            )
            return getattr(self._fallback, method)(ctx)

    def forecast_binary(self, ctx: QuestionContext) -> BinaryForecast:
        return self._run("forecast_binary", ctx)

    def forecast_numeric(self, ctx: QuestionContext) -> NumericForecast:
        return self._run("forecast_numeric", ctx)

    def forecast_multiple_choice(self, ctx: QuestionContext) -> MultipleChoiceForecast:
        return self._run("forecast_multiple_choice", ctx)

    def forecast_date(self, ctx: QuestionContext) -> NumericForecast:
        return self._run("forecast_date", ctx)


def build_forecaster(config) -> Forecaster:
    """Pick the forecaster implied by the configuration."""
    fallback = FallbackForecaster()
    if config.llm_configured:
        return ResilientForecaster(LlmForecaster(config.llm), fallback)
    return fallback
