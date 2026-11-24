"""The FutureEval bot: a forecasting-tools ``ForecastBot`` subclass.

The framework handles tournament/question fetching, skipping questions the
bot already forecasted, aggregating predictions, and posting the forecast
plus its rationale as a comment. This subclass supplies:

- research (AskNews when configured, otherwise a marked stub),
- forecasts from our own forecaster interface (LLM when a key is configured,
  deterministic base-rate fallback otherwise) -- see ``forecasters.py``.

Forecasts are generated entirely by the bot; no human reviews individual
forecasts before posting, as the tournament rules require.
"""

from __future__ import annotations

import logging
from datetime import datetime

from forecasting_tools import (
    BinaryQuestion,
    ConditionalPrediction,
    DateQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    PredictionAffirmed,
    ReasonedPrediction,
)

from .config import Config
from .forecasters import Forecaster, QuestionContext, build_forecaster
from .research import get_research

logger = logging.getLogger(__name__)


class FutureEvalBot(ForecastBot):
    """ForecastBot for the FutureEval seasonal tournament + MiniBench."""

    def __init__(self, config: Config, publish: bool = False, **kwargs):
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=publish,
            skip_previously_forecasted_questions=True,
            enable_summarize_research=False,
            **kwargs,
        )
        self._config = config
        self._forecaster: Forecaster = build_forecaster(config)
        logger.info("Forecaster in use: %s", self._forecaster.name)

    # -- research -----------------------------------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        return await get_research(question.question_text, self._config)

    # -- context ------------------------------------------------------------
    @staticmethod
    def _context(question: MetaculusQuestion, research: str) -> QuestionContext:
        ctx = QuestionContext(
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            research=research,
        )
        if isinstance(question, NumericQuestion):
            lower = (
                question.nominal_lower_bound
                if question.nominal_lower_bound is not None
                else question.lower_bound
            )
            upper = (
                question.nominal_upper_bound
                if question.nominal_upper_bound is not None
                else question.upper_bound
            )
            ctx.lower_bound = float(lower)
            ctx.upper_bound = float(upper)
            ctx.unit_of_measure = question.unit_of_measure or ""
        elif isinstance(question, DateQuestion):
            ctx.lower_bound = _as_timestamp(question.lower_bound)
            ctx.upper_bound = _as_timestamp(question.upper_bound)
        elif isinstance(question, MultipleChoiceQuestion):
            ctx.options = list(question.options)
        return ctx

    # -- binary ---------------------------------------------------------------
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        result = self._forecaster.forecast_binary(self._context(question, research))
        logger.info(
            "Binary forecast for %s: %.3f", question.page_url, result.probability
        )
        return ReasonedPrediction(
            prediction_value=result.probability, reasoning=result.rationale
        )

    # -- multiple choice ------------------------------------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        result = self._forecaster.forecast_multiple_choice(
            self._context(question, research)
        )
        predicted_options = [
            PredictedOption(option_name=name, probability=probability)
            for name, probability in result.probabilities.items()
        ]
        logger.info(
            "Multiple-choice forecast for %s: %s",
            question.page_url,
            result.probabilities,
        )
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(predicted_options=predicted_options),
            reasoning=result.rationale,
        )

    # -- numeric ----------------------------------------------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        result = self._forecaster.forecast_numeric(self._context(question, research))
        percentile_list = [
            Percentile(percentile=p, value=v) for p, v in result.percentiles
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            "Numeric forecast for %s: %s",
            question.page_url,
            [(p.percentile, p.value) for p in prediction.declared_percentiles],
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=result.rationale)

    # -- date -------------------------------------------------------------------
    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        result = self._forecaster.forecast_date(self._context(question, research))
        percentile_list = [
            Percentile(percentile=p, value=v) for p, v in result.percentiles
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            "Date forecast for %s: %s",
            question.page_url,
            [(p.percentile, p.value) for p in prediction.declared_percentiles],
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=result.rationale)

    # -- conditional --------------------------------------------------------------
    async def _run_forecast_on_conditional(
        self, question, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        """Forecast a conditional via its four sub-questions.

        Parent/child sub-questions that already carry a live forecast from
        this bot are affirmed rather than re-forecast; the yes/no branches
        are always forecast fresh.
        """
        parts: dict[str, ReasonedPrediction] = {}
        for label, sub in (
            ("parent", question.parent),
            ("child", question.child),
            ("yes", question.question_yes),
            ("no", question.question_no),
        ):
            previous = getattr(sub, "previous_forecasts", None)
            if label in ("parent", "child") and previous:
                parts[label] = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Affirming this bot's existing forecast on the "
                    f"{label} question.",
                )
            else:
                parts[label] = await self._make_prediction(sub, research)
        prediction = ConditionalPrediction(
            parent=parts["parent"].prediction_value,
            child=parts["child"].prediction_value,
            prediction_yes=parts["yes"].prediction_value,
            prediction_no=parts["no"].prediction_value,
        )
        reasoning = "\n\n".join(
            f"## {label.title()} question\n{part.reasoning}"
            for label, part in parts.items()
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


def _as_timestamp(value) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)
