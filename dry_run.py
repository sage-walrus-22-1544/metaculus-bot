#!/usr/bin/env python3
"""Offline dry-run: run the full research+forecast pipeline on fixture
questions and save the results. No Metaculus key is needed and nothing is
posted.

Also probes the live Metaculus API without credentials and records the
response verbatim, to document whether unauthenticated reads are possible.

Output: dry_run_output/sample_forecasts.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    DateQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from src.bot import FutureEvalBot  # noqa: E402
from src.config import load_config  # noqa: E402

FIXTURES_PATH = PROJECT_DIR / "fixtures" / "questions.json"
OUTPUT_PATH = PROJECT_DIR / "dry_run_output" / "sample_forecasts.json"

QUESTION_CLASSES = {
    "binary": BinaryQuestion,
    "numeric": NumericQuestion,
    "multiple_choice": MultipleChoiceQuestion,
    "date": DateQuestion,
}


def load_fixture_questions() -> list:
    raw = json.loads(FIXTURES_PATH.read_text())
    questions = []
    for entry in raw:
        entry = dict(entry)
        kind = entry.pop("type")
        for field in ("open_time", "close_time", "scheduled_resolution_time"):
            if entry.get(field):
                entry[field] = datetime.fromisoformat(entry[field])
        for field in ("lower_bound", "upper_bound"):
            if isinstance(entry.get(field), str) and kind == "date":
                entry[field] = datetime.fromisoformat(entry[field])
        questions.append(QUESTION_CLASSES[kind](**entry))
    return questions


def probe_unauthenticated_api() -> dict:
    """Try listing MiniBench questions with no credentials; record verbatim."""
    url = "https://www.metaculus.com/api/posts/?tournaments=minibench&statuses=open&limit=3"
    try:
        response = requests.get(url, timeout=30)
        return {
            "url": url,
            "status_code": response.status_code,
            "body_excerpt": response.text[:300],
        }
    except requests.RequestException as exc:
        return {"url": url, "error": str(exc)}


async def run_pipeline(questions: list) -> list[dict]:
    config = load_config(require_metaculus_key=False)
    bot = FutureEvalBot(config, publish=False)
    results = []
    for question in questions:
        research = await bot.run_research(question)
        if isinstance(question, BinaryQuestion):
            report = await bot._run_forecast_on_binary(question, research)
            forecast = {"probability": report.prediction_value}
        elif isinstance(question, MultipleChoiceQuestion):
            report = await bot._run_forecast_on_multiple_choice(question, research)
            forecast = {
                "probabilities": {
                    opt.option_name: opt.probability
                    for opt in report.prediction_value.predicted_options
                }
            }
        elif isinstance(question, DateQuestion):
            report = await bot._run_forecast_on_date(question, research)
            forecast = {
                "percentiles": [
                    {
                        "percentile": p.percentile,
                        "date": datetime.fromtimestamp(
                            p.value, tz=timezone.utc
                        ).isoformat(),
                    }
                    for p in report.prediction_value.declared_percentiles
                ]
            }
        elif isinstance(question, NumericQuestion):
            report = await bot._run_forecast_on_numeric(question, research)
            forecast = {
                "percentiles": [
                    {"percentile": p.percentile, "value": p.value}
                    for p in report.prediction_value.declared_percentiles
                ]
            }
        else:
            continue
        results.append(
            {
                "question": question.question_text,
                "type": type(question).__name__,
                "forecaster": bot._forecaster.name,
                "forecast": forecast,
                "rationale": report.reasoning,
            }
        )
    return results


def main() -> int:
    questions = load_fixture_questions()
    results = asyncio.run(run_pipeline(questions))
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Dry-run on synthetic fixture questions (fixtures/questions.json). "
            "Nothing was posted to Metaculus. The live API requires "
            "authentication for reads, as the probe below shows."
        ),
        "unauthenticated_api_probe": probe_unauthenticated_api(),
        "sample_forecasts": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUTPUT_PATH} with {len(results)} sample forecasts.")
    for item in results:
        print(f"- [{item['type']}] {item['question'][:70]}... -> {item['forecast']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
