#!/usr/bin/env python3
"""Run the Metaculus FutureEval bot.

Usage:
    .venv/bin/python run.py --check-config   # validate .env, print status, exit
    .venv/bin/python run.py --once           # one pass over both tournaments
    .venv/bin/python run.py --once --dry-run # one pass, forecasts NOT posted
    .venv/bin/python run.py --loop           # poll every POLL_INTERVAL_MINUTES
                                             # (default 5) and forecast new
                                             # questions as they appear

Tournament questions are only open for ~3 hours, so during the season the
bot should run with --loop (or be invoked every few minutes by cron).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bot import FutureEvalBot
from src.config import ConfigError, load_config

logger = logging.getLogger("metaculus_bot")


def _resolve_tournament_ids(config) -> list:
    """Prefer the forecasting-tools client constants (kept current by
    Metaculus); fall back to the configured slugs."""
    if config.tournament_slugs and _tournaments_overridden():
        return config.tournament_slugs
    try:
        from forecasting_tools import MetaculusClient

        return [
            MetaculusClient.CURRENT_AI_COMPETITION_ID,  # Fall 2026 seasonal
            MetaculusClient.CURRENT_MINIBENCH_ID,  # current MiniBench round
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read client constants (%s); using slugs.", exc)
        return config.tournament_slugs


def _tournaments_overridden() -> bool:
    import os

    return bool(os.getenv("TOURNAMENTS"))


async def _run_once(bot: FutureEvalBot, tournament_ids: list) -> None:
    all_reports = []
    for tournament_id in tournament_ids:
        logger.info("Forecasting on tournament: %s", tournament_id)
        reports = await bot.forecast_on_tournament(
            tournament_id, return_exceptions=True
        )
        all_reports.extend(reports)
    bot.log_report_summary(all_reports)
    errors = [r for r in all_reports if isinstance(r, BaseException)]
    if errors:
        logger.warning("%d question(s) errored this pass.", len(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Metaculus FutureEval bot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="single pass, then exit")
    mode.add_argument("--loop", action="store_true", help="poll continuously")
    mode.add_argument(
        "--check-config",
        action="store_true",
        help="validate configuration and exit (prints no secrets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute forecasts but do not post them to Metaculus",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        config = load_config(require_metaculus_key=True)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    if args.check_config:
        print("Configuration OK.")
        print(f"  Metaculus API key: set ({len(config.metaculus_api_key)} chars)")
        print(
            "  LLM: "
            + (
                f"{config.llm.model} @ {config.llm.base_url}"
                if config.llm_configured
                else "not configured -> deterministic fallback forecaster"
            )
        )
        print(
            "  AskNews research: "
            + ("configured" if config.asknews_configured else "not configured (stub)")
        )
        print(f"  Poll interval: {config.poll_interval_seconds // 60} min")
        print(f"  Tournaments (fallback slugs): {config.tournament_slugs}")
        return 0

    tournament_ids = _resolve_tournament_ids(config)
    bot = FutureEvalBot(config, publish=not args.dry_run)
    logger.info(
        "Starting bot (publish=%s, tournaments=%s)",
        not args.dry_run,
        tournament_ids,
    )

    if args.once:
        asyncio.run(_run_once(bot, tournament_ids))
        return 0

    logger.info(
        "Loop mode: polling every %d seconds. Ctrl-C to stop.",
        config.poll_interval_seconds,
    )
    while True:
        started = time.monotonic()
        try:
            asyncio.run(_run_once(bot, tournament_ids))
        except Exception:  # noqa: BLE001 - keep polling through failures
            logger.exception("Forecast pass failed; will retry next cycle.")
        elapsed = time.monotonic() - started
        time.sleep(max(5.0, config.poll_interval_seconds - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
