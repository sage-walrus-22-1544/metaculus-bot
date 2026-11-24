"""Research step: gather news context for a question before forecasting.

If AskNews credentials are configured (ASKNEWS_CLIENT_ID / ASKNEWS_SECRET --
free for registered bot builders), research comes from AskNews via the
forecasting-tools searcher. Otherwise a clearly-marked stub is returned so
the rest of the pipeline still runs end to end.
"""

from __future__ import annotations

import logging

from .config import Config

logger = logging.getLogger(__name__)

STUB_MESSAGE = (
    "(No news research: AskNews is not configured. Set ASKNEWS_CLIENT_ID and "
    "ASKNEWS_SECRET in .env to enable news research -- it is free for "
    "registered FutureEval bot builders. This forecast relies on the "
    "question text, background, and resolution criteria only.)"
)


async def get_research(question_text: str, config: Config) -> str:
    """Return research text for a question (never raises)."""
    if not config.asknews_configured:
        return STUB_MESSAGE
    try:
        from forecasting_tools import AskNewsSearcher

        searcher = AskNewsSearcher()
        research = await searcher.call_preconfigured_version(
            "asknews/news-summaries", question_text
        )
        if research:
            return research
        logger.warning("AskNews returned empty research; using stub.")
        return STUB_MESSAGE
    except Exception as exc:  # noqa: BLE001 - research must never kill a run
        logger.warning("AskNews research failed (%s); using stub.", exc)
        return STUB_MESSAGE
