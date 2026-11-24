"""Configuration for the Metaculus FutureEval bot.

All secrets are read from environment variables / the project ``.env`` file.
Nothing is hardcoded. If ``METACULUS_API_KEY`` is missing, loading the config
fails with a clear, actionable message (unless explicitly told not to require
it, e.g. for offline dry-runs on fixture questions).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_DIR / ".env"

# Slugs used only if the forecasting-tools client constants are unavailable.
FALLBACK_TOURNAMENT_SLUGS = ["fall-futureeval-2026", "minibench"]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass
class LlmConfig:
    """An OpenAI-compatible chat-completions endpoint."""

    model: str
    api_key: str
    base_url: str

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass
class Config:
    metaculus_api_key: str | None
    llm: LlmConfig | None
    asknews_client_id: str | None
    asknews_secret: str | None
    poll_interval_seconds: int = 300
    tournament_slugs: list[str] = field(default_factory=list)

    @property
    def asknews_configured(self) -> bool:
        return bool(self.asknews_client_id and self.asknews_secret)

    @property
    def llm_configured(self) -> bool:
        return self.llm is not None


def _clean(value: str | None) -> str | None:
    """Return the stripped value, treating blanks and placeholders as unset."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() in {"REPLACE_ME", "TODO", "NONE"}:
        return None
    return value


def _load_llm_config() -> LlmConfig | None:
    """Resolve an LLM endpoint from the environment, in priority order.

    1. Explicit generic settings: LLM_MODEL + LLM_API_KEY (+ LLM_BASE_URL).
    2. OPENROUTER_API_KEY (Metaculus offers free OpenRouter credits for the
       seasonal tournament via the participation form).
    3. OPENAI_API_KEY.
    4. ANTHROPIC_API_KEY (via Anthropic's OpenAI-compatible endpoint).
    Returns None when no LLM key is available; the bot then uses the
    deterministic fallback forecaster.
    """
    model = _clean(os.getenv("LLM_MODEL"))
    api_key = _clean(os.getenv("LLM_API_KEY"))
    base_url = _clean(os.getenv("LLM_BASE_URL"))
    if api_key:
        return LlmConfig(
            model=model or "openrouter/openai/gpt-4o",
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
        )

    openrouter_key = _clean(os.getenv("OPENROUTER_API_KEY"))
    if openrouter_key:
        return LlmConfig(
            model=model or "openrouter/openai/gpt-4o",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )

    openai_key = _clean(os.getenv("OPENAI_API_KEY"))
    if openai_key:
        return LlmConfig(
            model=model or "gpt-4o",
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
        )

    anthropic_key = _clean(os.getenv("ANTHROPIC_API_KEY"))
    if anthropic_key:
        return LlmConfig(
            model=model or "claude-sonnet-4-5",
            api_key=anthropic_key,
            base_url="https://api.anthropic.com/v1",
        )

    return None


def load_config(require_metaculus_key: bool = True) -> Config:
    """Load configuration from ``.env`` (if present) and the environment."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()

    metaculus_key = _clean(os.getenv("METACULUS_API_KEY")) or _clean(
        os.getenv("METACULUS_TOKEN")
    )
    if require_metaculus_key and not metaculus_key:
        raise ConfigError(
            "METACULUS_API_KEY is not set, so the bot cannot read questions "
            "or post forecasts.\n"
            f"Fix: create {ENV_PATH} containing a line\n"
            "    METACULUS_API_KEY=<your bot token>\n"
            "Get the token from your bot's settings page on Metaculus "
            "(see README.md -> 'Remaining user steps')."
        )
    if metaculus_key:
        # forecasting-tools' MetaculusClient reads the METACULUS_TOKEN env var.
        os.environ.setdefault("METACULUS_TOKEN", metaculus_key)

    raw_interval = _clean(os.getenv("POLL_INTERVAL_MINUTES"))
    poll_interval_seconds = 300
    if raw_interval:
        try:
            poll_interval_seconds = max(60, int(float(raw_interval) * 60))
        except ValueError as exc:
            raise ConfigError(
                f"POLL_INTERVAL_MINUTES={raw_interval!r} is not a number."
            ) from exc

    raw_tournaments = _clean(os.getenv("TOURNAMENTS"))
    if raw_tournaments:
        tournament_slugs = [s.strip() for s in raw_tournaments.split(",") if s.strip()]
    else:
        tournament_slugs = list(FALLBACK_TOURNAMENT_SLUGS)

    return Config(
        metaculus_api_key=metaculus_key,
        llm=_load_llm_config(),
        asknews_client_id=_clean(os.getenv("ASKNEWS_CLIENT_ID")),
        asknews_secret=_clean(os.getenv("ASKNEWS_SECRET")),
        poll_interval_seconds=poll_interval_seconds,
        tournament_slugs=tournament_slugs,
    )
