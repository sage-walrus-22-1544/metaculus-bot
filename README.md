# Metaculus FutureEval Forecasting Bot

An autonomous forecasting bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/)
bot tournaments — the seasonal Bot Tournament (prize pool ~$50k, running
Sep 2026 – Jan 2027 for Fall 2026) and the biweekly MiniBench (~$1k per
round, ~60 questions). Built on Metaculus's official
[`forecasting-tools`](https://pypi.org/project/forecasting-tools/) library
(the same library the official bot template uses).

**Status:** the code is complete and tested offline. It starts forecasting
the moment a bot API token is placed in `.env` — see
[Remaining user steps](#remaining-user-steps) below.

## How it works

```
poll (every 5 min) ──► fetch open questions from the seasonal tournament
                       and MiniBench (skipping ones already forecasted)
                    ──► for each new question:
                          1. research  (AskNews if configured, else a stub)
                          2. forecast  (LLM if configured, else fallback)
                          3. post the forecast + a rationale comment
                             that states the forecast value
```

- `run.py` — entry point: `--once`, `--loop`, `--dry-run`, `--check-config`.
- `src/config.py` — loads `.env`; fails with a clear message if
  `METACULUS_API_KEY` is missing. Nothing is hardcoded.
- `src/bot.py` — a `ForecastBot` subclass. The framework handles fetching
  questions, skipping previously-forecasted ones, and posting forecasts with
  their rationale comments. Questions are only open for **~3 hours**, which
  is why the bot polls every few minutes in `--loop` mode.
- `src/research.py` — news research via AskNews (free for registered bot
  builders) when `ASKNEWS_CLIENT_ID`/`ASKNEWS_SECRET` are set; otherwise a
  clearly-marked stub so the pipeline still runs.
- `src/forecasters.py` — the forecasting brain, behind a clean interface:
  - `LlmForecaster` — calls any OpenAI-compatible chat endpoint. Model,
    key, and base URL are configurable via environment variables
    (`LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL`, or simply
    `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`).
    It reasons about base rates and scenarios, then its final answer is
    parsed into a probability / percentile distribution / option split.
  - `FallbackForecaster` — a deterministic base-rate forecaster used when
    no LLM key is configured (binary = 50%, numeric percentiles spread
    across the allowed range, multiple choice = uniform). It also acts as
    a per-question safety net if an LLM call fails. Its rationales state
    plainly that they are base-rate fallback forecasts.
- `src/formatting.py` — pure helpers enforcing Metaculus's numeric
  invariants: binary probabilities clamped to [0.01, 0.99], numeric
  percentile values monotonic, full 201-point CDFs monotonic within [0, 1],
  multiple-choice probabilities floored and normalized to sum to 1.
- Forecasts are generated entirely by the bot — no human reviews individual
  forecasts before posting — and the bot never copies the community
  prediction (it is not even fetched), as the tournament rules require.

## Remaining user steps

Only a human can do these (account creation and the participation form are
the account holder's, and the password is the user's own choice — the agent
never creates accounts or invents passwords).

1. **Create a Metaculus account** (free): https://www.metaculus.com/
   — sign up with your email and a password you choose.
2. **Create your bot**: go to
   https://www.metaculus.com/futureeval/participate/ and follow step 1 —
   it redirects you to your settings page. Under **"My Forecasting Bots"**,
   click **"Create Bot"** and give it a name.
   ⚠️ Your **first** bot is your *primary* bot — the only one eligible for
   prizes. Make sure the first bot you create is the one this code runs as.
3. **Copy the bot's API token**: on the bot's settings page, copy its
   access token / API key.
4. **Hand the token over / put it in place**: create
   `~/workspace/metaculus-bot/.env` (copy `.env.example`) with the line
   ```
   METACULUS_API_KEY=<paste the bot token here>
   ```
   (The library also accepts the name `METACULUS_TOKEN`; this project
   standardizes on `METACULUS_API_KEY` and passes it through.)
5. **Submit the mandatory participation form** (required for prize
   eligibility, and it doubles as the application for free LLM credits):
   https://forms.gle/aQdYMq9Pisrf1v7d8
6. **Recommended extras** (also in `.env`):
   - `OPENROUTER_API_KEY` — from the free credits granted via the form, or
     your own key from https://openrouter.ai/keys. Without an LLM key the
     bot still runs, but only with the base-rate fallback forecaster.
   - `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET` — AskNews is free for
     registered bot builders and materially improves research quality.

## Starting the bot

```bash
cd ~/workspace/metaculus-bot
.venv/bin/python run.py --check-config   # verify .env (prints no secrets)
.venv/bin/python run.py --once --dry-run # one pass, forecasts NOT posted
.venv/bin/python run.py --loop           # poll every 5 min and post forecasts
```

Use `--loop` during the season: MiniBench and seasonal questions launch at
random hours and stay open only ~3 hours each. Most MiniBench questions
launch in the first few days of each 2-week round.

## Development / testing

```bash
.venv/bin/python -m pytest tests/ -q   # 37 unit tests
.venv/bin/python dry_run.py            # offline end-to-end run on fixture
                                       # questions -> dry_run_output/sample_forecasts.json
```

Note: the Metaculus API requires authentication for **all** reads (verified
2026-11-24: unauthenticated requests return `403 Permission Error: The API
is only available to authenticated users...`), so live questions cannot be
listed before the token exists. The dry-run therefore uses synthetic
fixture questions in `fixtures/questions.json`, clearly labeled as such.

## Project layout

```
metaculus-bot/
├── run.py                  # CLI entry point (poll / once / check-config)
├── dry_run.py              # offline pipeline run on fixtures
├── src/
│   ├── config.py           # .env loading and validation
│   ├── bot.py              # ForecastBot subclass (research + forecast hooks)
│   ├── research.py         # AskNews research or stub
│   ├── forecasters.py      # LLM forecaster + deterministic fallback
│   └── formatting.py       # probability / CDF invariants
├── tests/                  # pytest suite (formatting, forecasters, config, bot)
├── fixtures/questions.json # synthetic questions for the dry-run
├── dry_run_output/         # sample forecasts produced by dry_run.py
├── .env.example            # copy to .env and fill in
└── requirements.txt
```
