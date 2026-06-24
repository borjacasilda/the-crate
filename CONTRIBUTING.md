# Contributing

Thanks for your interest in The Crate. This is a focused, local-first project; the
notes below keep contributions consistent with the existing codebase.

## Getting set up

See [SETUP.md](SETUP.md). In short:

```bash
uv sync
cp .env.example .env          # set POSTGRES_PASSWORD
docker compose up -d
uv run python scripts/verify.py
```

## Project conventions

These are followed throughout — please match them:

- **Everything in English** — code, identifiers, comments, docstrings, UI copy, docs.
- **Comments explain the *why*, not the *what*.** Match the surrounding density and tone.
- **Isolated external clients.** Each third-party integration lives in one module and
  nobody else imports its HTTP client (`discogs.py`, `assistant/web_sources.py`,
  `assistant/vinyl_stores.py`). Degrade gracefully; never let an outage break a feature.
- **Idempotent SQL.** Schema changes are `CREATE … IF NOT EXISTS` / `ALTER … ADD COLUMN
  IF NOT EXISTS` in `database._SCHEMA_SQL`, applied on startup.
- **Type hints** use the string form for unions (`"str | None"`), matching the codebase.
- **No secrets in git.** Configuration goes through `.env` / `config.py`.

## Architecture in one line

Flat Python modules are the engine; `api.py` (+ `web/`) and `mcp_server.py` are thin
layers over them. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The module layout
is intentional — please do not relocate modules into packages without discussion, as
imports and the FastAPI asset routes depend on it.

## Tests

```bash
uv run python ab_tests/test_scoring.py      # scoring engine (no DB)
uv run python ab_tests/test_assistant.py    # assistant wiring (mock LLM, no Ollama)
uv run python ab_tests/test_api.py          # API routes (needs the database)
uv run python ab_tests/test_analysis.py     # full analysis (needs Essentia + DB)
```

Suites that need the database or the ML models skip cleanly when those are absent.
Add a test for any behaviour change and keep the relevant suite green.

## After a backend change

`api.py` builds its state at startup, so **restart `uvicorn`** after editing any
`.py` file. Edits under `web/` only need a browser refresh.

## Pull requests

1. Branch off `main`: `git checkout -b feature/your-change`.
2. Keep changes focused; update docs/tests alongside code.
3. Run `uv run python scripts/verify.py` and the relevant test suites.
4. Open a PR describing the change and how you verified it.
