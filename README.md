# The Crate

**A local audio-analysis and DJ-recommendation engine for electronic music.**
Analyse a record collection, find tracks that mix well together, recognise what is
playing live, and explore it all through a local web UI and an AI assistant.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> Runs entirely on your machine. No track audio, recommendations, or AI prompts
> leave your computer (Discogs/Resident Advisor lookups are opt-in and only send
> the search terms you ask for).

---

## What it is

The Crate turns a folder of records into a queryable, mixable library. It extracts
musical features from each track with [Essentia](https://essentia.upf.edu/) — tempo,
key (Camelot), energy, timbre, and (with the ML models) mood and genre embeddings —
then scores how well any two tracks transition. On top of that engine it offers:

- a **web UI** (FastAPI + vanilla JS) to browse crates, build sets, and run a live session,
- a **Live Mode** that recognises the record currently playing (Shazam-style audio
  fingerprinting, with an embedding fallback) and recommends what to play next,
- an **AI assistant** (a local LLM via [Ollama](https://ollama.com/)) that answers
  questions about the collection and the scene, and
- an **MCP server** that exposes the analysis engine to any MCP-compatible agent.

It is built for a DJ or collector who wants their own crate to be smart, fully local,
and private.

## Features

- **5-level analysis pipeline** that degrades gracefully — classic DSP at level 1,
  EffNet embeddings + mood + genre at level 5 — so it works with or without the ML models.
- **Harmonic + tempo recommendations**: "what mixes into this?" by Camelot key, BPM,
  energy direction, and EffNet "musical-world" similarity.
- **Live recognition**: landmark-hash fingerprinting first (exact, instant), embedding
  nearest-neighbour as a fallback for pitched/degraded playback.
- **Crates** ("maletas"): organise tracks into sets; one master library holds everything.
- **Sessions**: log a live set, review it, and rate each mix.
- **Discogs enrichment** (optional): cover art, label, year, styles.
- **Local AI assistant** (optional): a terse, data-engine-style curator over your library.
- **MCP server**: 27 tools exposing per-track analysis, harmonic matching, and more.

## How it works

The Crate is built around one idea: **your records should know how they fit together.**

**1 — Ingest & analysis (the stages).** Every track you add is standardised to a short
120-second, 16 kHz excerpt and analysed in up to five levels that *degrade gracefully* — if
the ML models aren't installed, the lower levels still run:

| Level | What it extracts | Needs models? |
|-------|------------------|:---:|
| 1 | Tempo (BPM), musical key in **Camelot** notation, energy, timbre | No — always runs |
| 2–3 | EffNet audio embedding — a 1280-D "fingerprint" of how it *sounds* | Yes |
| 4–5 | Mood and genre descriptors | Yes |

A fresh install with no models still gives you harmonic (key + tempo) mixing; adding the
models unlocks true "sounds-like" matching.

**2 — Recommendation (what mixes into this?).** Given a track, The Crate ranks the rest of
your crate by one score:

> base "sounds-like" similarity **×** harmonic/Camelot fit **×** tempo proximity **×** energy
> direction **×** mood / density

You steer it with a **mode** — `safe` (tight, obvious), `balanced` (default) or `creative`
(wider, more surprising) — and an **energy target** to lift or drop. Every factor has a
floor, so no single mismatch (a slightly off BPM, a neighbouring key) can veto an otherwise
excellent transition: you get a ranked shortlist, not one "correct" answer.

**3 — Live recognition.** In Live Mode it listens to what's playing and identifies it with a
recogniser chain — an exact **audio fingerprint** first (instant), then an **embedding
nearest-neighbour** fallback for pitched or noisy signals — then recommends the next track
with the same scoring.

**4 — Two separate "brains".** Recommendations ride on the 1280-D *audio* embedding (how a
track sounds); the AI assistant answers from a separate 768-D *text* embedding (a knowledge
base you feed it). They are never mixed. Full detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); worked use cases in [docs/EXAMPLES.md](docs/EXAMPLES.md).

## Architecture

The engine is a set of focused Python modules; the web UI and the MCP server are
thin layers over them.

| Layer | Files | Role |
|-------|-------|------|
| **Analysis engine** | `analyze.py` | Feature extraction, scoring, recommendations (Essentia) |
| **Ingest / library** | `crate.py`, `database.py` | The single ingest authority; PostgreSQL + pgvector store |
| **Live recognition** | `listener.py`, `fingerprint.py`, `recorder.py` | Capture + recognise + log a session |
| **Enrichment** | `discogs.py`, `enrich.py` | Optional Discogs metadata + cover art |
| **Web API + UI** | `api.py`, `web/` | FastAPI app serving JSON + the browser UI |
| **AI assistant** | `assistant/` | Local-LLM tool agent (PydanticAI → Ollama) + RAG |
| **Agent surface** | `mcp_server.py` | MCP server exposing the engine to external agents |
| **Config** | `config.py` | Single source of truth for paths, knobs, and env vars |

Two vector spaces back the recommendations: a 1280-D EffNet audio embedding (how a
track *sounds*) and a 768-D text embedding (the knowledge base), kept separate and
indexed with pgvector HNSW. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

Python ≥ 3.11 · [uv](https://docs.astral.sh/uv/) · FastAPI · Essentia / Essentia-TensorFlow ·
PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector) (Docker) ·
PydanticAI + [Ollama](https://ollama.com/) (optional) · MCP.

## Quick start

```bash
git clone https://github.com/borjacasilda/the-crate.git
cd the-crate
uv sync                              # install dependencies into .venv
cp .env.example .env                 # then set a Postgres password
docker compose up -d                 # start PostgreSQL + pgvector
uv run python analyze.py download    # fetch the Essentia ML models (optional, ~GBs)
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/crates>. **Full, step-by-step instructions (including
the AI assistant and troubleshooting) are in [SETUP.md](SETUP.md).**

## Configuration

A fresh clone runs with the template as-is: `cp .env.example .env` and you're ready for
local development — no editing required to get started. Beyond that:

- **`POSTGRES_PASSWORD`** — the one value you should set to your own strong string before
  any real use. It's required (everything is stored in this database); the template ships a
  placeholder so the app still starts out of the box.
- **`DISCOGS_ACCESS_TOKEN`** *(optional)* — only for cover-art / label enrichment. Get one
  at <https://www.discogs.com/settings/developers>; the app runs fine without it.
- **No cloud accounts, paid services or external API keys are required.** The AI assistant
  runs locally through [Ollama](https://ollama.com/) — nothing leaves your machine.

`.env` is git-ignored, so your credentials never reach the repository. All other settings
have sensible defaults (see the commented knobs in `.env.example`).

## Documentation

- **[SETUP.md](SETUP.md)** — get it running on your machine, step by step (macOS).
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the pieces fit together.
- **[docs/API.md](docs/API.md)** — the HTTP API (also live at `/docs` once running).
- **[docs/EXAMPLES.md](docs/EXAMPLES.md)** — common workflows, end to end.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — dev workflow, tests, conventions.
- **[LEGAL.md](LEGAL.md)** — third-party APIs, licences, and attributions.

## Testing

```bash
uv run python scripts/verify.py            # environment + dependency + DB check
uv run python ab_tests/test_scoring.py     # scoring engine (no DB needed)
uv run python ab_tests/test_assistant.py   # assistant wiring (mock LLM)
```

Suites that need the database skip cleanly when it is down. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) — see [LEGAL.md](LEGAL.md) for third-party attributions (Discogs,
Resident Advisor, Essentia models).
