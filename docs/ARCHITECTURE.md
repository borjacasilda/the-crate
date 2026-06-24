# Architecture

Local-first: one Python process, one local PostgreSQL + pgvector database, and an optional
local LLM (Ollama). Audio, library and prompts stay on the machine; Discogs and Resident
Advisor lookups are opt-in.

## Data flow

```
audio file ─▶ crate.py ─▶ analyze.py ─▶ database.py (PostgreSQL + pgvector)
             (ingest)     (features)    (tracks, embeddings, crates, sessions)
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
         api.py + web/                    listener.py                     mcp_server.py
   (browse, recommend, sessions,    (live recognition +              (engine exposed to
    assistant chat over SSE)         next-track recs)                  MCP agents)
```

The flat Python modules **are the engine**. `api.py` (with `web/`) and `mcp_server.py` are
thin layers over them — logic lives in the engine, not in those surfaces. (The flat layout
is intentional; modules are not packaged into subfolders, because imports and the FastAPI
asset routes depend on it.)

## Modules

| Module | Role |
|--------|------|
| `analyze.py` | Feature extraction (Essentia, the 5-level pipeline) + scoring / ranking |
| `crate.py` | The single ingest authority: standardise → store → analyse |
| `database.py` | All SQL; idempotent schema applied on startup; pgvector HNSW indexes |
| `listener.py`, `fingerprint.py`, `recorder.py` | Live recognition + recording |
| `discogs.py`, `enrich.py` | Optional Discogs metadata / cover enrichment |
| `assistant/` | PydanticAI tool-agent over Ollama + a text-RAG knowledge base |
| `api.py`, `web/` | FastAPI JSON API + vanilla-JS UI (Live Mode, assistant over SSE) |
| `mcp_server.py` | The engine exposed over the Model Context Protocol |
| `config.py` | Single source of truth for paths, knobs and env vars |

## Key concepts

**Single ingest authority (`crate.py`).** Everything entering the library is standardised
to a 120 s / 16 kHz `crate/<id>.wav`, then analysed. Nothing else inserts tracks. A "crate"
is a *logical* collection (a row in `crates`); all audio lives in `crate/` regardless, and
membership is tracked by `tracks.crate_id` / `crate_tracks`.

**Five-level analysis pipeline, graceful degradation.** Level 1 is classic DSP
(tempo / key / energy / timbre) and always works. Levels 2–5 add EffNet embeddings, mood and
genre and require the ML models in `models/`. Recommendations and harmonic matching still
work at level 1 (key + tempo). `THECRATE_MAX_LEVEL` caps the level.

**Two vector spaces, never mixed.** A 1280-D EffNet *audio* embedding (how a track *sounds*,
and the fallback for live recognition) and a 768-D *text* embedding (the assistant's
knowledge base). Both are stored in pgvector with HNSW indexes and use two-stage retrieval:
an approximate nearest-neighbour shortlist, then an exact re-score.

**Recommendation score** = an immutable EffNet base similarity multiplied by tunable
modifiers (BPM proximity, harmonic / Camelot relationship, energy direction, a "mixable"
tempo window, mood and density). The mode (`safe` / `balanced` / `creative`) and an energy
target shift the weights. Every modifier has a floor, so no single one can veto an otherwise
strong match.

**Live recognition is a pluggable recogniser chain:** a landmark-hash fingerprint (exact,
instant) → an EffNet nearest-neighbour match (a fuzzy fallback for pitched or degraded
playback) → a recommend-only pass when nothing is recognised.

## Design principles

- **Local-first & private** — your data and prompts stay on your machine.
- **Graceful degradation** — every ML level, every external API, and the LLM is optional;
  the layer below still works without it.
- **Isolated integrations** — one module per external service; an outage there never breaks
  a core feature.
- **Idempotent storage** — the schema and the ingest are safe to re-run.

> The full HTTP API is documented in [API.md](API.md), and is auto-generated as interactive
> Swagger at `/docs` whenever the server is running.
