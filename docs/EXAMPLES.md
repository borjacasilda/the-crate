# Examples & common workflows

End-to-end walkthroughs of what The Crate is actually for. They assume you've finished
[SETUP.md](../SETUP.md) and have the server running:

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/crates>.

---

## 1. Turn a folder of records into a smart library

1. Open **Collection** (`/crates`). Everything you import lives in one master library;
   "crates" are just named sub-collections (your *maletas*).
2. Import audio (drag-and-drop in the UI, or the `/import` endpoint). On ingest, each file
   is standardised to a 120 s / 16 kHz excerpt and analysed:
   - **Level 1 (always):** tempo (BPM), musical key in **Camelot** notation, energy, timbre.
   - **Levels 2–5 (if the ML models are present):** EffNet audio embeddings, mood and genre.
3. Browse the crate: every track now shows BPM, key and energy, and is ready to be matched.

> No models downloaded? It still works — you get tempo/key/energy and harmonic matching.
> Run `uv run python analyze.py download` to unlock "sounds-like" recommendations.

## 2. "What mixes into this track?"

The core question. From a track, ask for the next one:

- **Harmonic + tempo:** candidates are ranked by Camelot compatibility, BPM proximity and
  energy direction (lift / hold / drop).
- **"Sounds-like" (with models):** the EffNet embedding finds tracks in the same
  *musical world*, beyond just key and tempo.
- **Modes:** `safe` (tight, obvious blends), `balanced` (default), `creative` (wider,
  more surprising). An energy target nudges the picks up or down.

Every factor has a floor, so one mismatched dimension never vetoes an otherwise great
transition — you get a ranked shortlist, not a single "correct" answer.

## 3. Live Mode — recognise what's playing and get the next track

Open **Live Mode** (`/listen`), pick an input (audio interface > built-in > iPhone), and
start a set:

1. The Crate listens and recognises the record currently playing — first by **audio
   fingerprint** (exact, instant), then by **embedding nearest-neighbour** if the signal is
   pitched or noisy.
2. Once a track is locked, it recommends what to play next from your crate, with the
   harmonic/energy reasoning above.
3. Optional **AI re-check:** a one-line confirmation from the local LLM on whether the picks
   are solid and which is strongest.

## 4. Record, review and rate a session

- Use **Record** (`/record`) to capture a full take, or let Live Mode log the set.
- Review it under **Sessions** (`/sessions`): the tracklist it recognised, in order.
- Rate each transition so the history reflects what actually worked.

## 5. Enrich with Discogs (optional)

Open **Discogs** (`/discogs`) with a `DISCOGS_ACCESS_TOKEN` set to pull **cover art, label,
year and styles**. Enrichment also unlocks *label-level* similarity ("labels like Token")
because a label's sound is the average of its tracks' embeddings.

## 6. Ask the local AI assistant

A terse, data-engine-style curator over your library and the live scene (a local LLM via
Ollama — nothing leaves your machine). One tool-calling agent routes each question:

| You ask… | It uses… |
|----------|----------|
| "recommend something like *Kwartz – Impulse*" | audio similarity over your crate |
| "artists like Oscar Mulero", "labels like Token" | artist / label centroid similarity |
| "tracks 130–136 BPM in 8A" | catalogue metadata filter |
| "who is Dasha Rush?", "history of the Birmingham sound" | your ingested knowledge base (RAG) |
| "where is Surgeon playing?", "events in Berlin this weekend" | live web (Resident Advisor) |
| "is the new Lewis Fautzi in stock?" | live record-shop check + Discogs marketplace |

Add your own reference websites on the **Knowledge** page (`/knowledge`); the assistant can
search them too. Out-of-scope (non-music) questions are politely refused.

## 7. Use it from another agent (MCP)

`mcp_server.py` exposes the analysis engine over the **Model Context Protocol**, so any
MCP-compatible client can call per-track analysis, harmonic matching and recommendations as
tools — the same engine, a different surface. See [ARCHITECTURE.md](ARCHITECTURE.md).
