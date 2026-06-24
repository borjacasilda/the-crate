# HTTP API

The web layer is a FastAPI app (`api.py`). When the server is running, the **full,
authoritative, interactive reference is auto-generated**:

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

Use those as the source of truth. Pages that serve both HTML and JSON switch on the
`Accept` header (`Accept: application/json` returns data). Below is a map of the main route
groups.

## Pages (HTML)

| Route | Page |
|-------|------|
| `GET /crates` | Collection (all crates) |
| `GET /crate?id=<id>` | One crate: tracks, On-Spot pile, management |
| `GET /listen` | Live Mode dashboard |
| `GET /sessions`, `GET /session?id=<id>` | Saved sessions + detail |
| `GET /knowledge`, `GET /discogs`, `GET /record` | Knowledge base, enrichment, recording |

## Library & tracks

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` / `POST` | `/crates` | List / create crates |
| `PATCH` / `DELETE` | `/crates/{id}` | Rename / delete a crate (tracks re-home to default) |
| `GET` | `/tracks?crate=<id>` | A crate's tracks with features |
| `PATCH` | `/tracks/{id}` | Edit filename / label / on-spot flag |
| `POST` | `/tracks/add-to-crate`, `/tracks/remove-from-crate`, `/tracks/delete` | Membership / deletion |
| `POST` | `/import` | Upload + ingest audio files |
| `GET` | `/tracks/{id}/audio`, `/waveform`, `/cover` | Stream preview / waveform / cover art |

## Recommendations & analysis

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/affinity?track_id=<id>` | Cosine similarity of every crate track vs one |
| `GET` | recommendation routes (see `/docs`) | Next-track suggestions with mode + energy target |

## Live Mode

Start/stop a live session, stream recognition + next-track recommendations, and persist the
result. See the `/listen` page wiring and `/docs` for the exact routes.

## Knowledge base & reference sites

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/kb/sources` | Registered reference websites |
| `POST` | `/kb/source` | Add a reference website (URL + topic) |
| `DELETE` | `/kb/sources/{id}` | Remove a reference website |

## Assistant

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/assistant/status` | Ollama up?, active model, RAM, model catalog |
| `POST` | `/assistant/model` | Switch the active chat model (must be pulled) |
| `POST` | `/assistant/pull` | Download a model (streamed progress) |
| `GET` / `PATCH` | `/assistant/profile` | Location + Live-Mode re-check toggle |
| `POST` | `/assistant/confirm-recs` | Agent re-check of the current live picks |
| `POST` | `/chat` | Stream the assistant's answer (SSE) |

> This overview is intentionally compact. For request/response schemas and every route,
> open `/docs` while the server runs.
