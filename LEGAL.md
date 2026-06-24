# Legal & attributions

The Crate is released under the [MIT License](LICENSE). This document covers the
third-party software, models, and APIs it relies on.

## Dependencies

Python dependencies are declared in [`pyproject.toml`](pyproject.toml) and pinned in
`uv.lock`. To list them with their installed versions:

```bash
uv pip list
```

Key components and their licences:

| Component | Purpose | License |
|-----------|---------|---------|
| [Essentia](https://essentia.upf.edu/) / Essentia-TensorFlow | Audio analysis & ML models | AGPL-3.0 (library) |
| [FastAPI](https://fastapi.tiangolo.com/) / [Uvicorn](https://www.uvicorn.org/) | Web API server | MIT / BSD-3 |
| [pgvector](https://github.com/pgvector/pgvector) | Vector similarity in PostgreSQL | PostgreSQL License |
| [PydanticAI](https://ai.pydantic.dev/) | LLM tool-agent framework | MIT |
| [librosa](https://librosa.org/), [soundfile](https://github.com/bastibe/python-soundfile), [sounddevice](https://python-sounddevice.readthedocs.io/) | Audio I/O / DSP | ISC / BSD |
| [MCP SDK](https://modelcontextprotocol.io/) | Agent tool server | MIT |

> **Note on Essentia:** the Essentia library is AGPL-3.0. Pretrained models carry
> their own licences (mostly CC BY-NC-SA from the MTG). They are downloaded at runtime
> into `models/` and are **not** redistributed in this repository. Review the model
> licences at <https://essentia.upf.edu/models.html> before any commercial use.

## External APIs (all optional, opt-in)

### Discogs
- Docs: <https://www.discogs.com/developers/>
- Used for: cover art, label, year, and styles enrichment.
- Auth: a personal access token via `.env` (`DISCOGS_ACCESS_TOKEN`).
- Attribution: this project uses the Discogs API; Discogs® is a trademark of its owner.
- Rate limits: ~60 requests/minute authenticated.

### Resident Advisor
- Used for: live events and artist/label lookups.
- RA has **no official public API**; The Crate queries RA's own GraphQL endpoint and
  falls back to a domain-scoped web search. This is best-effort and can break if RA
  changes; treat it as informational. Respect RA's terms of service.

### Ollama (local)
- Docs: <https://ollama.com/>
- Runs language models entirely on your machine; no data leaves your computer.

## Privacy

The Crate is local-first. Your audio, library, sessions, and AI prompts stay on your
machine. The only outbound traffic is the optional Discogs / Resident Advisor /
web-search lookups, which send just the search terms for the request you make.
