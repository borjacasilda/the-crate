"""
assistant/embed_text.py — text embeddings for the knowledge-base RAG.

Wraps Ollama's /api/embed with nomic-embed-text (768-D). This is the TEXT
vector space, kept entirely separate from the 1280-D AUDIO space: documents and
questions live here, tracks/artists live there, and the two never mix.

nomic-embed-text is a task-prefixed model: documents must be embedded with a
"search_document:" prefix and queries with "search_query:". Honouring that
convention is the single biggest lever on retrieval quality, so it is baked in
here rather than left to callers.
"""
import httpx

from assistant import models as registry
from assistant import ollama_client

EMBED_MODEL = registry.EMBED_MODEL          # "nomic-embed-text"
EMBED_DIM = 768
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _embed(inputs: list, timeout: float = 120.0) -> list:
    """Call Ollama's batch embed endpoint; return one vector per input.

    Raises RuntimeError("ollama-down") / ("embed-model-missing") so the caller
    can surface an actionable message instead of a raw stack trace.
    """
    if not inputs:
        return []
    if not ollama_client.is_up():
        raise RuntimeError("ollama-down")
    if not ollama_client.has_model(EMBED_MODEL):
        raise RuntimeError("embed-model-missing")
    r = httpx.post(f"{ollama_client.OLLAMA_HOST}/api/embed",
                   json={"model": EMBED_MODEL, "input": inputs}, timeout=timeout)
    r.raise_for_status()
    vecs = r.json().get("embeddings") or []
    if len(vecs) != len(inputs):
        raise RuntimeError(f"embed-count-mismatch:{len(vecs)}!={len(inputs)}")
    return vecs


def embed_documents(texts: list) -> list:
    """Embed chunks for storage (search_document: prefix)."""
    return _embed([_DOC_PREFIX + t for t in texts])


def embed_query(text: str) -> list:
    """Embed a single question for retrieval (search_query: prefix)."""
    return _embed([_QUERY_PREFIX + text])[0]
