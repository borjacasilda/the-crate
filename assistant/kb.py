"""
assistant/kb.py — knowledge-base ingestion pipeline (Phase 2 — RAG).

Turns a document the user picked on their own machine into retrievable
knowledge, in five steps:

    extract text → MUSIC-RELEVANCE GATE → chunk → embed (nomic) → store

The gate is the hard requirement: before anything is written to the index, the
local LLM (the same Qwen we already run) classifies whether the text is about
music. If it is not, ingestion is refused and nothing lands in the RAG — the
knowledge base stays a clean, on-topic corpus.

Everything is local: text extraction is pure-Python, embeddings come from
Ollama, and the gate reuses the on-device model. No data leaves the machine.
"""
import functools
import hashlib
import io
import re

from assistant import agent as assistant_agent
from assistant import embed_text, ollama_client

# Chunking: ~1200 chars (~300 tokens) with 200-char overlap so a fact split
# across a boundary still survives in at least one chunk.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
# How much of a document the gate actually reads (head + a middle slice). Enough
# to judge the topic without paying to run the whole thing through the LLM.
GATE_HEAD = 4000
GATE_MID = 2000

# Suggested categories — a SEED list, not a fixed enum. The classifier is told to
# pick the best fit OR propose a new short kebab-case one; the user can edit it
# afterwards (auto + editable). This keeps the vocabulary consistent without ever
# closing it to new kinds of knowledge (books, scenes, gear, theory…).
SUGGESTED_CATEGORIES = [
    "artist", "dj", "producer", "label", "genre", "music-theory", "history",
    "book", "scene", "club", "festival", "gear", "interview", "review",
    "tutorial", "liner-notes", "discography", "general",
]


def _norm_category(c: str) -> str:
    """Normalise a category to lowercase kebab-case so 'DJ', 'dj' and 'D J' merge."""
    c = (c or "").strip().lower()
    c = re.sub(r"[\s_]+", "-", c)
    c = re.sub(r"[^a-z0-9-]", "", c).strip("-")
    return c or "general"


def _norm_tags(tags) -> list:
    """Clean a tag list: lowercase, de-duplicate, cap at 8."""
    out, seen = [], set()
    for t in tags or []:
        t = re.sub(r"\s+", " ", str(t).strip().lower())
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out[:8]


# ── text extraction ───────────────────────────────────────────────────────────
def extract_text(data: bytes, filename: str) -> str:
    """Pull plain text out of an uploaded file by extension.

    Supports .txt/.md (decode), .pdf (pypdf), .docx (python-docx). Unknown types
    are tried as UTF-8 text. Raises ValueError when nothing readable comes out.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        text = _extract_pdf(data)
    elif name.endswith(".docx"):
        text = _extract_docx(data)
    else:                                    # .txt/.md/.markdown/unknown
        text = data.decode("utf-8", errors="ignore")
    text = _normalise(text)
    if not text.strip():
        raise ValueError("no readable text could be extracted from this file")
    return text


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def _normalise(text: str) -> str:
    """Collapse runaway whitespace; keep paragraph breaks for chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_like_filename(s: str) -> bool:
    """True for raw filenames (so we prefer the LLM's clean title over them)."""
    return bool(re.search(r"\.(txt|md|markdown|pdf|docx)$", s.strip(), re.I))


# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list:
    """Split into overlapping chunks, preferring paragraph then sentence breaks
    so chunks read as coherent passages rather than mid-word cuts."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            brk = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if brk > size * 0.5:             # only honour a break past the halfway
                end = start + brk + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── music-relevance gate ──────────────────────────────────────────────────────
class _Relevance:
    """Plain result holder (kept import-light; PydanticAI model built lazily).
    Carries both the gate decision AND the auto-classification."""
    def __init__(self, is_music_related: bool, confidence: float, reason: str,
                 category: str = "general", tags: list = None, title: str = ""):
        self.is_music_related = is_music_related
        self.confidence = confidence
        self.reason = reason
        self.category = category
        self.tags = tags or []
        self.title = title


_GATE_PROMPT = (
    "You are the gatekeeper AND librarian for a knowledge base specialised in "
    "electronic music and DJ culture. In ONE pass, decide whether the text is "
    "primarily about music, and if so, classify it.\n\n"
    "ACCEPT if it is about: artists/producers/DJs, tracks/releases/albums, record "
    "labels, genres and scenes, clubs/festivals/venues, music history and culture, "
    "music production, gear/synths/drum machines, DJing/mixing, or music theory.\n"
    "REJECT anything not primarily about music (politics, sports, cooking, generic "
    "programming, business, science unrelated to sound, personal notes, etc.). Be "
    "strict: if it is not clearly and mostly about music, reject it.\n\n"
    "When ACCEPTED, also return:\n"
    "- category: the SINGLE best fit. Prefer one of these suggested categories: "
    + ", ".join(SUGGESTED_CATEGORIES) + ". If none truly fits, propose a new short "
    "lowercase kebab-case category (e.g. 'sound-design').\n"
    "- tags: 1-5 short lowercase keywords (artists, places, eras, techniques).\n"
    "- title: a concise human-readable title for the document.\n\n"
    "Return is_music_related, confidence 0.0-1.0, a short reason, category, tags, "
    "title. For rejected text, category/tags/title may be empty."
)


def _gate_sample(text: str) -> str:
    """A representative slice of the document for the classifier to judge."""
    if len(text) <= GATE_HEAD + GATE_MID:
        return text
    mid = len(text) // 2
    return text[:GATE_HEAD] + "\n…\n" + text[mid:mid + GATE_MID]


@functools.lru_cache(maxsize=4)
def _gate_agent(model_tag: str):
    from pydantic import BaseModel, Field, field_validator
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    class MusicRelevance(BaseModel):
        is_music_related: bool
        confidence: float = Field(ge=0.0, le=1.0)
        reason: str
        category: str = "general"
        tags: list[str] = Field(default_factory=list)
        title: str = ""

        @field_validator("confidence", mode="before")
        @classmethod
        def _clamp_confidence(cls, v):
            # Treat confidence as a MAGNITUDE of certainty and clamp to [0,1].
            # Smaller local models (notably the instruct builds) sometimes encode
            # "definitely NOT music" as a negative score (-1.0); abs() maps that to
            # high certainty in the rejection, and clamping guards any overshoot —
            # so a quirky value degrades gracefully instead of exhausting retries.
            try:
                return min(1.0, abs(float(v)))
            except (TypeError, ValueError):
                return 0.0

    model = OpenAIChatModel(
        model_tag,
        provider=OpenAIProvider(base_url=ollama_client.OPENAI_BASE, api_key="ollama"),
    )
    return Agent(model, output_type=MusicRelevance, system_prompt=_GATE_PROMPT,
                 retries=2)


async def is_music_related(text: str, model_tag: str = None) -> "_Relevance":
    """Ask the local LLM whether `text` belongs in a music knowledge base.

    Raises RuntimeError("ollama-down") / ("model-missing:tag") so the API can
    return an actionable message. The decision itself never raises — a model
    that fails to answer is treated as a rejection by the caller.
    """
    if not ollama_client.ensure_up():            # start Ollama on demand if it was quit
        raise RuntimeError("ollama-down")
    tag = model_tag or assistant_agent.active_model()
    if not ollama_client.has_model(tag):
        raise RuntimeError(f"model-missing:{tag}")
    agent = _gate_agent(tag)
    result = await agent.run("TEXT TO CLASSIFY:\n\n" + _gate_sample(text))
    o = result.output
    return _Relevance(bool(o.is_music_related), float(o.confidence), o.reason,
                      category=_norm_category(o.category), tags=_norm_tags(o.tags),
                      title=(o.title or "").strip())


# ── full ingest pipeline ──────────────────────────────────────────────────────
class GateRejected(Exception):
    """Raised when the music-relevance gate refuses a document."""
    def __init__(self, reason: str, confidence: float):
        super().__init__(reason)
        self.reason = reason
        self.confidence = confidence


class KbFull(Exception):
    """Raised when the knowledge base is at capacity. The user must delete some
    documents before adding new ones — keeps the rudimentary RAG from bloating."""
    def __init__(self, chunks: int, cap: int):
        super().__init__(f"knowledge base full ({chunks}/{cap} chunks)")
        self.chunks = chunks
        self.cap = cap


async def ingest_text(text: str, title: str = None, source_type: str = "upload",
                      source_url: str = None, category: str = None,
                      model_tag: str = None) -> dict:
    """Gate+classify → chunk → embed → store. Returns a summary dict.

    The single LLM pass both gates (music?) and classifies (category, tags, a
    clean title). `category`, when given, forces the category (manual override);
    otherwise the auto-classification is used (auto + editable later). Raises
    GateRejected when not music-related (nothing is stored), ValueError on
    empty/duplicate content.
    """
    import config
    import database

    # Capacity guard FIRST — before the LLM gate and embedding spend. When the KB
    # is full we refuse outright so the user frees space instead of growing it.
    used = database.kb_stats().get("chunks", 0)
    if used >= config.KB_MAX_CHUNKS:
        raise KbFull(used, config.KB_MAX_CHUNKS)

    text = _normalise(text)
    if len(text.strip()) < 40:
        raise ValueError("text is too short to ingest")

    chash = content_hash(text)
    existing = database.kb_document_by_hash(chash)
    if existing:
        raise ValueError("this exact document is already in the knowledge base")

    verdict = await is_music_related(text, model_tag=model_tag)
    if not verdict.is_music_related:
        raise GateRejected(verdict.reason, verdict.confidence)

    # Title: a clean user/paste title wins; otherwise use the LLM's suggestion.
    final_title = (title or "").strip()
    if not final_title or _looks_like_filename(final_title):
        final_title = verdict.title or final_title or "Untitled"
    final_category = _norm_category(category) if category else verdict.category

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("no chunks produced from text")
    vectors = embed_text.embed_documents(chunks)
    payload = [{"text": c, "embedding": v, "token_count": _est_tokens(c)}
               for c, v in zip(chunks, vectors)]
    meta = {"gate_reason": verdict.reason, "suggested_title": verdict.title,
            "suggested_category": verdict.category}
    doc_id = database.insert_kb_document(
        title=final_title, chunks=payload, model_name=embed_text.EMBED_MODEL,
        source_type=source_type, source_url=source_url, content_hash=chash,
        category=final_category, tags=verdict.tags, meta=meta)
    return {"doc_id": doc_id, "title": final_title, "n_chunks": len(chunks),
            "category": final_category, "tags": verdict.tags,
            "confidence": verdict.confidence, "reason": verdict.reason}


async def ingest_file(data: bytes, filename: str, title: str = None,
                      model_tag: str = None) -> dict:
    """Extract text from an uploaded file, then run the text ingest pipeline."""
    text = extract_text(data, filename)
    return await ingest_text(text, title=title or filename,
                             source_type="upload", model_tag=model_tag)
