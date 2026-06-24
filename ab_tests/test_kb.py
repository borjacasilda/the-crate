"""
test_kb.py — The Crate, Phase 2: knowledge-base RAG
===================================================
Two layers:
  • PURE units (no Ollama, no DB): chunking, text extraction, hashing.
  • E2E (needs Ollama + DB): the music-relevance GATE (accept music / reject
    non-music), full ingest → retrieve → delete, and the duplicate guard.

The E2E layer skips cleanly when Ollama is down so the unit layer always runs.

Run:  uv run python ab_tests/test_kb.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import database
from assistant import embed_text, kb, ollama_client


# ── pure units ────────────────────────────────────────────────────────────────
def test_chunking():
    short = "Jeff Mills is a Detroit techno DJ and producer of great renown."
    assert kb.chunk_text(short) == [short]                  # short → single chunk
    # a long break-free string forces fixed-size cuts WITH overlap
    flat = "abcdefghij " * 400                              # ~4400 chars, no breaks
    chunks = kb.chunk_text(flat)
    assert len(chunks) >= 3, f"expected several chunks, got {len(chunks)}"
    assert all(len(c) <= kb.CHUNK_CHARS + 5 for c in chunks)
    # consecutive chunks share a tail/head (a fact on a boundary survives)
    assert chunks[0][-20:] in chunks[1], "chunks do not overlap"
    print(f"  ✓ chunking: short→1 · long→{len(chunks)} bounded, overlapping chunks")


def test_extract_and_hash():
    txt = "Underground Resistance is a Detroit techno collective.".encode("utf-8")
    out = kb.extract_text(txt, "ur.txt")
    assert "Underground Resistance" in out
    h1, h2 = kb.content_hash("same text"), kb.content_hash("same text")
    assert h1 == h2 and h1 != kb.content_hash("other text")
    try:
        kb.extract_text(b"", "empty.txt"); assert False, "empty should raise"
    except ValueError:
        pass
    print("  ✓ extract: .txt decode · hash deterministic · empty→ValueError")


def test_normalisers():
    # open vocabulary, but consistent: 'DJ', 'dj', 'D J' all collapse
    assert kb._norm_category("DJ") == "dj"
    assert kb._norm_category("Music Theory") == "music-theory"
    assert kb._norm_category("  ") == "general"
    assert kb._norm_category("Sound_Design!") == "sound-design"     # new cat survives
    assert kb._norm_tags(["Detroit", "detroit", " 909 ", ""]) == ["detroit", "909"]
    assert len(kb._norm_tags([str(i) for i in range(20)])) == 8     # capped
    print("  ✓ normalisers: category kebab-case+open · tags deduped/capped")


# ── E2E (Ollama + DB) ─────────────────────────────────────────────────────────
def _ready():
    return database.DB_AVAILABLE and ollama_client.is_up() \
        and ollama_client.has_model(embed_text.EMBED_MODEL)


MUSIC = ("Robert Hood is a Detroit techno producer, a founding member of "
         "Underground Resistance, and the architect of minimal techno through his "
         "M-Plant label. His stripped-back, funk-driven tracks like Minus and "
         "Internal Empire stripped house and techno down to their rhythmic core.")
COOKING = ("To make a Bolognese ragu, gently brown minced beef with soffritto of "
           "onion, carrot and celery, deglaze with red wine, add tomato passata and "
           "milk, then simmer slowly for two to three hours before serving over "
           "fresh tagliatelle with grated Parmesan cheese.")


def test_gate():
    if not _ready():
        print("  ⚠ Ollama/DB not ready — skipping gate test"); return
    music = asyncio.run(kb.is_music_related(MUSIC))
    cook = asyncio.run(kb.is_music_related(COOKING))
    assert music.is_music_related, f"music wrongly rejected: {music.reason}"
    assert not cook.is_music_related, f"cooking wrongly accepted: {cook.reason}"
    # the same pass also classified: a sensible category + at least one tag + title
    assert music.category and music.category == kb._norm_category(music.category)
    assert music.tags, "expected auto-tags on the music doc"
    assert music.title, "expected an auto-title on the music doc"
    print(f"  ✓ gate+classify: music→accept ({music.confidence:.2f}) "
          f"cat='{music.category}' tags={music.tags[:3]} · cooking→reject")


def test_ingest_retrieve_delete():
    if not _ready():
        print("  ⚠ Ollama/DB not ready — skipping ingest E2E"); return
    title = "__test__ Robert Hood bio"
    doc_id = None
    try:
        res = asyncio.run(kb.ingest_text(MUSIC, title=title, source_type="test"))
        doc_id = res["doc_id"]
        assert res["n_chunks"] >= 1
        assert res["category"], "ingest did not auto-assign a category"
        # retrieval: a question about the doc must surface its chunk near the top
        qvec = embed_text.embed_query("who is robert hood and what is M-Plant")
        hits = database.search_kb_chunks(qvec, n=5)
        assert hits and any(h["doc_id"] == doc_id for h in hits), \
            "ingested doc not retrieved"
        top = hits[0]
        assert float(top["cosine_distance"]) < 0.6, "top hit too far to be useful"
        assert "category" in top, "search did not return category"
        # editable: rename the category, then category-filtered retrieval finds it
        assert database.update_kb_document(doc_id, category="dj", tags=["detroit", "minimal"])
        scoped = database.search_kb_chunks(qvec, n=5, category="dj")
        assert any(h["doc_id"] == doc_id for h in scoped), "category filter lost the doc"
        miss = database.search_kb_chunks(qvec, n=5, category="__nope__")
        assert not any(h["doc_id"] == doc_id for h in miss), "wrong-category filter leaked"
        cats = {c["category"] for c in database.kb_categories()}
        assert "dj" in cats, "kb_categories missing the edited category"
        # the non-music gate refuses and stores NOTHING
        before = database.kb_stats()["documents"]
        try:
            asyncio.run(kb.ingest_text(COOKING, title="__test__ ragu"))
            assert False, "cooking should have been gate-rejected"
        except kb.GateRejected:
            pass
        assert database.kb_stats()["documents"] == before, "rejected doc leaked in"
        # duplicate guard
        try:
            asyncio.run(kb.ingest_text(MUSIC, title=title, source_type="test"))
            assert False, "duplicate should raise"
        except ValueError:
            pass
        print(f"  ✓ ingest E2E: stored {res['n_chunks']} chunks (cat='{res['category']}') · "
              f"retrieved (d={float(top['cosine_distance']):.3f}) · category edit+filter · "
              f"cooking refused · dup guarded")
    finally:
        if doc_id:
            database.delete_kb_document(doc_id)


if __name__ == "__main__":
    print("\nThe Crate — knowledge-base tests\n" + "─" * 40)
    test_chunking()
    test_extract_and_hash()
    test_normalisers()
    test_gate()
    test_ingest_retrieve_delete()
    print("─" * 40 + "\nAll knowledge-base tests passed.\n")
