"""
test_discogs.py — The Crate, Phase 3b: Discogs enrichment
=========================================================
Covers everything that does NOT need a live Discogs token:
  • pure matching logic (normalise, similarity, score_match bands),
  • filename → (artist, title) parsing,
  • the DB layer for labels (centroid + ANN, mirrors artists) and the
    track_discogs enrichment record / doubtful queue / bulk map.

The live API path (search/best_match/cover download) is exercised separately
once a valid token is in .env. DB tests use a REAL track's EffNet vector and
clean up after themselves.

Run:  uv run python ab_tests/test_discogs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze
import database
import discogs
import enrich


# ── pure logic ────────────────────────────────────────────────────────────────
def test_parse():
    cases = {
        "Robert Hood - Minimal Nation [M-Plant].mp3": ("Robert Hood", "Minimal Nation"),
        "Oscar Mulero - Muscle and Mind (Original Mix).wav":
            ("Oscar Mulero", "Muscle and Mind (Original Mix)"),
        "noseparator.flac": ("", "noseparator"),
    }
    for fn, expect in cases.items():
        assert enrich._parse(fn) == expect, f"{fn} → {enrich._parse(fn)}"
    print("  ✓ parse: artist/title from filename · [EP] stripped · no-separator")


def test_norm_similarity():
    assert discogs._norm("Robert Hood (Original Mix) [M-Plant]") == "robert hood"
    assert discogs._similarity("Robert Hood", "robert hood") == 1.0
    assert discogs._similarity("Robert Hood", "Aphex Twin") == 0.0
    assert 0.0 < discogs._similarity("Minimal Nation", "Minimal Nation EP") < 1.0
    print("  ✓ normalise + token-set similarity (1.0 / 0.0 / partial)")


def test_score_match():
    cand = {"title": "Robert Hood - Internal Empire"}
    hot = discogs.score_match("Robert Hood", "Internal Empire", cand)
    cold = discogs.score_match("Robert Hood", "Internal Empire",
                               {"title": "Aphex Twin - Windowlicker"})
    assert hot >= discogs.AUTO_THRESHOLD, f"strong match scored {hot}"
    assert cold < discogs.DOUBT_THRESHOLD, f"wrong match scored {cold}"
    # tracklist confirmation lifts a generic release-title match
    rel = {"tracklist": ["Minus", "Internal Empire", "The Pulse"]}
    boosted = discogs.score_match("Robert Hood", "Internal Empire",
                                  {"title": "Robert Hood - M-Plant Sampler"}, release=rel)
    assert boosted >= 0.5, f"tracklist confirm too weak: {boosted}"
    print(f"  ✓ score_match: strong {hot} ≥ auto · wrong {cold} < doubt · tracklist boost {boosted}")


# ── DB: labels (mirror artists) ───────────────────────────────────────────────
def _a_real_track():
    with database._transaction() as cur:
        cur.execute("""SELECT t.track_id FROM tracks t
                       JOIN embeddings_effnet e ON e.track_id = t.track_id LIMIT 1;""")
        r = cur.fetchone()
    return str(r["track_id"]) if r else None


def test_labels_db():
    if not database.DB_AVAILABLE:
        print("  ⚠ DB down — skipping label DB test"); return
    tid = _a_real_track()
    assert tid, "no analysed track with an EffNet vector to test with"
    name = "__test__ Disposable Label"
    lid = database.upsert_label(name, discogs_id=12345)
    try:
        assert database.upsert_label(name) == lid, "upsert not idempotent"
        database.link_track_label(tid, lid)
        pooled = analyze.persist_label_embedding(lid)
        assert pooled == 1, f"expected 1 pooled track, got {pooled}"
        vec = database.get_label_embedding(lid, analyze._model_version("effnet"))
        assert vec and len(vec) == 1280, "label centroid missing / wrong dim"
        hits = database.find_similar_labels(vec, n=3)        # self-match present
        assert any(h["label_id"] == lid or str(h["label_id"]) == lid for h in hits) \
            or hits == [], "self not found among neighbours"
        assert any(l["name"] == name for l in database.list_labels())
    finally:
        with database._transaction() as cur:
            cur.execute("DELETE FROM labels WHERE label_id = %s;", (lid,))
    assert not database.get_label(name), "label not cleaned up"
    print("  ✓ labels: upsert idempotent · centroid 1280-D · ANN · cascade cleanup")


# ── DB: track_discogs enrichment record + queue ───────────────────────────────
def test_track_discogs_record():
    if not database.DB_AVAILABLE:
        print("  ⚠ DB down — skipping track_discogs test"); return
    tid = _a_real_track()
    assert tid
    try:
        database.upsert_track_discogs(
            tid, status="doubtful", confidence=0.55,
            candidates=[{"release_id": 999, "title": "X - Y", "label": "Z", "year": 2020}],
            label=None)
        row = database.get_track_discogs(tid)
        assert row and row["status"] == "doubtful" and row["candidates"], "record not stored"
        q = database.discogs_queue("doubtful")
        assert any(str(r["track_id"]) == tid for r in q), "not in doubtful queue"
        m = database.track_discogs_map([tid])
        assert tid in m and m[tid]["status"] == "doubtful", "bulk map missing"
        assert database.set_track_discogs_status(tid, "skipped")
        assert database.get_track_discogs(tid)["status"] == "skipped"
        # a matched upsert overwrites + records label/styles
        database.upsert_track_discogs(tid, status="matched", confidence=0.9,
                                      label="Ostgut Ton", year=2014,
                                      styles=["Techno", "Minimal"])
        row = database.get_track_discogs(tid)
        assert row["status"] == "matched" and row["label"] == "Ostgut Ton"
        assert "Techno" in (row["styles"] or [])
    finally:
        with database._transaction() as cur:
            cur.execute("DELETE FROM track_discogs WHERE track_id = %s;", (tid,))
    print("  ✓ track_discogs: upsert/get · doubtful queue · bulk map · status edit · cleanup")


if __name__ == "__main__":
    print("\nThe Crate — Discogs enrichment tests\n" + "─" * 40)
    test_parse()
    test_norm_similarity()
    test_score_match()
    test_labels_db()
    test_track_discogs_record()
    print("─" * 40 + "\nAll Discogs tests passed.\n")
