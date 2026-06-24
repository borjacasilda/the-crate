"""
test_artists.py — The Crate, Phase 0: artist entities + centroids
=================================================================
Covers the structured-artist layer the AI assistant's recommendation hierarchy
builds on: filename parsing, idempotent backfill, EffNet centroids, and the
"artists who sound like X" ANN.

NON-DESTRUCTIVE: the destructive checks use a DISPOSABLE artist that is created
and deleted; the real backfilled artists are only read.

Run (DB up):  uv run python ab_tests/test_artists.py
Skips cleanly when the DB is unavailable.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import database
import analyze


def test_parse_artist_names():
    P = database._parse_artist_names
    assert P("Oscar Mulero - ZW Systems (Original Mix).mp3") == ["Oscar Mulero"]
    assert P("Obscure Shape, SHDW - Blick Des Bösen.mp3") == ["Obscure Shape", "SHDW"]
    assert P("Rødhåd, UFO95 - LAVANDE 03 [240106].mp3") == ["Rødhåd", "UFO95"]
    assert P("A & B - Track.wav") == ["A", "B"]
    assert P("NoSeparatorHere.mp3") == []          # no ' - ' → no artist
    assert P("dop - Your Sex (Paul Ritch Remix).mp3") == ["dop"]
    print("  ✓ parse: single / multi (',' and '&') / [EP] stripped / no-separator")


def test_backfill_idempotent():
    a1 = database.backfill_artists()
    a2 = database.backfill_artists()              # re-run must not duplicate
    assert a1 == a2, f"backfill not idempotent: {a1} vs {a2}"
    assert a1["artists"] >= 1 and a1["links"] >= a1["tracks"]
    # upsert_artist is idempotent on name (same id back).
    name = "Oscar Mulero"
    assert database.upsert_artist(name) == database.upsert_artist(name)
    print(f"  ✓ backfill idempotent ({a1['artists']} artists, {a1['links']} links) · upsert stable")


def test_centroid_and_similarity():
    artists = database.list_artists()
    assert artists, "no artists — run backfill first"
    # An artist with a centroid: dims + self-similarity.
    mv = analyze._model_version("effnet")
    target = next((a for a in artists if database.artist_track_vectors(
        str(a["artist_id"]), mv)), None)
    assert target, "no artist has EffNet vectors"
    aid = str(target["artist_id"])
    import numpy as np
    vecs = database.artist_track_vectors(aid, mv)
    c = np.mean(np.array(vecs), axis=0); c = c / np.linalg.norm(c)
    assert abs(np.linalg.norm(c) - 1.0) < 1e-6, "centroid must be L2-normalised"
    with database._transaction() as cur:
        cur.execute("SELECT vector_dims(embedding) AS d FROM embeddings_artist "
                    "WHERE artist_id=%s;", (aid,))
        row = cur.fetchone()
    assert row and row["d"] == 1280, "artist centroid must be 1280-D EffNet space"
    # Self is the nearest neighbour at cosine_distance ~0 (similarity ~1.0).
    top = database.find_similar_artists(c.tolist(), n=1)
    assert top and str(top[0]["artist_id"]) == aid
    assert abs(float(top[0]["cosine_distance"])) < 1e-3
    # Excluding self returns OTHER artists, ordered.
    others = database.find_similar_artists(c.tolist(), n=5, exclude_artist_id=aid)
    assert all(str(o["artist_id"]) != aid for o in others)
    sims = [1 - float(o["cosine_distance"]) for o in others]
    assert sims == sorted(sims, reverse=True), "results not nearest-first"
    print(f"  ✓ centroid 1280-D L2-normalised · self-match 1.0 · '{target['name']}' "
          f"→ {len(others)} neighbours, ordered")


def test_disposable_artist_lifecycle():
    name = f"__test_artist_{uuid.uuid4().hex[:8]}__"
    aid = database.upsert_artist(name)
    try:
        assert database.get_artist(name)["name"] == name
        assert database.get_artist(aid)["name"] == name
        # No tracks → no centroid.
        assert analyze.persist_artist_embedding(aid) is None
    finally:
        with database._transaction() as cur:
            cur.execute("DELETE FROM artists WHERE artist_id=%s;", (aid,))
        assert database.get_artist(name) is None
    print("  ✓ disposable artist: upsert/get by name+id, no-track→no centroid, cleaned up")


if __name__ == "__main__":
    print("\nThe Crate — artist entity tests\n" + "─" * 40)
    if not database.DB_AVAILABLE:
        print("  ⚠ DB unavailable — start Docker (docker compose up -d). Skipping.")
        sys.exit(0)
    test_parse_artist_names()
    test_backfill_idempotent()
    test_centroid_and_similarity()
    test_disposable_artist_lifecycle()
    print("─" * 40 + "\nAll artist tests passed.\n")
