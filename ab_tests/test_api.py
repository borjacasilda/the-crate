"""
test_api.py — VinylID API + session-layer regression tests
==========================================================
End-to-end checks over the HTTP surface and the data-management layer built
on top of it: track listing & stage-2 fields, audio/waveform serving, affinity,
crate move/delete, and the per-crate session views + session embeddings.

NON-DESTRUCTIVE: the move/delete tests operate on a DISPOSABLE track (a copy of
an existing crate excerpt registered under a throwaway row) and clean up after
themselves — real crate data is never touched.

Requires the API running locally:
    uv run uvicorn api:app --host 127.0.0.1 --port 8000
Then:
    uv run python ab_tests/test_api.py

If the server (or DB) is down the suite SKIPS with a clear message rather than
failing, so it is safe to run in any environment.
"""

import shutil
import sys
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import database

BASE = "http://127.0.0.1:8000"


# ── helpers ──────────────────────────────────────────────────────────────────
def _api_up() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=3).json().get("db") is True
    except Exception:
        return False


def _any_track_id() -> str:
    rows = requests.get(f"{BASE}/tracks?crate=__active__").json()
    assert rows, "no tracks in the active crate to test against"
    return rows[0]["track_id"]


def _make_disposable_track() -> "tuple[str, Path]":
    """Register a throwaway track (copy of a real excerpt) — caller must delete."""
    src = next(Path("crate").glob("*.wav"))
    eid = uuid.uuid4().hex
    dummy = Path("crate") / f"{eid}.wav"
    shutil.copy(src, dummy)
    with database._transaction() as cur:
        cur.execute(
            """INSERT INTO tracks (crate_path, filename, crate_id)
               VALUES (%s, %s, %s) RETURNING track_id;""",
            (str(dummy.resolve()), f"TEST {eid[:8]} - Disposable.wav",
             database.active_crate_id()))
        tid = str(cur.fetchone()["track_id"])
    return tid, dummy


# ── tests ────────────────────────────────────────────────────────────────────
def test_health_and_crates():
    h = requests.get(f"{BASE}/health").json()
    assert h["db"] is True and h["pipeline_level"] >= 1
    # Dual route: HTML to a browser, JSON to fetch — and never cached without Vary.
    html = requests.get(f"{BASE}/crates", headers={"Accept": "text/html"})
    js = requests.get(f"{BASE}/crates", headers={"Accept": "application/json"})
    assert html.text.lstrip().startswith("<!doctype")
    assert isinstance(js.json(), list)
    assert js.headers.get("cache-control") == "no-store"
    assert "Accept" in js.headers.get("vary", "")
    print("  ✓ /health ok · /crates dual route (HTML/JSON) with no-store+Vary")


def test_tracks_stage2_fields():
    rows = requests.get(f"{BASE}/tracks?crate=__active__").json()
    t = rows[0]
    for f in ("bpm", "energy", "mood_aggressive", "danceability", "density",
              "brightness", "on_spot"):
        assert f in t, f"missing stage-2 field {f}"
    print(f"  ✓ /tracks exposes stage-2 fields ({len(rows)} tracks)")


def test_audio_and_waveform():
    tid = _any_track_id()
    full = requests.get(f"{BASE}/tracks/{tid}/audio")
    assert full.status_code == 200 and full.headers["content-type"] == "audio/wav"
    rng = requests.get(f"{BASE}/tracks/{tid}/audio",
                       headers={"Range": "bytes=0-1000"})
    assert rng.status_code == 206, "audio must honour Range for seeking"
    wf = requests.get(f"{BASE}/tracks/{tid}/waveform").json()["peaks"]
    assert len(wf) == 240 and all(0.0 <= v <= 1.0 for v in wf)
    # bins clamp
    clamped = requests.get(f"{BASE}/tracks/{tid}/waveform?bins=9999").json()["peaks"]
    assert len(clamped) == 600
    print(f"  ✓ audio (200 + 206 Range) · waveform (240 peaks, bins clamp 600)")


def test_affinity():
    tid = _any_track_id()
    aff = requests.get(f"{BASE}/affinity?track_id={tid}&crate=__active__").json()
    assert abs(aff[tid] - 1.0) < 1e-3, "self-affinity must be ~1.0"
    print(f"  ✓ /affinity self-similarity {aff[tid]:.3f} over {len(aff)} tracks")


def test_membership_and_delete():
    tid, dummy = _make_disposable_track()
    cleaned = False
    try:
        # A throwaway user crate to add into (many-to-many membership).
        cdest = database.create_crate("__test_dest__", genre="techno")
        r = requests.post(f"{BASE}/tracks/add-to-crate",
                          json={"track_ids": [tid], "crate": str(cdest)}).json()
        assert r["added"] == 1
        members = {str(t["track_id"]) for t in database.list_tracks(crate_id=str(cdest))}
        assert tid in members, "track was not added to the destination crate"
        # The track also stays in the master library (default crate = all tracks).
        master = {str(t["track_id"]) for t in
                  database.list_tracks(crate_id=database.default_crate_id())}
        assert tid in master, "track must also remain in the master library"
        # Remove from the user crate → gone from it, kept in master.
        rr = requests.post(f"{BASE}/tracks/remove-from-crate",
                           json={"track_ids": [tid], "crate": str(cdest)}).json()
        assert rr["removed"] == 1
        members2 = {str(t["track_id"]) for t in database.list_tracks(crate_id=str(cdest))}
        assert tid not in members2 and database.get_track(tid) is not None

        # Delete it through the API; row + WAV must both vanish.
        d = requests.post(f"{BASE}/tracks/delete", json={"track_ids": [tid]}).json()
        assert d["deleted"] == 1
        assert not dummy.exists(), "crate WAV not removed on delete"
        with database._transaction() as cur:
            cur.execute("SELECT count(*) AS c FROM tracks WHERE track_id=%s;", (tid,))
            assert cur.fetchone()["c"] == 0
        cleaned = True
        print("  ✓ add-to-crate + master kept · remove-from-crate · delete (row + WAV gone)")
    finally:
        if not cleaned and dummy.exists():
            dummy.unlink()
        if not cleaned:
            try: database.delete_track(tid)
            except Exception: pass
        # Drop the throwaway destination crate.
        with database._transaction() as cur:
            cur.execute("DELETE FROM crates WHERE name='__test_dest__';")


def test_validations():
    tid = _any_track_id()
    # empty batch → 422
    assert requests.post(f"{BASE}/tracks/delete", json={"track_ids": []}).status_code == 422
    assert requests.post(f"{BASE}/tracks/add-to-crate",
                         json={"track_ids": [tid], "crate": "no-existe-xyz"}).status_code == 404
    # bad crate filter on sessions → 404; bad session → 404
    assert requests.get(f"{BASE}/sessions?crate=no-existe-xyz",
                        headers={"Accept": "application/json"}).status_code == 404
    assert requests.get(f"{BASE}/sessions/00000000-0000-0000-0000-000000000000").status_code == 404
    print("  ✓ validations: 422 empty batch · 404 bad crate/session")


def test_sessions_and_embeddings():
    saved = requests.get(f"{BASE}/sessions",
                         headers={"Accept": "application/json"}).json()
    if not saved:
        print("  ⚠ no saved sessions — skipping session-detail checks")
        return
    s = saved[0]
    sid = s["session_id"]
    # Per-crate filter returns a subset of the global list.
    if s.get("crate_name"):
        crate_id = str(database.resolve_crate_id(s["crate_name"]))
        scoped = requests.get(f"{BASE}/sessions?crate={crate_id}",
                              headers={"Accept": "application/json"}).json()
        assert all(x["session_id"] for x in scoped)
        assert len(scoped) <= len(saved)
    # Detail + tracklist.
    detail = requests.get(f"{BASE}/sessions/{sid}").json()
    assert detail["name"] == s["name"] and "tracklist" in detail
    # Session embedding (centroid) exists for a saved session with tracks.
    with database._transaction() as cur:
        cur.execute("SELECT n_tracks, vector_dims(embedding) AS d "
                    "FROM embeddings_session WHERE session_id=%s;", (sid,))
        row = cur.fetchone()
    if row:
        assert row["d"] == 1280, "session centroid must be 1280-D EffNet space"
        print(f"  ✓ sessions: filter + detail · centroid {row['d']}-D "
              f"(pooled {row['n_tracks']} tracks)")
    else:
        print("  ✓ sessions: filter + detail (no centroid — session had no embeddings)")


if __name__ == "__main__":
    print("\nVinylID API + session tests\n" + "─" * 40)
    if not _api_up():
        print("  ⚠ API/DB not reachable at " + BASE + " — start the server first.")
        print("    uv run uvicorn api:app --host 127.0.0.1 --port 8000")
        sys.exit(0)
    test_health_and_crates()
    test_tracks_stage2_fields()
    test_audio_and_waveform()
    test_affinity()
    test_membership_and_delete()
    test_validations()
    test_sessions_and_embeddings()
    print("─" * 40 + "\nAll API tests passed.\n")
