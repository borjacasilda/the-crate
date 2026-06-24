"""
test_live_tracklist.py — Live Mode tracklist integrity (no audio, no DB).
========================================================================
Guards the bug where the API live worker logged EVERY identified pass straight to
the tracklist, so a single EffNet nearest-neighbour hit (the top recommendation)
appeared as a track that never played. The fix debounces fuzzy matches and de-dupes
consecutive re-locks (api._apply_recognition).

Drives the state machine directly with synthetic RecognitionResults — fast, no
Ollama/Essentia/DB. _live_recommendations is stubbed so the tracklist logic is
exercised in isolation.

Run:  uv run python ab_tests/test_live_tracklist.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api          # noqa: E402
import analyze      # noqa: E402
import listener     # noqa: E402
from listener import RecognitionResult  # noqa: E402

api._live_recommendations = lambda *a, **k: []   # isolate the state machine from DB scoring


def _res(track_id, strategy):
    return RecognitionResult(
        identified=True, strategy=strategy, confidence=0.9, track_id=track_id,
        crate_path=f"/crate/{track_id}.wav", filename=f"{track_id}.mp3",
        features=analyze.TrackFeatures(bpm=130.0, camelot="9A"))


def _reset():
    api.LIVE.update(running=True, status="searching", track=None, confidence=0.0,
                    strategy=None, locked_at=None, misses=0, recommendations=[],
                    tracklist=[], pending_save=False, pending_id=None, pending_n=0,
                    session_id=None, crate_id=None)


def _logged():
    return [(e["track_id"], e["detected_by"]) for e in api.LIVE["tracklist"]]


def test_effnet_neighbour_debounced():
    """A single EffNet nearest-neighbour hit (the top recommendation) must NOT reach
    the tracklist — only the fingerprinted plays do."""
    _reset()
    seq = [("RW1", "fingerprint"), ("RW1", "fingerprint"), ("DK", "effnet"),
           ("RW1", "fingerprint"), ("RW2", "fingerprint"), ("DK2", "effnet")]
    for tid, strat in seq:
        api._apply_recognition(_res(tid, strat))
    tl = _logged()
    assert [t for t, _ in tl] == ["RW1", "RW2"], tl
    assert all(d == "fingerprint" for _, d in tl), tl
    print("  ✓ spurious EffNet neighbours never reach the tracklist")


def test_stable_effnet_commits():
    """A GENUINE EffNet match (same track for STABLE_READS passes) DOES log, so
    pitched/degraded vinyl the fingerprint can't catch is still recovered."""
    _reset()
    for _ in range(listener.STABLE_READS):
        api._apply_recognition(_res("PV", "effnet"))
    assert _logged() == [("PV", "effnet")], _logged()
    # One pass alone would not have committed:
    _reset()
    api._apply_recognition(_res("PV", "effnet"))
    assert _logged() == [], _logged()
    print(f"  ✓ EffNet commits only after {listener.STABLE_READS} stable reads (1 pass logs nothing)")


def test_fingerprint_single_pass():
    """Fingerprint is exact → commits on the first pass (the speed path)."""
    _reset()
    api._apply_recognition(_res("FP", "fingerprint"))
    assert _logged() == [("FP", "fingerprint")], _logged()
    print("  ✓ fingerprint commits in a single pass")


def test_no_consecutive_duplicate():
    """A re-lock of the same record after a brief drop is not logged twice back-to-
    back, but the same track may reappear later in the set with a gap."""
    _reset()
    api._apply_recognition(_res("A", "fingerprint"))          # A locks + logs
    for _ in range(api.LIVE_GRACE_MISSES + 1):                # lose the lock
        api._apply_recognition(None)
    api._apply_recognition(_res("A", "fingerprint"))          # A re-acquires → no dup
    assert _logged() == [("A", "fingerprint")], _logged()
    api._apply_recognition(_res("B", "fingerprint"))          # B
    api._apply_recognition(_res("A", "fingerprint"))          # A again, with a gap → logs
    assert [t for t, _ in _logged()] == ["A", "B", "A"], _logged()
    print("  ✓ no consecutive duplicate; replay-with-gap still logs")


if __name__ == "__main__":
    print("\nThe Crate — live tracklist tests\n" + "─" * 40)
    test_effnet_neighbour_debounced()
    test_stable_effnet_commits()
    test_fingerprint_single_pass()
    test_no_consecutive_duplicate()
    print("─" * 40 + "\nAll live-tracklist tests passed.\n")
