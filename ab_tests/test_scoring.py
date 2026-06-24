"""
test_scoring.py — VinylID scoring-engine regression tests
=========================================================
Three focused, DB-free tests over the pure scoring layer in analyze.py,
covering the invariants checked in the 2026-06 spec review:

    Test 1 — test_harmonic_monotone()
        The harmonic modifier must be monotone over the Camelot relationship
        (same key >= adjacent >= relative >= dissonant) in both confidence
        modes.

    Test 1b — test_harmonic_confidence()
        Continuous confidence weighting (confidence = ks1 × ks2): atonal
        pairs ≈ neutral, tonal pairs ≈ full Camelot penalty, mixed pairs
        partially active, no cliff around the old 0.4 threshold, and the
        legacy binary gate still reproducible via the feature toggle.

    Test 2 — test_density_floor()
        The density modifier must never fall below config.DENSITY_MOD_FLOOR,
        even on an extreme spectral-complexity mismatch, and must stay 1.0
        for an identical pair.

    Test 3 — test_topk_safeguard()
        score_candidates() must find the true best total even when every
        track in the initial top-K window is penalised below a candidate
        sitting just OUTSIDE the window (the counterexample that breaks
        plain top-K truncation).

Usage
-----
    # From the project root:
    uv run python ab_tests/test_scoring.py

No database, no audio files, no model downloads — synthetic TrackFeatures only.
Nothing is written outside ab_tests/.
"""

import math
import sys
from pathlib import Path

# Project root on sys.path so `import analyze` works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze
from config import DENSITY_MOD_FLOOR


# ── Synthetic track factory ──────────────────────────────────────────────────
def _track(embedding=None, camelot="8A", key_strength=0.9,
           spectral_complexity=20.0, path="synthetic.wav") -> "analyze.TrackFeatures":
    """A TrackFeatures where every modifier except the one under test is neutral.

    Same BPM, same key, flat matching energy, >16 bars of mixable overlap and
    no mood/emotional data (those modifiers go neutral on None).
    """
    return analyze.TrackFeatures(
        path=path, duration=360.0,
        bpm=130.0, bpm_confidence=1.0,
        camelot=camelot, key_strength=key_strength,
        spectral_complexity=spectral_complexity,
        energy_curve=[0.5] * 60,
        intro_end=64.0, outro_start=300.0,
        effnet_embedding=embedding,
        mfcc_mean=[1.0] * 13,
    )


def _emb(cos: float) -> list:
    """A 3-D unit vector at the given cosine to the reference axis [1, 0, 0]."""
    return [cos, math.sqrt(max(0.0, 1.0 - cos * cos)), 0.0]


# ── Test 1: harmonic modifier order ──────────────────────────────────────────
def test_harmonic_monotone():
    base = _track(camelot="8A")
    cases = [("8A", "Same key"), ("9A", "Adjacent"),
             ("8B", "Relative (mood shift)"), ("3A", "Dissonant")]
    mods = []
    for cam, label in cases:
        other = _track(camelot=cam)
        assert analyze.key_relationship_label("8A", cam) == label, \
            f"{cam}: expected label {label}"
        mods.append(analyze._harmonic_mod_raw(base, other))
    assert mods == sorted(mods, reverse=True) and mods[0] > mods[-1], \
        f"harmonic modifier not monotone over relationships: {mods}"
    assert mods[0] == 1.0, f"same key must be a perfect 1.0, got {mods[0]}"
    print("  ✓ harmonic modifier: same key ≥ adjacent ≥ relative ≥ dissonant")


# ── Test 1b: harmonic confidence weighting ───────────────────────────────────
def test_harmonic_confidence():
    from config import KEY_STRENGTH_THRESHOLD

    def dissonant_mod(ks1, ks2):
        # 8A → 3A is Dissonant — the strongest Camelot penalty (raw 0.70).
        return analyze._harmonic_mod_raw(_track(camelot="8A", key_strength=ks1),
                                         _track(camelot="3A", key_strength=ks2))

    assert analyze.HARMONIC_CONTINUOUS_CONFIDENCE, \
        "tests assume the continuous mode is the default"

    # Two atonal tracks → practically neutral (confidence 0.015).
    both_atonal = dissonant_mod(0.1, 0.15)
    assert both_atonal > 0.99, f"atonal pair should be ~1.0, got {both_atonal:.4f}"

    # Two solidly tonal tracks → Camelot weighs heavily (confidence 0.765).
    both_tonal = dissonant_mod(0.9, 0.85)
    expected = 1.0 - (0.9 * 0.85) * (1.0 - 0.70)
    assert abs(both_tonal - expected) < 1e-9, \
        f"expected {expected:.4f}, got {both_tonal:.4f}"
    same_key = analyze._harmonic_mod_raw(_track(camelot="8A", key_strength=0.9),
                                         _track(camelot="8A", key_strength=0.85))
    assert same_key == 1.0 and same_key - both_tonal > 0.2, \
        "tonal pair must keep a strong Camelot spread"

    # Tonal + atonal → PARTIALLY active, neither neutral nor full penalty.
    mixed = dissonant_mod(0.8, 0.2)
    assert 0.9 < mixed < 1.0, \
        f"mixed pair should be partially active, got {mixed:.4f}"

    # Grey zone around the old threshold: continuous mode has no cliff.
    grey_below, grey_above = dissonant_mod(0.38, 0.42), dissonant_mod(0.42, 0.42)
    assert abs(grey_below - grey_above) < 0.02, \
        f"grey zone must be smooth: {grey_below:.4f} vs {grey_above:.4f}"

    # Exact threshold on both tracks: finite, in range, no special-casing.
    at_threshold = dissonant_mod(KEY_STRENGTH_THRESHOLD, KEY_STRENGTH_THRESHOLD)
    assert 0.0 < at_threshold <= 1.0 and math.isfinite(at_threshold)

    # Legacy binary gate, still reproducible via the toggle: hard cliff at the
    # threshold (documents the OLD behaviour the continuous mode smooths out).
    analyze.HARMONIC_CONTINUOUS_CONFIDENCE = False
    try:
        assert dissonant_mod(0.2, 0.9) == 1.0, "legacy: atonal → exactly neutral"
        assert dissonant_mod(0.38, 0.42) == 1.0, "legacy: 0.38 falls under the gate"
        assert dissonant_mod(0.42, 0.42) == 0.70, "legacy: 0.42 takes the full penalty"
        assert dissonant_mod(KEY_STRENGTH_THRESHOLD, KEY_STRENGTH_THRESHOLD) == 0.70, \
            "legacy: exactly-at-threshold is NOT atonal (strict <)"
    finally:
        analyze.HARMONIC_CONTINUOUS_CONFIDENCE = True
    print(f"  ✓ harmonic confidence: atonal {both_atonal:.3f} · mixed {mixed:.3f} · "
          f"tonal {both_tonal:.3f}; grey zone smooth ({grey_below:.3f}/{grey_above:.3f}); "
          f"legacy cliff (1.00/0.70) behind toggle")


# ── Test 2: density modifier floor ───────────────────────────────────────────
def test_density_floor():
    a = _track(spectral_complexity=7.0)
    b = _track(spectral_complexity=21.0)
    mod = analyze._density_mod_raw(a, b)
    assert mod >= DENSITY_MOD_FLOOR, \
        f"density modifier {mod:.3f} fell below floor {DENSITY_MOD_FLOOR}"
    # diff = 14/21 → raw 1/3 → floored: floor + (1-floor)/3.
    expected = DENSITY_MOD_FLOOR + (1.0 - DENSITY_MOD_FLOOR) / 3.0
    assert abs(mod - expected) < 1e-9, f"expected {expected:.4f}, got {mod:.4f}"

    # Worst case (raw 0.0) must land exactly ON the floor, never below.
    extreme = analyze._density_mod_raw(_track(spectral_complexity=1e-9),
                                       _track(spectral_complexity=30.0))
    assert abs(extreme - DENSITY_MOD_FLOOR) < 1e-6, \
        f"extreme mismatch must hit the floor exactly, got {extreme:.4f}"

    # Identical pair stays a perfect 1.0 (floor must not distort the top).
    same = analyze._density_mod_raw(_track(spectral_complexity=12.0),
                                    _track(spectral_complexity=12.0))
    assert same == 1.0, f"identical pair must score 1.0, got {same}"

    # And the floor propagates through mix_score: total never annihilated.
    s = analyze.mix_score(a, b, mode='balanced')
    assert s['density'] >= DENSITY_MOD_FLOOR and s['total'] > 0.0
    print(f"  ✓ density modifier: floor {DENSITY_MOD_FLOOR} holds (7 vs 21 → {mod:.3f}); identical → 1.0")


# ── Test 3: top-K truncation safeguard ───────────────────────────────────────
def test_topk_safeguard():
    # Seed at the reference axis; sc=20 so density penalises the near tracks.
    seed = _track(embedding=_emb(1.0), spectral_complexity=20.0, path="seed.wav")

    # Three NEAR candidates (base ≈ .99) hammered by density (sc=2 → raw 0.1),
    # one slightly FARTHER candidate (base = .90) with zero penalties.
    near = [_track(embedding=_emb(0.99 - i * 0.001), spectral_complexity=2.0,
                   path=f"penalised_{i}.wav") for i in range(3)]
    clean = _track(embedding=_emb(0.90), spectral_complexity=20.0,
                   path="clean.wav")
    pool = near + [clean]

    # Stand-in for the ANN index: nearest-first by true cosine, truncated at k.
    def fake_load(current, exclude_paths=None, crate="__active__",
                  k=analyze.RETRIEVAL_K):
        ranked = sorted(pool, key=lambda f: -analyze.cosine_sim(
            current.effnet_embedding, f.effnet_embedding))
        return [(f.path, f) for f in ranked[:k]]

    original = analyze._load_candidates
    analyze._load_candidates = fake_load
    try:
        # k=2: the initial window holds ONLY penalised tracks. Plain truncation
        # would return one of them; the safeguard must expand and find clean.wav.
        scored = analyze.score_candidates(seed, mode='balanced', k=2)
        best_path, _, best_s = max(scored, key=lambda x: x[2]['total'])

        window = fake_load(seed, k=2)
        window_best = max(analyze.mix_score(seed, f, mode='balanced')['total']
                          for _, f in window)
        assert best_s['total'] > window_best, \
            "counterexample is degenerate: the clean track does not beat the window"
        assert best_path == "clean.wav", \
            f"safeguard failed: best is {best_path} (total {best_s['total']:.3f})"
        # The bound must hold: nothing unfetched could have beaten the winner.
        assert best_s['total'] >= min(s['effnet_base'] for _, _, s in scored)
    finally:
        analyze._load_candidates = original
    print(f"  ✓ top-K safeguard: window of 2 penalised tracks expanded, "
          f"clean.wav wins ({best_s['total']:.3f} > {window_best:.3f})")


if __name__ == "__main__":
    print("\nVinylID scoring-engine tests\n" + "─" * 40)
    test_harmonic_monotone()
    test_harmonic_confidence()
    test_density_floor()
    test_topk_safeguard()
    print("─" * 40 + "\nAll 4 tests passed.\n")
