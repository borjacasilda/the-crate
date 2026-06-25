"""
test_analysis.py — The Crate pipeline validation
===============================================
Two focused, self-contained tests that verify the two most important operations
in the analysis pipeline and produce human-readable outputs.

    Test 1 — test_analyze_track()
        Calls analyze.extract_features() on a real audio file and validates
        that every Level-1 field is present, structurally correct, and
        within sensible numeric bounds.

    Test 2 — test_effnet_embedding()
        Verifies that the EffNet embedding is numerically valid (finite,
        unit-normalised, non-zero), then — when Postgres is up — persists it
        via database.py and confirms a full round-trip including a
        self-similarity search.

Usage
-----
    # Run both tests from the project root:
    python ab_tests/test_analysis.py

    # Run one test in isolation:
    python -c "
    import sys; sys.path.insert(0, '.')
    from ab_tests.test_analysis import test_analyze_track
    test_analyze_track()"

Outputs (all contained under ab_tests/)
    ab_tests/test_analysis.log              append-mode structured log
    ab_tests/output_tests/track_features.json
    ab_tests/output_tests/effnet_embedding.json

Nothing is written outside ab_tests/.
"""

import dataclasses
import datetime
import json
import logging
import math
import re
import sys
import time
import traceback
import uuid
from pathlib import Path

# ── Project root on sys.path so `import analyze` and `import database` work
# whether the script is invoked as `python ab_tests/test_analysis.py` (cwd =
# project root) or from inside ab_tests/ directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import analyze
import crate
import database

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
AB_TESTS_DIR = Path(__file__).parent          # ab_tests/
LOG_PATH     = AB_TESTS_DIR / "test_analysis.log"
OUTPUT_DIR   = AB_TESTS_DIR / "output_tests"

# Primary test audio path (as specified in the task).
# The file may not exist — _get_features() handles that gracefully.
TEST_AUDIO_PATH = Path("samples_tracks/Record-2026-0531-084103.wav")

# Level-1 fields that MUST be populated for every track, regardless of whether
# the ML models are installed.  Used by Test 1's structural assertions.
REQUIRED_L1_FIELDS = (
    "path", "duration", "bpm", "bpm_confidence",
    "key", "scale", "camelot", "key_strength",
    "danceability", "onset_rate", "loudness", "replay_gain",
    "dynamic_complexity", "spectral_centroid", "spectral_complexity",
    "spectral_flux", "spectral_rolloff", "zcr",
    "mfcc_mean", "bark_mean", "energy_curve", "complexity_curve",
    "intro_end", "outro_start", "pipeline_level",
)

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _setup_test_logging() -> logging.Logger:
    """Configure and return the 'the_crate.tests' logger.

    Attaches a single append-mode FileHandler to LOG_PATH.  Idempotent:
    calling this function multiple times (e.g. when tests are imported
    individually) never stacks duplicate handlers.

    Returns:
        The configured 'the_crate.tests' Logger instance.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # ensure output dir exists early
    AB_TESTS_DIR.mkdir(parents=True, exist_ok=True) # ensure log dir exists

    logger = logging.getLogger("the_crate.tests")
    if logger.handlers:
        return logger  # already configured — skip re-attachment

    logger.setLevel(logging.DEBUG)

    # NOTE: the format string uses %(test_name)s which is NOT a standard
    # LogRecord attribute.  It is injected by the LoggerAdapter returned by
    # _make_adapter(), which merges {"test_name": <name>} into every record's
    # extra dict before the formatter sees it.
    fmt = logging.Formatter(
        "%(asctime)s [%(test_name)s] %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Prevent propagation to the root logger (which may have a StreamHandler
    # installed by analyze.py's basicConfig call).  All test output to stdout
    # is handled by explicit print() calls, not via logging.
    logger.propagate = False
    return logger


def _make_adapter(logger: logging.Logger, test_name: str) -> logging.LoggerAdapter:
    """Wrap a logger so every emitted record carries test_name in its extra dict.

    The Formatter pattern '%(test_name)s' is resolved from this extra field,
    making every log line grep-able by test name without passing it as a string
    argument to every log call.

    Args:
        logger:    The base 'the_crate.tests' logger.
        test_name: Short uppercase label, e.g. 'TEST_ANALYZE_TRACK'.

    Returns:
        A LoggerAdapter that injects {'test_name': test_name} into every record.
    """
    return logging.LoggerAdapter(logger, {"test_name": test_name})


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED FEATURE CACHE
# ══════════════════════════════════════════════════════════════════════════════
# analyze.extract_features() is expensive (10-60 s depending on pipeline level
# and track length).  We call it at most once per process by caching the result
# at module level.  None means "not yet attempted"; the sentinel _FEATURES_FAILED
# means "attempted but failed or audio file not found" so we don't retry.

_FEATURES_CACHE: "analyze.TrackFeatures | None" = None
_FEATURES_FAILED: bool = False
_RESOLVED_AUDIO_PATH: "Path | None" = None  # the path actually used (for display)


def _get_features() -> "analyze.TrackFeatures | None":
    """Return the TrackFeatures for the test audio file, caching the result.

    Resolves TEST_AUDIO_PATH against the project root so the test works when
    invoked from any working directory.  Falls back to the first .mp3/.wav/.flac
    found in samples_tracks/ if TEST_AUDIO_PATH does not exist — this lets the
    test suite run on the actual repo even though the named file is absent.

    Returns:
        A fully-populated TrackFeatures, or None if no audio file was found or
        feature extraction failed.  Sets _FEATURES_FAILED so subsequent calls
        skip the expensive extraction attempt.
    """
    global _FEATURES_CACHE, _FEATURES_FAILED, _RESOLVED_AUDIO_PATH
    if _FEATURES_CACHE is not None:
        return _FEATURES_CACHE
    if _FEATURES_FAILED:
        return None

    # 1) Try the path as-is (works when cwd is project root).
    candidate = TEST_AUDIO_PATH
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate

    # 2) If primary path missing, fall back to first audio file in samples_tracks/.
    if not candidate.exists():
        samples_dir = _PROJECT_ROOT / "samples_tracks"
        audio_exts  = {".wav", ".mp3", ".flac"}
        fallbacks   = sorted(
            p for p in samples_dir.iterdir()
            if p.is_file() and p.suffix.lower() in audio_exts
        ) if samples_dir.is_dir() else []
        if fallbacks:
            candidate = fallbacks[0]
            print(
                f"\n  ⚠  TEST_AUDIO_PATH not found — using fallback:\n"
                f"     {candidate.name}\n"
            )
        else:
            print(
                f"\n  ⚠  SKIP: no audio files found.\n"
                f"     Looked for:   {TEST_AUDIO_PATH}\n"
                f"     Fallback dir: {samples_dir}\n"
            )
            _FEATURES_FAILED = True
            return None

    _RESOLVED_AUDIO_PATH = candidate

    try:
        _FEATURES_CACHE = analyze.extract_features(str(candidate))
    except Exception:
        _FEATURES_FAILED = True
        tb = traceback.format_exc()
        # Print prominently so the exception is never invisible, then also log
        # it if the test logger has been initialised (may not be on very first
        # call before _setup_test_logging() runs).
        print(f"\n  ✗  extract_features() raised an exception:\n{tb}", flush=True)
        _base_logger = logging.getLogger("the_crate.tests")
        if _base_logger.handlers:
            # Adapter needs test_name in extra dict for the formatter.
            _make_adapter(_base_logger, "_GET_FEATURES").error(
                "extract_features raised:\n%s", tb)
        return None

    return _FEATURES_CACHE


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_separator(title: str) -> None:
    """Print a double-rule banner around a test title.

    Args:
        title: Short title string, e.g. 'TEST 1 — ANALYZE TRACK'.
    """
    width = 52
    print("\n" + "═" * width)
    print(f" {title}")
    print("═" * width)


def _dash() -> str:
    """Return the single-rule footer line used at the end of each test."""
    return "─" * 52


def _opt(value, fmt: str = "") -> str:
    """Format an optional (possibly-None) value for display.

    Args:
        value: The value to display; None renders as '—'.
        fmt:   Optional format spec applied to non-None values, e.g. '.4f'.

    Returns:
        Formatted string or '—'.
    """
    if value is None:
        return "—"
    return format(value, fmt) if fmt else str(value)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — ANALYZE TRACK
# ══════════════════════════════════════════════════════════════════════════════

def test_analyze_track() -> bool:
    """Validate that extract_features() produces a correct TrackFeatures record.

    Runs extract_features() on the test audio file, checks structural presence
    of all Level-1 fields, validates numeric ranges, and writes the full feature
    dict to OUTPUT_DIR/track_features.json.

    Returns:
        True if all assertions passed, False if any failed (failures are
        collected and reported rather than raising on the first one).

    Raises:
        Exception: Re-raised if an unexpected exception (not an assertion
        failure) occurs, after logging the full traceback.
    """
    logger = _setup_test_logging()
    log    = _make_adapter(logger, "TEST_ANALYZE_TRACK")

    log.info("Starting — audio: %s", TEST_AUDIO_PATH)

    _print_separator("TEST 1 — ANALYZE TRACK")

    failures: list[str] = []  # collect all assertion failures before reporting

    # ── Step 1: extract features and time the call ──────────────────────────
    try:
        t0 = time.perf_counter()
        f  = _get_features()
        elapsed = time.perf_counter() - t0
    except Exception:
        log.error("extract_features raised an exception:\n%s", traceback.format_exc())
        print("✗ FAIL — exception during extraction (see ab_tests/test_analysis.log)")
        raise

    if f is None:
        msg = "No audio file available — test skipped."
        log.warning(msg)
        print(f"  ⚠  SKIP: {msg}")
        return False

    log.info("Extraction complete — elapsed: %.2f s  pipeline_level: %d",
             elapsed, f.pipeline_level)

    # ── Step 2: structural presence of all required Level-1 fields ──────────
    log.info("Step 2 — validating structural completeness (%d required fields)",
             len(REQUIRED_L1_FIELDS))

    for field in REQUIRED_L1_FIELDS:
        val = getattr(f, field, "__missing__")
        if val == "__missing__" or val is None:
            msg = f"Field '{field}' is missing or None"
            failures.append(msg)
            log.warning("ASSERT FAIL — %s", msg)

    # ── Step 3: numeric range assertions ────────────────────────────────────
    log.info("Step 3 — validating numeric ranges")

    def _check(condition: bool, field: str, expected: str, actual) -> None:
        """Register a failure if condition is False; log pass/fail."""
        if not condition:
            msg = f"Range check '{field}': expected {expected}, got {actual!r}"
            failures.append(msg)
            log.warning("ASSERT FAIL — %s", msg)

    _check(60.0 <= f.bpm <= 220.0,
           "bpm", "[60.0, 220.0]", f.bpm)

    # RhythmExtractor2013 (multifeature) returns a correlation score, not a
    # probability — values commonly exceed 1.0 (e.g. 3.0 is normal).
    # We only assert the floor; there is no meaningful upper bound.
    _check(f.bpm_confidence >= 0.0,
           "bpm_confidence", ">= 0.0", f.bpm_confidence)

    _check(f.key in analyze.KEY_INDEX,
           "key", f"one of {set(analyze.KEY_INDEX.keys())}", f.key)

    _check(f.scale in {"major", "minor"},
           "scale", "{'major','minor'}", f.scale)

    # camelot must be a valid code like "8A" / "12B" or the sentinel "?".
    camelot_ok = (f.camelot == "?" or
                  bool(re.match(r'^([1-9]|1[0-2])[AB]$', f.camelot)))
    _check(camelot_ok, "camelot", r"r'^([1-9]|1[0-2])[AB]$' or '?'", f.camelot)

    _check(0.0 <= f.key_strength <= 1.0,
           "key_strength", "[0.0, 1.0]", f.key_strength)

    # Essentia's DSP danceability can exceed 1.0 (it is not a probability).
    _check(0.0 <= f.danceability <= 3.0,
           "danceability", "[0.0, 3.0]", f.danceability)

    _check(f.pipeline_level in {1, 2, 3, 4, 5},
           "pipeline_level", "{1,2,3,4,5}", f.pipeline_level)

    _check(f.duration > 0.0,
           "duration", "> 0.0", f.duration)

    if f.mfcc_mean is not None:
        _check(len(f.mfcc_mean) == 13,
               "mfcc_mean length", "13", len(f.mfcc_mean))

    if f.bark_mean is not None:
        _check(len(f.bark_mean) == 27,
               "bark_mean length", "27", len(f.bark_mean))

    if f.energy_curve is not None:
        _check(len(f.energy_curve) > 0,
               "energy_curve length", "> 0", len(f.energy_curve))
        bad_frames = [v for v in f.energy_curve if v < 0.0]
        _check(not bad_frames,
               "energy_curve values", "all >= 0.0",
               f"{len(bad_frames)} negative values" if bad_frames else "ok")

    n_assertions = len(REQUIRED_L1_FIELDS) + 11  # 25 structural + 11 range checks

    # ── Step 4: persist to JSON ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "track_features.json"
    payload  = dataclasses.asdict(f)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    size_bytes = out_path.stat().st_size
    log.info("JSON written — path: %s  size: %d bytes", out_path, size_bytes)

    # ── Step 5: human-readable stdout summary ───────────────────────────────
    audio_name = _RESOLVED_AUDIO_PATH.name if _RESOLVED_AUDIO_PATH else str(TEST_AUDIO_PATH)
    dur_min    = f.duration / 60

    # EffNet embedding stats (optional)
    if f.effnet_embedding is not None:
        import numpy as np  # already a transitive dep via analyze / essentia
        emb_arr  = np.array(f.effnet_embedding)
        emb_norm = float(np.linalg.norm(emb_arr))
        emb_str  = f"1280-D vector  ✓  (L2 norm: {emb_norm:.4f})"
    else:
        emb_str  = "—  (model not installed)"

    # TempoCNN display: shown only when DSP confidence was too low to trust.
    if f.bpm_cnn is not None:
        tempo_cnn_str = f"{f.bpm_cnn:.2f} BPM"
    else:
        tempo_cnn_str = "—  (DSP confidence was sufficient)"

    # energy_curve stats
    if f.energy_curve:
        import numpy as np
        ec      = f.energy_curve
        ec_str  = (f"{len(ec)} frames  "
                   f"[min {min(ec):.4f}  mean {float(np.mean(ec)):.4f}  max {max(ec):.4f}]")
    else:
        ec_str = "—"

    print(f" File:            {audio_name}")
    print(f" Duration:        {dur_min:.2f} min")
    print(f" BPM:             {f.bpm:.1f}  (confidence {f.bpm_confidence:.2f})")
    print(f" Key:             {f.key} {f.scale}  →  Camelot {f.camelot}")
    print(f" Key strength:    {f.key_strength:.2f}")
    print(f" Danceability:    {f.danceability:.2f}")
    print(f" Onset rate:      {f.onset_rate:.2f} /s")
    print(f" Loudness:        {f.loudness:.1f}")
    print(f" Dynamic cmplx:   {f.dynamic_complexity:.2f}")
    print(f" Centroid:        {f.spectral_centroid:.0f} Hz")
    print(f" Complexity:      {f.spectral_complexity:.1f}")
    print(f" Rolloff:         {f.spectral_rolloff:.0f} Hz")
    print(f" Intro ends:      {f.intro_end:.1f} s")
    print(f" Outro starts:    {f.outro_start:.1f} s")
    print(f" Pipeline level:  {f.pipeline_level}/5")
    print(f" EffNet embed:    {emb_str}")
    print(f" TempoCNN BPM:    {tempo_cnn_str}")
    print(f" Mood aggress.:   {_opt(f.mood_aggressive, '.2f')}")
    print(f" Danceability NN: {_opt(f.danceability_nn, '.2f')}")
    print(f" energy_curve:    {ec_str}")
    print(f" Breakdowns:      (breakdown detection runs in Test 2)")
    print(_dash())

    # relative path for tidy display
    try:
        display_path = out_path.relative_to(Path.cwd())
    except ValueError:
        display_path = out_path

    print(f" Output:  {display_path}  ({size_bytes:,} bytes)")
    print(f" Elapsed: {elapsed:.1f} s")

    if failures:
        for msg in failures:
            print(f"   ✗ {msg}")
        print(f" Result:  ✗ FAIL  ({len(failures)} of {n_assertions} assertions failed)")
        log.warning("TEST RESULT: FAIL — %d/%d assertions failed", len(failures), n_assertions)
    else:
        print(f" Result:  ✓ PASS  (all {n_assertions} assertions passed)")
        log.info("TEST RESULT: PASS — %d assertions passed", n_assertions)

    print("═" * 52)
    return len(failures) == 0


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — EFFNET EMBEDDING + METADATA PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def test_effnet_embedding() -> bool:
    """Validate EffNet embedding extraction and (when DB is up) round-trip persistence.

    Step A validates the embedding numerically: correct dimension, all-finite
    values, unit L2 norm, non-zero vector.

    Step B (skipped when database.DB_AVAILABLE is False) inserts a synthetic
    test row, upserts the embedding, retrieves the row, runs a self-similarity
    search, and cleans up.

    Returns:
        True if all applicable assertions passed.  Returns False (not raises)
        on assertion failures so the caller can continue to the final tally.

    Raises:
        Exception: Re-raised for unexpected non-assertion exceptions after
        logging the full traceback.
    """
    import numpy as np

    logger = _setup_test_logging()
    log    = _make_adapter(logger, "TEST_EFFNET_EMBED")

    log.info("Starting — database available: %s", database.DB_AVAILABLE)

    _print_separator("TEST 2 — EFFNET EMBEDDING + METADATA PERSISTENCE")

    failures: list[str]      = []
    skipped:  list[str]      = []
    track_id_inserted: str   = ""  # set in Step B for cleanup in finally block
    n_assertions              = 0

    # ── STEP A — EXTRACTION ──────────────────────────────────────────────────
    # Re-use the cached features so extract_features is called at most once
    # per process run.
    f = _get_features()
    if f is None:
        msg = "No audio file available — Test 2 skipped entirely."
        log.warning(msg)
        print(f"  ⚠  SKIP: {msg}")
        print("═" * 52)
        return False

    emb = f.effnet_embedding

    # A1 — embedding presence check (skip rest of Step A if None)
    n_assertions += 1
    if emb is None:
        msg = ("EffNet model not available — pipeline_level is 1.  "
               "Install `essentia-tensorflow`; models download automatically on first use.")
        log.warning(msg)
        skip_msg = "EffNet embedding is None — all numeric embedding assertions skipped"
        skipped.append(skip_msg)
        log.warning("SKIPPED (no EffNet model): %s", skip_msg)
        # Mark DB sub-steps skipped too — no embedding to persist.
        skipped.append("All DB persistence assertions (no embedding to persist)")
        _print_embedding_summary(
            f, emb=None, db_used=False, track_id="",
            failures=failures, skipped=skipped, n_assertions=n_assertions,
            cosine_distance=None,
        )
        # Return True: missing model is a skip, not a test failure.
        return True

    emb_arr = np.array(emb, dtype=np.float64)

    # A2 — dimension
    n_assertions += 1
    if len(emb) != 1280:
        msg = f"effnet_embedding length: expected 1280, got {len(emb)}"
        failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
    else:
        log.info("Step A2 — dim OK: %d", len(emb))

    # A3 — all finite (no NaN or Inf)
    n_assertions += 1
    non_finite = [i for i, x in enumerate(emb) if not math.isfinite(x)]
    if non_finite:
        msg = f"effnet_embedding has {len(non_finite)} non-finite values at indices {non_finite[:5]}"
        failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
    else:
        log.info("Step A3 — all %d values finite", len(emb))

    # A4 — L2 norm close to 1.0 (unit vector, normalised by analyze.py)
    n_assertions += 1
    norm = float(np.linalg.norm(emb_arr))
    if not math.isclose(norm, 1.0, abs_tol=1e-3):
        msg = f"L2 norm expected 1.0 ± 1e-3, got {norm:.6f}"
        failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
    else:
        log.info("Step A4 — L2 norm: %.6f ✓", norm)

    # A5 — non-zero vector (norm of the raw, un-normalised vector would be >1e-6;
    # since we only have the normalised version the norm check above already
    # guarantees this, but we keep the assertion explicit as documentation).
    n_assertions += 1
    if norm <= 1e-6:
        msg = f"Embedding appears to be the zero vector (norm {norm:.2e})"
        failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
    else:
        log.info("Step A5 — non-zero vector ✓")

    # ── STEP B — METADATA ROUND-TRIP ────────────────────────────────────────
    db_used        = database.DB_AVAILABLE
    cosine_distance: float | None = None
    metadata_stored: dict         = {}

    if not db_used:
        skip_reason = ("database.DB_AVAILABLE is False — "
                       "start Docker (`docker compose up -d`) to enable DB tests")
        log.warning("Step B SKIPPED — %s", skip_reason)
        skipped.append("All DB persistence assertions (DB unavailable)")
        print(f"\n  ⚠  DB persistence skipped: {skip_reason}\n")
    else:
        try:
            # B1 — insert synthetic test row
            n_assertions += 1
            test_run_id   = str(uuid.uuid4())
            synthetic_path = "ab_tests/synthetic/test_embed_track.wav"
            track_id_inserted = database.insert_track(
                crate_path=synthetic_path,
                filename="test_embed_track.wav",
                duration=f.duration,
            )
            log.info("Step B1 — DB row inserted  track_id: %s", track_id_inserted)

            # B2 — build metadata dict and update features
            n_assertions += 1
            metadata_stored = {
                "bpm":            f.bpm,
                "key":            f.key,
                "scale":          f.scale,
                "camelot":        f.camelot,
                "pipeline_level": f.pipeline_level,
                "mood_aggressive":  f.mood_aggressive,
                "danceability_nn":  f.danceability_nn,
                # Extra traceability keys required by the spec.
                "test_run_id":    test_run_id,
                "test_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "source_notes":   "synthetic test record — not a real vinyl track",
            }
            database.update_track_features(
                track_id_inserted, metadata_stored, f.pipeline_level)
            log.info("Step B2 — features updated with test metadata (run_id: %s)",
                     test_run_id)

            # B3 — upsert the EffNet embedding
            n_assertions += 1
            # MODEL_VERSION is defined in crate.py (derived from the REGISTRY),
            # fall back to the hard-coded constant if crate failed to import.
            model_version = getattr(
                __import__("crate", fromlist=["MODEL_VERSION"]),
                "MODEL_VERSION",
                "discogs-effnet-bs64-1",
            )
            database.upsert_effnet_embedding(
                track_id_inserted, emb, model_version)
            log.info("Step B3 — embedding upserted  model: %s  dim: %d",
                     model_version, len(emb))

            # B4 — retrieve and assert round-trip correctness
            n_assertions += 2  # crate_path + bpm round-trip
            row = database.get_track(track_id_inserted)

            if row.get("crate_path") != synthetic_path:
                msg = (f"crate_path round-trip mismatch: "
                       f"expected {synthetic_path!r}, got {row.get('crate_path')!r}")
                failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
            else:
                log.info("Step B4a — crate_path round-trip ✓")

            retrieved_bpm = (row.get("features") or {}).get("bpm")
            if retrieved_bpm is None or not math.isclose(retrieved_bpm, f.bpm, rel_tol=1e-2):
                msg = (f"BPM round-trip mismatch: "
                       f"expected {f.bpm:.2f}, got {retrieved_bpm}")
                failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
            else:
                log.info("Step B4b — BPM round-trip ✓ (%.2f)", retrieved_bpm)

            # B5 — self-similarity search (the track must find itself first)
            n_assertions += 1
            results = database.find_similar_effnet(emb, n=1, exclude_track_id=None)
            if not results:
                msg = "find_similar_effnet returned no results"
                failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
            else:
                cosine_distance = results[0].get("cosine_distance", 999.0)
                if cosine_distance >= 0.01:
                    msg = (f"Self-similarity cosine_distance expected < 0.01, "
                           f"got {cosine_distance:.6f}")
                    failures.append(msg); log.warning("ASSERT FAIL — %s", msg)
                else:
                    log.info("Step B5 — self-similarity ✓  cosine_distance: %.8f",
                             cosine_distance)

        except Exception:
            log.error("Step B raised an unexpected exception:\n%s",
                      traceback.format_exc())
            print("✗ FAIL — unexpected exception in Step B "
                  "(see ab_tests/test_analysis.log)")
            raise
        finally:
            # B6 — always clean up the synthetic row so DB state is not polluted
            if track_id_inserted:
                try:
                    database.delete_track(track_id_inserted)
                    log.info("Step B6 — cleanup: synthetic test row deleted (%s)",
                             track_id_inserted)
                except Exception as e:
                    log.warning("Cleanup delete_track failed: %s", e)

    # ── STEP C — persist embedding JSON and print summary ───────────────────
    _print_embedding_summary(
        f,
        emb=emb,
        db_used=db_used,
        track_id=track_id_inserted,
        failures=failures,
        skipped=skipped,
        n_assertions=n_assertions,
        cosine_distance=cosine_distance,
        metadata_stored=metadata_stored,
        model_version=model_version if db_used else "discogs-effnet-bs64-1",
    )

    if failures:
        log.warning("TEST RESULT: FAIL — %d/%d assertions failed, %d skipped",
                    len(failures), n_assertions, len(skipped))
    else:
        log.info("TEST RESULT: PASS — %d assertions passed, %d skipped",
                 n_assertions - len(skipped), len(skipped))

    return len(failures) == 0


def _print_embedding_summary(
    f: "analyze.TrackFeatures",
    *,
    emb,
    db_used: bool,
    track_id: str,
    failures: list,
    skipped: list,
    n_assertions: int,
    cosine_distance: "float | None",
    metadata_stored: dict = None,
    model_version: str = "discogs-effnet-bs64-1",
) -> None:
    """Print the stdout summary block for Test 2 and write effnet_embedding.json.

    Separated from the main test body so the summary is always printed even
    when Step B raises.

    Args:
        f:                The TrackFeatures record.
        emb:              The effnet_embedding list (or None).
        db_used:          Whether DB persistence was attempted.
        track_id:         The inserted (and deleted) test track_id.
        failures:         List of assertion failure message strings.
        skipped:          List of skip reason strings.
        n_assertions:     Total number of assertions attempted.
        cosine_distance:  Result of the self-similarity search, or None.
        metadata_stored:  The dict passed to update_track_features, or None.
        model_version:    String model identifier.
    """
    import numpy as np

    metadata_stored = metadata_stored or {}

    # Embedding stats (only if available)
    if emb is not None:
        emb_arr       = np.array(emb, dtype=np.float64)
        norm          = float(np.linalg.norm(emb_arr))
        finite_count  = sum(1 for x in emb if math.isfinite(x))
        preview_vals  = [f"{x:.4f}" for x in emb[:8]]
        preview_str   = "[" + ", ".join(preview_vals) + ", ...]"
    else:
        norm = finite_count = 0
        preview_str = "—"

    # DB persistence display
    db_label = "✓ enabled" if db_used else "✗ unavailable (start Docker)"
    tid_short = (track_id[:4] + "..." + track_id[-4:]) if len(track_id) > 8 else track_id

    print(f" Embedding dim:      {len(emb) if emb else '—'}")
    print(f" L2 norm:            {norm:.4f}  {'✓' if emb and abs(norm - 1.0) < 1e-3 else '✗'}")
    print(f" Finite values:      {finite_count} / {len(emb) if emb else 0}  "
          f"{'✓' if emb and finite_count == len(emb) else '✗'}")
    print(f" Zero vector check:  {'✓  (norm: ' + f'{norm:.4f})' if emb else '—'}")

    print(f"\n DB persistence:     {db_label}")
    if db_used and track_id:
        bpm_disp = f"{metadata_stored.get('bpm', 0.0):.1f}"
        key_disp = f"{metadata_stored.get('key', '?')} {metadata_stored.get('scale', '')}".strip()
        cam_disp = metadata_stored.get("camelot", "?")
        extra_keys = [k for k in ("test_run_id", "test_timestamp", "source_notes")
                      if k in metadata_stored]
        print(f" Track inserted:     {tid_short}")
        print(f" Metadata stored:    bpm={bpm_disp}  key={key_disp}  camelot={cam_disp}")
        print(f" Extra metadata:     {', '.join(extra_keys)}  ✓")
        print(f" Embedding upserted: {model_version}  dim={len(emb) if emb else 0}")
        if cosine_distance is not None:
            cd_ok = cosine_distance < 0.01
            print(f" Self-similarity:    cosine_distance = {cosine_distance:.8f}  "
                  f"{'✓' if cd_ok else '✗'}")
        print(f" Cleanup:            synthetic row deleted  ✓")
    elif not db_used:
        print(f" (DB sub-steps skipped — see log for details)")

    # Write embedding JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "effnet_embedding.json"
    payload = {
        "track_id":      track_id if track_id else "N/A (DB unavailable)",
        "model_version": model_version,
        "dim":           len(emb) if emb else 0,
        "l2_norm":       round(norm, 6),
        "embedding":     emb if emb else [],
        "metadata":      metadata_stored,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    size_bytes = out_path.stat().st_size

    print(_dash())

    try:
        display_path = out_path.relative_to(Path.cwd())
    except ValueError:
        display_path = out_path

    print(f" Output:  {display_path}")
    print(f"          ({size_bytes:,} bytes — first 8 values shown below)")
    print(f" Preview: {preview_str}")

    # n_assertions tracks only ATTEMPTED assertions; skipped ones were never
    # counted in n_assertions, so we must NOT subtract len(skipped) from it.
    n_passed = n_assertions - len(failures)

    # Special case: embedding was None — all assertions skipped, nothing failed.
    embedding_skipped = emb is None

    if failures:
        for msg in failures:
            print(f"   ✗ {msg}")
        print(f" Result:  ✗ FAIL  ({len(failures)} assertions failed, "
              f"{len(skipped)} skipped)")
    elif embedding_skipped:
        # Not a failure — model simply not installed.
        print(f" Result:  ⚠ SKIPPED  (EffNet model not installed — "
              f"install `essentia-tensorflow` and models download automatically)")
    else:
        print(f" Result:  ✓ PASS  ({n_passed} assertions passed, "
              f"{len(skipped)} skipped)")

    print("═" * 52)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — pure logic, no audio / models / network (DB tests gated)
# ══════════════════════════════════════════════════════════════════════════════
# Tests 3–8 exercise the scoring, harmonic, tempo, breakdown, standardisation and
# persistence logic directly with synthetic inputs, so they are fast and
# deterministic and run even when no audio file or ML model is present. They
# share a compact scaffold (_unit_test) rather than repeating Test 1/2's full
# banner boilerplate.

def _unit_test(test_name: str, title: str, body) -> bool:
    """Run a compact unit test.

    `body(check)` performs the test, calling check(condition, message) for each
    assertion; failures are collected and reported together rather than aborting
    on the first. Returns True iff every assertion passed.

    Args:
        test_name: grep label for the log, e.g. 'TEST_CAMELOT'.
        title:     banner title printed to stdout.
        body:      callable taking a single `check(cond, msg)` callback.
    """
    logger = _setup_test_logging()
    log    = _make_adapter(logger, test_name)
    log.info("Starting")
    _print_separator(title)

    failures: list[str] = []
    passed = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal passed
        if cond:
            passed += 1
        else:
            failures.append(msg)
            log.warning("ASSERT FAIL — %s", msg)

    try:
        body(check)
    except Exception:
        log.error("%s raised an unexpected exception:\n%s",
                  test_name, traceback.format_exc())
        print(f"   💥 exception during {test_name} (see ab_tests/test_analysis.log)")
        raise

    total = passed + len(failures)
    if failures:
        for m in failures:
            print(f"   ✗ {m}")
        print(f" Result:  ✗ FAIL  ({len(failures)} of {total} assertions failed)")
        log.warning("TEST RESULT: FAIL — %d/%d failed", len(failures), total)
    else:
        print(f" Result:  ✓ PASS  (all {total} assertions passed)")
        log.info("TEST RESULT: PASS — %d assertions", total)
    print("═" * 52)
    return not failures


def _synthetic_track(**overrides) -> "analyze.TrackFeatures":
    """Build a TrackFeatures with plausible defaults for scoring tests.

    Defaults describe a generic 130 BPM, 8A, mid-energy techno tool with a
    Level-2 EffNet embedding; pass overrides to vary a single dimension under test.
    """
    base = dict(
        path="synthetic.wav", duration=120.0,
        bpm=130.0, bpm_confidence=1.0, key="A", scale="minor", camelot="8A",
        key_strength=0.8, energy_curve=[0.5] * 120, spectral_complexity=5.0,
        intro_end=16.0, outro_start=104.0,
        mfcc_mean=[1.0, 0.0, 0.0], effnet_embedding=[1.0, 0.0, 0.0],
        pipeline_level=2,
    )
    base.update(overrides)
    return analyze.TrackFeatures(**base)


def test_camelot_and_keys() -> bool:
    """to_camelot mapping + key_relationship_label across every reachable relation."""
    def body(check):
        # ── to_camelot: a sample of the wheel + the atonal sentinel ──
        check(analyze.to_camelot("A", "minor") == "8A", "A minor → 8A")
        check(analyze.to_camelot("C", "major") == "8B", "C major → 8B")
        check(analyze.to_camelot("Bb", "major") == "6B", "Bb major → 6B (enharmonic)")
        check(analyze.to_camelot("???", "minor") == "?",
              "unknown key → '?' sentinel (was: collapsed to index 0 → 5A)")

        # ── key_relationship_label: every branch that can actually fire ──
        check(analyze.key_relationship_label("8A", "8A") == "Same key", "8A/8A = Same key")
        check(analyze.key_relationship_label("8A", "9A") == "Adjacent", "8A/9A = Adjacent")
        check(analyze.key_relationship_label("8A", "7A") == "Adjacent", "8A/7A = Adjacent")
        check(analyze.key_relationship_label("8A", "8B") == "Relative (mood shift)",
              "8A/8B = Relative")
        check(analyze.key_relationship_label("8A", "2B") == "Dissonant", "8A/2B = Dissonant")
        check(analyze.key_relationship_label("?", "8A") == "Unknown", "'?' = Unknown")
        check(analyze.key_relationship_label("", "8A") == "Unknown", "empty = Unknown")
        # NOTE: the 'Energy boost (+7)' branch requires diff == 7, but
        # diff = min(|d|, 12-|d|) maxes at 6, so that branch is unreachable —
        # documented here rather than asserted. (See summary / known issues.)
    return _unit_test("TEST_CAMELOT", "TEST 3 — CAMELOT & KEY RELATIONSHIPS", body)


def test_bpm_compat() -> bool:
    """bpm_compatibility thresholds + half/double folding, and bpm_delta sign/fold."""
    def body(check):
        bc = analyze.bpm_compatibility
        check(bc(130, 130) == 1.0, "±0 BPM → 1.0")
        check(bc(130, 133) == 1.0, "±3 BPM (≤4) → 1.0")
        check(abs(bc(130, 136) - 0.75) < 1e-9, "±6 BPM → 0.75 (mid of 4..8 ramp)")
        check(abs(bc(130, 140) - 0.375) < 1e-9, "±10 BPM → 0.375 (in 8..16 ramp)")
        check(bc(130, 150) == 0.0, "±20 BPM (>16) → 0.0 floor")
        check(bc(0, 130) == 0.0 and bc(130, 0) == 0.0, "zero BPM → 0.0")
        # Half/double folding: a track at half/double tempo is the same pulse.
        check(bc(130, 65) == 1.0, "130 vs 65 (half-time) → 1.0")
        check(bc(130, 260) == 1.0, "130 vs 260 (double-time) → 1.0")

        bd = analyze.bpm_delta
        check(abs(bd(130, 133) - 3.0) < 1e-9, "bpm_delta 130→133 = +3.0")
        check(abs(bd(130, 65)) < 1e-9, "bpm_delta 130→65 folds to ~0")
        check(bd(130, 120) < 0, "bpm_delta 130→120 is negative (slower)")
    return _unit_test("TEST_BPM", "TEST 4 — BPM COMPATIBILITY & FOLDING", body)


def test_mix_score() -> bool:
    """The two-stage scoring guarantees: immutable base, bounded modifiers, fallbacks."""

    def body(check):
        t1 = _synthetic_track(effnet_embedding=[1.0, 0.0, 0.0])
        t2 = _synthetic_track(effnet_embedding=[0.8, 0.6, 0.0], bpm=150.0,
                              camelot="2B", energy_curve=[0.9] * 120,
                              spectral_complexity=20.0)
        expected_base = analyze.cosine_sim(t1.effnet_embedding, t2.effnet_embedding)

        # (1) Immutable base: with every modifier disabled, total == effnet_base.
        all_off = analyze.ModifierStrengths(bpm=0, harmonic=0, energy=0,
                                            transition=0, mood=0, emotional=0, density=0)
        s_off = analyze.mix_score(t1, t2, strengths=all_off)
        check(abs(s_off["effnet_base"] - expected_base) < 1e-9,
              "effnet_base == cosine of embeddings")
        check(abs(s_off["total"] - expected_base) < 1e-9,
              "all modifiers off → total collapses to effnet_base")

        # (2) A modifier can only pull the score DOWN, never above the base.
        s_on = analyze.mix_score(t1, t2, mode="balanced")
        check(s_on["total"] <= s_on["effnet_base"] + 1e-9,
              "balanced total never exceeds the immutable base")
        for m in analyze.MODIFIER_NAMES:
            check(0.0 <= s_on[m] <= 1.0, f"modifier '{m}' is bounded in [0,1]")

        # (3) EffNet base is floored at 0 (opposite-pointing embeddings).
        t_neg = _synthetic_track(effnet_embedding=[-1.0, 0.0, 0.0])
        check(analyze.mix_score(t1, t_neg, strengths=all_off)["effnet_base"] == 0.0,
              "negative cosine floored to 0.0")

        # (4) MFCC fallback when embeddings are absent.
        m1 = _synthetic_track(effnet_embedding=None, mfcc_mean=[1.0, 0.0, 0.0])
        m2 = _synthetic_track(effnet_embedding=None, mfcc_mean=[0.0, 1.0, 0.0])
        s_mfcc = analyze.mix_score(m1, m2, strengths=all_off)
        check(s_mfcc["timbre_source"] == "mfcc", "no embedding → MFCC fallback base")
        check(abs(s_mfcc["effnet_base"]) < 1e-9, "orthogonal MFCCs → base ~0")

        # (5) Mode presets resolve to their configured strengths.
        s_safe = analyze.mix_score(t1, t2, mode="safe")
        check(s_safe["modifier_strengths"].bpm == 1.5, "safe mode amplifies bpm to 1.5")
        check(s_safe["modifier_strengths"].mood == 0.0, "safe mode disables mood")
        s_creative = analyze.mix_score(t1, t2, mode="creative")
        check(s_creative["mood_mode"] == "contrast", "creative mode uses mood contrast")
    return _unit_test("TEST_MIX_SCORE", "TEST 5 — TWO-STAGE MIX SCORE", body)


def test_breakdowns() -> bool:
    """detect_breakdowns: a sharp dip is HIGH, brief dips are filtered, flat → NONE."""
    def body(check):
        # A clear 10s drop to silence in an otherwise busy track.
        energy = [1.0] * 20 + [0.0] * 10 + [1.0] * 20
        cmplx  = [1.0] * 20 + [0.0] * 10 + [1.0] * 20
        events, reliability = crate.detect_breakdowns(energy, cmplx)
        check(len(events) == 1, f"one breakdown detected (got {len(events)})")
        check(reliability == "HIGH", f"sharp drop → HIGH (got {reliability})")
        if events:
            check(events[0]["duration_sec"] >= crate.BREAKDOWN_MIN_SECONDS,
                  "breakdown meets the minimum-duration filter")

        # A 2-second dip (< BREAKDOWN_MIN_SECONDS) must be filtered out.
        short = [1.0] * 20 + [0.0] * 2 + [1.0] * 20
        ev2, rel2 = crate.detect_breakdowns(short, short)
        check(ev2 == [] and rel2 == "NONE", "sub-4s dip filtered → NONE")

        # Flat curves carry no breakdown information (normalise to zeros) → NONE.
        flat = [0.7] * 40
        ev3, rel3 = crate.detect_breakdowns(flat, flat)
        check(ev3 == [] and rel3 == "NONE", "flat curves → NONE")

        # Empty input is handled.
        check(crate.detect_breakdowns([], []) == ([], "NONE"), "empty curves → NONE")
    return _unit_test("TEST_BREAKDOWNS", "TEST 6 — BREAKDOWN DETECTION", body)


def test_standardisation() -> bool:
    """crate._standardize / _best_window: mono, 16kHz, ≤120s, picks the busy window."""
    import numpy as np

    def body(check):
        sr = crate.ML_SAMPLE_RATE
        target = crate.TARGET_SECONDS

        # 200s stereo signal, silent except a loud, spectrally-busy burst at 80–120s.
        n = 200 * sr
        audio = np.zeros((n, 2), dtype=np.float32)
        rng = np.random.default_rng(0)
        busy = slice(80 * sr, 120 * sr)
        audio[busy, :] = rng.standard_normal((40 * sr, 2)).astype(np.float32) * 0.5

        excerpt, start, end = crate._standardize(audio, sr)
        check(excerpt.ndim == 1, "excerpt is mono (1-D)")
        check(excerpt.dtype == np.float32, "excerpt is float32")
        check(len(excerpt) <= target * sr, f"excerpt ≤ {target}s")
        check(end - start == target, f"window length == {target}s")
        # The chosen window must overlap the busy region (80–120s), not the silence.
        check(start < 120 and end > 80, f"window {start}-{end}s overlaps the busy burst")

        # A clip shorter than one second must raise rather than emit garbage.
        raised = False
        try:
            crate._standardize(np.zeros(sr // 2, dtype=np.float32), sr)
        except ValueError:
            raised = True
        check(raised, "sub-1s input raises ValueError")
    return _unit_test("TEST_STANDARDISE", "TEST 7 — CRATE STANDARDISATION", body)


def test_db_crud() -> bool:
    """Track + session CRUD round-trip. Skipped (returns True) when the DB is down."""
    def body(check):
        if not database.DB_AVAILABLE:
            print("  ⚠  SKIP: database.DB_AVAILABLE is False — start Docker for DB tests.")
            return

        synthetic_path = f"ab_tests/synthetic/crud_{uuid.uuid4().hex}.wav"
        track_id = None
        session_id = None
        try:
            # ── tracks ──
            track_id = database.insert_track(synthetic_path, "crud_test.wav", 120.0)
            check(bool(track_id), "insert_track returns an id")
            check(database.insert_track(synthetic_path, "crud_test.wav", 120.0) == track_id,
                  "re-insert same crate_path is idempotent (same id)")
            check(database.get_track(track_id)["crate_path"] == synthetic_path,
                  "get_track round-trips crate_path")
            check(database.get_track_by_path(synthetic_path)["track_id"] == track_id,
                  "get_track_by_path resolves the row")
            check(database.get_track_by_path("ab_tests/does-not-exist.wav") is None,
                  "get_track_by_path misses cleanly (None)")

            database.update_track_features(track_id, {"bpm": 128.0, "duration": 120.0}, 3)
            row = database.get_track(track_id)
            check(row["analyzed_at"] is not None, "update_track_features stamps analyzed_at")
            check(row["pipeline_level"] == 3, "pipeline_level persisted")
            check(abs((row["features"] or {}).get("bpm", 0) - 128.0) < 1e-6,
                  "features JSONB round-trips bpm")

            total, analyzed, pending = database.count_tracks()
            check(total >= 1 and analyzed >= 1, "count_tracks reflects the analysed row")
            check(total == analyzed + pending, "count_tracks is internally consistent")
            check(any(r["track_id"] == track_id for r in database.list_tracks()),
                  "list_tracks includes the row")

            # ── sessions ──
            session_id = database.create_session()
            pos = database.log_track_played(session_id, track_id, detected_by="manual")
            check(pos == 1, "first logged track is position 1")
            sess = database.get_session(session_id)
            check(len(sess["tracklist"]) == 1, "get_session rebuilds the live tracklist")
            closed = database.close_session(session_id)
            check(len(closed) == 1, "close_session returns the snapshot")
            check(database.get_session(session_id)["ended_at"] is not None,
                  "closed session has ended_at")
        finally:
            # Clean up so the DB isn't polluted (session cascade-deletes its plays).
            if session_id:
                with database._transaction() as cur:
                    cur.execute("DELETE FROM mix_sessions WHERE session_id = %s;",
                                (session_id,))
            if track_id:
                database.delete_track(track_id)
                check(database.get_track_by_path(synthetic_path) is None,
                      "delete_track removes the row")
    return _unit_test("TEST_DB_CRUD", "TEST 8 — DATABASE CRUD & SESSIONS", body)


def test_energy_and_temperature() -> bool:
    """Directional energy target, Camelot energy direction, and temperature sampling."""
    def body(check):
        # ── camelot_energy_direction: signed single fifth-step on the wheel ──
        ced = analyze.camelot_energy_direction
        check(ced("8A", "9A") == 1, "8A→9A = +1 (up a fifth)")
        check(ced("8A", "7A") == -1, "8A→7A = -1 (down a fifth)")
        check(ced("12A", "1A") == 1, "12A→1A wraps to +1")
        check(ced("1A", "12A") == -1, "1A→12A wraps to -1")
        check(ced("8A", "8B") == 0, "8A→8B (diff letter) = 0")
        check(ced("8A", "10A") == 0, "8A→10A (two steps) = 0")
        check(ced("?", "8A") == 0 and ced("8A", "") == 0, "unknown/empty = 0")

        # ── energy_compatibility honours the target direction ──
        ec = analyze.energy_compatibility
        cur  = _synthetic_track(energy_curve=[0.50] * 120, camelot="8A")
        up   = _synthetic_track(energy_curve=[0.75] * 120, camelot="8A")  # rises
        down = _synthetic_track(energy_curve=[0.32] * 120, camelot="8A")  # falls
        check(ec(cur, up, target=+0.30) > ec(cur, up, target=-0.30),
              "a rising move scores higher under 'up' than 'down'")
        check(ec(cur, down, target=-0.30) > ec(cur, down, target=+0.30),
              "a falling move scores higher under 'down' than 'up'")
        # target=0.0 reproduces the old flat-is-ideal sweet spot (same-camelot → no nudge).
        flat = _synthetic_track(energy_curve=[0.52] * 120, camelot="8A")
        check(ec(cur, flat, target=0.0) == 1.0, "near-flat move at target 0 → 1.0")

        # ── energy_target flows through mix_score's energy modifier ──
        s_up = analyze.mix_score(cur, up, strengths=analyze.ModifierStrengths(energy_target=+0.30))
        s_dn = analyze.mix_score(cur, up, strengths=analyze.ModifierStrengths(energy_target=-0.30))
        check(s_up["energy"] >= s_dn["energy"],
              "mix_score energy modifier favours the rising candidate under 'up'")

        # ── sample_by_score: deterministic at temp 0, sampled subset above ──
        scored = [(f"t{i}", None, {"total": t})
                  for i, t in enumerate([0.9, 0.7, 0.5, 0.3, 0.1])]
        top2 = analyze.sample_by_score(scored, 2, temperature=0.0)
        check([x[0] for x in top2] == ["t0", "t1"], "temp 0 → deterministic top-2 in order")
        keys = {x[0] for x in scored}
        sampled = analyze.sample_by_score(scored, 3, temperature=1.0)
        check(len(sampled) == 3 and len({x[0] for x in sampled}) == 3,
              "temp>0 returns 3 distinct picks")
        check(all(x[0] in keys for x in sampled), "sampled picks all come from the input")
        check(len(analyze.sample_by_score(scored, 99, temperature=1.0)) == len(scored),
              "n >= population returns everything")

        # ── the dead 'Energy boost (+7)' label is gone; map has no stale entry ──
        check("Energy boost (+7)" not in analyze.HARMONIC_MOD_MAP,
              "stale 'Energy boost (+7)' removed from HARMONIC_MOD_MAP")
    return _unit_test("TEST_ENERGY_TEMP", "TEST 9 — ENERGY DIRECTION & TEMPERATURE", body)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run both tests and print a final pass/fail tally.

    Each test is called independently so a failure in Test 1 does not prevent
    Test 2 from running.  Exceptions that escape a test function are caught
    here so the other test still executes.
    """
    # Ensure directories exist before the first log write.
    AB_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = _setup_test_logging()
    log    = _make_adapter(logger, "RUNNER")
    log.info("═══ Test run started ═══")

    results: dict[str, bool | None] = {}

    for name, fn in [("test_analyze_track",     test_analyze_track),
                     ("test_effnet_embedding",  test_effnet_embedding),
                     ("test_camelot_and_keys",  test_camelot_and_keys),
                     ("test_bpm_compat",        test_bpm_compat),
                     ("test_mix_score",         test_mix_score),
                     ("test_breakdowns",        test_breakdowns),
                     ("test_standardisation",   test_standardisation),
                     ("test_db_crud",           test_db_crud),
                     ("test_energy_and_temperature", test_energy_and_temperature)]:
        try:
            results[name] = fn()
        except Exception:
            results[name] = None   # None = crashed (different from False = failed)
            log.error("Test %s raised an unhandled exception:\n%s",
                      name, traceback.format_exc())

    # Final tally
    passed  = sum(1 for v in results.values() if v is True)
    failed  = sum(1 for v in results.values() if v is False)
    crashed = sum(1 for v in results.values() if v is None)
    total   = len(results)

    print(f"\n{'═'*52}")
    print(f" FINAL TALLY — {passed}/{total} passed"
          + (f"  {failed} failed" if failed else "")
          + (f"  {crashed} crashed" if crashed else ""))
    for name, result in results.items():
        icon = "✓" if result is True else ("✗" if result is False else "💥")
        print(f"   {icon}  {name}")
    print(f" Log: {LOG_PATH}")
    print(f"{'═'*52}\n")

    log.info("═══ Test run complete — %d/%d passed ═══", passed, total)

    # Exit with a non-zero code if anything failed or crashed so CI can pick
    # it up as a failure.
    if failed or crashed:
        sys.exit(1)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
## TEST NOTES
# ══════════════════════════════════════════════════════════════════════════════
#
# ── ASSERTION INVENTORY ──────────────────────────────────────────────────────
#
# TEST 1 — test_analyze_track()   (36 total)
#
#   Structural (25) — one per REQUIRED_L1_FIELDS entry:
#     path, duration, bpm, bpm_confidence, key, scale, camelot, key_strength,
#     danceability, onset_rate, loudness, replay_gain, dynamic_complexity,
#     spectral_centroid, spectral_complexity, spectral_flux, spectral_rolloff,
#     zcr, mfcc_mean, bark_mean, energy_curve, complexity_curve,
#     intro_end, outro_start, pipeline_level
#     → each asserts the field exists on the dataclass and is not None.
#
#   Numeric range (11):
#     bpm             in [60.0, 220.0]
#     bpm_confidence  >= 0.0  (RhythmExtractor2013 multifeature returns a
#                               correlation score — values > 1.0 are normal)
#     key             one of analyze.KEY_INDEX keys
#     scale           in {'major', 'minor'}
#     camelot         matches r'^([1-9]|1[0-2])[AB]$' or '?'
#     key_strength    in [0.0, 1.0]
#     danceability    in [0.0, 3.0]   (DSP danceability, not a probability)
#     pipeline_level  in {1, 2, 3, 4, 5}
#     duration        > 0.0
#     mfcc_mean       length == 13
#     bark_mean       length == 27
#     energy_curve    length > 0 AND all values >= 0.0
#
#   None skipped when EffNet is unavailable — all 36 are Level-1 assertions.
#
# TEST 2 — test_effnet_embedding()   (up to 12 total)
#
#   STEP A — Extraction (5 assertions):
#     A1  effnet_embedding is not None
#     A2  len(effnet_embedding) == 1280
#     A3  all(math.isfinite(x) for x in emb)  → no NaN / Inf
#     A4  L2 norm == 1.0 ± 1e-3               → unit-normalised
#     A5  norm > 1e-6                          → not the zero vector
#
#   STEP B — DB persistence (7 assertions), SKIPPED when DB unavailable:
#     B1  insert_track succeeds, returns a non-empty UUID string
#     B2  update_track_features succeeds without raising
#     B3  upsert_effnet_embedding succeeds without raising
#     B4a get_track returns crate_path matching the synthetic path
#     B4b get_track returns features["bpm"] matching original to 2 d.p.
#     B5  find_similar_effnet(n=1) returns ≥ 1 result
#         first result cosine_distance < 0.01   (self-similarity)
#     B6  delete_track completes (cleanup — always attempted in finally)
#
#   Skipped when EffNet unavailable (pipeline_level == 1):
#     A2–A5 and all of Step B (no embedding to check or persist).
#
#   Skipped when DB unavailable (database.DB_AVAILABLE is False):
#     B1–B6 (all seven DB persistence assertions).
#
# ── ab_tests/ DIRECTORY CONTRACT ────────────────────────────────────────────
#
#   ab_tests/
#     test_analysis.py          This file. Lives here permanently.
#     test_analysis.log         Append-mode log. Never truncated.
#                               Lines are grep-able by [TEST_NAME].
#     output_tests/             Created at runtime if missing.
#       track_features.json     Full dataclasses.asdict() of the
#                               TrackFeatures record. Re-written each run.
#       effnet_embedding.json   Embedding + metadata snapshot. Re-written
#                               each run. Allows manual inspection of
#                               the 1280-D vector and round-trip values.
#
#   Nothing is written outside ab_tests/. sys.path is patched at the top
#   of the module so `import analyze` and `import database` resolve to the
#   project root regardless of the working directory.
#
# ── HOW TO ADD A NEW TEST ────────────────────────────────────────────────────
#
#   1. Define a function with this signature:
#          def test_<name>() -> bool:
#
#   2. At the top of the function, call:
#          logger = _setup_test_logging()
#          log    = _make_adapter(logger, "TEST_<NAME>")
#          log.info("Starting — ...")
#
#   3. Collect assertion failures into a `failures: list[str]` rather than
#      raising immediately. Use a helper like _check() for range checks so
#      all failures surface in a single run.
#
#   4. Write any output files to OUTPUT_DIR (not to the project root or any
#      other location). Log the path and size after writing.
#
#   5. Print a human-readable stdout summary between _print_separator(title)
#      and a final "✓ PASS / ✗ FAIL" line.
#
#   6. Register the new function in the `main()` loop:
#          for name, fn in [..., ("test_<name>", test_<name>)]:
#
#   7. Add its assertion inventory to the ## TEST NOTES block.
