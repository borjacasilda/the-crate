"""
The Crate — Crate Management Layer
--------------------------------
Gets audio INTO the system and curates the collection. This module owns the
INGEST + STANDARDISATION boundary and nothing else:

    * It does NOT touch Postgres directly — every read/write goes through
      database.py (the system of record).
    * It does NOT extract musical features — that is analyze.py's job; crate.py
      only calls extract_features() and hands the result to database.py.

WHY ./crate/ IS THE SINGLE SOURCE OF TRUTH
==========================================
Every track, regardless of origin (live capture, a USB stick, a folder), is
boiled down to ONE standardised artefact: a 120 s, mono, 16 kHz float32 WAV
named <excerpt_id>.wav under ./crate/. That uniformity is what makes live
recognition simple:

    The listener never has to care where a track came from or in what format.
    It captures a slice of live audio, runs the SAME standardisation, then
    fingerprints/embeds it and matches against the excerpts already sitting in
    ./crate/. Identical sample rate, channel count, and feature pipeline on both
    sides means a live snippet and its stored excerpt land in the exact same
    embedding space — no per-format normalisation, no rate mismatch, no "did we
    analyse this the same way" doubt. One folder, one format, one comparison.

PIPELINE (identical for all three input sources)
    raw audio -> mono -> 16 kHz resample -> pick best 120 s window
              -> write <excerpt_id>.wav to ./crate/ -> trigger analysis

INPUT SOURCES (exactly three — hard efficiency rule)
    1. LIVE CAPTURE   via sounddevice (built-in mic or any interface/sound card
                      that appears as a system input device).
    2. FILE IMPORT    one .wav/.mp3/.flac from any mounted storage.
    3. FOLDER IMPORT  batch file-import of a directory (USB drive / phone).

CLI
    add-rec   [--device N] [--label "artist - title"]
    add-file  <path> [--label "artist - title"]
    add-folder <path>
    list      [--pending]
    remove    <excerpt_id>
    analyze   (runs analyze_pending)
    health    (runs crate_health)

See the ## REFACTOR NOTES block at the end for the assumptions made about the
analyze.py and database.py interfaces that you should verify.
"""
import argparse
import json
import logging
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# config.py is the single source for paths/rates (see its docstring). We only
# import what we use here — the crate is always written at ML_SAMPLE_RATE, never
# the full 44.1 kHz rate, so SAMPLE_RATE is intentionally not imported.
from config import CRATE_DIR, GENRE_PROFILES, ML_SAMPLE_RATE, PROJECT_ROOT

# database.py is the ONLY persistence path. Importing it runs its one-time
# connect guard; if Postgres is down it stays importable but DB_AVAILABLE=False
# and every op raises DBUnavailableError — we surface that, never silently swap
# in a JSON file.
import database

# ════════════════════════════════════════════════════════════
#  LOGGING  (app.log at project root, append mode)
# ════════════════════════════════════════════════════════════
logger = logging.getLogger("thecrate.crate")

# Operations log path. Project root (NOT inside ./crate/) per spec, so it
# survives a crate wipe and sits next to docker-compose.yml / .env.
APP_LOG = PROJECT_ROOT / "app.log"


def _setup_file_logging() -> None:
    """Attach a single append-mode FileHandler to the 'thecrate' logger family.

    Attaching at 'thecrate' (the parent of thecrate.crate AND thecrate.db) means
    crate, database, and analyze chatter all land in one app.log. Idempotent: we
    tag our handler and bail if it's already present, so re-imports don't pile up
    duplicate handlers (which would multiply every log line).
    """
    parent = logging.getLogger("thecrate")
    for h in parent.handlers:
        # Our handler is marked so a second import is a no-op.
        if getattr(h, "_thecrate_crate_file", False):
            return
    handler = logging.FileHandler(APP_LOG, mode="a", encoding="utf-8")
    handler._thecrate_crate_file = True  # type: ignore[attr-defined]  # dedup marker
    # Timestamped + grep-friendly: "<ts> [thecrate.crate] INFO add SUCCESS ...".
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    parent.addHandler(handler)
    parent.setLevel(logging.INFO)
    # Stop here, don't bubble up to the root logger. analyze.py calls
    # logging.basicConfig() which installs a root StreamHandler; without this,
    # every thecrate.* record would print to the console (via root) AND be logged
    # to app.log (via our handler), double-printing the banner. We keep console
    # output deliberate (explicit print()s) and route all logging to app.log.
    parent.propagate = False


_setup_file_logging()  # Run at import so even a bare `import crate` logs to file.


# ════════════════════════════════════════════════════════════
#  ANALYSIS BACKEND SELECTION  (module preferred, subprocess fallback)
# ════════════════════════════════════════════════════════════
# Prefer importing analyze.py as a module: extract_features() then reuses the
# SAME process, so Essentia's TensorFlow models load once and are cached across
# every track in a batch instead of being re-instantiated per track. If the
# import explodes (e.g. essentia not installed in this interpreter), we fall
# back to shelling out to a fresh Python that does the extraction and prints the
# feature dict as JSON — slower (cold models per call) but keeps crate.py
# working. The active path is reported in the startup banner.
try:
    import analyze  # noqa: F401  (used below; also probed for ModelManager)
    ANALYSIS_MODE = "module"
    # EffNet graph filename -> model_version string the DB keys embeddings on
    # (e.g. 'discogs-effnet-bs64-1'). Pulled from analyze's registry so it can
    # never drift from the model actually producing the vectors.
    MODEL_VERSION = Path(analyze.ModelManager.REGISTRY["effnet"][0]).stem
except Exception as _imp_err:  # Wide net: ImportError OR any import-time failure.
    analyze = None
    ANALYSIS_MODE = "subprocess"
    # Fallback constant must match analyze.py's REGISTRY['effnet'] filename stem.
    MODEL_VERSION = "discogs-effnet-bs64-1"
    logger.warning("analyze import failed (%s) — using subprocess fallback for "
                   "feature extraction", _imp_err)

# Marker the subprocess prints before its JSON payload so we can separate it from
# extract_features()'s human progress prints ("📂 Loading...") on stdout.
_SUBPROC_MARKER = "<<<THECRATE_FEATURES>>>"
_SUBPROC_SNIPPET = (
    "import sys, json, dataclasses, analyze;"
    "prior = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else None;"
    "f = analyze.extract_features(sys.argv[1], bpm_prior=prior);"
    "print('" + _SUBPROC_MARKER + "' + json.dumps(dataclasses.asdict(f)))"
)


# ════════════════════════════════════════════════════════════
#  STANDARDISATION CONSTANTS
# ════════════════════════════════════════════════════════════
TARGET_SECONDS = 120        # Length of every crate excerpt.
TRIM_FRACTION = 0.10        # Skip first/last 10% (intro/outro) when picking a window.
BREAKDOWN_THRESHOLD = 0.4   # combined_score below this = candidate breakdown second.
BREAKDOWN_MIN_SECONDS = 4   # Runs shorter than this are filtered (brief dips).
BREAKDOWN_SHARP = 0.2       # Any breakdown second below this => HIGH reliability.

# Audio extensions we accept for file/folder import.
AUDIO_EXTS = {".wav", ".mp3", ".flac"}


# ════════════════════════════════════════════════════════════
#  LAZY AUDIO-LIBRARY IMPORTS
# ════════════════════════════════════════════════════════════
# soundfile/sounddevice/librosa are imported lazily so the module stays usable
# (list/health/remove/analyze) even when they aren't installed — only the ops
# that actually need them fail, with an actionable message. They are NOT yet in
# pyproject.toml; install with:  uv add soundfile sounddevice librosa
def _soundfile():
    """Return the soundfile module or raise a clear, actionable error."""
    try:
        import soundfile as sf
        return sf
    except ImportError as e:  # Re-raise with the fix the user needs.
        raise RuntimeError("soundfile is required for WAV/FLAC/MP3 I/O — "
                           "install it: `uv add soundfile`") from e


def _sounddevice():
    """Return the sounddevice module or raise a clear, actionable error."""
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as e:  # OSError: PortAudio shared lib missing.
        raise RuntimeError("sounddevice is required for live capture — "
                           "install it: `uv add sounddevice`") from e


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a 1-D float array, using librosa ONLY when rates differ.

    Args:
        audio: mono float32 samples.
        sr_in: source sample rate.
        sr_out: target sample rate.
    Returns:
        Resampled float32 array (the input unchanged when sr_in == sr_out, so
        already-16 kHz files never pull in librosa at all).
    """
    if sr_in == sr_out:
        return audio  # No resample needed — skip the heavy import entirely.
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("librosa is required to resample to 16 kHz — "
                           "install it: `uv add librosa`") from e
    return librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)


# ════════════════════════════════════════════════════════════
#  CRATE PATH / ID HELPERS
# ════════════════════════════════════════════════════════════
# IMPORTANT IDENTITY MODEL (see REFACTOR NOTES): crate.py identifies an excerpt
# by its EXCERPT_ID — the crate filename stem (a uuid4 we mint before writing the
# file). This is DISTINCT from database.py's track_id, the uuid primary key it
# mints on insert_track(); the DB gives us no way to inject ours or rename
# crate_path afterwards. We therefore key the crate by excerpt_id and bridge to
# the DB row via the unique crate_path. All database.py calls that need the DB
# primary key (track_id) resolve it via _db_row().
def _ensure_crate_dir() -> None:
    """Create ./crate/ if missing (idempotent). Called by every ingest path."""
    CRATE_DIR.mkdir(parents=True, exist_ok=True)


def _crate_path(excerpt_id: str) -> Path:
    """Absolute path of the excerpt for a given excerpt_id (crate filename stem)."""
    return CRATE_DIR / f"{excerpt_id}.wav"


def _excerpt_id(crate_path: str) -> str:
    """The excerpt_id for a stored excerpt = its filename without extension."""
    return Path(crate_path).stem


def _db_row(excerpt_id: str):
    """Resolve the database row for an excerpt_id via its unique crate_path.

    Args:
        excerpt_id: crate filename stem.
    Returns:
        The track row dict (including the DB's own 'track_id' primary key), or
        None if no row references this excerpt.
    """
    return database.get_track_by_path(str(_crate_path(excerpt_id)))


# ════════════════════════════════════════════════════════════
#  LIGHTWEIGHT PER-SECOND ANALYSIS  (window selection only)
# ════════════════════════════════════════════════════════════
# Deliberately NOT analyze.py's full pipeline: window selection runs on EVERY
# import before we even know if a track is worth keeping, so it must be cheap.
# One RMS + one FFT per second is plenty to find the "busiest" stretch.
def _per_second_features(audio: np.ndarray, sr: int) -> tuple:
    """Compute per-second RMS energy and spectral flux for window selection.

    Args:
        audio: mono float32 samples.
        sr: sample rate (one analysis frame == one second == `sr` samples).
    Returns:
        (rms, flux) — two float arrays of length floor(len(audio)/sr). flux[0] is
        0 (no previous frame to diff against).
    """
    n = len(audio) // sr  # Whole seconds; a trailing partial second is ignored.
    rms = np.zeros(n, dtype=np.float64)
    flux = np.zeros(n, dtype=np.float64)
    if n == 0:
        return rms, flux
    window = np.hanning(sr)  # Hann taper so frame edges don't fake spectral change.
    prev_mag = None
    for i in range(n):
        seg = audio[i * sr:(i + 1) * sr]
        rms[i] = float(np.sqrt(np.mean(seg * seg)))      # Loudness envelope.
        mag = np.abs(np.fft.rfft(seg * window))          # Magnitude spectrum.
        if prev_mag is not None:
            # Spectral flux = how much the spectrum changed since last second;
            # high during busy/percussive sections, low in sparse breakdowns.
            flux[i] = float(np.linalg.norm(mag - prev_mag))
        prev_mag = mag
    return rms, flux


def _best_window(rms: np.ndarray, flux: np.ndarray, n_sec: int,
                 target: int = TARGET_SECONDS) -> tuple:
    """Find the start/end (in seconds) of the most energetic+busy `target`s window.

    Algorithm (per spec):
        1. score per second = RMS * spectral_flux.
        2. ignore the first/last TRIM_FRACTION of the track (intro/outro).
        3. slide a `target`-second window; pick the one maximising mean score.

    Args:
        rms: per-second RMS array.
        flux: per-second spectral-flux array.
        n_sec: number of whole seconds in the track.
        target: desired window length in seconds.
    Returns:
        (start_sec, end_sec). For tracks at/under `target` seconds the whole
        track is returned. When the trimmed middle is itself shorter than the
        window, the trim is relaxed to the full track so we still return a window.
    """
    if n_sec <= target:
        return 0, n_sec  # Too short to slide — keep everything we have.

    score = rms * flux                       # Per-second desirability.
    start_lo = int(TRIM_FRACTION * n_sec)    # Skip intro.
    start_hi = n_sec - int(TRIM_FRACTION * n_sec)  # End of the keep-region.
    if start_hi - start_lo < target:
        # The middle 80% can't hold a full window — search the whole track instead.
        start_lo, start_hi = 0, n_sec

    last_start = start_hi - target           # Last valid window start index.
    if last_start < start_lo:                # Defensive: clamp into a valid range.
        last_start = max(0, n_sec - target)
        start_lo = min(start_lo, last_start)

    # Prefix sums turn each window's total into an O(1) lookup; mean is
    # proportional to the sum because the window length is fixed, so we maximise
    # the sum directly.
    prefix = np.concatenate(([0.0], np.cumsum(score)))
    best_start, best_score = start_lo, -1.0
    for s in range(start_lo, last_start + 1):
        window_sum = prefix[s + target] - prefix[s]
        if window_sum > best_score:
            best_score, best_start = window_sum, s
    return best_start, best_start + target


def _crossfade_concat(parts: list, sr: int, fade_sec: float = 0.5) -> np.ndarray:
    """Concatenate audio parts with short equal-power crossfades at each splice.

    A hard splice between two arbitrary points of a track produces a click and a
    spurious spectral event that downstream DSP (breakdown detection, beat
    tracking) would misread as real. A 0.5 s equal-power crossfade (cos/sin
    envelopes, constant perceived loudness through the joint) suppresses both.

    Args:
        parts: list of mono float32 arrays, in playback order.
        sr: sample rate of the parts.
        fade_sec: crossfade length per splice.
    Returns:
        One mono float32 array. Each splice shortens the total by `fade_sec`.
    """
    n_fade = int(fade_sec * sr)
    out = parts[0]
    for seg in parts[1:]:
        if n_fade > 0 and len(out) >= n_fade and len(seg) >= n_fade:
            t = np.linspace(0.0, np.pi / 2.0, n_fade, dtype=np.float32)
            joint = out[-n_fade:] * np.cos(t) + seg[:n_fade] * np.sin(t)
            out = np.concatenate([out[:-n_fade], joint, seg[n_fade:]])
        else:                                   # Degenerate tiny parts: hard join.
            out = np.concatenate([out, seg])
    return out.astype(np.float32)


def _composite_window(rms: np.ndarray, flux: np.ndarray, n_sec: int,
                      target: int = TARGET_SECONDS) -> list:
    """Pick three non-overlapping `target/3`-second segments: peak, mid, low.

    The contiguous best-window keeps only the busiest stretch of a track; this
    composite alternative samples its dynamic RANGE instead — the most intense
    segment, an average one, and the sparsest one — so the stored excerpt (and
    its averaged EffNet embedding) represents the WHOLE track. That widens live
    recognition coverage: the listener can match a record even while its intro
    or breakdown is playing, parts the contiguous window deliberately discards.

    Trade-off (why this is opt-in, not the default): even with crossfaded
    splices the excerpt is not a real continuous passage, so the DSP-derived
    energy curve, breakdown map and mix-zone bars describe the collage, not the
    record. Use it for crates where recognition coverage matters more than
    structural analysis fidelity.

    Selection: score = RMS x flux per second, intro/outro trimmed exactly like
    _best_window. Peak = max-mean segment; low = min-mean among non-overlapping
    candidates; mid = candidate whose mean is closest to the median. Segments
    are returned in CHRONOLOGICAL order to preserve the track's narrative arc.

    Args:
        rms: per-second RMS array.
        flux: per-second spectral-flux array.
        n_sec: number of whole seconds in the track.
        target: total excerpt length the three segments should sum to.
    Returns:
        list of (start_sec, end_sec) tuples, chronologically ordered; empty list
        when the track is too short to hold three disjoint segments (caller
        should fall back to _best_window).
    """
    seg = target // 3
    start_lo = int(TRIM_FRACTION * n_sec)
    start_hi = n_sec - int(TRIM_FRACTION * n_sec)
    if start_hi - start_lo < 3 * seg:           # Can't fit 3 disjoint segments.
        return []

    score = rms * flux
    prefix = np.concatenate(([0.0], np.cumsum(score)))
    starts = np.arange(start_lo, start_hi - seg + 1)
    means = (prefix[starts + seg] - prefix[starts]) / seg

    chosen = []

    def _take(idx: int) -> None:
        """Accept candidate `idx` and mask every start overlapping it."""
        chosen.append(int(starts[idx]))
        overlap = np.abs(starts - starts[idx]) < seg
        means[overlap] = np.nan                 # NaN = no longer selectable.

    _take(int(np.nanargmax(means)))                             # 1) peak segment.
    if np.all(np.isnan(means)):
        return []
    _take(int(np.nanargmin(means)))                             # 2) low segment.
    if np.all(np.isnan(means)):
        return []
    median = np.nanmedian(means)
    _take(int(np.nanargmin(np.abs(means - median))))            # 3) mid segment.

    return [(s, s + seg) for s in sorted(chosen)]


# ════════════════════════════════════════════════════════════
#  AUDIO LOADING + STANDARDISATION
# ════════════════════════════════════════════════════════════
def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse a (frames, channels) array to mono by averaging channels."""
    if audio.ndim > 1:
        return audio.mean(axis=1)  # Average rather than drop a channel.
    return audio


def _load_file(path: str) -> tuple:
    """Decode an audio file to a float32 array + its native sample rate.

    Tries soundfile first (handles WAV/FLAC, and MP3 on libsndfile >= 1.1). Falls
    back to librosa for formats an older libsndfile can't decode (e.g. MP3),
    which routes through audioread.

    Args:
        path: path to a .wav/.mp3/.flac file.
    Returns:
        (audio, sr) where audio is float32 (possibly multi-channel — caller mono-
        downmixes in _standardize).
    """
    sf = _soundfile()
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        return audio, sr
    except Exception as e:  # Most likely an MP3 on a libsndfile without MP3 support.
        logger.warning("soundfile could not decode %s (%s) — trying librosa",
                       Path(path).name, e)
        try:
            import librosa
        except ImportError as le:
            raise RuntimeError(f"cannot decode {Path(path).name}: install librosa "
                               "for MP3 support (`uv add librosa`)") from le
        # sr=None preserves the native rate; mono=False keeps channels for the
        # standard downmix path.
        audio, sr = librosa.load(path, sr=None, mono=False)
        # librosa returns (channels, frames); transpose to (frames, channels) to
        # match soundfile's layout that _to_mono expects.
        if audio.ndim > 1:
            audio = audio.T
        return audio.astype(np.float32), int(sr)


def _standardize(audio: np.ndarray, sr: int, strategy: str = "best") -> tuple:
    """Run the full standardisation pipeline on a raw audio array.

    mono -> 16 kHz -> 120 s excerpt. Shared by ALL three input sources so the
    crate is guaranteed uniform regardless of origin.

    Args:
        audio: raw samples (mono or multi-channel) as float.
        sr: native sample rate of `audio`.
        strategy: '"best"' (default) = the single most energetic+busy contiguous
            window; '"composite"' = three crossfaded peak/mid/low segments
            covering the track's dynamic range (see _composite_window for the
            recognition-coverage vs analysis-fidelity trade-off). Composite
            falls back to "best" automatically on tracks too short to hold it.
    Returns:
        (excerpt, start_sec, end_sec): excerpt is mono float32 at ML_SAMPLE_RATE,
        at most TARGET_SECONDS long; start/end locate it in the source (for the
        composite strategy they span first-segment start to last-segment end).
    Raises:
        ValueError: if the input is shorter than one second of audio.
    """
    audio = _to_mono(audio).astype(np.float32)        # 1) downmix to mono.
    audio = _resample(audio, sr, ML_SAMPLE_RATE)      # 2) resample to 16 kHz.

    n_sec = len(audio) // ML_SAMPLE_RATE
    if n_sec < 1:
        # Below a second there is nothing meaningful to window or analyse.
        raise ValueError(f"audio too short ({len(audio)} samples @ {ML_SAMPLE_RATE} Hz)")

    rms, flux = _per_second_features(audio, ML_SAMPLE_RATE)  # 3a) cheap curves.

    if strategy == "composite":
        segments = _composite_window(rms, flux, n_sec)       # 3b-alt) 3 segments.
        if segments:
            parts = [audio[s * ML_SAMPLE_RATE:e * ML_SAMPLE_RATE] for s, e in segments]
            excerpt = _crossfade_concat(parts, ML_SAMPLE_RATE)
            logger.info("composite window segments=%s (chronological, crossfaded)",
                        segments)
            return excerpt, segments[0][0], segments[-1][1]
        logger.info("composite window: track too short — falling back to best window")

    start, end = _best_window(rms, flux, n_sec)              # 3b) pick window.
    excerpt = audio[start * ML_SAMPLE_RATE:end * ML_SAMPLE_RATE]
    return excerpt, start, end


def _write_excerpt(excerpt_id: str, excerpt: np.ndarray) -> Path:
    """Write a standardised excerpt to ./crate/<excerpt_id>.wav as 16 kHz float32.

    Args:
        excerpt_id: crate filename stem.
        excerpt: mono float32 samples at ML_SAMPLE_RATE.
    Returns:
        The path written.
    """
    _ensure_crate_dir()
    sf = _soundfile()
    path = _crate_path(excerpt_id)
    # subtype FLOAT => 32-bit float WAV, lossless for our already-float samples.
    sf.write(str(path), excerpt, ML_SAMPLE_RATE, subtype="FLOAT")
    return path


# ════════════════════════════════════════════════════════════
#  BREAKDOWN DETECTION  (stored in features JSONB)
# ════════════════════════════════════════════════════════════
def _normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns zeros for a flat (no-range) curve."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:        # Flat curve carries no breakdown information.
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def detect_breakdowns(energy_curve: list, complexity_curve: list) -> tuple:
    """Detect breakdowns from a track's per-second energy + complexity curves.

    Reuses the curves analyze.py already produced (energy_curve = per-second RMS,
    complexity_curve = per-second spectral complexity) — no second pass over the
    audio. Algorithm:
        * normalise both curves to 0-1.
        * combined_score = RMS * spectral_complexity.
        * a breakdown is combined_score < BREAKDOWN_THRESHOLD for >=
          BREAKDOWN_MIN_SECONDS consecutive seconds (shorter dips are dropped).

    Args:
        energy_curve: per-second RMS values (analyze.py's energy_curve).
        complexity_curve: per-second spectral complexity (analyze.py's
            complexity_curve).
    Returns:
        (events, reliability):
          events       list of {'start_sec', 'end_sec', 'duration_sec'} dicts.
          reliability  'HIGH' if any breakdown dips below BREAKDOWN_SHARP (sharp,
                       unambiguous drop), 'MEDIUM' if breakdowns exist but are
                       shallow, 'NONE' if there are none / curves unusable.
    """
    if not energy_curve or not complexity_curve:
        return [], "NONE"
    n = min(len(energy_curve), len(complexity_curve))  # Align lengths defensively.
    rms = _normalize01(np.asarray(energy_curve[:n], dtype=np.float64))
    comp = _normalize01(np.asarray(complexity_curve[:n], dtype=np.float64))
    # If either proxy is flat its normalised form is all zeros, which would flag
    # the WHOLE track as one breakdown — meaningless, so bail out.
    if not rms.any() or not comp.any():
        return [], "NONE"

    combined = rms * comp
    below = combined < BREAKDOWN_THRESHOLD  # Boolean per-second "is this quiet?".

    events, sharp = [], False
    run_start = None
    for i, is_low in enumerate(list(below) + [False]):  # Sentinel flushes a final run.
        if is_low and run_start is None:
            run_start = i                               # Run begins.
        elif not is_low and run_start is not None:
            length = i - run_start                      # Run ends at i (exclusive).
            if length >= BREAKDOWN_MIN_SECONDS:         # Filter brief dips.
                events.append({
                    "start_sec": float(run_start),
                    "end_sec": float(i),
                    "duration_sec": float(length),
                })
                # Sharp if the deepest second of this run is a hard drop.
                if combined[run_start:i].min() < BREAKDOWN_SHARP:
                    sharp = True
            run_start = None

    if not events:
        return [], "NONE"
    return events, ("HIGH" if sharp else "MEDIUM")


# ════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION DISPATCH  (module vs subprocess)
# ════════════════════════════════════════════════════════════
def _extract_features(path: str, bpm_prior: tuple = None):
    """Extract features for one crate WAV, returning analyze.py's asdict() form.

    Routes through the in-process module when available (models stay warm) or the
    subprocess fallback otherwise. Returns None on failure (logged), so callers
    can leave the track pending and move on rather than crash a batch.

    Args:
        path: crate WAV to analyse.
        bpm_prior: optional (lo, hi) plausible-BPM range from the owning crate,
            forwarded to analyze.extract_features for metrical-octave folding.
    """
    if ANALYSIS_MODE == "module":
        try:
            return asdict(analyze.extract_features(path, bpm_prior=bpm_prior))
        except Exception as e:  # Don't let one bad track kill the caller.
            logger.error("extract_features (module) FAILED on %s: %s",
                         Path(path).name, e, exc_info=True)
            return None
    return _extract_features_subprocess(path, bpm_prior=bpm_prior)


def _extract_features_subprocess(path: str, bpm_prior: tuple = None):
    """Fallback extraction: run analyze.extract_features in a fresh interpreter.

    The child prints the feature dict as JSON behind _SUBPROC_MARKER; we ignore
    its human progress prints and parse only the marked line.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROC_SNIPPET, path]
            + ([str(bpm_prior[0]), str(bpm_prior[1])] if bpm_prior else []),
            cwd=str(PROJECT_ROOT),   # so `import analyze` resolves.
            capture_output=True, text=True, timeout=1800,  # 30 min hard cap.
        )
    except Exception as e:
        logger.error("extract_features (subprocess) FAILED to launch on %s: %s",
                     Path(path).name, e)
        return None
    if proc.returncode != 0:
        logger.error("extract_features (subprocess) FAILED on %s (rc=%d): %s",
                     Path(path).name, proc.returncode, proc.stderr.strip()[-500:])
        return None
    for line in proc.stdout.splitlines():
        if line.startswith(_SUBPROC_MARKER):
            return json.loads(line[len(_SUBPROC_MARKER):])
    logger.error("extract_features (subprocess) produced no features for %s",
                 Path(path).name)
    return None


def _analyze_and_persist(crate_path: Path, db_track_id: str) -> int:
    """Analyse one crate WAV, persist features + breakdowns + embedding to the DB.

    Args:
        crate_path: the ./crate/<id>.wav to analyse.
        db_track_id: the DATABASE primary key (from insert_track / a track row),
            not the crate filename stem — this is what database.py keys on.
    Returns:
        The pipeline level actually reached, or 0 if extraction failed (the track
        stays pending so a later analyze_pending() retries it).
    """
    name = crate_path.name
    # The BPM prior comes from the track's OWNING crate (learned stats once the
    # crate is mature, genre seed before that). Best-effort: any failure here
    # simply means no folding, never a blocked analysis.
    bpm_prior = None
    try:
        row = database.get_track(db_track_id)
        bpm_prior = database.crate_bpm_prior(row.get("crate_id"))
    except Exception as e:
        logger.warning("bpm prior unavailable for %s: %s", name, e)
    fdict = _extract_features(str(crate_path), bpm_prior=bpm_prior)
    if fdict is None:
        logger.error("analyze FAILED track=%s reason=extraction-returned-none", name)
        return 0

    level = int(fdict.get("pipeline_level", 1))

    # Breakdown detection reuses the curves analyze.py already computed, then we
    # stash the result as extra JSONB keys. analyze.py's loader ignores unknown
    # keys, so this metadata rides along without disturbing rehydration.
    events, reliability = detect_breakdowns(
        fdict.get("energy_curve"), fdict.get("complexity_curve"))
    fdict["breakdowns"] = events
    fdict["breakdown_count"] = len(events)
    fdict["breakdown_reliability"] = reliability

    # Persist features (also stamps analyzed_at => track leaves the pending set).
    database.update_track_features(db_track_id, fdict, level)

    # Fan every embedding vector (Level 2–5: EffNet + genre/mood-theme/instrument)
    # out to its pgvector table. analyze.persist_embeddings owns this so the
    # per-model model_version strings come straight from its REGISTRY and stay in
    # sync with the model that produced them. In the subprocess fallback (analyze
    # never imported) we can still persist at least the EffNet vector via the
    # hard-coded MODEL_VERSION — the others are unavailable without the REGISTRY.
    if analyze is not None:
        analyze.persist_embeddings(db_track_id, fdict)
    else:
        emb = fdict.get("effnet_embedding")
        if emb:
            database.upsert_effnet_embedding(db_track_id, emb, MODEL_VERSION)

    # Acoustic fingerprint (Shazam-style landmarks) for exact live recognition.
    # Best-effort like everything else post-extraction: a fingerprint failure
    # must never un-analyse a track; the `fingerprint` CLI command backfills.
    n_landmarks = _fingerprint_track(crate_path, db_track_id)

    # Link the track to its artist entities and refresh their EffNet centroids
    # (Phase 0). Best-effort: a failure here never un-analyses the track.
    if analyze is not None:
        try:
            trow = database.get_track(db_track_id)
            for nm in database._parse_artist_names(trow.get("filename") if trow else ""):
                aid = database.upsert_artist(nm)
                database.link_track_artist(db_track_id, aid)
                analyze.persist_artist_embedding(aid)   # recompute centroid w/ new track
        except Exception as e:
            logger.warning("artist linking failed for %s: %s", db_track_id, e)

    logger.info("analyze SUCCESS track=%s level=%d bpm=%.1f key=%s camelot=%s "
                "breakdowns=%d reliability=%s embedding=%s landmarks=%d",
                name, level, fdict.get("bpm", 0.0), fdict.get("key", "?"),
                fdict.get("camelot", "?"), len(events), reliability,
                "yes" if fdict.get("effnet_embedding") else "no", n_landmarks)
    return level


def _fingerprint_track(crate_path: Path, db_track_id: str) -> int:
    """Extract + store landmark hashes for one excerpt. Returns landmark count.

    Reads the standardised WAV (already mono 16 kHz — fingerprint.SR), extracts
    the constellation hashes, and replaces the track's row set atomically.
    Best-effort: returns 0 on any failure and logs, never raises.
    """
    try:
        import fingerprint
        sf = _soundfile()
        audio, sr = sf.read(str(crate_path), dtype="float32")
        audio = _to_mono(audio)
        if sr != fingerprint.SR:                 # Excerpts are 16 kHz by contract,
            audio = _resample(audio, sr, fingerprint.SR)   # but never trust a file.
        hashes = fingerprint.extract_hashes(audio)
        return database.replace_fingerprints(db_track_id, hashes)
    except Exception as e:
        logger.warning("fingerprint FAILED track=%s reason=%s", crate_path.name, e)
        return 0


# ════════════════════════════════════════════════════════════
#  INGEST CORE  (shared by all three input sources)
# ════════════════════════════════════════════════════════════
def _ingest(audio: np.ndarray, sr: int, display_name: str,
            strategy: str = "best", crate_id: str = None) -> tuple:
    """Standardise raw audio, write the crate WAV, and insert the DB row.

    The ONE code path every source funnels through after acquiring samples, so
    standardisation + persistence are written exactly once.

    Args:
        audio: raw samples from any source.
        sr: native sample rate.
        display_name: human label / source filename stored as tracks.filename
            (also the value folder-import dedups on).
        strategy: excerpt window strategy, "best" | "composite" (see _standardize).
        crate_id: owning crate; None resolves to the active/default crate.
    Returns:
        (excerpt_id, crate_path, db_track_id):
          excerpt_id    crate filename stem (crate.py's public id).
          crate_path    path of the written excerpt.
          db_track_id   database.py's primary key for follow-up calls.
    """
    excerpt, start, end = _standardize(audio, sr, strategy=strategy)
    excerpt_id = uuid.uuid4().hex                    # Crate id == filename stem.
    crate_path = _write_excerpt(excerpt_id, excerpt)
    duration = len(excerpt) / ML_SAMPLE_RATE
    # insert_track is idempotent on crate_path; our fresh uuid never collides, so
    # this is always a real insert. It returns the DB's own primary key.
    db_track_id = database.insert_track(
        str(crate_path), filename=display_name, duration=duration,
        crate_id=crate_id)
    # Many-to-many membership: recording/importing into a USER crate files the
    # track there too (no-op for the default master crate, which holds all).
    database.add_tracks_to_crate(crate_id, [db_track_id])
    logger.info("ingest SUCCESS excerpt=%s file=%s window=%d-%ds dur=%.0fs db_id=%s",
                excerpt_id, display_name, start, end, duration, db_track_id)
    return excerpt_id, crate_path, db_track_id


# ════════════════════════════════════════════════════════════
#  LIVE CAPTURE
# ════════════════════════════════════════════════════════════
def list_input_devices() -> list:
    """Return (and pretty-print) the system's input-capable audio devices.

    Returns:
        list of (index, name, default_samplerate) for every device exposing at
        least one input channel — covers the built-in mic and any external
        interface / sound card the OS sees as an input.
    """
    sd = _sounddevice()
    devices = sd.query_devices()
    default_in = sd.default.device[0] if sd.default.device else None
    rows = []
    print("\nInput devices:")
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue  # Output-only device — not a capture source.
        marker = " (default)" if idx == default_in else ""
        rate = int(dev["default_samplerate"])
        print(f"  [{idx}] {dev['name']}{marker}  — {dev['max_input_channels']} ch "
              f"@ {rate} Hz")
        rows.append((idx, dev["name"], rate))
    print()
    return rows


def _capture(device_index=None) -> tuple:
    """Record live mono audio from a device until the user presses Enter.

    Args:
        device_index: input device index, or None for the system default.
    Returns:
        (audio, sr): the recording as a mono float32 array at the device's native
        rate (standardisation downmixes/resamples it like any other source).
    """
    sd = _sounddevice()
    # Resolve the device's native rate; capturing at the device default avoids
    # PortAudio errors from interfaces that don't support 16 kHz directly.
    info = sd.query_devices(device_index if device_index is not None
                            else sd.default.device[0], "input")
    sr = int(info["default_samplerate"])
    logger.info("device SELECTED index=%s name=%s rate=%d",
                device_index if device_index is not None else "default",
                info["name"], sr)
    print(f"\n● Recording from [{info['name']}] @ {sr} Hz — press Enter to stop...")

    chunks = []  # Accumulate callback blocks; concatenated once at the end.

    def _cb(indata, frames, time_info, status):
        if status:  # Over/underflows are logged but don't abort the capture.
            logger.warning("capture stream status: %s", status)
        chunks.append(indata.copy())  # copy(): indata is reused by PortAudio.

    # InputStream (not sd.rec) because the take length is open-ended — we stop on
    # user input rather than a pre-set duration.
    with sd.InputStream(samplerate=sr, channels=1, device=device_index,
                        dtype="float32", callback=_cb):
        try:
            input()  # Blocks the main thread; the callback fills `chunks` meanwhile.
        except (EOFError, KeyboardInterrupt):
            pass     # Ctrl-D / Ctrl-C is a normal "stop now".
    print("■ Stopped.\n")

    if not chunks:
        raise ValueError("no audio captured")
    audio = np.concatenate(chunks, axis=0).reshape(-1)  # Flatten to mono 1-D.
    return audio, sr


# ════════════════════════════════════════════════════════════
#  PUBLIC OPERATIONS
# ════════════════════════════════════════════════════════════
def add_from_recording(device_index=None, label=None, strategy: str = "best",
                       crate: str = None) -> str:
    """Capture live audio, standardise to a 120 s excerpt, store, and analyse.

    The full recording is held in memory only — just the standardised excerpt is
    written to ./crate/; the raw take is discarded when this returns.

    Args:
        device_index: input device index; None uses the system default device.
        label: optional "artist - title"; defaults to a timestamped name.
        strategy: "best" (contiguous window) | "composite" (peak/mid/low segments).
        crate: crate name/id to file the record in; None = the active crate.
    Returns:
        The excerpt_id (crate filename stem) of the new excerpt.
    """
    label = label or f"recording-{datetime.now():%Y%m%d-%H%M%S}"
    try:
        crate_id = database.resolve_crate_id(crate)
        audio, sr = _capture(device_index)            # Live take (in memory only).
        excerpt_id, crate_path, db_id = _ingest(audio, sr, label, strategy=strategy,
                                                crate_id=crate_id)
        _analyze_and_persist(crate_path, db_id)       # Trigger analysis immediately.
        logger.info("add-rec SUCCESS excerpt=%s label=%s", excerpt_id, label)
        return excerpt_id
    except Exception as e:
        logger.error("add-rec FAILED label=%s reason=%s", label, e, exc_info=True)
        raise


def add_from_file(path, label=None, strategy: str = "best",
                  crate: str = None) -> str:
    """Import one audio file: standardise, store, insert, and analyse.

    Args:
        path: path to a .wav/.mp3/.flac on any mounted storage.
        label: optional "artist - title"; defaults to the source filename.
        strategy: "best" (contiguous window) | "composite" (peak/mid/low segments).
        crate: crate name/id to file the record in; None = the active crate.
    Returns:
        The excerpt_id (crate filename stem) of the new excerpt.
    """
    path = str(path)
    label = label or Path(path).name
    try:
        crate_id = database.resolve_crate_id(crate)
        audio, sr = _load_file(path)
        excerpt_id, crate_path, db_id = _ingest(audio, sr, label, strategy=strategy,
                                                crate_id=crate_id)
        _analyze_and_persist(crate_path, db_id)       # Trigger analysis immediately.
        logger.info("add-file SUCCESS excerpt=%s src=%s", excerpt_id, Path(path).name)
        return excerpt_id
    except Exception as e:
        logger.error("add-file FAILED src=%s reason=%s", Path(path).name, e,
                     exc_info=True)
        raise


def add_from_folder(folder_path, strategy: str = "best", crate: str = None) -> list:
    """Batch-import every audio file in a folder, skipping ones already imported.

    "Already imported" is decided by filename: if a track row already carries a
    source filename matching a file here, it's skipped (so re-running over the
    same USB stick is a no-op). Analysis is DEFERRED — this only ingests; run
    analyze_pending() (or the `analyze` CLI command) afterwards. Deferring keeps a
    big import fast and lets the idempotent analyser sweep everything in one pass.

    Args:
        folder_path: directory of audio files (non-recursive).
    Returns:
        list of excerpt_ids that were newly added this call.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error("add-folder FAILED path=%s reason=not-a-directory", folder)
        raise NotADirectoryError(f"{folder} is not a directory")
    crate_id = database.resolve_crate_id(crate)   # Once for the whole batch.

    # Existing source filenames already in the DB — our skip set.
    existing = {row["filename"] for row in database.list_tracks()}
    # Non-recursive, sorted, only the three accepted extensions.
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

    added = []
    for f in files:
        if f.name in existing:
            logger.info("add-folder SKIP file=%s reason=already-in-db", f.name)
            continue
        try:
            audio, sr = _load_file(str(f))
            excerpt_id, _crate_path, _db_id = _ingest(audio, sr, f.name, strategy=strategy,
                                                      crate_id=crate_id)
            added.append(excerpt_id)
            existing.add(f.name)  # Guard against duplicate names within this run.
        except Exception as e:    # One bad file shouldn't abort the whole folder.
            logger.error("add-folder FAILED file=%s reason=%s", f.name, e,
                         exc_info=True)

    logger.info("add-folder SUCCESS folder=%s added=%d skipped=%d",
                folder, len(added), len(files) - len(added))
    if added:
        print(f"Imported {len(added)} track(s). Run `analyze` to extract features.")
    return added


def analyze_pending(crate: str = None) -> int:
    """Analyse every crate track that has no features yet. Idempotent.

    Finds tracks whose analyzed_at is NULL (the DB's "pending" marker), analyses
    each, and persists results. Safe to re-run: already-analysed tracks are never
    touched, and a track whose extraction fails simply stays pending for next time.

    Args:
        crate: crate name/id to restrict the sweep to; None sweeps ALL crates
            (the historical behaviour — pending work is pending work).
    Returns:
        Count of tracks newly analysed this call.
    """
    crate_id = database.resolve_crate_id(crate) if crate else None
    pending = [row for row in database.list_tracks(crate_id=crate_id)
               if row["analyzed_at"] is None]
    if not pending:
        logger.info("analyze-pending: nothing to do")
        return 0

    logger.info("analyze-pending: %d track(s) to analyse", len(pending))
    done = 0
    for row in pending:
        crate_path = Path(row["crate_path"])
        if not crate_path.exists():
            # Row references a file that's gone — flag it; don't try to analyse.
            logger.error("analyze FAILED excerpt=%s reason=missing-file path=%s",
                         _excerpt_id(row["crate_path"]), crate_path)
            continue
        if _analyze_and_persist(crate_path, row["track_id"]) > 0:
            done += 1
    logger.info("analyze-pending SUCCESS analysed=%d of %d", done, len(pending))
    return done


def upgrade_pipeline(target_level: int = 5, crate: str = None) -> int:
    """Re-analyse tracks whose pipeline_level is below target_level.

    Looks for already-analysed tracks that reached a lower pipeline level
    (e.g. L2 because TF models weren't available at ingest time) and re-runs
    the full analysis pipeline on their existing crate excerpt. Safe to call
    repeatedly — tracks already at or above target_level are never touched.

    Args:
        target_level: desired minimum pipeline level (default 5).
        crate: crate name/id to restrict the sweep to; None = all crates.
    Returns:
        Count of tracks upgraded this call.
    """
    crate_id = database.resolve_crate_id(crate) if crate else None
    candidates = database.list_tracks_below_level(target_level, crate_id=crate_id)
    if not candidates:
        logger.info("upgrade-pipeline: all tracks already at level >= %d", target_level)
        return 0

    logger.info("upgrade-pipeline: %d track(s) below L%d", len(candidates), target_level)
    done = 0
    for row in candidates:
        crate_path = Path(row["crate_path"])
        if not crate_path.exists():
            logger.error("upgrade FAILED excerpt=%s reason=missing-file path=%s",
                         _excerpt_id(row["crate_path"]), crate_path)
            continue
        old_level = row.get("pipeline_level", "?")
        if _analyze_and_persist(crate_path, row["track_id"]) > 0:
            done += 1
            logger.info("upgrade OK excerpt=%s %s -> L%d",
                        _excerpt_id(row["crate_path"]), old_level, target_level)
    logger.info("upgrade-pipeline SUCCESS upgraded=%d of %d", done, len(candidates))
    return done


def fingerprint_pending() -> int:
    """Backfill landmark hashes for analyzed tracks that have none. Idempotent.

    Covers tracks analyzed before fingerprinting existed (new tracks are
    fingerprinted inline by _analyze_and_persist). Safe to re-run: tracks that
    already have landmarks are never touched.

    Returns:
        Count of tracks fingerprinted this call.
    """
    pending = database.tracks_without_fingerprints()
    if not pending:
        logger.info("fingerprint-pending: nothing to do")
        return 0
    logger.info("fingerprint-pending: %d track(s) to fingerprint", len(pending))
    done = 0
    for row in pending:
        crate_path = Path(row["crate_path"])
        if not crate_path.exists():
            logger.error("fingerprint FAILED excerpt=%s reason=missing-file path=%s",
                         _excerpt_id(row["crate_path"]), crate_path)
            continue
        n = _fingerprint_track(crate_path, row["track_id"])
        if n > 0:
            done += 1
            print(f"  ✔ {row['filename']}: {n} landmarks")
    logger.info("fingerprint-pending SUCCESS fingerprinted=%d of %d",
                done, len(pending))
    return done


def list_crate(show_pending=False, crate: str = None) -> None:
    """Pretty-print a crate: totals plus per-track BPM/key/Camelot/level.

    Args:
        show_pending: when True, list ONLY tracks still awaiting analysis (the
            "what's left to do" view); when False, list every track.
        crate: crate name/id to list; None = the ACTIVE crate. Pass "all" to
            list every crate together.
    """
    try:
        if crate == "all":
            crate_id, scope = None, "all crates"
        else:
            crate_id = database.resolve_crate_id(crate)
            row = database.get_crate(crate_id) if crate_id else None
            scope = f"crate '{row['name']}' ({row['genre']})" if row else "crate"
        total, analyzed, pending = database.count_tracks(crate_id=crate_id)
        rows = database.list_tracks(crate_id=crate_id)
    except database.DBUnavailableError as e:
        # Read-only status command: degrade gracefully instead of a traceback.
        print(f"\nCrate unavailable — {e}\n")
        return
    if show_pending:
        rows = [r for r in rows if r["analyzed_at"] is None]

    print(f"\n{scope} — {total} tracks | {analyzed} analyzed | {pending} pending")
    if show_pending:
        print("(showing pending only)")
    print("─" * 78)
    if not rows:
        print("  (empty)\n")
        return

    # Header row for the fixed-width table below.
    print(f"  {'excerpt':<12} {'BPM':>6} {'key':>10} {'cam':>4} {'lvl':>3}  filename")
    for r in rows:
        excerpt_id = _excerpt_id(r["crate_path"])[:12]  # Stem is what `remove` takes.
        feats = r.get("features") or {}           # NULL until analysed.
        if r["analyzed_at"] is None:
            bpm, key, camelot, lvl = "  —  ", "pending", " — ", "—"
        else:
            bpm = f"{feats.get('bpm', 0.0):.1f}"
            key = f"{feats.get('key', '?')} {feats.get('scale', '')}".strip()
            camelot = feats.get("camelot", "?")
            lvl = str(r.get("pipeline_level", 1))
        print(f"  {excerpt_id:<12} {bpm:>6} {key:>10} {camelot:>4} {lvl:>3}  "
              f"{r['filename']}")
    print()


def remove(excerpt_id) -> None:
    """Delete a track's crate WAV and its database row (after confirmation).

    Args:
        excerpt_id: crate filename stem (as shown by list_crate / returned by add_*).

    Prompts before deleting because this is irreversible — the excerpt is gone
    and the DB row (plus its embeddings, via ON DELETE CASCADE) is removed.
    """
    excerpt_id = str(excerpt_id)
    crate_path = _crate_path(excerpt_id)
    row = _db_row(excerpt_id)  # Resolve the DB primary key via the unique crate_path.

    if row is None and not crate_path.exists():
        print(f"No such track: {excerpt_id}")
        logger.warning("remove NOOP excerpt=%s reason=not-found", excerpt_id)
        return

    # Confirm — show the friendly filename when we have a row.
    label = row["filename"] if row else crate_path.name
    answer = input(f"Delete '{label}' ({excerpt_id})? This cannot be undone [y/N]: ")
    if answer.strip().lower() not in ("y", "yes"):
        print("Aborted.")
        logger.info("remove ABORTED excerpt=%s", excerpt_id)
        return

    # DB first: if it fails we keep the file and surface the error rather than
    # orphaning a row that points at a deleted excerpt.
    if row is not None:
        database.delete_track(row["track_id"])
    if crate_path.exists():
        crate_path.unlink()
    print(f"Removed {excerpt_id}.")
    logger.info("remove SUCCESS excerpt=%s file=%s", excerpt_id, label)


def crate_health() -> dict:
    """Return a snapshot of crate + database health.

    Returns:
        dict with: total, analyzed, pending (track counts); db_reachable (bool);
        crate_size_mb (float, total size of ./crate/). Never raises — a dead DB
        yields zero counts and db_reachable=False so a status caller stays alive.
    """
    db_reachable = False
    total = analyzed = pending = 0
    try:
        db_reachable = database.health_check()  # True only if DB up AND pgvector present.
        total, analyzed, pending = database.count_tracks()
    except database.DBUnavailableError:
        db_reachable = False  # Expected when Docker isn't running.

    return {
        "total": total,
        "analyzed": analyzed,
        "pending": pending,
        "db_reachable": db_reachable,
        "crate_size_mb": _crate_dir_size_mb(),
    }


# ════════════════════════════════════════════════════════════
#  SIZE / BANNER HELPERS
# ════════════════════════════════════════════════════════════
def _crate_dir_size_mb() -> float:
    """Total size of all files in ./crate/ in megabytes (0.0 if it doesn't exist)."""
    if not CRATE_DIR.exists():
        return 0.0
    total_bytes = sum(p.stat().st_size for p in CRATE_DIR.glob("*") if p.is_file())
    return total_bytes / (1024 * 1024)


def _human_size(mb: float) -> str:
    """Format a size in MB as a human string, promoting to GB past 1024 MB."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.1f} MB"


def startup_banner() -> None:
    """Print and log the concise three-line crate status block.

    Reports track counts, the analysis pipeline level, DB connectivity, the
    analysis backend in use (module vs subprocess), and the crate size. Tolerant
    of a down database — it degrades to zeros + "disconnected" rather than raising.
    """
    _ensure_crate_dir()
    health = crate_health()
    db_state = "connected" if health["db_reachable"] else "disconnected"

    # Pipeline level only knowable when analyze imported in-process; the
    # subprocess backend can't cheaply probe it, so show '?'.
    if ANALYSIS_MODE == "module":
        try:
            level = f"{analyze.ModelManager.pipeline_level()}/5"
        except Exception:
            level = "?/5"
    else:
        level = "?/5"

    line1 = (f"[The Crate Crate] tracks: {health['total']} | "
             f"analyzed: {health['analyzed']} | pending: {health['pending']}")
    line2 = (f"[The Crate Crate] pipeline: level {level} | DB: {db_state} | "
             f"analysis: {ANALYSIS_MODE}")
    line3 = (f"[The Crate Crate] crate dir: ./crate/ "
             f"({_human_size(health['crate_size_mb'])})")
    for line in (line1, line2, line3):
        print(line)
        logger.info(line)


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    """Configure the argparse CLI mirroring the public operations."""
    p = argparse.ArgumentParser(
        prog="crate.py",
        description="The Crate — crate management (ingest + standardisation).")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("add-rec", help="Record live audio into the crate.")
    sp.add_argument("--device", type=int, default=None,
                    help="input device index (default: system default)")
    sp.add_argument("--label", type=str, default=None,
                    help='optional "artist - title"')
    sp.add_argument("--crate", type=str, default=None,
                    help="crate name to file into (default: active crate)")

    sp = sub.add_parser("add-file", help="Import one .wav/.mp3/.flac file.")
    sp.add_argument("path", type=str)
    sp.add_argument("--label", type=str, default=None,
                    help='optional "artist - title"')
    sp.add_argument("--crate", type=str, default=None,
                    help="crate name to file into (default: active crate)")

    sp = sub.add_parser("add-folder", help="Batch-import a folder of audio.")
    sp.add_argument("path", type=str)
    sp.add_argument("--crate", type=str, default=None,
                    help="crate name to file into (default: active crate)")

    sp = sub.add_parser("list", help="List crate contents.")
    sp.add_argument("--pending", action="store_true",
                    help="show only tracks awaiting analysis")
    sp.add_argument("--crate", type=str, default=None,
                    help="crate to list (default: active; 'all' = every crate)")

    sp = sub.add_parser("new-crate", help="Create a logical crate (name + genre).")
    sp.add_argument("name", type=str)
    sp.add_argument("--genre", type=str, default=None,
                    help="genre profile for the BPM prior (default: techno). "
                         "Known: " + ", ".join(sorted(GENRE_PROFILES)))

    sub.add_parser("crates", help="List all crates with track counts.")

    sp = sub.add_parser("use", help="Set the ACTIVE crate for future commands.")
    sp.add_argument("name", type=str)

    sp = sub.add_parser("remove", help="Delete a track by its excerpt_id.")
    sp.add_argument("excerpt_id", type=str)

    sp = sub.add_parser("analyze", help="Analyse all pending tracks (idempotent).")
    sp.add_argument("--crate", type=str, default=None,
                    help="restrict the sweep to one crate (default: all crates)")

    sp = sub.add_parser("upgrade",
                        help="Re-analyse tracks below a target pipeline level.")
    sp.add_argument("--level", type=int, default=5,
                    help="target pipeline level (default: 5)")
    sp.add_argument("--crate", type=str, default=None,
                    help="restrict the sweep to one crate (default: all crates)")

    sub.add_parser("fingerprint",
                   help="Backfill acoustic fingerprints for analyzed tracks "
                        "that have none (new tracks are fingerprinted at ingest).")

    sub.add_parser("health", help="Print crate + DB health.")
    sub.add_parser("devices", help="List input devices for live capture.")
    return p


def main():
    """CLI entry point. Prints the startup banner, then dispatches the command."""
    startup_banner()
    args = build_parser().parse_args()
    if args.command is None:
        build_parser().print_help()
        return

    # The ops log their own SUCCESS/FAILED detail; here we just turn any escaped
    # exception into a clean one-line CLI message instead of a raw traceback.
    try:
        if args.command == "add-rec":
            print("excerpt_id:", add_from_recording(device_index=args.device,
                                                    label=args.label, crate=args.crate))
        elif args.command == "add-file":
            print("excerpt_id:", add_from_file(args.path, label=args.label,
                                              crate=args.crate))
        elif args.command == "add-folder":
            ids = add_from_folder(args.path, crate=args.crate)
            print(f"added {len(ids)} track(s):")
            for tid in ids:
                print(" ", tid)
        elif args.command == "list":
            list_crate(show_pending=args.pending, crate=args.crate)
        elif args.command == "remove":
            remove(args.excerpt_id)
        elif args.command == "analyze":
            print(f"analysed {analyze_pending(crate=args.crate)} track(s).")
        elif args.command == "upgrade":
            n = upgrade_pipeline(target_level=args.level, crate=args.crate)
            print(f"upgraded {n} track(s) to L{args.level}.")
        elif args.command == "fingerprint":
            print(f"fingerprinted {fingerprint_pending()} track(s).")
        elif args.command == "new-crate":
            cid = database.create_crate(args.name, genre=args.genre)
            print(f"crate '{args.name}' ready ({cid})")
        elif args.command == "crates":
            active = database.active_crate_id()
            for c in database.list_crates():
                star = " ←" if str(c["crate_id"]) == active else ""
                print(f"  {c['name']:<24} {c['genre']:<12} "
                      f"{c['n_analyzed']}/{c['n_tracks']} analysed  "
                      f"seed {c['bpm_seed_lo']:.0f}-{c['bpm_seed_hi']:.0f} BPM{star}")
        elif args.command == "use":
            database.set_active_crate(args.name)
            print(f"active crate -> '{args.name}'")
        elif args.command == "health":
            h = crate_health()
            print(f"\ntracks:        {h['total']}")
            print(f"analyzed:      {h['analyzed']}")
            print(f"pending:       {h['pending']}")
            print(f"DB reachable:  {h['db_reachable']}")
            print(f"crate size:    {_human_size(h['crate_size_mb'])}\n")
        elif args.command == "devices":
            list_input_devices()
    except database.DBUnavailableError as e:
        print(f"\n✗ {e}")
    except Exception as e:
        # Already logged with full context inside the op; keep the console terse.
        print(f"\n✗ {args.command} failed: {e}")


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════
#  REFACTOR NOTES
# ════════════════════════════════════════════════════════════
# STRUCTURE (top -> bottom):
#   logging setup -> analysis-backend selection -> standardisation constants ->
#   lazy audio-lib imports -> path/id helpers -> lightweight per-second analysis
#   (window) -> load + standardise + write -> breakdown detection -> feature
#   extraction dispatch -> ingest core -> live capture -> public ops -> banner ->
#   CLI -> these notes.
#
# SHARED HELPERS EXTRACTED (each piece of logic lives in exactly one place):
#   * _standardize()   the mono->16k->best-window pipeline, used by all 3 sources.
#   * _ingest()        standardise + write WAV + insert row, used by all 3 add_* ops.
#   * _analyze_and_persist()  extract -> breakdowns -> update_track_features +
#                      upsert embedding; shared by add_* (inline) and analyze_pending.
#   * _extract_features()  the single module-vs-subprocess switch.
#   * _db_row()/_crate_path()/_excerpt_id()  the one bridge between crate ids and DB rows.
#   * _per_second_features() + _best_window()  the lightweight window selector,
#                      deliberately independent of analyze.py's heavy pipeline.
#
# ASSUMPTIONS ABOUT analyze.py (VERIFY THESE):
#   * The module is named analyze.py (the brief said "analysis.py" — it does not
#     exist; analyze.py exposes the same extract_features()/get_features()).
#   * extract_features(path) -> a TrackFeatures dataclass whose asdict() includes:
#       bpm, key, scale, camelot, duration, pipeline_level, energy_curve (per-sec
#       RMS), complexity_curve (per-sec spectral complexity), and effnet_embedding
#       (L2-normalised 1280-D list or None). Breakdown detection REUSES
#       energy_curve + complexity_curve rather than recomputing.
#   * ModelManager.pipeline_level() and ModelManager.REGISTRY['effnet'] exist
#     (used for the banner level and the embedding model_version string).
#   * Importing analyze.py has no heavy side effects (no model load, no DB) — it
#     only imports essentia and probes for TensorFlow. Confirmed by reading it.
#
# ASSUMPTIONS ABOUT database.py (VERIFY THESE):
#   * insert_track(crate_path, filename, duration) -> str  AUTO-generates the UUID
#     primary key and does NOT accept a caller-supplied id; there is also no
#     update_track_path(). CONSEQUENCE: a crate file cannot be named after the DB
#     primary key without either modifying database.py or leaving crate_path
#     stale. So crate.py's identifier is the EXCERPT_ID — the FILENAME STEM (a
#     uuid4 we mint) — kept deliberately distinct in name from database.py's
#     track_id (the DB primary key); we bridge the two via the UNIQUE crate_path
#     (_db_row). The DB primary key (track_id) remains the id the rest of the
#     system uses (embeddings, sessions); the listener locates audio via a row's
#     crate_path, so it never needs the excerpt_id. If you'd rather crate
#     filenames equal the DB primary key, add insert_track(track_id=...) OR
#     update_track_path() to database.py and I will switch the naming over —
#     it's a ~10-line change here.
#   * update_track_features(track_id, features_dict, pipeline_level) stores the
#     dict verbatim as JSONB. We add 'breakdowns'/'breakdown_count'/
#     'breakdown_reliability' keys; analyze.py's loader ignores unknown keys, so
#     this is safe. duration_sec is refreshed from features['duration'].
#   * list_tracks() rows expose: crate_path, filename, analyzed_at (NULL ==
#     pending), pipeline_level, features (dict|None), track_id (DB primary key).
#   * count_tracks() -> (total, analyzed, pending); get_track_by_path() -> row|None;
#     delete_track() CASCADEs embeddings; health_check() -> bool; upsert_effnet_
#     embedding(track_id, vector, model_version); DBUnavailableError exists.
#   * Folder-import dedup keys on tracks.filename. If you import the SAME source
#     file once via add-file with a --label and again via add-folder, the label
#     and the basename differ, so it can be added twice — documented edge case.
#
# OTHER NOTES:
#   * config.py was CREATED by this change (it didn't exist); it centralises
#     CRATE_DIR/SAMPLE_RATE/ML_SAMPLE_RATE/CACHE_FILE as the brief specified.
#   * soundfile / sounddevice / librosa are NOT yet in pyproject.toml. They are
#     imported lazily so list/health/remove/analyze work without them; ingest
#     needs soundfile (+ librosa to resample), capture needs sounddevice. Install:
#     uv add soundfile sounddevice librosa
#   * Crate excerpts are stored at 16 kHz (ML_SAMPLE_RATE). analyze.py re-decodes
#     them at 44.1 kHz for its classic DSP and at 16 kHz for ML — both work on a
#     16 kHz source; the DSP simply gains no information above 8 kHz, which is an
#     accepted trade for one uniform, recognition-ready store.
