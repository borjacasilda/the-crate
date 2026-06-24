"""
The Crate — Acoustic Fingerprinting (Shazam-style landmark hashing)
-----------------------------------------------------------------
Pure-DSP module: audio in, (hash, time-offset) pairs out. No database, no
Essentia, no model — just numpy + scipy (already present via librosa). Storage
and matching live in database.py; the live recogniser lives in listener.py.

HOW IT WORKS (Wang 2003, "An Industrial-Strength Audio Search Algorithm")
=========================================================================
1. SPECTROGRAM   STFT of the mono 16 kHz signal → magnitude in dB.
2. CONSTELLATION Local maxima of the spectrogram ("peaks"): the loudest
                 time-frequency points, robust to noise/EQ because a peak
                 survives playback through a speaker + room + mic.
3. HASHING       Each anchor peak is paired with up to FANOUT later peaks in a
                 target zone. The triple (f_anchor, f_target, dt) packs into one
                 int — a "landmark". A track yields thousands of landmarks.
4. MATCHING      The same extraction runs on a live snippet. Matching hashes
                 vote with (t_track - t_query); a true match concentrates votes
                 on ONE offset (the snippet's position inside the track), noise
                 scatters them. Vote count of the best bin = match strength.

WHY EXACT-MATCH BEATS SIMILARITY FOR "WHAT IS PLAYING"
======================================================
EffNet answers "what does this SOUND like" (fuzzy, can confuse twins). Hashes
answer "is this THE SAME recording" (binary): only the same audio produces the
same peak constellation at a consistent time alignment. The crate's excerpts
are the reference corpus — tiny and controlled, ideal for this technique.

TUNING NOTES
============
The defaults below target the crate's 16 kHz mono excerpts:
  * NFFT=1024 / HOP=512 → 31.25 frames/s, 15.6 Hz/bin — Shazam-class resolution.
  * PEAKS_PER_SECOND=18 caps landmark density (storage) while keeping matches
    plentiful: a 10 s clean snippet typically aligns 50–300 votes.
  * MIN_VOTES=12 accepted votes ≈ astronomically unlikely by chance (each vote
    is a 32-bit hash AND offset agreement), but reachable within ~5–8 s of audio
    even over a built-in mic.
"""
import numpy as np

# ── STFT grid ─────────────────────────────────────────────────────────────────
SR = 16000          # The one rate everything speaks: crate excerpts + live buffer.
NFFT = 1024         # 64 ms window @ 16 kHz → 513 bins, 15.6 Hz each.
HOP = 512           # 50% overlap → 31.25 frames/s. dt and offsets are in FRAMES.

# ── Constellation ─────────────────────────────────────────────────────────────
PEAK_NEIGHBORHOOD = (15, 15)   # (time frames, freq bins) a peak must dominate.
PEAKS_PER_SECOND = 18          # Density cap: strongest peaks win.

# ── Landmark pairing ──────────────────────────────────────────────────────────
FANOUT = 8          # Pairs per anchor peak.
TARGET_T_MIN = 1    # Target zone: 1 frame (32 ms) …
TARGET_T_MAX = 96   # … to 96 frames (~3 s) after the anchor.

# ── Matching ──────────────────────────────────────────────────────────────────
# Minimum aligned votes for a confident ID. Every vote = full 32-bit hash match
# AND offset agreement, so false positives need a conspiracy; true positives on
# a ≥8 s snippet land in the hundreds.
MIN_VOTES = 12
# Votes at which confidence saturates to 1.0 (linear below).
CONFIDENCE_SATURATION = 60


def _stft_mag_db(audio: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram in dB, shape (n_frames, NFFT//2+1).

    Hand-rolled framing (stride trick + Hann + rfft) instead of librosa.stft:
    zero extra imports, and this module must stay cheap to import for the
    ingest hook in crate.py.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < NFFT:
        audio = np.pad(audio, (0, NFFT - len(audio)))
    n_frames = 1 + (len(audio) - NFFT) // HOP
    # (n_frames, NFFT) view without copying the signal.
    strides = (audio.strides[0] * HOP, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, NFFT), strides=strides)
    window = np.hanning(NFFT).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    return 20.0 * np.log10(mag + 1e-10)


def _find_peaks(spec_db: np.ndarray) -> "list[tuple[int, int]]":
    """Constellation map: (t_frame, f_bin) of the strongest local maxima.

    A point qualifies when it dominates its PEAK_NEIGHBORHOOD and clears a
    floor 10 dB above the spectrogram median (kills silence/noise-floor
    "peaks"). The PEAKS_PER_SECOND densest survivors win, strongest first —
    bounding storage per track regardless of programme material.
    """
    from scipy import ndimage   # Transitive dep via librosa; lazy keeps import light.

    local_max = ndimage.maximum_filter(spec_db, size=PEAK_NEIGHBORHOOD) == spec_db
    floor = np.median(spec_db) + 10.0
    candidates = np.argwhere(local_max & (spec_db > floor))   # (t, f) pairs.
    if len(candidates) == 0:
        return []
    # Keep the strongest N overall (N scales with duration).
    duration_sec = spec_db.shape[0] * HOP / SR
    cap = max(int(PEAKS_PER_SECOND * duration_sec), 1)
    if len(candidates) > cap:
        strengths = spec_db[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[np.argsort(strengths)[::-1][:cap]]
    # Time order is what the pairing loop needs.
    return [tuple(p) for p in candidates[np.argsort(candidates[:, 0])]]


def _pack(f1: int, f2: int, dt: int) -> int:
    """(f_anchor, f_target, dt_frames) → one int. 10+10+12 bits, fits BIGINT."""
    return (int(f1) << 22) | (int(f2) << 12) | int(dt)


def extract_hashes(audio_16k: np.ndarray) -> "list[tuple[int, int]]":
    """Fingerprint a mono 16 kHz signal → [(hash, t_anchor_frame), ...].

    The SAME function runs at ingest (over the 120 s excerpt) and at recognition
    (over the live snippet) — identical parameters on both sides is what makes
    the hashes comparable at all.

    Args:
        audio_16k: mono float array at SR (16 kHz). Other rates are the
            caller's job to resample first (crate._resample).
    Returns:
        List of (hash, t_offset) landmark tuples; empty when the signal is too
        short or too quiet to yield peaks.
    """
    peaks = _find_peaks(_stft_mag_db(audio_16k))
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        paired = 0
        for t2, f2 in peaks[i + 1:]:
            dt = t2 - t1
            if dt < TARGET_T_MIN:
                continue
            if dt > TARGET_T_MAX:
                break                      # Peaks are time-sorted — zone exhausted.
            hashes.append((_pack(f1, f2, dt), int(t1)))
            paired += 1
            if paired >= FANOUT:
                break
    return hashes


def confidence_from_votes(votes: int) -> float:
    """Map an aligned-vote count to a 0..1 confidence (linear, saturating)."""
    return max(0.0, min(1.0, votes / float(CONFIDENCE_SATURATION)))
