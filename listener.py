"""
The Crate — Live Listener
-----------------------
The module that closes the loop: it turns audio coming off the decks into crate
entries (capture) and into live recognition + recommendations (active).

Two modes, mirroring how a vinyl DJ actually works:

    1. CAPTURE  (crate building) — listen to a record from when you start until
       you stop (press Enter), keep the most characteristic 120 s, and file it
       in the crate. This is how music GETS IN. It is a thin wrapper over
       crate.add_from_recording(): the same standardisation funnel (mono / 16 kHz
       / best-120 s window) every other crate source goes through, so a captured
       record is shape-identical to an imported one.

    2. ACTIVE   (live recognition) — listen continuously to what is PLAYING,
       recognise it against the crate (everything that can sound here is already
       in the crate), log it to a session tracklist, and surface the next-record
       recommendations live. This is where the playlist gets built.

Recognition is a PLUGGABLE CHAIN of strategies tried in order, so a strategy can
be swapped or dropped after real-world testing without touching the loop:

    FingerprintRecogniser — landmark-hash (Shazam-style) EXACT matching against
                            the hashes extracted from every excerpt at ingest
                            (fingerprint.py + the `fingerprints` table). First
                            in the chain: one pass to a confident ID, robust
                            even over a laptop mic. Bypasses the debounce.
    EffnetRecogniser      — embed the live audio and nearest-neighbour match it
                            against the crate's EffNet vectors. Fuzzy fallback:
                            survives what breaks hashes (e.g. repitched vinyl).
    RecommendOnlyRecogniser — never identifies; just embeds what's playing so
                            candidates can still be surfaced when ID fails.

CLI:
    uv run python listener.py devices
    uv run python listener.py capture [--device N] [--label "Artist - Title"]
    uv run python listener.py active  [--device N] [--mode safe|balanced|creative]
                                      [--threshold F] [--window S] [--interval S]
                                      [--stable N] [--line-in] [--top N]
"""
import argparse
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

import analyze
import crate
import database
import fingerprint
from config import ML_SAMPLE_RATE

logger = logging.getLogger("thecrate.listener")

# ── Active-mode tuning (all overridable from the CLI) ─────────────────────────
RECOG_WINDOW_SECONDS  = 30.0   # Rolling buffer length embedded each recognition pass.
RECOG_INTERVAL_SECONDS = 5.0   # Seconds between recognition passes.
RECOG_MIN_SECONDS     = 8.0    # Don't attempt recognition until this much audio is buffered.
# Max cosine distance (0 = identical direction, 2 = opposite) for an EffNet match
# to count as a confident ID. Line-in matches sit very low; a built-in mic adds
# acoustic noise and needs this looser. Exposed as --threshold for field tuning.
RECOG_COSINE_MAX      = 0.15
# A candidate must be the nearest match this many consecutive passes before it is
# accepted as a NEW track — debounces flicker between two close neighbours.
STABLE_READS          = 2


# ════════════════════════════════════════════════════════════
#  ROLLING AUDIO BUFFER
# ════════════════════════════════════════════════════════════
class RollingBuffer:
    """Thread-safe ring buffer holding the most recent `seconds` of mono audio.

    The capture callback (PortAudio thread) appends; the recognition loop (main
    thread) snapshots. A lock guards the shared array so a snapshot never tears.
    """

    def __init__(self, seconds: float, sr: int):
        self._maxlen = int(seconds * sr)
        self._sr = sr
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray) -> None:
        """Append a 1-D float32 block, dropping anything older than `seconds`."""
        with self._lock:
            self._buf = np.concatenate([self._buf, chunk])[-self._maxlen:]

    def snapshot(self) -> np.ndarray:
        """Return a copy of the current buffer (safe to process without the lock held)."""
        with self._lock:
            return self._buf.copy()

    def seconds(self) -> float:
        """How many seconds of audio are currently buffered."""
        with self._lock:
            return len(self._buf) / self._sr


# ════════════════════════════════════════════════════════════
#  RECOGNITION STRATEGIES  (pluggable chain)
# ════════════════════════════════════════════════════════════
@dataclass
class RecognitionResult:
    """Outcome of one recognition pass.

    identified=True means a specific crate track was matched (track_id/features
    populated). identified=False is the recommend-only fallback: no ID, but it
    still carries the live embedding so candidates can be surfaced.
    """
    identified: bool
    strategy: str
    confidence: float                      # 0..1, higher = surer.
    track_id: "str | None" = None
    crate_path: "str | None" = None
    filename: "str | None" = None
    features: "analyze.TrackFeatures | None" = None
    distance: "float | None" = None        # cosine distance of the match, when applicable.


class Recogniser:
    """Base strategy. Subclasses return a RecognitionResult or None (no match).

    `embedding` may be the 1280-D vector itself OR a zero-arg callable that
    computes it on demand (see _resolve_embedding). The live loop passes a
    cached callable so the ~1 s EffNet inference is only paid when a strategy
    actually needs it — with fingerprint first in the chain, the common case
    (exact hit) never touches TensorFlow at all.
    """
    name = "base"

    def recognise(self, embedding, audio_16k, sr) -> "RecognitionResult | None":
        raise NotImplementedError


def _resolve_embedding(embedding):
    """Vector-or-callable → vector (or None). The lazy-embedding unwrapper."""
    return embedding() if callable(embedding) else embedding


class EffnetRecogniser(Recogniser):
    """Identify the playing track by EffNet-embedding nearest-neighbour search.

    Embeds the live buffer (already done upstream, passed in as `embedding`),
    runs find_similar_effnet(n=1), and accepts the hit only if its cosine
    distance is within `threshold`. Reuses the crate's HNSW index — no new store.
    """
    name = "effnet"

    def __init__(self, threshold: float = RECOG_COSINE_MAX):
        self.threshold = threshold

    def recognise(self, embedding, audio_16k, sr):
        if embedding is None or not database.DB_AVAILABLE:
            return None
        embedding = _resolve_embedding(embedding)   # Pay the inference only here.
        if embedding is None:
            return None
        results = database.find_similar_effnet(embedding, n=1)
        if not results:
            return None
        top = results[0]
        dist = float(top.get("cosine_distance", 2.0))
        if dist > self.threshold:
            return None                       # Nearest crate track isn't close enough.
        feats = analyze._hydrate(top["features"]) if top.get("features") else None
        # Confidence: 1.0 at distance 0, 0.0 at the threshold boundary.
        confidence = max(0.0, 1.0 - dist / self.threshold) if self.threshold > 0 else 1.0
        return RecognitionResult(
            identified=True, strategy=self.name, confidence=confidence,
            track_id=str(top["track_id"]), crate_path=top["crate_path"],
            filename=top.get("filename"), features=feats, distance=dist)


class FingerprintRecogniser(Recogniser):
    """Landmark-hash (Shazam-style) EXACT recognition. First in the chain.

    Extracts constellation hashes from the live snippet (fingerprint.py — the
    same parameters used at ingest) and runs the offset-vote match in Postgres
    (database.match_fingerprints). A hit means "this IS that recording", not
    "this sounds like it" — so a confident fingerprint ID needs no debounce and
    lands in a single pass (~the snippet length, vs 2 stable EffNet reads).

    Accepts only when the best (track, offset) bin collects >= min_votes aligned
    matches: each vote is a full 32-bit hash AND offset agreement, so the false-
    positive odds are negligible while ~8 s of even mic-quality audio clears it.
    """
    name = "fingerprint"

    def __init__(self, min_votes: int = None):
        self.min_votes = min_votes if min_votes is not None else fingerprint.MIN_VOTES

    def recognise(self, embedding, audio_16k, sr):
        if audio_16k is None or not database.DB_AVAILABLE:
            return None
        if sr != fingerprint.SR:        # Contract: callers hand us 16 kHz audio.
            logger.warning("fingerprint recogniser: expected %d Hz, got %s — skipping",
                           fingerprint.SR, sr)
            return None
        query = fingerprint.extract_hashes(audio_16k)
        if len(query) < self.min_votes:        # Too quiet/short to ever clear the bar.
            return None
        rows = database.match_fingerprints(query, top=1)
        if not rows:
            return None
        top = rows[0]
        votes = int(top["votes"])
        if votes < self.min_votes:
            return None
        feats = analyze._hydrate(top["features"]) if top.get("features") else None
        return RecognitionResult(
            identified=True, strategy=self.name,
            confidence=fingerprint.confidence_from_votes(votes),
            track_id=str(top["track_id"]), crate_path=top["crate_path"],
            filename=top.get("filename"), features=feats,
            distance=None)   # Exact-match strategy: cosine distance not applicable.


class RecommendOnlyRecogniser(Recogniser):
    """Fallback that never identifies but keeps the loop useful.

    When no strategy could name the track, this wraps the live embedding in a
    minimal TrackFeatures so the loop can still surface nearest-neighbour
    candidates. identified=False, so the active loop shows candidates WITHOUT
    logging a session entry (we don't know what to log).
    """
    name = "recommend_only"

    def recognise(self, embedding, audio_16k, sr):
        embedding = _resolve_embedding(embedding)
        if embedding is None:
            return None
        feats = analyze.TrackFeatures(effnet_embedding=embedding)
        return RecognitionResult(
            identified=False, strategy=self.name, confidence=0.0, features=feats)


class RecogniserChain:
    """Try each strategy in order; return the first that produces a result.

    With RecommendOnlyRecogniser last, the chain effectively always returns
    something once an embedding exists — identified when a real ID strategy
    succeeded, non-identifying otherwise.
    """

    def __init__(self, recognisers):
        self.recognisers = recognisers

    def recognise(self, embedding, audio_16k, sr):
        for r in self.recognisers:
            try:
                res = r.recognise(embedding, audio_16k, sr)
            except Exception as e:                # One bad strategy never aborts the chain.
                logger.warning("recogniser '%s' failed: %s", r.name, e)
                res = None
            if res is not None:
                return res
        return None

    @classmethod
    def default(cls, threshold: float = RECOG_COSINE_MAX) -> "RecogniserChain":
        """The standard chain: fingerprint (exact) → EffNet (fuzzy) → recommend-only.

        Fingerprint goes FIRST: when the recording is in the crate it answers in
        one pass with near-zero false-positive odds. EffNet remains the fallback
        for degraded signals (heavy room bleed, pitch-shifted playback — hashes
        break under repitching, embeddings survive it).
        """
        return cls([FingerprintRecogniser(),
                    EffnetRecogniser(threshold),
                    RecommendOnlyRecogniser()])


# ════════════════════════════════════════════════════════════
#  RECOMMENDATION RENDERING  (reuses analyze.py scoring)
# ════════════════════════════════════════════════════════════
def _recommend(current: "analyze.TrackFeatures", mode: str, top: int,
               exclude_path: str = None, energy_target: float = 0.0,
               temperature: float = 0.0, crate_id: str = None) -> list:
    """Score the best candidate crate tracks against `current` and return the top.

    Reuses analyze.score_candidates() (two-stage retrieval — HNSW top-K with
    the exact-winner safeguard — then mix_score) + analyze.sample_by_score(),
    so the live engine and the offline `next` command rank (and sample)
    identically. `energy_target` biases the energy direction; `temperature`
    adds variety. Returns [(path, features, score)].
    """
    strengths = analyze._ensure_strengths(mode)
    strengths.energy_target = energy_target
    scored = analyze.score_candidates(
        current, mode=mode, strengths=strengths,
        exclude_paths=[exclude_path] if exclude_path else None,
        crate=crate_id if crate_id else "__active__")
    return analyze.sample_by_score(scored, top, temperature)


def _print_recommendations(current: "analyze.TrackFeatures", mode: str, top: int,
                           exclude_path: str = None, energy_target: float = 0.0,
                           temperature: float = 0.0, crate_id: str = None) -> None:
    """Print the top next-track picks for `current` in the live one-line style."""
    from pathlib import Path
    picks = _recommend(current, mode, top, exclude_path=exclude_path,
                       energy_target=energy_target, temperature=temperature,
                       crate_id=crate_id)
    if not picks:
        print("     (no other analysed tracks in the crate yet)")
        return
    for i, (path, feats, s) in enumerate(picks, 1):
        bpm_d = analyze.bpm_delta(current.bpm, feats.bpm)
        key_rel = analyze.key_relationship_label(current.camelot, feats.camelot)
        _, dir_label, energy_pct = analyze.energy_direction(current, feats)
        tip = analyze.mix_tip(s, key_rel, bpm_d, dir_label)
        star = "  ★" if s["total"] >= analyze.PERFECT_MIX_THRESHOLD else ""
        print(f"     {i}. {Path(path).name}{star}")
        print(f"        total {s['total']:.2f} · {feats.bpm:.0f} BPM (Δ{bpm_d:+.1f}) · "
              f"{feats.camelot} ({key_rel}) · {dir_label} {energy_pct:+.0f}%")
        print(f"        ▶ {tip}")


# ════════════════════════════════════════════════════════════
#  MODE 1 — CAPTURE  (add a record to the crate)
# ════════════════════════════════════════════════════════════
def listen_capture(device_index=None, label: str = None,
                   composite: bool = False, crate_name: str = None) -> str:
    """Capture a record live and file it in the crate.

    Thin wrapper over crate.add_from_recording(): records until you press Enter,
    keeps a 120 s excerpt, stores the standardised excerpt, and analyses it.
    The whole take is listened to; only the most characteristic slice is kept —
    exactly your Mode 1 requirement.

    Args:
        device_index: input device index; None uses the system default.
        label: optional "Artist - Title"; defaults to a timestamped name.
        composite: False (default) keeps the single busiest contiguous window;
            True keeps three crossfaded peak/mid/low segments covering the
            track's dynamic range — wider live-recognition coverage, at the
            cost of analysis fidelity (see crate._composite_window).
    Returns:
        The excerpt_id of the new excerpt.
    """
    strategy = "composite" if composite else "best"
    print("\n🎙️  CAPTURE — add a record to the crate")
    print(f"   window strategy: {strategy}")
    print("   Drop the needle, let it play, press Enter when you've heard enough.\n")
    excerpt_id = crate.add_from_recording(device_index=device_index, label=label,
                                          strategy=strategy, crate=crate_name)
    print(f"\n✅  Filed in the crate as {excerpt_id}\n")
    return excerpt_id


def import_file(path: str, label: str = None, composite: bool = False,
                crate_name: str = None) -> str:
    """Import one WAV/MP3/FLAC into the crate (Mode 1, file door).

    Same funnel as live capture — crate.add_from_file() standardises, stores
    and analyses immediately — exposed here so the whole "add to the crate"
    mode lives behind one CLI regardless of the input door.

    Args:
        path: audio file on any mounted storage.
        label: optional "Artist - Title"; defaults to the filename.
        composite: window strategy, as in listen_capture().
    Returns:
        The excerpt_id of the new excerpt.
    """
    strategy = "composite" if composite else "best"
    print(f"\n📥  IMPORT — {path}  (window strategy: {strategy})\n")
    excerpt_id = crate.add_from_file(path, label=label, strategy=strategy,
                                     crate=crate_name)
    print(f"\n✅  Filed in the crate as {excerpt_id}\n")
    return excerpt_id


def import_folder(folder: str, composite: bool = False, defer: bool = False,
                  crate_name: str = None) -> list:
    """Batch-import a folder into the crate (Mode 1, folder door).

    crate.add_from_folder() ingests fast and defers the heavy analysis; by
    default this then runs crate.analyze_pending() so the mode is complete in
    one command. Pass defer=True to keep the original two-step behaviour.

    Args:
        folder: directory of audio files (non-recursive).
        composite: window strategy, as in listen_capture().
        defer: True skips the analysis sweep (run it later via crate CLI).
    Returns:
        list of newly added excerpt_ids.
    """
    strategy = "composite" if composite else "best"
    print(f"\n📦  FOLDER IMPORT — {folder}  (window strategy: {strategy})\n")
    added = crate.add_from_folder(folder, strategy=strategy, crate=crate_name)
    print(f"   {len(added)} new track(s) ingested.")
    if added and not defer:
        print("   Analysing pending tracks…")
        n = crate.analyze_pending()
        print(f"\n✅  {n} track(s) analysed and ready.\n")
    elif defer:
        print("   Analysis deferred — run `python crate.py analyze` when ready.\n")
    return added


# ════════════════════════════════════════════════════════════
#  MODE 2 — ACTIVE  (live recognition + recommendations)
# ════════════════════════════════════════════════════════════
def listen_active(device_index=None, mode: str = "balanced",
                  threshold: float = RECOG_COSINE_MAX,
                  window: float = RECOG_WINDOW_SECONDS,
                  interval: float = RECOG_INTERVAL_SECONDS,
                  top: int = 3, energy: str = "stable",
                  temperature: float = 0.0, crate_name: str = None,
                  stable_reads: int = STABLE_READS) -> str:
    """Listen to what's playing, recognise it, log a session, recommend the next track.

    Opens a mix session, captures a rolling buffer, and every `interval` seconds
    embeds the buffer and runs it through the recogniser chain. When a NEW track
    is confidently identified (stable for `stable_reads` passes — fingerprint
    hits skip the debounce entirely, they are exact), it is logged to the
    session (detected_by=<recogniser strategy>) and its top next-track picks are
    printed. Press Enter to stop — the session is then closed and the full
    tracklist printed.

    Args:
        device_index: input device index; None uses the system default.
        mode: scoring preset for the recommendations ('safe'|'balanced'|'creative').
        threshold: max EffNet cosine distance for a confident match.
        window: rolling-buffer length in seconds (audio embedded each pass).
        interval: seconds between recognition passes.
        top: number of next-track recommendations to show per recognised track.
        energy: wanted energy direction for the recs ('up'|'stable'|'down').
        temperature: 0.0 = best picks; >0 samples for variety.
        crate_name: crate the RECOMMENDATIONS draw from (and the session is
            tagged with); None = the active crate. Recognition deliberately
            searches ALL crates — you can play any record you own.
        stable_reads: consecutive passes an EFFNET candidate must stay nearest
            before being accepted (debounces flicker between close neighbours).
            Fingerprint IDs bypass this — an aligned-offset hash match cannot
            flicker. 1 is sensible for clean line-in; keep 2 for a built-in mic.
    Returns:
        The session_id (UUID str) of the logged set, or "" if it couldn't start.
    """
    # ── Pre-flight: the active loop needs both the crate (DB) and the EffNet model. ──
    if not database.DB_AVAILABLE:
        print("\n⚠️  Database unavailable — start Docker: docker compose up -d\n")
        return ""
    if analyze.ModelManager.get("effnet") is None:
        print("\n⚠️  EffNet model unavailable — install essentia-tensorflow and run "
              "`python analyze.py download`. Active recognition needs the embedding.\n")
        return ""

    sd = crate._sounddevice()
    # Resolve the device's native rate — capture there and resample to 16 kHz for
    # the model, the same way crate._capture avoids unsupported-rate PortAudio errors.
    info = sd.query_devices(device_index if device_index is not None
                            else sd.default.device[0], "input")
    sr = int(info["default_samplerate"])

    energy_target = analyze.ENERGY_TARGETS.get(energy, 0.0)   # 'up'/'stable'/'down' → value.
    chain = RecogniserChain.default(threshold)
    buffer = RollingBuffer(window, sr)
    crate_id = database.resolve_crate_id(crate_name)
    session_id = database.create_session(crate_id=crate_id)

    def _cb(indata, frames, time_info, status):
        if status:
            logger.warning("capture stream status: %s", status)
        buffer.append(indata.copy().reshape(-1))   # copy: PortAudio reuses indata.

    # Stop on Enter without blocking the recognition loop: a daemon thread waits
    # on input() and trips the event the loop polls.
    stop = threading.Event()

    def _wait_enter():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        stop.set()
    threading.Thread(target=_wait_enter, daemon=True).start()

    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  🎧  ACTIVE LISTENING — [{info['name']}] @ {sr} Hz")
    print(f"  mode: {mode} · energy: {energy} · temp: {temperature:.1f} · "
          f"window: {window:.0f}s · every {interval:.0f}s · match ≤ {threshold:.2f} · "
          f"stable ×{stable_reads} (fingerprint ×1)")
    print(f"  Press Enter to stop and save the set.")
    print(f"{sep}\n")
    logger.info("active START session=%s device=%s rate=%d mode=%s threshold=%.2f",
                session_id, info["name"], sr, mode, threshold)

    current_id = None          # The track currently believed to be playing.
    pending_id, pending_n = None, 0   # Debounce: candidate + consecutive-read count.
    n_logged = 0

    try:
        with sd.InputStream(samplerate=sr, channels=1, device=device_index,
                            dtype="float32", callback=_cb):
            while not stop.is_set():
                # Sleep in short slices so Enter stops us promptly, not after a full interval.
                slept = 0.0
                while slept < interval and not stop.is_set():
                    time.sleep(0.2)
                    slept += 0.2
                if stop.is_set():
                    break
                if buffer.seconds() < min(RECOG_MIN_SECONDS, window):
                    continue

                audio = buffer.snapshot()
                audio_16k = crate._resample(audio, sr, ML_SAMPLE_RATE)

                # Lazy embedding: fingerprint (first in the chain) needs only the
                # raw audio. The ~1 s EffNet inference runs at most once per pass
                # and ONLY if a downstream strategy asks for the vector.
                emb_cache = []

                def _embed_once(_a=audio_16k):
                    if not emb_cache:
                        emb_cache.append(analyze.embed_effnet(_a))
                    return emb_cache[0]

                res = chain.recognise(_embed_once, audio_16k, ML_SAMPLE_RATE)
                if res is None:
                    continue

                if not res.identified:
                    continue                  # recommend-only fallback: stay quiet until ID.

                if res.track_id == current_id:
                    continue                  # Same track still playing — nothing new.

                # Debounce: an EffNet candidate must repeat before committing
                # (close neighbours can flicker). Fingerprint IDs are exact —
                # offset-aligned hash votes can't flicker — so they commit on
                # the FIRST pass: that single-read path is the speed win.
                needed = 1 if res.strategy == "fingerprint" else stable_reads
                if res.track_id == pending_id:
                    pending_n += 1
                else:
                    pending_id, pending_n = res.track_id, 1
                if pending_n < needed:
                    continue

                current_id = res.track_id
                pending_id, pending_n = None, 0
                n_logged += 1
                _on_new_track(session_id, res, mode, top, n_logged,
                              energy_target=energy_target, temperature=temperature,
                              crate_id=crate_id)
    finally:
        tracklist = database.close_session(session_id)
        logger.info("active STOP session=%s tracks=%d", session_id, len(tracklist))
        _print_set_summary(tracklist)

    return session_id


def _on_new_track(session_id: str, res: "RecognitionResult", mode: str,
                  top: int, position: int, energy_target: float = 0.0,
                  temperature: float = 0.0, crate_id: str = None) -> None:
    """Log a newly-recognised track to the session and print its next-track picks."""
    from pathlib import Path
    database.log_track_played(session_id, res.track_id, detected_by=res.strategy)
    name = res.filename or (Path(res.crate_path).name if res.crate_path else res.track_id)
    conf_pct = int(res.confidence * 100)
    print(f"▶ [{position:02d}] NOW PLAYING — {name}   "
          f"(match {conf_pct}% · dist {res.distance:.3f})")
    if res.features is not None:
        f = res.features
        print(f"     {f.bpm:.0f} BPM · {f.camelot} ({f.key} {f.scale}) · "
              f"energy {analyze.track_energy(f):.3f} · level {f.pipeline_level}/5")
        print(f"  🎚️  next ({mode}):")
        _print_recommendations(f, mode, top, exclude_path=res.crate_path,
                               energy_target=energy_target, temperature=temperature,
                               crate_id=crate_id)
    print()
    logger.info("active RECOGNISED session=%s pos=%d track=%s dist=%.4f",
                session_id, position, res.track_id, res.distance or -1.0)


def _print_set_summary(tracklist: list) -> None:
    """Print the closed session's ordered tracklist."""
    from pathlib import Path
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  🎶  SET COMPLETE — {len(tracklist)} track(s) recognised")
    print(f"{sep}")
    for row in tracklist:
        pos = row.get("position", "?")
        name = row.get("filename") or (Path(row["crate_path"]).name
                                       if row.get("crate_path") else row.get("track_id"))
        played = row.get("played_at")
        when = played.strftime("%H:%M:%S") if hasattr(played, "strftime") else str(played)
        print(f"   [{pos:>2}] {when}  {name}")
    print(f"{sep}\n")


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════
def cmd_devices() -> None:
    """List selectable audio input devices (crate.list_input_devices() prints them)."""
    crate.list_input_devices()   # already pretty-prints index / name / rate.
    print("   Use --device <index> with `capture` or `active`.\n")


def build_parser() -> argparse.ArgumentParser:
    """Configure the listener CLI: devices / capture / active."""
    p = argparse.ArgumentParser(
        prog="listener.py",
        description="The Crate — live listener (capture into the crate, or recognise live).")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("devices", help="List audio input devices.")

    sp = sub.add_parser("capture", help="Record a record and file it in the crate (Mode 1).")
    sp.add_argument("--device", type=int, default=None, help="input device index")
    sp.add_argument("--label", type=str, default=None, help='"Artist - Title"')
    sp.add_argument("--composite", action="store_true",
                    help="excerpt = 3 crossfaded peak/mid/low segments instead of "
                         "one contiguous window (wider recognition coverage)")
    sp.add_argument("--crate", type=str, default=None,
                    help="crate to file into (default: active crate)")

    sp = sub.add_parser("add", help="Import one WAV/MP3/FLAC into the crate (Mode 1).")
    sp.add_argument("file", help="path to the audio file")
    sp.add_argument("--label", type=str, default=None, help='"Artist - Title"')
    sp.add_argument("--composite", action="store_true",
                    help="composite excerpt strategy (see capture --composite)")
    sp.add_argument("--crate", type=str, default=None,
                    help="crate to file into (default: active crate)")

    sp = sub.add_parser("add-folder", help="Batch-import a folder into the crate (Mode 1).")
    sp.add_argument("folder", help="directory of audio files (non-recursive)")
    sp.add_argument("--composite", action="store_true",
                    help="composite excerpt strategy (see capture --composite)")
    sp.add_argument("--defer", action="store_true",
                    help="ingest only; skip the analysis sweep")
    sp.add_argument("--crate", type=str, default=None,
                    help="crate to file into (default: active crate)")

    sp = sub.add_parser("active", help="Live recognition + recommendations (Mode 2).")
    sp.add_argument("--device", type=int, default=None, help="input device index")
    sp.add_argument("--mode", choices=list(analyze.MODE_CONFIG.keys()), default="balanced")
    # The four recognition knobs default to None so --line-in can fill whichever
    # the user did not set explicitly (an explicit flag always wins the preset).
    sp.add_argument("--threshold", type=float, default=None,
                    help=f"max EffNet cosine distance for a confident match "
                         f"(default {RECOG_COSINE_MAX})")
    sp.add_argument("--window", type=float, default=None,
                    help=f"rolling buffer length (seconds) embedded each pass "
                         f"(default {RECOG_WINDOW_SECONDS:.0f})")
    sp.add_argument("--interval", type=float, default=None,
                    help=f"seconds between recognition passes "
                         f"(default {RECOG_INTERVAL_SECONDS:.0f})")
    sp.add_argument("--stable", type=int, default=None,
                    help=f"consecutive passes an EffNet match must repeat before "
                         f"committing (default {STABLE_READS}; fingerprint IDs "
                         f"always commit in 1)")
    sp.add_argument("--line-in", action="store_true", dest="line_in",
                    help="clean-signal preset (interface/line-in): window 15s, "
                         "interval 3s, threshold 0.10, stable 1 — ~6-9s to ID. "
                         "Not recommended over a built-in mic.")
    sp.add_argument("--top", type=int, default=3, help="recommendations per track")
    sp.add_argument("--energy", choices=["up", "stable", "down"], default="stable",
                    help="wanted energy direction for the recommendations")
    sp.add_argument("--temperature", type=float, default=0.0,
                    help="recommendation variety: 0 = best picks, higher = adventurous")
    sp.add_argument("--crate", type=str, default=None,
                    help="crate the recommendations draw from (default: active). "
                         "Recognition always searches ALL crates.")
    return p


# Per-knob (standard_default, line_in_preset). Resolution order for each knob:
# explicit CLI flag > --line-in preset > standard default.
_ACTIVE_KNOBS = {
    "threshold": (RECOG_COSINE_MAX, 0.10),
    "window":    (RECOG_WINDOW_SECONDS, 15.0),
    "interval":  (RECOG_INTERVAL_SECONDS, 3.0),
    "stable":    (STABLE_READS, 1),
}


def _resolve_active_knobs(args) -> dict:
    """Fill unset recognition knobs from the --line-in preset or the defaults."""
    out = {}
    for knob, (standard, preset) in _ACTIVE_KNOBS.items():
        explicit = getattr(args, knob)
        out[knob] = explicit if explicit is not None else (preset if args.line_in
                                                           else standard)
    return out


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "devices":
        cmd_devices()
    elif args.command == "capture":
        listen_capture(device_index=args.device, label=args.label,
                       composite=args.composite, crate_name=args.crate)
    elif args.command == "add":
        import_file(args.file, label=args.label, composite=args.composite,
                    crate_name=args.crate)
    elif args.command == "add-folder":
        import_folder(args.folder, composite=args.composite, defer=args.defer,
                      crate_name=args.crate)
    elif args.command == "active":
        knobs = _resolve_active_knobs(args)
        listen_active(device_index=args.device, mode=args.mode,
                      threshold=knobs["threshold"], window=knobs["window"],
                      interval=knobs["interval"], stable_reads=knobs["stable"],
                      top=args.top, energy=args.energy,
                      temperature=args.temperature, crate_name=args.crate)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
