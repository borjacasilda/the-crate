"""
The Crate — Audio Capture Engine (for the local API)
--------------------------------------------------
The stateful core behind api.py: owns the microphone/interface while a take is
being recorded or the active-listening placeholder is running.

WHY A SEPARATE MODULE
=====================
crate._capture() blocks on input() — perfect for the CLI, useless for HTTP where
"start" and "stop" arrive as two different requests. This module re-implements
ONLY the capture state machine (start → callback fills chunks → stop returns
audio); everything downstream (standardise, write excerpt, DB insert, analysis)
still goes through crate._ingest() / crate._analyze_and_persist() — the single
ingest funnel. No ingest logic is duplicated here.

CONCURRENCY MODEL
=================
One global engine, one audio activity at a time (the device is exclusive anyway):
    IDLE ──start_recording()──▶ RECORDING ──stop()──▶ IDLE  (returns audio)
    IDLE ──start_listening()──▶ LISTENING ──stop()──▶ IDLE
A lock guards every transition. The PortAudio callback runs on its own thread
and only appends to a list / ring buffer — it never touches the state machine.

Level metering: the callback keeps the RMS + peak of the most recent block so
the UI can poll a VU meter without copying the whole take.
"""
import logging
import threading
import time

import numpy as np

logger = logging.getLogger("thecrate")


def _sounddevice():
    """Return the sounddevice module or raise a clear, actionable error."""
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as e:  # OSError: PortAudio shared lib missing.
        raise RuntimeError("sounddevice is required for capture — "
                           "install it: `uv add sounddevice`") from e


class CaptureBusyError(RuntimeError):
    """Raised when a start is requested while another capture is active."""


class CaptureEngine:
    """Single-owner state machine over the system's audio input.

    States: 'idle' | 'recording' | 'listening'. All public methods are
    thread-safe; FastAPI handlers may call them from any worker thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "idle"
        self._stream = None          # The live sd.InputStream while not idle.
        self._chunks = []            # Recording: every callback block (full take).
        self._sr = None              # Capture rate (device native).
        self._device_index = None
        self._device_name = None
        self._started_at = None      # time.monotonic() at stream start.
        # VU meter: written by the PortAudio callback, read by status pollers.
        # A plain float store/load is atomic in CPython — no extra lock needed.
        self._level_rms = 0.0
        self._level_peak = 0.0
        # Listening mode keeps only a rolling tail (it must run for hours
        # without growing); recording keeps everything (it IS the take).
        self._listen_tail = None     # np.ndarray ring, guarded by _tail_lock.
        self._tail_lock = threading.Lock()
        self._listen_tail_seconds = 30.0

    # ── Introspection ─────────────────────────────────────────────────────────
    def status(self) -> dict:
        """Snapshot for the API: state, device, elapsed seconds, current level."""
        with self._lock:
            elapsed = (time.monotonic() - self._started_at) if self._started_at else 0.0
            return {
                "state": self._state,
                "device_index": self._device_index,
                "device_name": self._device_name,
                "elapsed_seconds": round(elapsed, 1),
                "level_rms": round(self._level_rms, 4),
                "level_peak": round(self._level_peak, 4),
                "clipping": self._level_peak >= 0.999,
            }

    # ── Device helpers ────────────────────────────────────────────────────────
    def input_devices(self, rescan: bool = True) -> list:
        """Input-capable devices as [{index, name, channels, samplerate, default}].

        PortAudio snapshots the device list at init, so an audio interface
        plugged in AFTER startup is invisible to the cached list. When the
        engine is idle we tear PortAudio down and re-init to pick up
        hot-plugged hardware; while capturing we serve the cached snapshot
        (re-init would kill the open stream). Pass rescan=False on cheap
        polling paths (e.g. /health) that don't need hot-plug freshness.
        """
        sd = _sounddevice()
        if rescan and self._state == "idle":
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass        # The cached list is still better than an error.
        default_in = sd.default.device[0] if sd.default.device else None
        out = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] < 1:
                continue
            out.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "samplerate": int(dev["default_samplerate"]),
                "default": idx == default_in,
            })
        return out

    def level_test(self, device_index=None, seconds: float = 2.0) -> dict:
        """Capture a short probe and report its levels (gain-staging check).

        Grabs `seconds` of audio from the device and returns RMS / peak /
        clipping — the A/B tool for comparing the built-in mic against the
        interface before committing a take. Requires the engine to be idle
        (the device is exclusive).
        """
        with self._lock:
            if self._state != "idle":
                raise CaptureBusyError(f"capture engine is busy: {self._state}")
            self._state = "leveltest"   # Reserve the device for the probe.
        try:
            sd = _sounddevice()
            info = sd.query_devices(device_index if device_index is not None
                                    else sd.default.device[0], "input")
            sr = int(info["default_samplerate"])
            frames = int(seconds * sr)
            probe = sd.rec(frames, samplerate=sr, channels=1,
                           dtype="float32", device=device_index)
            sd.wait()
            probe = probe.reshape(-1)
            rms = float(np.sqrt(np.mean(probe ** 2)))
            peak = float(np.max(np.abs(probe))) if len(probe) else 0.0
            return {
                "device_index": device_index,
                "device_name": info["name"],
                "samplerate": sr,
                "seconds": seconds,
                "rms": round(rms, 4),
                "peak": round(peak, 4),
                "clipping": peak >= 0.999,
                # Quick human verdict for the UI: silent probe usually means the
                # wrong device or a muted channel; clipping means too much gain.
                "verdict": ("clipping — lower the gain" if peak >= 0.999 else
                            "no signal — check device/cable" if rms < 1e-4 else
                            "ok"),
            }
        finally:
            with self._lock:
                self._state = "idle"

    # ── Recording (full take, unbounded) ──────────────────────────────────────
    def start_recording(self, device_index=None) -> dict:
        """Open the input stream and start accumulating the full take."""
        with self._lock:
            if self._state != "idle":
                raise CaptureBusyError(f"capture engine is busy: {self._state}")
            sd = _sounddevice()
            info = sd.query_devices(device_index if device_index is not None
                                    else sd.default.device[0], "input")
            sr = int(info["default_samplerate"])
            chunks = []

            def _cb(indata, frames, time_info, status):
                if status:  # Over/underflows are logged but don't abort the take.
                    logger.warning("recording stream status: %s", status)
                block = indata.copy().reshape(-1)
                chunks.append(block)
                self._level_rms = float(np.sqrt(np.mean(block ** 2)))
                self._level_peak = float(np.max(np.abs(block)))

            stream = sd.InputStream(samplerate=sr, channels=1,
                                    device=device_index, dtype="float32",
                                    callback=_cb)
            stream.start()
            self._stream, self._chunks, self._sr = stream, chunks, sr
            self._device_index, self._device_name = device_index, info["name"]
            self._started_at = time.monotonic()
            self._state = "recording"
            logger.info("api-recording START device=%s rate=%d", info["name"], sr)
        return self.status()

    def stop_recording(self) -> tuple:
        """Close the stream and return the take as (audio, sr).

        Raises:
            RuntimeError: when not recording, or when no audio arrived (dead
                device / zero-length take) — callers surface this to the UI.
        """
        with self._lock:
            if self._state != "recording":
                raise RuntimeError("not recording")
            self._stream.stop(); self._stream.close()
            chunks, sr = self._chunks, self._sr
            elapsed = time.monotonic() - self._started_at
            self._reset_locked()
        if not chunks:
            raise RuntimeError("no audio captured — check the input device")
        audio = np.concatenate(chunks)
        logger.info("api-recording STOP seconds=%.1f samples=%d", elapsed, len(audio))
        return audio, sr

    def cancel_recording(self) -> None:
        """Close the stream and discard the take."""
        with self._lock:
            if self._state != "recording":
                raise RuntimeError("not recording")
            self._stream.stop(); self._stream.close()
            n = sum(len(c) for c in self._chunks)
            self._reset_locked()
        logger.info("api-recording CANCEL discarded_samples=%d", n)

    # ── Active listening (placeholder: hears, keeps a rolling tail, does nothing) ──
    def start_listening(self, device_index=None) -> dict:
        """Open the input stream in listening mode (recognition NOT wired yet).

        Keeps only the most recent ~30 s in a ring buffer — the exact shape the
        future recogniser chain will consume (listener.RECOG_WINDOW_SECONDS), so
        wiring recognition later is: snapshot the tail, embed, match. For now it
        only feeds the VU meter.
        """
        with self._lock:
            if self._state != "idle":
                raise CaptureBusyError(f"capture engine is busy: {self._state}")
            sd = _sounddevice()
            info = sd.query_devices(device_index if device_index is not None
                                    else sd.default.device[0], "input")
            sr = int(info["default_samplerate"])
            maxlen = int(self._listen_tail_seconds * sr)
            with self._tail_lock:
                self._listen_tail = np.zeros(0, dtype=np.float32)

            def _cb(indata, frames, time_info, status):
                if status:
                    logger.warning("listening stream status: %s", status)
                block = indata.copy().reshape(-1)
                with self._tail_lock:
                    self._listen_tail = np.concatenate(
                        [self._listen_tail, block])[-maxlen:]
                self._level_rms = float(np.sqrt(np.mean(block ** 2)))
                self._level_peak = float(np.max(np.abs(block)))

            stream = sd.InputStream(samplerate=sr, channels=1,
                                    device=device_index, dtype="float32",
                                    callback=_cb)
            stream.start()
            self._stream, self._sr = stream, sr
            self._device_index, self._device_name = device_index, info["name"]
            self._started_at = time.monotonic()
            self._state = "listening"
            logger.info("api-listening START device=%s rate=%d (placeholder mode)",
                        info["name"], sr)
        return self.status()

    def listening_tail(self) -> tuple:
        """(copy of the rolling tail, sr) — the future recogniser's input."""
        with self._tail_lock:
            tail = self._listen_tail.copy() if self._listen_tail is not None \
                else np.zeros(0, dtype=np.float32)
        return tail, self._sr

    def stop_listening(self) -> None:
        """Close the listening stream and drop the tail."""
        with self._lock:
            if self._state != "listening":
                raise RuntimeError("not listening")
            self._stream.stop(); self._stream.close()
            self._reset_locked()
            with self._tail_lock:
                self._listen_tail = None
        logger.info("api-listening STOP")

    # ── Internal ──────────────────────────────────────────────────────────────
    def _reset_locked(self) -> None:
        """Return to idle. Caller MUST hold self._lock."""
        self._stream = None
        self._chunks = []
        self._sr = None
        self._device_index = None
        self._device_name = None
        self._started_at = None
        self._level_rms = 0.0
        self._level_peak = 0.0
        self._state = "idle"


# The process-wide engine instance api.py imports. One process, one mic.
ENGINE = CaptureEngine()
