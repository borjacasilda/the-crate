"""
The Crate — Local Web API (FastAPI)
---------------------------------
The browser-facing front door for the vinyl ripping station. Phase 1 scope:

    RECORD   form (crate / artist / title / device / formats) → record until
             STOP → archive full take (FLAC/WAV/MP3) → ingest through
             crate._ingest() → analyse → results.
    LISTEN   active-listening placeholder: opens the device, fills a rolling
             tail, shows a VU meter — recognition deliberately NOT wired yet.
    LIBRARY  read-only crate/track listing so results are verifiable in-browser.

DESIGN RULES
============
* No ingest logic lives here. Everything funnels through crate._ingest() +
  crate._analyze_and_persist() — the same path the CLI uses. The API only owns
  HTTP plumbing and the capture state machine (recorder.py).
* Analysis is slow (~30-60 s/track) and Essentia's TF graphs are not
  thread-safe → all post-stop processing runs on a 1-worker executor (same
  pattern as mcp_server.py) and the client polls GET /jobs/{id}.
* One audio activity at a time (recording XOR listening XOR level test) —
  enforced by recorder.ENGINE, surfaced to HTTP as 409 Conflict.
* Crate resolution happens at START time, not at stop — a bad crate name must
  fail BEFORE the needle drops, not after the take.

RUN
===
    uv run python api.py                  # http://127.0.0.1:8000
    uv run uvicorn api:app --port 8000    # equivalent
"""
import logging
import math
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, RedirectResponse, Response,
                               StreamingResponse)
from pydantic import BaseModel, Field
from scipy.signal import resample_poly

import config
import database
import crate
import analyze
import discogs
import enrich
import listener
from recorder import ENGINE, CaptureBusyError
from assistant import agent as assistant_agent
from assistant import models as assistant_models
from assistant import ollama_client

logger = logging.getLogger("thecrate")

@asynccontextmanager
async def _lifespan(app):
    """Startup: cap the threadpool to the DB pool, ensure dirs, probe MP3 support."""
    global MP3_SUPPORTED
    # Cap the AnyIO worker threadpool that runs our sync `def` routes so concurrent
    # DB-bound requests can't outnumber the connection pool and trigger PoolError
    # (H4). Leave a few connections for the listener / analysis / MCP background
    # threads that borrow from the same pool. Both sides move together via the
    # THECRATE_DB_POOL_MAX knob.
    import anyio
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = max(1, database.POOL_MAX - 4)
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        MP3_SUPPORTED = "MP3" in sf.available_formats()
    except Exception:
        MP3_SUPPORTED = False
    logger.info("api START threadpool=%d db_pool=%d recordings_dir=%s mp3=%s",
                limiter.total_tokens, database.POOL_MAX,
                config.RECORDINGS_DIR, MP3_SUPPORTED)
    yield
    # Ordered shutdown (A3): signal the live worker to stop and stop accepting new
    # executor work, so a reload/quit doesn't leave a recognition loop + queued
    # analysis running against a tearing-down process. wait=False: don't block the
    # shutdown on an in-flight take/import — daemon threads exit with the process.
    # The DB pool is left to process teardown on purpose (no close hook, and the
    # reconnect layer must stay usable for any borrow still in flight).
    with _LIVE_LOCK:
        LIVE["running"] = False
    _EXECUTOR.shutdown(wait=False)
    _COVER_EXECUTOR.shutdown(wait=False)
    logger.info("api STOP — live worker signalled, executors shut down")


app = FastAPI(title="The Crate API", version="0.1.0",
              description="Local vinyl ripping + analysis station",
              lifespan=_lifespan)


@app.middleware("http")
async def _no_cache_web_assets(request, call_next):
    """Stop the browser serving a STALE shared script or page from disk cache — the
    real cause of "the inline edit doesn't work": an old util.js (without the
    edit/rename wiring) lingers in cache while the file on disk is already fixed.
    This is a local app, so revalidation costs nothing; force it for every .js/.css
    and HTML page so an edit to the front-end is always picked up on reload."""
    resp = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css")) or \
       "text/html" in resp.headers.get("content-type", ""):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


WEB_DIR = Path(__file__).resolve().parent / "web"

# Serialises every Essentia/TF touch (analysis is not thread-safe) AND the
# post-stop pipeline as a whole, so two quick takes queue instead of colliding.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thecrate-api")

# Bounded pool for fire-and-forget Discogs cover re-searches after inline edits
# (A5). Was an unbounded thread-per-edit; this caps concurrency (and idle threads)
# so a rapid batch of edits can't spawn a thread storm hammering Discogs. Separate
# from _EXECUTOR because it does NO Essentia/TF work — it must not queue behind a
# take/import on the single TF worker.
_COVER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="thecrate-cover")

# ── In-memory job registry ────────────────────────────────────────────────────
# job_id → {state, detail, result, error}. Lost on restart by design: a job is
# only the *processing* of a take; the durable artifacts (recordings/, crate/,
# DB rows) all land before 'done'. _JOBS_LOCK guards multi-key mutations.
JOBS: dict = {}
_JOBS_LOCK = threading.Lock()

# Metadata of the take currently being recorded (set at start, consumed at
# stop). One slot is enough: ENGINE enforces a single concurrent recording.
_CURRENT_TAKE: dict = {}

# Probed once at startup; drives the MP3 checkbox in the UI.
MP3_SUPPORTED = False

EXPORT_FORMATS = ("flac", "wav", "mp3")


def _crate_file(row: dict) -> Path:
    """Resolve a track's excerpt on disk by basename under CRATE_DIR (M8).

    crate_path is stored ABSOLUTE, so moving the project folder would 404 every
    audio/waveform read that trusts it. CRATE_DIR is location-independent (config)
    and excerpt basenames are unique, so re-root by basename to survive a move.
    """
    return config.CRATE_DIR / Path(row["crate_path"]).name


def _register_job(job_id: str, payload: dict) -> None:
    """Insert a job and evict the oldest FINISHED ones so JOBS can't grow forever (M4).

    Active jobs (queued / running) are never evicted; only done/failed entries past
    config.JOBS_MAX are dropped, oldest first (dict preserves insertion order).
    """
    with _JOBS_LOCK:
        JOBS[job_id] = payload
        if len(JOBS) > config.JOBS_MAX:
            for jid in list(JOBS):
                if len(JOBS) <= config.JOBS_MAX:
                    break
                if jid != job_id and JOBS[jid].get("state") in ("done", "failed"):
                    del JOBS[jid]


# ════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ════════════════════════════════════════════════════════════
class NewCrate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    genre: str = "techno"
    description: str = Field(default="", max_length=2000)


class StartRecording(BaseModel):
    device_index: "int | None" = None
    crate: "str | None" = None            # Existing crate name/id…
    new_crate: "NewCrate | None" = None   # …or create one (mutually exclusive).
    artist: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    keep_full_take: bool = True
    formats: list = ["flac"]              # Subset of EXPORT_FORMATS.


class StartListening(BaseModel):
    device_index: "int | None" = None
    crate: "str | None" = None    # Crate the session draws recommendations from.


class ListeningParams(BaseModel):
    """Partial update from the /listen dashboard — omitted fields keep their
    current value. One optional field per registered knob; a new dashboard
    parameter = one field here + one entry in LISTENING_PARAM_CATALOG."""
    # Recommendation mode — the preset of modifier strengths (analyze.MODE_CONFIG).
    # It is the BASE the energy direction and the penalizer toggles then layer on top.
    mode: "str | None" = Field(default=None, pattern="^(safe|balanced|creative)$")
    energy: "str | None" = Field(default=None, pattern="^(up|stable|down)$")
    # {modifier_name: bool} — partial; merged into the current toggle state.
    modifiers: "dict | None" = None


# ════════════════════════════════════════════════════════════
#  STATUS / DEVICES
# ════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    """First stop when anything misbehaves: DB, models, devices, formats."""
    db_ok = False
    try:
        db_ok = database.health_check()
    except Exception:
        pass
    try:
        level = analyze.ModelManager.pipeline_level()
    except Exception:
        level = 1
    try:
        n_devices = len(ENGINE.input_devices(rescan=False))   # /health polls every 5s.
    except Exception:
        n_devices = 0
    return {
        "db": db_ok,
        "pipeline_level": level,
        "input_devices": n_devices,
        "mp3_export": MP3_SUPPORTED,
        "recordings_dir": str(config.RECORDINGS_DIR),
        "capture": ENGINE.status(),
    }


@app.get("/devices")
def devices():
    try:
        return ENGINE.input_devices()
    except RuntimeError as e:
        raise HTTPException(500, str(e))


# Input source auto-selection + picker. Priority for the DEFAULT source:
#   1. interface — any input that is neither the built-in mic nor a phone
#      (the vinyl signal path: USB interface / mixer). Wins OUTRIGHT: it is
#      selected with NO picker, exactly as when only the built-in mic exists.
#   2. builtin   — the host machine's own microphone: the sensible default
#      whenever the DJ actually has a choice to make.
#   3. iphone    — Continuity / portable phone mic: offered, never auto-default.
# A picker (dropdown) is shown ONLY when there is a real choice to make: a
# built-in mic AND at least one external (phone/other) input, with no interface
# to settle it. Interface-present or built-in-only → fixed, no asking.
_IPHONE_RE = re.compile(r"iphone|ipad|ipod|android", re.I)   # NB: not "phone" — collides with "microphone"
_BUILTIN_RE = re.compile(r"macbook|built-?in|imac|mac mini|mac studio", re.I)


def _device_kind(name: str) -> str:
    if _IPHONE_RE.search(name):
        return "iphone"
    if _BUILTIN_RE.search(name):
        return "builtin"
    return "interface"


def _classify_devices(rescan: bool = True) -> list:
    """Input devices tagged with their kind (interface | builtin | iphone)."""
    try:
        devs = ENGINE.input_devices(rescan=rescan)
    except RuntimeError:
        return []
    return [{**d, "kind": _device_kind(d["name"])} for d in devs]


def _select_input(devs: list) -> tuple:
    """(selected_device | None, show_picker) per interface > built-in > iPhone.

    An interface is selected outright (no picker). Otherwise the built-in mic is
    the default, and a picker is offered only when an external input also exists.
    With neither interface nor built-in, fall back to the first/default device.
    """
    if not devs:
        return None, False
    interface = next((d for d in devs if d["kind"] == "interface"), None)
    if interface is not None:
        return interface, False                       # interface wins → fixed
    builtin = next((d for d in devs if d["kind"] == "builtin"), None)
    if builtin is not None:
        has_external = any(d["kind"] != "builtin" for d in devs)
        return builtin, has_external                  # default built-in; pick only if a choice
    default = next((d for d in devs if d["default"]), devs[0])
    return default, len(devs) > 1


def _auto_input_device(rescan: bool = True) -> "dict | None":
    """The input The Crate defaults to right now (interface > built-in > iPhone)."""
    sel, _ = _select_input(_classify_devices(rescan))
    return sel


@app.get("/devices/auto")
def device_auto():
    """Auto-selected input + whether to offer a picker (hot-plug aware).

    Returns the chosen device's fields plus `picker` (should the UI show a
    dropdown?) and the full classified `devices` list, so the page builds the
    selector without a second round-trip.
    """
    devs = _classify_devices()
    sel, picker = _select_input(devs)
    if sel is None:
        raise HTTPException(500, "no input devices available")
    return {**sel, "picker": picker, "devices": devs}


@app.get("/devices/{index}/level")
def device_level(index: int, seconds: float = 2.0):
    """Gain-staging probe: capture `seconds` and report RMS/peak/clipping."""
    try:
        return ENGINE.level_test(device_index=index,
                                 seconds=max(0.5, min(seconds, 10.0)))
    except CaptureBusyError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
#  CRATES
# ════════════════════════════════════════════════════════════
@app.get("/genres")
def genres():
    """Genre profiles with their BPM seed range — drives the new-crate form so
    the style choice and the tempo prior the models use can never drift apart."""
    return [
        {"genre": g, "bpm_lo": p["bpm_seed"][0], "bpm_hi": p["bpm_seed"][1]}
        for g, p in config.GENRE_PROFILES.items()
    ]


def _crate_json(c: dict) -> dict:
    return {"crate_id": str(c["crate_id"]), "name": c["name"],
            "genre": c["genre"], "description": c.get("description") or "",
            "bpm_lo": c.get("bpm_seed_lo"), "bpm_hi": c.get("bpm_seed_hi"),
            "n_tracks": c.get("n_tracks"), "n_analyzed": c.get("n_analyzed")}


# Dual routes serve two representations of one URL, so they must NEVER be
# cached without varying on Accept — otherwise the browser reuses the cached
# PAGE for the fetch() that expects JSON ("Unexpected token '<'").
_DUAL_ROUTE_HEADERS = {"Cache-Control": "no-store", "Vary": "Accept"}


@app.get("/crates")
def crates(request: Request, response: Response):
    """Dual route: the browser (Accept: text/html) gets the collection PAGE;
    fetch()/API clients get the JSON listing. Same URL, the user-facing one."""
    if "text/html" in request.headers.get("accept", ""):
        return FileResponse(WEB_DIR / "crates.html", headers=_DUAL_ROUTE_HEADERS)
    response.headers.update(_DUAL_ROUTE_HEADERS)
    try:
        active = None
        try:
            active = str(database.active_crate_id())
        except Exception:
            pass
        return [{**_crate_json(c), "active": str(c["crate_id"]) == active,
                 "is_default": c["name"] == config.DEFAULT_CRATE_NAME}
                for c in database.list_crates()]
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))


@app.post("/crates", status_code=201)
def create_crate(body: NewCrate):
    try:
        cid = database.create_crate(body.name, genre=body.genre,
                                    description=body.description or None)
        return {"crate_id": str(cid), "name": body.name, "genre": body.genre,
                "description": body.description}
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))


@app.get("/crates/{crate_id}")
def crate_detail(crate_id: str):
    try:
        row = database.get_crate(crate_id)
        if row is None:
            raise HTTPException(404, "no such crate")
        total, analyzed, pending = database.count_tracks(crate_id=crate_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(404, "no such crate")
    out = _crate_json(row)
    out.update({"n_tracks": total, "n_analyzed": analyzed, "n_pending": pending,
                "is_default": row["name"] == config.DEFAULT_CRATE_NAME})
    return out


@app.delete("/crates/{crate_id}")
def delete_crate(crate_id: str):
    """Delete a crate; its tracks are re-homed to the default crate (never lost).

    409 when the target is the default crate (the safety net), 404 when it does
    not exist.
    """
    try:
        res = database.delete_crate(crate_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    if not res["deleted"]:
        code = 404 if res["reason"] == "no such crate" else 409
        raise HTTPException(code, res["reason"])
    return {"deleted": True, "rehomed": res["rehomed"]}


class RenameBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@app.patch("/crates/{crate_id}")
def rename_crate_route(crate_id: str, body: RenameBody):
    """Rename a crate. 409 if the name is taken, 400 for the default library (its
    name is the is_default sentinel — renaming it would break default detection),
    404 if it does not exist."""
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "name cannot be empty")
    try:
        crate = database.get_crate(crate_id)
        if crate is None:
            raise HTTPException(404, "no such crate")
        if crate["name"] == config.DEFAULT_CRATE_NAME:
            raise HTTPException(400, "the default library cannot be renamed")
        ok = database.rename_crate(crate_id, name)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(404, "no such crate")
    if not ok:
        raise HTTPException(409, f"a crate named '{name}' already exists")
    return {"crate_id": crate_id, "name": name}


# ════════════════════════════════════════════════════════════
#  RECORDING FLOW
# ════════════════════════════════════════════════════════════
def _sanitize_filename(text: str) -> str:
    """Strip path separators / control chars so 'AC/DC' can't escape the dir."""
    clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", text).strip()
    return clean[:150] or "untitled"


def _save_full_take(audio: np.ndarray, sr: int, artist: str, title: str,
                    formats: list) -> list:
    """Archive the raw take to RECORDINGS_DIR in each requested format.

    FLAC/WAV are written as 16-bit PCM (vinyl's S/N never exceeds 16-bit
    range; halves the size vs float). MP3 is written CBR at libsndfile's best
    quality when the bundled lame supports it. Returns the written paths.
    """
    import soundfile as sf
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{_sanitize_filename(artist)} - {_sanitize_filename(title)} [{stamp}]"
    # Defensive clip: PortAudio float32 should already be in [-1, 1], but a hot
    # interface can exceed it and integer encoders wrap instead of clipping.
    audio = np.clip(audio, -1.0, 1.0)
    written = []
    for fmt in formats:
        path = config.RECORDINGS_DIR / f"{base}.{fmt}"
        if fmt == "flac":
            sf.write(str(path), audio, sr, subtype="PCM_16")
        elif fmt == "wav":
            sf.write(str(path), audio, sr, subtype="PCM_16")
        elif fmt == "mp3":
            try:  # Best quality CBR when this soundfile exposes the knobs.
                with sf.SoundFile(str(path), "w", samplerate=sr, channels=1,
                                  format="MP3", bitrate_mode="CONSTANT",
                                  compression_level=0.0) as f:
                    f.write(audio)
            except TypeError:           # Older soundfile: no bitrate kwargs.
                sf.write(str(path), audio, sr, format="MP3")
        written.append(str(path))
        logger.info("full-take SAVED %s", path.name)
    return written


def _strip_heavy(features: dict) -> dict:
    """Drop the big vectors/curves from a features dict for UI consumption."""
    heavy = {"effnet_embedding", "genre_discogs400", "jamendo_moodtheme_vector",
             "jamendo_instrument", "energy_curve", "complexity_curve",
             "mfcc_mean", "bark_mean", "emotional_vector", "breakdowns"}
    return {k: v for k, v in features.items() if k not in heavy}


def _process_take(job_id: str, audio: np.ndarray, sr: int, meta: dict) -> None:
    """The post-STOP pipeline, run on the 1-worker executor.

    Order is deliberate: the irreplaceable artifact (the full take) is written
    FIRST, so an ingest/analysis failure can never cost the rip itself.
    """
    # Grab the job dict under the lock so the lookup can't race _register_job's
    # eviction; mutating the returned object afterwards is safe (active jobs are
    # never evicted, and we hold the reference regardless). (A2)
    with _JOBS_LOCK:
        job = JOBS[job_id]
    try:
        if meta["keep_full_take"] and meta["formats"]:
            job["state"] = "saving"
            job["recordings"] = _save_full_take(
                audio, sr, meta["artist"], meta["title"], meta["formats"])

        job["state"] = "ingesting"      # mono → 16 kHz → 120 s window → DB row.
        label = f"{meta['artist']} - {meta['title']}"
        excerpt_id, crate_path, db_track_id = crate._ingest(
            audio, sr, label, strategy="best", crate_id=meta["crate_id"])

        job["state"] = "analyzing"      # Full L1–L5 pipeline + embeddings.
        crate._analyze_and_persist(crate_path, db_track_id)

        row = database.get_track(db_track_id)
        feats = row.get("features") or {}
        job["result"] = {
            "excerpt_id": excerpt_id,
            "track_id": str(db_track_id),
            "label": label,
            "bpm": feats.get("bpm"),
            "key": f"{feats.get('key', '?')} {feats.get('scale', '')}".strip(),
            "camelot": feats.get("camelot"),
            "pipeline_level": row.get("pipeline_level"),
            "duration_excerpt": feats.get("duration"),
        }
        job["state"] = "done"
        logger.info("api-job DONE id=%s excerpt=%s", job_id, excerpt_id)
    except Exception as e:
        job["state"] = "failed"
        job["error"] = str(e)
        logger.error("api-job FAILED id=%s reason=%s", job_id, e, exc_info=True)


@app.post("/recordings/start")
def start_recording(body: StartRecording):
    # Validate EVERYTHING before touching the device — a bad crate name or
    # format must fail before the needle drops.
    bad = [f for f in body.formats if f not in EXPORT_FORMATS]
    if bad:
        raise HTTPException(422, f"unknown formats: {bad}")
    if "mp3" in body.formats and not MP3_SUPPORTED:
        raise HTTPException(422, "mp3 export not supported by this libsndfile")
    if body.crate and body.new_crate:
        raise HTTPException(422, "give either 'crate' or 'new_crate', not both")
    try:
        if body.new_crate:
            crate_id = database.create_crate(body.new_crate.name,
                                             genre=body.new_crate.genre)
        else:
            crate_id = database.resolve_crate_id(body.crate)  # None → active.
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))

    # No device in the request → auto-pick (interface > built-in > iPhone).
    device_index = body.device_index
    if device_index is None:
        auto = _auto_input_device()
        device_index = auto["index"] if auto else None
    try:
        status = ENGINE.start_recording(device_index=device_index)
    except CaptureBusyError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

    _CURRENT_TAKE.clear()
    _CURRENT_TAKE.update({
        "crate_id": str(crate_id) if crate_id else None,
        "artist": body.artist.strip(), "title": body.title.strip(),
        "keep_full_take": body.keep_full_take,
        "formats": list(dict.fromkeys(body.formats)),   # Dedup, keep order.
    })
    return status


@app.get("/recordings/current")
def recording_status():
    return ENGINE.status()


@app.post("/recordings/current/stop")
def stop_recording():
    try:
        audio, sr = ENGINE.stop_recording()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    meta = dict(_CURRENT_TAKE)
    _CURRENT_TAKE.clear()

    job_id = uuid.uuid4().hex[:12]
    _register_job(job_id, {"state": "queued", "result": None, "error": None,
                           "recordings": [], "label": f"{meta['artist']} - {meta['title']}",
                           "take_seconds": round(len(audio) / sr, 1)})
    _EXECUTOR.submit(_process_take, job_id, audio, sr, meta)
    return {"job_id": job_id, "take_seconds": JOBS[job_id]["take_seconds"]}


@app.post("/recordings/current/cancel")
def cancel_recording():
    try:
        ENGINE.cancel_recording()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    _CURRENT_TAKE.clear()
    return {"cancelled": True}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


# ════════════════════════════════════════════════════════════
#  FILE IMPORT (hardware door: USB / phone / disk via the browser picker)
# ════════════════════════════════════════════════════════════
AUDIO_EXTS = {".mp3", ".wav", ".flac"}


def _process_import(job_id: str, paths: list, names: list, crate_id: str,
                    tmp_dir: str) -> None:
    """Ingest+analyse each uploaded file sequentially on the 1-worker executor.

    Mirrors crate.add_from_folder semantics: dedup by source filename (re-running
    an import over the same USB stick is a no-op), one failure never aborts the
    batch. The temp dir is always cleaned at the end.
    """
    import shutil
    with _JOBS_LOCK:                       # consistent read vs eviction (A2)
        job = JOBS[job_id]
    imported, moved, skipped, failed = [], [], [], []
    try:
        existing = {row["filename"]: row for row in database.list_tracks()}
        for i, (tmp_path, original) in enumerate(zip(paths, names), 1):
            job["state"] = "importing"
            job["detail"] = f"{i}/{len(paths)} — {original}"
            if original in existing:
                # Known recording: never re-analyse. Just ADD it to the target
                # crate's membership (a track can live in several crates),
                # keeping its analysis/embeddings/fingerprints intact.
                row = existing[original]
                if database.add_tracks_to_crate(crate_id, [str(row["track_id"])]):
                    moved.append(original)
                else:
                    skipped.append(original)         # already a member (or default)
                continue
            try:
                crate.add_from_file(tmp_path, label=original, crate=crate_id)
                existing[original] = {"crate_id": crate_id}
                imported.append(original)
            except Exception as e:
                logger.error("api-import FAILED file=%s reason=%s", original, e,
                             exc_info=True)
                failed.append({"file": original, "error": str(e)})
        job["result"] = {"imported": imported, "moved": moved, "skipped": skipped,
                         "failed": failed, "crate_id": crate_id}
        job["state"] = "done"
        logger.info("api-import DONE id=%s ok=%d moved=%d skip=%d fail=%d",
                    job_id, len(imported), len(moved), len(skipped), len(failed))
    except Exception as e:
        job["state"] = "failed"
        job["error"] = str(e)
        logger.error("api-import FAILED id=%s reason=%s", job_id, e, exc_info=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/import")
async def import_files(crate_id: str = Form(alias="crate"),
                       files: "list[UploadFile]" = File(...)):
    """Import audio files into a crate (the 'from hardware' door of the UI).

    The browser file picker IS the hardware picker on a local app: USB sticks,
    mounted phones, SD cards all appear as volumes. Uploads are staged to a temp
    dir, then a single job ingests+analyses them sequentially (1-worker rule).
    """
    import shutil
    import tempfile
    bad = [f.filename for f in files
           if Path(f.filename or "").suffix.lower() not in AUDIO_EXTS]
    if bad:
        raise HTTPException(422, f"unsupported files (need .mp3/.wav/.flac): {bad}")
    if len(files) > config.IMPORT_MAX_FILES:
        raise HTTPException(413, f"too many files ({len(files)}); the limit is "
                                 f"{config.IMPORT_MAX_FILES} per import.")
    try:
        resolved = database.resolve_crate_id(crate_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    if resolved is None:
        raise HTTPException(404, f"no such crate: {crate_id}")

    tmp_dir = tempfile.mkdtemp(prefix="thecrate-import-")
    max_bytes = int(config.IMPORT_MAX_FILE_MB * 1024 * 1024)
    max_total_bytes = int(config.IMPORT_MAX_TOTAL_MB * 1024 * 1024)
    paths, names = [], []
    total_written = 0                             # cumulative across the whole upload (A4)
    for f in files:
        original = Path(f.filename).name          # Strip any client path part.
        dest = Path(tmp_dir) / _sanitize_filename(original)
        # Stream to disk in 1 MB chunks instead of f.read() the whole file into RAM
        # (M5): a big WAV rip would otherwise be fully buffered. Enforce the per-file
        # AND the aggregate cap as we go so an oversized/runaway upload is rejected
        # (413) without buffering it or filling the disk.
        written = 0
        try:
            with dest.open("wb") as out:
                while chunk := await f.read(1024 * 1024):
                    written += len(chunk)
                    total_written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(
                            413, f"{original} exceeds the "
                                 f"{config.IMPORT_MAX_FILE_MB:g} MB per-file import limit.")
                    if total_written > max_total_bytes:
                        raise HTTPException(
                            413, f"this import exceeds the "
                                 f"{config.IMPORT_MAX_TOTAL_MB:g} MB total upload limit.")
                    out.write(chunk)
        except HTTPException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        paths.append(str(dest))
        names.append(original)

    job_id = uuid.uuid4().hex[:12]
    _register_job(job_id, {"state": "queued", "detail": f"0/{len(paths)}",
                           "result": None, "error": None,
                           "label": f"import × {len(paths)}"})
    _EXECUTOR.submit(_process_import, job_id, paths, names, str(resolved), tmp_dir)
    return {"job_id": job_id, "files": len(paths)}


# ════════════════════════════════════════════════════════════
#  LIVE MODE — recognition worker + session tracklist
# ════════════════════════════════════════════════════════════
# Once listening starts, a background thread consumes the CaptureEngine's
# rolling tail every LIVE_INTERVAL seconds and runs the same recogniser chain
# as the CLI (fingerprint → EffNet → recommend-only, listener.py). Everything
# expensive is EVENT-driven, not per-cycle:
#   * recommendations are computed only on a lock CHANGE (≈16 ms:
#     score_candidates = HNSW top-K + mix_score) or a dashboard-param change;
#   * the crate was analysed at ingest — there is nothing to "pre-analyse";
#   * fingerprint hashing is pure DSP (no TF); the EffNet fallback is lazy and
#     serialised through _EXECUTOR (Essentia/TF is not thread-safe).
# Lock state machine: searching → locked → (LIVE_GRACE_MISSES misses) →
# searching. The grace period stops the banner flickering mid-blend, where the
# outgoing record fades but is still the one playing.
LIVE_INTERVAL_SECONDS = 4.0     # Recognition cadence.
LIVE_WINDOW_SECONDS = 12.0      # Tail slice fed to the chain.
LIVE_MIN_AUDIO_SECONDS = 8.0    # Below this the hash count can't clear MIN_VOTES.
LIVE_GRACE_MISSES = 2           # Lock survives this many missed cycles.
LIVE_EMBED_TIMEOUT = 8.0        # Max wait (s) for the EffNet embed before skipping a pass (M2).

LIVE = {
    "running": False, "status": "idle",      # idle | searching | locked
    "track": None, "confidence": 0.0, "strategy": None, "locked_at": None,
    "misses": 0, "recommendations": [], "tracklist": [],
    "session_id": None, "session_started_at": None,
    "crate_id": None,                         # Recommendations draw from here.
    "pending_save": False,                    # Stopped with tracks, awaiting consent.
    "pending_id": None, "pending_n": 0,       # Fuzzy-match debounce (see _apply_recognition).
}
_LIVE_LOCK = threading.Lock()
_LIVE_THREAD = None


def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample the capture-rate tail to the 16 kHz the recognisers expect."""
    sr = int(round(sr))
    if sr == config.ML_SAMPLE_RATE:
        return audio.astype(np.float32)
    g = math.gcd(config.ML_SAMPLE_RATE, sr)
    return resample_poly(audio, config.ML_SAMPLE_RATE // g, sr // g).astype(np.float32)


def _track_json(res) -> dict:
    """Public shape of an identified track (banner + tracklist entries)."""
    f = res.features
    return {
        "track_id": res.track_id,
        "filename": res.filename,
        "bpm": getattr(f, "bpm", None) if f else None,
        "camelot": getattr(f, "camelot", None) if f else None,
    }


def _live_recommendations(features, exclude_path: str,
                          crate_id: "str | None" = None) -> list:
    """Top-3 next-track picks for the locked track, honouring dashboard params.

    Pure DB + numpy (no TF) — safe to call from the worker thread or a request
    handler (the PATCH /listening/params refresh).
    """
    with _PARAMS_LOCK:
        mode = LISTENING_PARAMS.get("mode", "balanced")
        energy = LISTENING_PARAMS["energy"]
        mods = dict(LISTENING_PARAMS.get("modifiers", {}))
    # Mode is the BASE preset of modifier strengths; the energy direction and the
    # penalizer toggles below layer on top of it (a toggle only ever disables).
    strengths = analyze._ensure_strengths(mode)
    strengths.energy_target = analyze.ENERGY_TARGETS.get(energy, 0.0)
    # Disabled penalizers → strength 0.0 (neutral 1.0 in mix_score, no penalty).
    for m, on in mods.items():
        if not on and hasattr(strengths, m):
            setattr(strengths, m, 0.0)
    # On Spot is a SET-ASIDE pile: per the DJ's policy it must NOT surface in the
    # main recommendation. Exclude those crate_paths alongside the playing track
    # (same scope the scoring runs over: the given crate, else the active one).
    try:
        scope_id = crate_id or database.active_crate_id()
        on_spot = [r["crate_path"] for r in database.list_tracks(crate_id=scope_id)
                   if r.get("on_spot")]
    except Exception:
        on_spot = []
    exclude = ([exclude_path] if exclude_path else []) + on_spot
    scored = analyze.score_candidates(
        features, mode=mode, strengths=strengths,
        exclude_paths=exclude or None,
        crate=crate_id if crate_id else "__active__")
    picks = analyze.sample_by_score(scored, 3, 0.0)
    # score_candidates returns the on-disk crate_path, whose basename is a content
    # hash (e.g. 0c1d…wav) — NOT a name. Resolve each path back to its track row so a
    # recommendation shows the SAME label every other list shows (tracks.filename =
    # "Artist - Title [EP]"), plus a track_id (for the cover thumbnail + preview) and
    # whether a Discogs cover exists. Only 3 picks, recomputed on a lock change.
    by_path = {}
    for path, _f, _s in picks:
        try:
            by_path[path] = database.get_track_by_path(path)
        except Exception:
            by_path[path] = None
    ids = [str(r["track_id"]) for r in by_path.values() if r]
    try:
        dmap = database.track_discogs_map(ids) if ids else {}
    except Exception:
        dmap = {}
    out = []
    for path, f, s in picks:
        row = by_path.get(path) or {}
        tid = str(row["track_id"]) if row.get("track_id") is not None else None
        bpm_d = analyze.bpm_delta(features.bpm, f.bpm)
        key_rel = analyze.key_relationship_label(features.camelot, f.camelot)
        _, dir_label, energy_pct = analyze.energy_direction(features, f)
        out.append({
            "track_id": tid,
            "filename": row.get("filename") or Path(path).name,
            "has_cover": bool((dmap.get(tid) or {}).get("has_cover")),
            "total": round(s["total"], 3),
            "bpm": f.bpm, "bpm_delta": round(bpm_d, 1),
            "camelot": f.camelot, "key_relationship": key_rel,
            "energy_direction": dir_label, "energy_pct": round(energy_pct),
            "tip": analyze.mix_tip(s, key_rel, bpm_d, dir_label),
        })
    return out


def _apply_recognition(res) -> None:
    """Fold one chain pass into the lock state machine.

    Fuzzy (EffNet) matches are DEBOUNCED before they may lock + log: the nearest
    crate neighbour is essentially the top recommendation, so a single transitional
    hit would otherwise land in the tracklist as a track that never played. A new
    EffNet candidate must repeat for listener.STABLE_READS consecutive passes;
    fingerprint is exact (offset-aligned hash votes cannot flicker) so it commits in
    one — fast where it is safe, patient where it is risky. The tracklist also
    de-dupes: a record is never logged twice back-to-back (a re-lock after a brief
    drop), though the same track may reappear later in the set with a gap between.

    Concurrency (A1): only the single _live_worker thread calls this, so _LIVE_LOCK
    here guards READERS (the 700 ms status poll, PATCH /listening/params), not
    re-entrancy. The two DB touches on a new lock — the recommendation recompute and
    the session log — are therefore done OUTSIDE the lock: the banner/tracklist commit
    under a short lock, then the I/O runs unlocked, then the recs are written back
    under a second short lock (guarded so a stop/relock mid-compute can't store stale
    picks). This keeps the lock hold time at in-memory speed regardless of DB latency.
    """
    recompute = None        # set to a snapshot when a new lock needs recs + a log
    with _LIVE_LOCK:
        if res is not None and res.identified:
            current = LIVE["track"]["track_id"] if LIVE["track"] else None
            if res.track_id == current:
                LIVE["pending_id"], LIVE["pending_n"] = None, 0   # still the locked track
            else:
                # New candidate — debounce fuzzy hits before committing/logging.
                needed = 1 if res.strategy == "fingerprint" else listener.STABLE_READS
                if res.track_id == LIVE["pending_id"]:
                    LIVE["pending_n"] += 1
                else:
                    LIVE["pending_id"], LIVE["pending_n"] = res.track_id, 1
                if LIVE["pending_n"] < needed:
                    return                        # not stable yet: hold the lock, log nothing
                LIVE["pending_id"], LIVE["pending_n"] = None, 0
                # Stable → commit the new lock now; defer the DB work (recs + log).
                LIVE["track"] = _track_json(res)
                LIVE["locked_at"] = datetime.now().isoformat(timespec="seconds")
                LIVE["recommendations"] = []      # filled by the unlocked recompute below
                # Dedup: skip when the same record is already the last logged entry
                # (a re-lock); a genuine replay later in the set is not adjacent, so
                # it still logs.
                tl = LIVE["tracklist"]
                should_log = not tl or tl[-1]["track_id"] != res.track_id
                if should_log:
                    tl.append({**LIVE["track"], "position": len(tl) + 1,
                               "identified_at": LIVE["locked_at"],
                               "detected_by": res.strategy})
                recompute = {
                    "features": res.features, "crate_path": res.crate_path,
                    "crate_id": LIVE["crate_id"], "session_id": LIVE["session_id"],
                    "track_id": res.track_id, "strategy": res.strategy, "log": should_log,
                }
            LIVE["status"] = "locked"
            LIVE["confidence"] = round(res.confidence, 3)
            LIVE["strategy"] = res.strategy
            LIVE["misses"] = 0
        elif LIVE["status"] == "locked":
            LIVE["misses"] += 1
            if LIVE["misses"] > LIVE_GRACE_MISSES:
                # Blend/lock lost: banner fades, tracklist stays.
                LIVE["status"] = "searching"
                LIVE["track"] = None
                LIVE["confidence"] = 0.0
                LIVE["strategy"] = None
                LIVE["recommendations"] = []
                LIVE["pending_id"], LIVE["pending_n"] = None, 0   # drop a stale half-vote
        else:
            LIVE["status"] = "searching"

    if recompute is None:
        return
    # ── DB I/O for the new lock, OUTSIDE _LIVE_LOCK (A1) ──
    try:
        recs = _live_recommendations(recompute["features"], recompute["crate_path"],
                                     recompute["crate_id"])
    except Exception as e:
        logger.warning("live recommendations failed: %s", e)
        recs = []
    if recompute["log"] and recompute["session_id"]:
        try:
            database.log_track_played(recompute["session_id"], recompute["track_id"],
                                      detected_by=recompute["strategy"])
        except Exception as e:
            logger.warning("session log failed: %s", e)
    # Store the picks only if the SAME track is still locked — a stop/relock could
    # have moved on while we computed (mirrors the PATCH refresh guard).
    with _LIVE_LOCK:
        if LIVE["track"] and LIVE["track"]["track_id"] == recompute["track_id"]:
            LIVE["recommendations"] = recs


def _make_live_embedder(audio16):
    """Lazy, time-bounded, single-shot EffNet embedder for one recognition pass.

    TF stays serialised through the shared 1-worker executor, but the wait is capped
    at LIVE_EMBED_TIMEOUT so a busy import queue can't stall live recognition (M2):
    on timeout we give up THIS pass (return None) and the chain degrades to
    fingerprint / recommend-only. Memoised so the two chain stages that ask for the
    embedding don't submit the work twice.
    """
    cache = {}

    def embed():
        if "v" in cache:
            return cache["v"]
        fut = _EXECUTOR.submit(analyze.embed_effnet, audio16)
        try:
            cache["v"] = fut.result(timeout=LIVE_EMBED_TIMEOUT)
        except FuturesTimeout:
            fut.cancel()                 # no-op if already running; frees a queued slot
            cache["v"] = None
            logger.debug("live embed skipped this pass: executor busy (>%.0fs)",
                         LIVE_EMBED_TIMEOUT)
        return cache["v"]
    return embed


def _live_worker():
    """Recognition loop: tail → 16 kHz → chain → state machine, every interval."""
    chain = listener.RecogniserChain.default()
    logger.info("live worker started (interval=%.1fs window=%.1fs)",
                LIVE_INTERVAL_SECONDS, LIVE_WINDOW_SECONDS)
    while LIVE["running"]:
        try:
            tail, sr = ENGINE.listening_tail()
            if tail is not None and sr and len(tail) >= LIVE_MIN_AUDIO_SECONDS * sr:
                window = tail[-int(LIVE_WINDOW_SECONDS * sr):]
                audio16 = _to_16k(window, sr)
                # Lazy embedding: TF only runs if fingerprint misses, serialised
                # through the single-worker executor and time-bounded so a busy
                # import queue can't stall live recognition (M2).
                embed = _make_live_embedder(audio16)
                res = chain.recognise(embed, audio16, config.ML_SAMPLE_RATE)
                _apply_recognition(res)
        except Exception as e:
            logger.warning("live worker cycle failed: %s", e)
        time.sleep(LIVE_INTERVAL_SECONDS)
    logger.info("live worker stopped")


# ════════════════════════════════════════════════════════════
#  ACTIVE LISTENING (device + live recognition endpoints)
# ════════════════════════════════════════════════════════════
# ── Live dashboard parameters ─────────────────────────────────────────────────
# Server-side state behind the on-screen control panel at /listen. The catalog
# is the single registration point for a knob: GET /listening/params returns it
# and the page renders the panel FROM it, so a future parameter (mode,
# temperature, crate…) needs no HTML change — one catalog entry + one field on
# ListeningParams. Values survive session stop/start (server lifetime) and the
# recogniser loop, once wired, reads them on every recommendation cycle — so a
# change mid-session applies to the very next recommendation.
# 'energy' maps to analyze.ENERGY_TARGETS ('up' +0.30 / 'stable' 0.0 /
# 'down' −0.30) → ModifierStrengths.energy_target.
# Stage-2 modifiers the DJ can switch off live (energy is excluded — it has its
# own direction control above). Turning one off sets its strength to 0.0, i.e.
# neutral 1.0 in mix_score, so it stops penalising recommendations.
_TOGGLEABLE_MODIFIERS = ("bpm", "harmonic", "transition", "mood", "emotional", "density")

LISTENING_PARAM_CATALOG = [
    {
        # Broad preset (analyze.MODE_CONFIG): sets every modifier's strength at once.
        # Energy direction + penalizer toggles below then layer on top of it.
        "name": "mode", "label": "Mode", "type": "choice",
        "default": "balanced",
        "options": [
            {"value": "safe",     "label": "Safe"},
            {"value": "balanced", "label": "Balanced"},
            {"value": "creative", "label": "Creative"},
        ],
        "hint": "Safe = tight (BPM/key amplified). Creative = the EffNet vibe leads. "
                "Energy and the penalizers below fine-tune this preset.",
    },
    {
        "name": "energy", "label": "Energy", "type": "choice",
        "default": "stable",
        "options": [
            {"value": "up",     "label": "↑ Up"},
            {"value": "stable", "label": "→ Steady"},
            {"value": "down",   "label": "↓ Down"},
        ],
        "hint": "Desired energy direction for the next transition.",
    },
    {
        "name": "modifiers", "label": "Penalizers", "type": "toggles",
        "default": {m: True for m in _TOGGLEABLE_MODIFIERS},
        "options": [
            {"value": "bpm",        "label": "BPM"},
            {"value": "harmonic",   "label": "Harmonic"},
            {"value": "transition", "label": "Transition"},
            {"value": "mood",       "label": "Mood"},
            {"value": "emotional",  "label": "Emotional"},
            {"value": "density",    "label": "Layers"},
        ],
        "hint": "Turn a penalizer off to drop it from the live recommendation.",
    },
]
LISTENING_PARAMS = {p["name"]: p["default"] for p in LISTENING_PARAM_CATALOG}
_PARAMS_LOCK = threading.Lock()


@app.get("/listening/params")
def listening_params():
    """Current dashboard values + the catalog the panel renders itself from."""
    with _PARAMS_LOCK:
        values = dict(LISTENING_PARAMS)
    return {"values": values, "catalog": LISTENING_PARAM_CATALOG}


@app.patch("/listening/params")
def update_listening_params(body: ListeningParams):
    """Partial update from the dashboard. Returns the full confirmed state.

    When a track is locked, the recommendations are recomputed right here so a
    mid-session energy change refreshes the banner without waiting for the next
    lock (pure DB + numpy, ~16 ms — no TF involved).
    """
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(422, "no parameters provided")
    with _PARAMS_LOCK:
        for k, v in updates.items():
            # modifiers is a dict of toggles → MERGE (keep untouched ones),
            # accepting only known modifier names; everything else replaces.
            if k == "modifiers" and isinstance(v, dict):
                cur = dict(LISTENING_PARAMS.get("modifiers", {}))
                for mk, mv in v.items():
                    if mk in _TOGGLEABLE_MODIFIERS:
                        cur[mk] = bool(mv)
                LISTENING_PARAMS["modifiers"] = cur
            else:
                LISTENING_PARAMS[k] = v
        values = dict(LISTENING_PARAMS)
    with _LIVE_LOCK:
        locked = LIVE["status"] == "locked" and LIVE["track"] is not None
        track_id = LIVE["track"]["track_id"] if locked else None
        live_crate = LIVE["crate_id"]
    if locked:
        try:
            row = database.get_track(track_id)
            if row and row.get("features"):
                feats = analyze._hydrate(row["features"])
                recs = _live_recommendations(feats, row["crate_path"], live_crate)
                with _LIVE_LOCK:
                    if LIVE["track"] and LIVE["track"]["track_id"] == track_id:
                        LIVE["recommendations"] = recs
        except Exception as e:
            logger.warning("recommendation refresh on param change failed: %s", e)
    return {"values": values}


@app.post("/listening/start")
def start_listening(body: StartListening):
    global _LIVE_THREAD
    # Idempotent: if a listening session is already running, return its status
    # instead of erroring. This is what made repeated "LISTEN ON" clicks (e.g.
    # after a page reload that left the server listening) fail with
    # "capture engine is busy: listening" — now they just resume.
    if ENGINE.status()["state"] == "listening":
        relaunch = False
        with _LIVE_LOCK:
            if not LIVE["running"]:        # engine listening but worker gone → relaunch
                LIVE["running"] = True
                relaunch = True
        if relaunch:
            _LIVE_THREAD = threading.Thread(target=_live_worker, daemon=True,
                                            name="thecrate-live")
            _LIVE_THREAD.start()
        return ENGINE.status()
    # No device in the request → auto-pick (interface > built-in > iPhone).
    device_index = body.device_index
    if device_index is None:
        auto = _auto_input_device()
        device_index = auto["index"] if auto else None
    try:
        out = ENGINE.start_listening(device_index=device_index)
    except CaptureBusyError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    # Anything unnamed by now was never consented to — clear before starting.
    try:
        database.purge_unnamed_sessions()
    except Exception as e:
        logger.warning("purge_unnamed_sessions failed: %s", e)
    # Resolve the crate the session draws from (falls back to the active one).
    crate_id = None
    try:
        crate_id = (database.resolve_crate_id(body.crate) if body.crate
                    else database.active_crate_id())
    except Exception as e:
        logger.warning("live crate resolution failed: %s", e)
    with _LIVE_LOCK:
        LIVE.update(running=True, status="searching", track=None,
                    confidence=0.0, strategy=None, locked_at=None, misses=0,
                    recommendations=[], tracklist=[], pending_save=False,
                    pending_id=None, pending_n=0,
                    crate_id=str(crate_id) if crate_id else None,
                    session_started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            LIVE["session_id"] = database.create_session(crate_id=crate_id)
        except Exception as e:
            logger.warning("create_session failed (tracklist will be "
                           "memory-only): %s", e)
            LIVE["session_id"] = None
    _LIVE_THREAD = threading.Thread(target=_live_worker, daemon=True,
                                    name="thecrate-live")
    _LIVE_THREAD.start()
    return out


@app.get("/listening/status")
def listening_status():
    s = ENGINE.status()
    with _PARAMS_LOCK:
        s["params"] = dict(LISTENING_PARAMS)   # Dashboard state, for UI sync.
    with _LIVE_LOCK:
        s["live"] = {k: LIVE[k] for k in
                     ("status", "track", "confidence", "strategy", "locked_at",
                      "recommendations", "tracklist", "session_started_at",
                      "crate_id", "pending_save")}
    return s


@app.post("/listening/stop")
def stop_listening():
    with _LIVE_LOCK:
        LIVE["running"] = False               # Worker exits on its next cycle.
    try:
        ENGINE.stop_listening()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    with _LIVE_LOCK:
        has_tracks = len(LIVE["tracklist"]) > 0
        LIVE["pending_save"] = has_tracks and LIVE["session_id"] is not None
        LIVE["status"] = "idle"
        if not LIVE["pending_save"] and LIVE["session_id"]:
            # Empty session — nothing to ask consent for, drop it silently.
            try:
                database.delete_session(LIVE["session_id"])
            except Exception:
                pass
            LIVE["session_id"] = None
        return {"stopped": True, "n_tracks": len(LIVE["tracklist"]),
                "pending_save": LIVE["pending_save"],
                "default_name": "Tracklist " + datetime.now().strftime("%Y-%m-%d %H:%M")}


# ════════════════════════════════════════════════════════════
#  SESSIONS — consented Live Mode tracklists
# ════════════════════════════════════════════════════════════
class SaveSession(BaseModel):
    name: str = Field(default="", max_length=120)


@app.post("/sessions/save")
def save_pending_session(body: SaveSession):
    """Persist the just-stopped session under a UNIQUE user-chosen name."""
    with _LIVE_LOCK:
        sid = LIVE["session_id"] if LIVE["pending_save"] else None
    if sid is None:
        raise HTTPException(409, "no session awaiting save")
    name = body.name.strip() or "Tracklist " + datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        ok = database.save_session(sid, name)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    if not ok:
        raise HTTPException(409, f"a session named '{name}' already exists")
    # Build the session centroid (mean of its tracks' EffNet vectors) for future
    # agentic similarity. Pure SQL + numpy, no TF — but best-effort: a failure
    # here must never undo a save the user already consented to.
    try:
        analyze.persist_session_embedding(sid)
    except Exception as e:
        logger.warning("session embedding failed for %s: %s", sid, e)
    with _LIVE_LOCK:
        LIVE.update(pending_save=False, session_id=None, tracklist=[])
    return {"saved": True, "name": name, "session_id": sid}


@app.get("/sessions/{session_id}/similar")
def similar_sessions(session_id: str, n: int = 5):
    """Sessions whose sonic centroid resembles this one (for future agents).

    Reads this session's stored centroid and runs an ANN search over the other
    session centroids — 'sets that feel like this set'. 404 when the session has
    no centroid yet (no identified tracks with embeddings).
    """
    try:
        mv = analyze._model_version("effnet")
        vecs = database.session_track_vectors(session_id, mv)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    if not vecs:
        raise HTTPException(404, "session has no centroid")
    import numpy as _np
    centroid = _np.mean(_np.array(vecs), axis=0)
    norm = _np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    rows = database.find_similar_sessions(centroid.tolist(), n=n,
                                          exclude_session_id=session_id)
    return [{
        "session_id": str(r["session_id"]), "name": r["name"],
        "n_tracks": r["n_tracks"], "crate_name": r.get("crate_name"),
        "similarity": round(1.0 - float(r["cosine_distance"]), 4),
        "tracklist": r.get("tracklist"),
    } for r in rows]


@app.post("/sessions/discard")
def discard_pending_session():
    """The user declined — delete the session and its play log."""
    with _LIVE_LOCK:
        sid = LIVE["session_id"]
        LIVE.update(pending_save=False, session_id=None, tracklist=[])
    if sid:
        try:
            database.delete_session(sid)
        except Exception as e:
            logger.warning("discard session failed: %s", e)
    return {"discarded": True}


@app.get("/sessions")
def sessions(request: Request, response: Response, crate: "str | None" = None):
    """Dual route like /crates: browsers get the page, fetch() gets JSON.

    ?crate=X scopes the JSON to that crate's sessions (the per-crate SESSIONS
    view); omitted = every saved session.
    """
    if "text/html" in request.headers.get("accept", ""):
        return FileResponse(WEB_DIR / "sessions.html", headers=_DUAL_ROUTE_HEADERS)
    response.headers.update(_DUAL_ROUTE_HEADERS)
    try:
        crate_id = database.resolve_crate_id(crate) if crate else None
        rows = database.list_sessions(crate_id=crate_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except KeyError:
        raise HTTPException(404, f"no such crate: {crate}")
    return [{
        "session_id": str(r["session_id"]), "name": r["name"],
        "started_at": str(r["started_at"]),
        "ended_at": str(r["ended_at"]) if r["ended_at"] else None,
        "crate_name": r.get("crate_name"), "n_tracks": r["n_tracks"],
    } for r in rows]


@app.get("/sessions/{session_id}")
def session_detail(session_id: str):
    try:
        s = database.get_session(session_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except KeyError:
        raise HTTPException(404, "no such session")
    except Exception:
        raise HTTPException(404, "no such session")
    tl = s.get("tracklist", [])
    # Stamp the harmonic key on each played track (the snapshot stores only
    # filename/time/method) so the session view shows the set's key flow.
    cams = database.track_camelots([t.get("track_id") for t in tl])
    return {
        "session_id": str(s["session_id"]), "name": s.get("name"),
        "started_at": str(s["started_at"]),
        "ended_at": str(s["ended_at"]) if s.get("ended_at") else None,
        "tracklist": [{
            "position": t["position"],
            "track_id": str(t["track_id"]) if t.get("track_id") else None,
            "filename": t.get("filename"),
            "played_at": str(t["played_at"]),
            "detected_by": t.get("detected_by"),
            "camelot": cams.get(str(t["track_id"])) if t.get("track_id") else None,
            "rating": t.get("rating"),
        } for t in tl],
    }


class TrackRating(BaseModel):
    # null clears the rating; otherwise good/bad. Same optional+pattern idiom as
    # ListeningParams.energy — None bypasses the pattern, any value must match.
    rating: "str | None" = Field(default=None, pattern="^(good|bad)$")


@app.patch("/sessions/{session_id}/track/{position}")
def rate_session_track(session_id: str, position: int, body: TrackRating):
    """Tag a played track in a session as a good/bad mix (green/red), or clear it.

    Per-track only (position = set order). The rating lives on the normalised
    session_tracks row, so it persists for saved sets and survives the track later
    being deleted from the crate."""
    try:
        ok = database.set_session_track_rating(session_id, position, body.rating)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    if not ok:
        raise HTTPException(404, "no such session track")
    return {"session_id": session_id, "position": position, "rating": body.rating}


@app.patch("/sessions/{session_id}")
def rename_session_route(session_id: str, body: RenameBody):
    """Rename a saved session. 409 if the name is taken, 404 if it does not exist."""
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "name cannot be empty")
    try:
        ok = database.rename_session(session_id, name)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except KeyError:
        raise HTTPException(404, "no such session")
    if not ok:
        raise HTTPException(409, f"a session named '{name}' already exists")
    return {"session_id": session_id, "name": name}


# ════════════════════════════════════════════════════════════
#  LIBRARY (read-only)
# ════════════════════════════════════════════════════════════
@app.get("/tracks")
def tracks(crate: "str | None" = None):
    """Track listing. crate accepts a name, an id, or '__active__'.

    Includes the stage-2 scoring descriptors the Live Mode reference list
    sorts on (energy, aggressiveness, danceability, density, brightness).
    """
    try:
        if crate == "__active__":
            crate_id = database.active_crate_id()
        else:
            crate_id = database.resolve_crate_id(crate) if crate else None
        rows = database.list_tracks(crate_id=crate_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    dmap = database.track_discogs_map([str(r["track_id"]) for r in rows])
    out = []
    for r in rows:
        feats = r.get("features") or {}
        curve = feats.get("energy_curve") or []
        d = dmap.get(str(r["track_id"])) or {}
        out.append({
            "track_id": str(r["track_id"]),
            "excerpt_id": Path(r["crate_path"]).stem,
            "filename": r["filename"],
            "analyzed": r["analyzed_at"] is not None,
            "pipeline_level": r.get("pipeline_level"),
            "bpm": feats.get("bpm"),
            "camelot": feats.get("camelot"),
            "on_spot": bool(r.get("on_spot")),
            "energy": round(float(np.mean(curve)), 4) if curve else None,
            "mood_aggressive": feats.get("mood_aggressive"),
            "danceability": feats.get("danceability_nn"),
            "rhythmic_density": feats.get("onset_rate"),   # onsets/s — varies within 4/4 techno
            "density": feats.get("spectral_complexity"),
            "brightness": feats.get("timbre_bright"),
            # Discogs enrichment (Phase 3b): cover thumbnail + label/year/styles
            "has_cover": bool(d.get("has_cover")),
            "label": d.get("label"),
            "year": d.get("year"),
            "styles": d.get("styles") or [],
        })
    return out


@app.get("/affinity")
def affinity(track_id: str, crate: "str | None" = None):
    """Cosine similarity ('Afinidad') of every crate track vs `track_id`.

    One SQL pass over the EffNet embeddings; the Live Mode list fetches this
    once per lock change and sorts client-side.
    """
    try:
        if crate == "__active__":
            crate_id = database.active_crate_id()
        else:
            crate_id = database.resolve_crate_id(crate) if crate else None
        rows = database.crate_affinity(track_id, crate_id=crate_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(404, f"affinity unavailable: {e}")
    return {r["track_id"]: round(r["affinity"], 4) for r in rows}


@app.get("/tracks/{track_id}/audio")
def track_audio(track_id: str):
    """Stream the crate excerpt for in-list preview (Quick Look-style).

    Serves the canonical 16 kHz WAV from crate/ — FileResponse honours Range
    requests, so the <audio> element can seek without downloading everything.
    """
    try:
        row = database.get_track(track_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except Exception:
        row = None
    if row is None:
        raise HTTPException(404, "no such track")
    p = _crate_file(row)
    if not p.exists():
        raise HTTPException(404, "audio file missing from crate/")
    return FileResponse(p, media_type="audio/wav")


# Waveform peaks for the preview player, computed once per track and cached.
# The excerpt is a small 16 kHz mono WAV already on disk, so peak extraction is
# a cheap numpy bucket-max — but caching avoids redoing it on every replay.
# OrderedDict as a bounded LRU: re-inserting on hit moves a key to the end, and
# the oldest is evicted past the cap so the cache can't grow without limit.
from collections import OrderedDict
WAVEFORM_BINS = 240
WAVEFORM_CACHE_MAX = 256
_WAVEFORM_CACHE: "OrderedDict" = OrderedDict()


@app.get("/tracks/{track_id}/waveform")
def track_waveform(track_id: str, bins: int = WAVEFORM_BINS):
    """Downsampled |amplitude| peaks (0–1) for the deejay-style waveform."""
    bins = max(40, min(bins, 600))
    cache_key = (track_id, bins)
    if cache_key in _WAVEFORM_CACHE:
        _WAVEFORM_CACHE.move_to_end(cache_key)        # LRU touch.
        return {"peaks": _WAVEFORM_CACHE[cache_key]}
    try:
        row = database.get_track(track_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except Exception:
        row = None
    if row is None:
        raise HTTPException(404, "no such track")
    p = _crate_file(row)
    if not p.exists():
        raise HTTPException(404, "audio file missing from crate/")
    import soundfile as sf
    audio, _ = sf.read(str(p), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    if n == 0:
        return {"peaks": [0.0] * bins}
    # Bucket-max of |amplitude|, normalised to the track's own peak.
    edges = np.linspace(0, n, bins + 1, dtype=int)
    peaks = [float(np.max(np.abs(audio[edges[i]:edges[i + 1]])))
             if edges[i + 1] > edges[i] else 0.0 for i in range(bins)]
    top = max(peaks) or 1.0
    peaks = [round(v / top, 3) for v in peaks]
    _WAVEFORM_CACHE[cache_key] = peaks
    while len(_WAVEFORM_CACHE) > WAVEFORM_CACHE_MAX:
        _WAVEFORM_CACHE.popitem(last=False)           # Evict the oldest.
    return {"peaks": peaks}


def _refresh_cover_bg(track_id: str) -> None:
    """Fire-and-forget Discogs cover re-search after an inline edit, OFF the request
    thread (Discogs is slow and may be rate-limited / unauthorised). Best-effort:
    enrich.refresh_cover does nothing when Discogs is unconfigured or a cover already
    exists, and any failure is swallowed so the edit response is never affected."""
    def _work():
        try:
            import enrich
            enrich.refresh_cover(track_id)
        except Exception as e:
            logger.warning("cover refresh failed for %s: %s", track_id, e)
    _COVER_EXECUTOR.submit(_work)          # bounded pool, not a fresh thread (A5)


class TrackUpdate(BaseModel):
    """Partial track update: ON SPOT flag, the display filename (the inline-edited
    'Artist - Title [EP].ext' label), and/or the Discogs label (the Label column)."""
    on_spot: "bool | None" = None
    filename: "str | None" = Field(default=None, max_length=300)
    label: "str | None" = Field(default=None, max_length=200)


@app.patch("/tracks/{track_id}")
def update_track(track_id: str, body: TrackUpdate):
    """Update a track's ON SPOT flag, display filename and/or Discogs label. Every
    list reads tracks.filename + track_discogs.label, so an edit propagates to all
    of them. After a filename/label edit, if the track has no cover art we kick off
    a best-effort Discogs cover re-search with the updated values (see _refresh_cover_bg)."""
    if body.on_spot is None and body.filename is None and body.label is None:
        raise HTTPException(422, "no fields provided")
    updated = False
    try:
        if body.filename is not None:
            name = body.filename.strip()
            if not name:
                raise HTTPException(422, "filename cannot be empty")
            updated = database.rename_track(track_id, name)
            if updated:
                # Re-derive artist entities from the edited label and refresh
                # the affected centroids (Phase 0 stays consistent on rename).
                try:
                    for aid in database.relink_track_artists(track_id, name):
                        analyze.persist_artist_embedding(aid)
                except Exception as e:
                    logger.warning("artist relink failed for %s: %s", track_id, e)
        if body.label is not None:
            # Set ONLY the label (preserves cover/year/styles), then re-link the
            # label entity so its sonic centroid (similar_labels) stays correct.
            database.set_track_label(track_id, body.label)
            try:
                for lid in database.relink_track_label(track_id, body.label.strip()):
                    analyze.persist_label_embedding(lid)
            except Exception as e:
                logger.warning("label relink failed for %s: %s", track_id, e)
            updated = True
        if body.on_spot is not None:
            updated = database.set_track_on_spot(track_id, body.on_spot) or updated
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except HTTPException:
        raise
    except Exception:
        updated = False                       # Malformed UUID etc. → 404 below.
    if not updated:
        raise HTTPException(404, "no such track")
    # A metadata edit can change the Discogs match — if the track still has no cover,
    # try to fetch one in the background with the corrected artist/title/label.
    if body.filename is not None or body.label is not None:
        _refresh_cover_bg(track_id)
    return {"track_id": track_id, "on_spot": body.on_spot,
            "filename": body.filename, "label": body.label}


class TrackBatch(BaseModel):
    """A set of track_ids targeted by a bulk crate-management action."""
    track_ids: list = Field(default_factory=list)


class TrackMove(TrackBatch):
    crate: str = Field(min_length=1)          # Destination crate name or id.


@app.post("/tracks/delete")
def delete_tracks(body: TrackBatch):
    """Permanently delete tracks: DB row (CASCADE wipes embeddings/fingerprints)
    + the crate WAV. Best-effort per id; returns how many actually went."""
    if not body.track_ids:
        raise HTTPException(422, "no track_ids provided")
    deleted, failed = [], []
    for tid in body.track_ids:
        try:
            row = database.get_track(tid)
            if row is None:
                failed.append(tid); continue
            database.delete_track(tid)        # CASCADE removes the embedding rows.
            p = _crate_file(row)
            if p.exists():
                p.unlink()
            deleted.append(tid)
        except Exception as e:
            logger.warning("delete track %s failed: %s", tid, e)
            failed.append(tid)
    return {"deleted": len(deleted), "failed": len(failed)}


@app.post("/tracks/remove-from-crate")
def remove_from_crate(body: TrackMove):
    """Drop tracks' membership in ONE user crate WITHOUT deleting them: the
    records stay in the master library (Vinyl Collection) and any other crate
    they belong to. `crate` = the crate to remove them from."""
    if not body.track_ids:
        raise HTTPException(422, "no track_ids provided")
    try:
        cid = database.resolve_crate_id(body.crate)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except KeyError:
        raise HTTPException(404, f"no such crate: {body.crate}")
    removed, failed = [], []
    for tid in body.track_ids:
        try:
            if database.remove_track_from_crate(tid, str(cid)):
                removed.append(tid)
            else:
                failed.append(tid)
        except Exception as e:
            logger.warning("remove-from-crate %s failed: %s", tid, e)
            failed.append(tid)
    return {"removed": len(removed), "failed": len(failed)}


@app.post("/tracks/add-to-crate")
def add_to_crate(body: TrackMove):
    """ADD tracks to a user crate (membership; a track can live in several crates
    at once). Analysis, embeddings and fingerprints stay keyed to the same
    track_id — nothing is recomputed."""
    if not body.track_ids:
        raise HTTPException(422, "no track_ids provided")
    try:
        dest = database.resolve_crate_id(body.crate)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except KeyError:
        raise HTTPException(404, f"no such crate: {body.crate}")
    if dest is None:
        raise HTTPException(404, f"no such crate: {body.crate}")
    added = database.add_tracks_to_crate(str(dest), [str(t) for t in body.track_ids])
    return {"added": added, "crate_id": str(dest)}


@app.get("/tracks/{track_id}")
def track_detail(track_id: str):
    try:
        row = database.get_track(track_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    except Exception:
        row = None
    if row is None:
        raise HTTPException(404, "no such track")
    return {
        "track_id": str(row["track_id"]),
        "excerpt_id": Path(row["crate_path"]).stem,
        "filename": row["filename"],
        "analyzed_at": str(row["analyzed_at"]) if row["analyzed_at"] else None,
        "pipeline_level": row.get("pipeline_level"),
        "features": _strip_heavy(row.get("features") or {}),
    }


class ArtistRename(BaseModel):
    """Global artist rename: the entity + every track label that credits it."""
    old: str = Field(min_length=1, max_length=200)
    new: str = Field(min_length=1, max_length=200)


@app.post("/artists/rename")
def artist_rename(body: ArtistRename):
    """Rename an artist GLOBALLY — the artists row AND every track's display label —
    so it shows up across every list and in similar_artists, not just on the clicked
    track. Merges into an existing artist of the same name. Refreshes the survivor's
    audio centroid afterwards (track set may have grown via a merge)."""
    try:
        res = database.rename_artist(body.old, body.new)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    if res is None:
        raise HTTPException(404, f"unknown artist '{body.old}' (or empty new name)")
    try:
        analyze.persist_artist_embedding(res["artist_id"])
    except Exception as e:
        logger.warning("centroid refresh after artist rename failed for %s: %s",
                       res["artist_id"], e)
    return res


# ════════════════════════════════════════════════════════════
#  AI ASSISTANT (Phase 1 — chat over the existing engine)
# ════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    context: "dict | None" = None        # {track_id?, crate_id?, session_id?}
    model: "str | None" = None


class ModelRequest(BaseModel):
    model: str = Field(min_length=1)


class ProfilePatch(BaseModel):
    # The user's physical location (for events/what's-on) and the Live-Mode
    # agent re-check toggle. Both optional — send only what changes.
    location: "str | None" = Field(default=None, max_length=120)
    recheck: "bool | None" = None


def _sse(payload: dict) -> str:
    import json as _json
    return f"data: {_json.dumps(payload)}\n\n"


@app.get("/assistant/status")
def assistant_status():
    """Ollama up?, active model, RAM, and the model catalog with suitability."""
    up = ollama_client.is_up()
    installed = ollama_client.installed_models() if up else set()
    catalog = assistant_models.catalog(installed)
    active = assistant_agent.active_model()
    # Flag a "thinking" build as the active chat model: it reasons on every turn, so it is
    # markedly slower for live chat. We don't override the user's choice — just surface it
    # so the UI can warn (the model picker already notes it at selection time).
    meta = next((m for m in catalog
                 if active in (m["tag"], f"{m['tag']}:latest")), None)
    active_instant = bool(meta and meta.get("instant"))
    return {
        "ollama_up": up,
        "ollama_installed": True,        # binary present in this environment
        "active_model": active,
        "active_instant": active_instant,
        "active_warning": (None if active_instant else
                           "This model 'thinks' on every turn — slower for live chat; "
                           "qwen3:4b-instruct is the fast default."),
        "ram_gb": round(assistant_models.system_ram_gb(), 1),
        "models": catalog,
        "embed_model": assistant_models.EMBED_MODEL,
        "embed_ready": ("nomic-embed-text" in installed
                        or "nomic-embed-text:latest" in installed),
        "kb": database.kb_stats(),
    }


@app.post("/assistant/model")
def assistant_set_model(body: ModelRequest):
    """Switch the active chat model (must be pulled)."""
    if not ollama_client.has_model(body.model):
        raise HTTPException(409, f"model '{body.model}' is not installed")
    assistant_agent.set_active_model(body.model)
    return {"active_model": body.model}


@app.post("/assistant/pull")
def assistant_pull(body: ModelRequest):
    """Download a model via Ollama, streaming progress (SSE)."""
    def gen():
        for msg in ollama_client.pull_stream(body.model):
            yield _sse({"status": msg})
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/assistant/profile")
def assistant_profile():
    """The assistant's remembered profile: the user's physical location (for
    events/what's-on) and the Live-Mode re-check toggle."""
    from assistant import profile
    return profile.get()


@app.patch("/assistant/profile")
def assistant_profile_set(body: ProfilePatch):
    """Update the location and/or the re-check toggle (send only what changes).
    Location is also settable by the agent in chat (set_user_location)."""
    from assistant import profile
    if body.location is not None:
        profile.set_location(body.location)
    if body.recheck is not None:
        profile.set_recheck(body.recheck)
    return profile.get()


@app.post("/assistant/confirm-recs")
def assistant_confirm_recs():
    """Agent re-check of the CURRENT live recommendations (the ASSISTANT toggle).

    Only runs when re-check is ON and a track is locked — otherwise returns a
    disabled/idle note. One-shot local LLM pass; out of process, never touches
    the Essentia executor."""
    from assistant import profile, recheck
    if not profile.get_recheck():
        return {"enabled": False, "note": "Re-check is off — the agent answers your chat only."}
    with _LIVE_LOCK:
        locked = LIVE["status"] == "locked" and LIVE["track"] is not None
        track = dict(LIVE["track"]) if locked else None
        recs = list(LIVE["recommendations"]) if locked else []
    if not locked:
        return {"enabled": True, "note": "Waiting for a locked track to review."}
    out = recheck.confirm(track, recs)
    return {"enabled": True, "track": track.get("filename"), **out}


@app.post("/chat")
async def chat(body: ChatRequest):
    """Stream the assistant's answer (SSE: {delta}|{done}|{error})."""
    async def gen():
        try:
            async for delta in assistant_agent.run_stream(
                    body.message, body.context, body.model):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except RuntimeError as e:
            m = str(e)
            if m == "ollama-down":
                hint = ("The local LLM (Ollama) isn't running. Start it with "
                        "`brew services start ollama` (or `ollama serve`).")
            elif m.startswith("model-missing:"):
                hint = (f"Model '{m.split(':', 1)[1]}' isn't installed yet — "
                        f"pick and download one from the model menu.")
            else:
                hint = f"Assistant error: {m}"
            yield _sse({"error": hint})
        except Exception as e:
            logger.warning("chat failed: %s", e)
            yield _sse({"error": f"Assistant error: {e}"})
    return StreamingResponse(gen(), media_type="text/event-stream")


# ── knowledge base (Phase 2 — RAG) ────────────────────────────────────────────
class KbText(BaseModel):
    # title optional: when blank, the classifier's auto-title is used.
    title: str = Field(default="", max_length=300)
    text: str = Field(min_length=40, max_length=200_000)
    source_url: "str | None" = None
    category: "str | None" = None        # optional manual override; else auto


class KbDocEdit(BaseModel):
    title: "str | None" = None
    category: "str | None" = None
    tags: "list[str] | None" = None


def _kb_error(e: Exception):
    """Map an ingest exception onto a meaningful HTTP error for the UI."""
    from assistant import kb as kb_mod
    if isinstance(e, kb_mod.GateRejected):
        raise HTTPException(422, {"rejected": True, "reason": e.reason,
                                  "confidence": round(e.confidence, 2)})
    if isinstance(e, kb_mod.KbFull):
        # 409 Conflict — the KB is at capacity; the UI shows a banner asking the
        # user to delete documents before adding new ones.
        raise HTTPException(409, {"full": True, "chunks": e.chunks, "cap": e.cap})
    if isinstance(e, RuntimeError):
        m = str(e)
        if m == "ollama-down":
            raise HTTPException(503, "The local LLM (Ollama) isn't running. "
                                     "Start it with `ollama serve`.")
        if m == "embed-model-missing":
            raise HTTPException(503, "The embedding model 'nomic-embed-text' isn't "
                                     "installed. Pull it from the assistant menu.")
        if m.startswith("model-missing:"):
            raise HTTPException(503, f"Model '{m.split(':', 1)[1]}' isn't installed.")
    if isinstance(e, ValueError):
        raise HTTPException(409, str(e))
    logger.warning("kb ingest failed: %s", e)
    raise HTTPException(500, f"Ingestion failed: {e}")


@app.get("/kb/docs")
def kb_list(category: str = None):
    """Every ingested document (newest first), optionally filtered by category."""
    stats = database.kb_stats()
    # Capacity for the UI gauge + full-banner (rudimentary RAG guard).
    stats["max_chunks"] = config.KB_MAX_CHUNKS
    stats["max_file_mb"] = config.KB_MAX_FILE_MB
    stats["full"] = stats.get("chunks", 0) >= config.KB_MAX_CHUNKS
    return {"documents": database.list_kb_documents(category=category or None),
            "stats": stats, "categories": database.kb_categories()}


@app.get("/kb/categories")
def kb_categories():
    """Distinct categories with counts — drives the filter and the suggestions."""
    from assistant import kb as kb_mod
    return {"categories": database.kb_categories(),
            "suggested": kb_mod.SUGGESTED_CATEGORIES}


@app.post("/kb/text")
async def kb_ingest_text(body: KbText):
    """Ingest pasted text. The single gate+classify pass runs first; non-music
    text is refused (422) and nothing is stored. Category is auto-assigned (or
    forced via body.category) and editable afterwards."""
    from assistant import kb as kb_mod
    try:
        return await kb_mod.ingest_text(body.text, title=body.title,
                                        source_type="paste", source_url=body.source_url,
                                        category=body.category)
    except Exception as e:
        _kb_error(e)


@app.post("/kb/file")
async def kb_ingest_file(file: UploadFile = File(...),
                         title: str = Form(default="")):
    """Ingest a file the user picked on their machine (.txt/.md/.pdf/.docx).
    Same music-relevance gate; non-music files are refused (422)."""
    from assistant import kb as kb_mod
    data = await file.read()
    if not data:
        raise HTTPException(400, "the uploaded file is empty")
    # Reject an oversized file up front, before extraction/embedding spend.
    max_bytes = int(config.KB_MAX_FILE_MB * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(413, f"file is too large ({len(data) // (1024*1024)} MB); "
                                 f"the limit is {config.KB_MAX_FILE_MB:g} MB per file.")
    try:
        return await kb_mod.ingest_file(data, file.filename,
                                        title=title or file.filename)
    except Exception as e:
        _kb_error(e)


@app.patch("/kb/docs/{doc_id}")
def kb_edit(doc_id: str, body: KbDocEdit):
    """Edit a document's curation fields (title / category / tags) — the
    'editable' half of auto+editable. Category is normalised to kebab-case."""
    from assistant import kb as kb_mod
    cat = kb_mod._norm_category(body.category) if body.category is not None else None
    tags = kb_mod._norm_tags(body.tags) if body.tags is not None else None
    if not database.update_kb_document(doc_id, title=body.title, category=cat, tags=tags):
        raise HTTPException(404, "document not found")
    return {"updated": doc_id, "category": cat, "tags": tags}


@app.delete("/kb/docs/{doc_id}")
def kb_delete(doc_id: str):
    """Remove a document and its chunks from the knowledge base."""
    if not database.delete_kb_document(doc_id):
        raise HTTPException(404, "document not found")
    return {"deleted": doc_id}


# ── reference web sources (assistant web scouting) ────────────────────────────
class WebSourceReq(BaseModel):
    """A website the assistant may search live + a MANDATORY topic (what it is for)."""
    url: str = Field(min_length=4, max_length=500)
    topic: str = Field(min_length=2, max_length=200)
    note: "str | None" = Field(default=None, max_length=500)


@app.get("/kb/sources")
def kb_list_sources():
    """Registered reference websites (newest first) for the Knowledge UI."""
    try:
        return {"sources": database.list_web_sources()}
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))


@app.post("/kb/source")
def kb_add_source(body: WebSourceReq):
    """Register a reference website (URL + topic). Best-effort: fetch the page now
    and embed a snapshot into the web cache so it is searchable immediately, even
    before the first live query (the seed degrades gracefully if the fetch fails)."""
    url, topic = body.url.strip(), body.topic.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    try:
        sid = database.insert_web_source(url, topic, (body.note or "").strip() or None)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    seeded = 0
    try:                                          # best-effort snapshot → web cache
        from assistant import web_sources, embed_text, kb as kb_mod
        text = web_sources.fetch_page_text(url)
        if text and len(text) >= 40:
            chunks = kb_mod.chunk_text(text)[:8]   # cap the seed so one page cannot flood
            vecs = embed_text.embed_documents(chunks)
            seeded = database.insert_web_cache([
                {"source_id": sid, "query": None, "title": topic, "url": url,
                 "text": c, "embedding": v} for c, v in zip(chunks, vecs)])
            database.evict_web_cache(config.WEB_CACHE_MAX_CHUNKS)
    except Exception as e:
        logger.warning("web source seed failed for %s: %s", url, e)
    return {"source_id": sid, "url": url, "topic": topic, "seeded_chunks": seeded}


@app.delete("/kb/sources/{source_id}")
def kb_delete_source(source_id: str):
    """Remove a reference website and its cached rows (ON DELETE CASCADE)."""
    try:
        ok = database.delete_web_source(source_id)
    except database.DBUnavailableError as e:
        raise HTTPException(503, str(e))
    if not ok:
        raise HTTPException(404, "no such source")
    return {"deleted": source_id}


# ── Discogs enrichment (Phase 3b) ─────────────────────────────────────────────
class DiscogsConfirm(BaseModel):
    track_id: str
    release_id: int


@app.get("/discogs/status")
def discogs_status():
    """Is Discogs configured, and how many tracks are matched/doubtful/etc."""
    counts = {}
    for st in ("matched", "confirmed", "doubtful", "unmatched"):
        counts[st] = len(database.discogs_queue(st))
    return {"configured": discogs.is_configured(),
            "pending": len(database.tracks_pending_discogs()), "counts": counts}


@app.post("/discogs/enrich")
def discogs_enrich(limit: int = None, track_id: str = None):
    """Run enrichment, streaming per-track progress (SSE). `track_id` enriches a
    single track; otherwise the whole pending queue (optionally capped)."""
    if not discogs.is_configured():
        raise HTTPException(503, "Discogs is not configured. Add a valid "
                                 "DISCOGS_ACCESS_TOKEN (personal access token) to .env.")

    def gen():
        if track_id:
            r = enrich.enrich_track(track_id)
            yield _sse(r)
            yield _sse({"done": True, "counts": {r.get("status", "error"): 1}})
        else:
            for ev in enrich.iter_enrich(limit):
                yield _sse(ev)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/discogs/queue")
def discogs_queue_route(status: str = "doubtful"):
    """The review queue — tracks (default doubtful) with their candidate releases."""
    rows = database.discogs_queue(status)
    for r in rows:
        r["track_id"] = str(r["track_id"])
    return {"status": status, "tracks": rows}


@app.post("/discogs/confirm")
def discogs_confirm(body: DiscogsConfirm):
    """Apply a release the user picked for a doubtful track (status=confirmed)."""
    if not discogs.is_configured():
        raise HTTPException(503, "Discogs is not configured.")
    try:
        return enrich.confirm_match(body.track_id, body.release_id)
    except Exception as e:
        logger.warning("discogs confirm failed: %s", e)
        raise HTTPException(500, f"Confirm failed: {e}")


@app.post("/discogs/skip/{track_id}")
def discogs_skip(track_id: str):
    """Dismiss a doubtful track so it leaves the queue (status=skipped)."""
    if not database.set_track_discogs_status(track_id, "skipped"):
        raise HTTPException(404, "no enrichment row for this track")
    return {"skipped": track_id}


@app.get("/tracks/{track_id}/cover", include_in_schema=False)
def track_cover(track_id: str):
    """Serve a track's downloaded Discogs cover (404 when not matched yet)."""
    p = config.COVERS_DIR / f"{track_id}.jpg"
    if not p.exists():
        raise HTTPException(404, "no cover")
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


# ════════════════════════════════════════════════════════════
#  PAGES (sober static HTML — design pass comes later)
# ════════════════════════════════════════════════════════════
@app.get("/chat.js", include_in_schema=False)
def chat_js():
    """Shared right-rail chat panel — included on every page."""
    return FileResponse(WEB_DIR / "chat.js", media_type="text/javascript")


@app.get("/theme.css", include_in_schema=False)
def theme_css():
    """Shared design system — single source of truth for every page."""
    return FileResponse(WEB_DIR / "theme.css", media_type="text/css")


@app.get("/util.js", include_in_schema=False)
def util_js():
    """Shared front-end helpers (j/post/esc/parseLabel) — one source of truth."""
    return FileResponse(WEB_DIR / "util.js", media_type="text/javascript")


@app.get("/ui.js", include_in_schema=False)
def ui_js():
    """Unified UI feedback layer (toast/confirmDialog) + bfcache state guard."""
    return FileResponse(WEB_DIR / "ui.js", media_type="text/javascript")


@app.get("/player.js", include_in_schema=False)
def player_js():
    """Shared preview player — the SAME mini-player on every track list."""
    return FileResponse(WEB_DIR / "player.js", media_type="text/javascript")


@app.get("/keycol.js", include_in_schema=False)
def keycol_js():
    """Shared KEY (Camelot) column — header dropdown to hide/sort, every list."""
    return FileResponse(WEB_DIR / "keycol.js", media_type="text/javascript")


@app.get("/", include_in_schema=False)
def page_index():
    """Home is the crate collection — pick a box first, everything else is
    scoped inside it (Live Mode, Sessions)."""
    return RedirectResponse("/crates", status_code=307)


@app.get("/record", include_in_schema=False)
def page_record():
    return FileResponse(WEB_DIR / "record.html")


@app.get("/listen", include_in_schema=False)
def page_listen():
    return FileResponse(WEB_DIR / "listen.html")


@app.get("/knowledge", include_in_schema=False)
def page_knowledge():
    """The knowledge manager — ingest documents from your machine into the RAG."""
    return FileResponse(WEB_DIR / "knowledge.html")


@app.get("/discogs", include_in_schema=False)
def page_discogs():
    """The Discogs enrichment console — run matching + review the doubtful queue."""
    return FileResponse(WEB_DIR / "discogs.html")


@app.get("/session", include_in_schema=False)
def page_session():
    """One session's detail (?id=<uuid>): its tracklist, tracks previewable."""
    return FileResponse(WEB_DIR / "session.html")


@app.get("/library", include_in_schema=False)
def page_library_redirect():
    """Legacy URL — the collection moved to /crates."""
    return RedirectResponse("/crates", status_code=308)


@app.get("/crate/new", include_in_schema=False)
def page_crate_new():
    """New-crate form (name, description, style → BPM seed)."""
    return FileResponse(WEB_DIR / "crate-new.html")


@app.get("/crate", include_in_schema=False)
def page_crate():
    """Inside one crate (?id=<uuid>): tracks, record-into, import-into."""
    return FileResponse(WEB_DIR / "crate.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    # reload=False on purpose: a reload re-imports Essentia/TF (slow) and would
    # orphan a live capture stream.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
