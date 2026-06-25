"""
The Crate — Shared Configuration Constants
----------------------------------------
The single import point for paths and sample rates used across the project
(crate.py, listener.py, and any future tooling). Keeping them here means the
canonical audio store and the standardisation rates are defined ONCE, so a
change (e.g. moving the crate, bumping the excerpt rate) never has to be hunted
down in three modules.

analyze.py imports SAMPLE_RATE / ML_SAMPLE_RATE from here (it historically had
its own copies; unified 2026-06). All modules import from config.py.
"""
import os
from pathlib import Path

# Project root = the directory this file lives in. Resolved so the crate path is
# absolute regardless of the caller's working directory (the CLI may be invoked
# from anywhere).
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Sample rates ──────────────────────────────────────────────────────────────
# Essentia's full-resolution rate (CD/vinyl rips). analyze.py decodes at this
# rate for its classic DSP chain. Kept here so listener.py can reuse it.
SAMPLE_RATE = 44100
# The rate EVERY crate excerpt is stored at: it matches the EffNet/TempoCNN model
# input rate, so the same 16 kHz file feeds both DSP analysis and ML embedding
# without a second decode/resample at recognition time.
ML_SAMPLE_RATE = 16000

# ── Paths ───────────────────────────────────────────────────────────────────
# The canonical audio store. Every track in the system lives here as one
# standardised <excerpt_id>.wav excerpt. This is the single source of truth the
# live listener fingerprints/embeds against.
CRATE_DIR = PROJECT_ROOT / "crate"

# Where FULL vinyl takes are archived (FLAC/WAV/MP3) by the recording API.
# Unlike CRATE_DIR (small excerpts, must stay local/fast), this is bulk archival
# storage — point THECRATE_RECORDINGS_DIR at an external drive if the internal
# disk is tight. The recommendation engine never reads from here; deleting the
# folder loses the archived rips but breaks nothing.
RECORDINGS_DIR = Path(os.environ.get("THECRATE_RECORDINGS_DIR",
                                     PROJECT_ROOT / "recordings"))

# Discogs cover art downloaded by the enrichment pipeline (Phase 3b). One image
# per matched track, shown as a thumbnail in the track lists. Pure cache: safe to
# delete, re-downloads on the next enrich. Lives beside recordings.
COVERS_DIR = Path(os.environ.get("THECRATE_COVERS_DIR",
                                 PROJECT_ROOT / "covers"))

# Knowledge base (RAG) capacity guard. The KB is deliberately rudimentary local
# context for the assistant — user-supplied docs (an Ableton manual, label notes,
# important chats). Chunks are the unit that grows: one 768-d vector each, scanned
# on every search. Cap the total so a runaway ingest cannot bloat memory or slow
# every query; when full, ingestion is refused and the UI asks the user to delete
# knowledge first. KB_MAX_FILE_MB rejects an oversized single file up front, before
# extraction/embedding. Both tunable via env.
KB_MAX_CHUNKS = int(os.environ.get("THECRATE_KB_MAX_CHUNKS", "4000"))
KB_MAX_FILE_MB = float(os.environ.get("THECRATE_KB_MAX_FILE_MB", "8"))

# Hardware-import door (POST /import) guards. Audio files are large, so the per-file
# cap is generous (a long WAV rip can be ~100 MB) but bounded so one upload can't OOM
# the box; the count cap bounds a runaway batch. Both env-overridable.
IMPORT_MAX_FILE_MB = float(os.environ.get("THECRATE_IMPORT_MAX_FILE_MB", "200"))
IMPORT_MAX_FILES = int(os.environ.get("THECRATE_IMPORT_MAX_FILES", "200"))
# Aggregate cap across a single upload (per-file × count alone could stage ~40 GB
# in the temp dir before failing). Deliberately generous — well above any realistic
# crate import — so it only catches a pathological/runaway batch, never normal use.
IMPORT_MAX_TOTAL_MB = float(os.environ.get("THECRATE_IMPORT_MAX_TOTAL_MB", "8192"))

# Cap on the in-memory API job registry (api.JOBS): each finished recording/import is a
# tiny status dict the browser polls. Past this many, the oldest FINISHED jobs are
# evicted so a long-lived server can't leak memory.
JOBS_MAX = int(os.environ.get("THECRATE_JOBS_MAX", "200"))

# Live "scene" lookup for the assistant (events, artists, releases). Resident
# Advisor has no official API, so we use RA's own GraphQL endpoint as the primary
# source; if its schema changes and a query fails we fall back to a plain web
# search scoped to the domains in WEB_SOURCES. WEB_SOURCES is USER-EXPANDABLE
# (comma-separated env) — the agent points at es.ra.co by default, add more as you
# like, e.g. "es.ra.co,ra.co,residentadvisor.net".
RA_GRAPHQL = os.environ.get("THECRATE_RA_GRAPHQL", "https://ra.co/graphql")
WEB_SOURCES = [s.strip() for s in os.environ.get(
    "THECRATE_WEB_SOURCES", "es.ra.co").split(",") if s.strip()]

# Record shops the assistant checks for VINYL IN STOCK right now. Stock detection
# is shop-specific (each shop's HTML and stock signal differ), so a shop only works
# if assistant/vinyl_stores.py has an adapter for it; this list just selects which
# adapters are queried. Hardwax (the reference) is the one fully parsed; deejay.de /
# decks.de / hhv.de are adapter stubs pending bespoke handling (see that module), so
# only "hardwax" is on by default. Env-overridable: THECRATE_VINYL_STORES.
VINYL_STORES = [s.strip() for s in os.environ.get(
    "THECRATE_VINYL_STORES", "hardwax").split(",") if s.strip()]

# Cap on the assistant's web-search cache (web_cache table: 768-D embeddings of live
# results + page snapshots from registered reference sources). Bounds memory/scan
# cost like KB_MAX_CHUNKS does for the curated KB; oldest rows are evicted past it.
WEB_CACHE_MAX_CHUNKS = int(os.environ.get("THECRATE_WEB_CACHE_MAX", "2000"))

# HTTP timeouts for the assistant's live web lookups. Deliberately tight: a dead/slow
# source should fail FAST, because the chat turn blocks on it — a 15-20s stall is the
# difference between "the assistant is thinking" and "the assistant is broken". Normal
# responses land in 1-3s; raise via env on a slow connection. WEB covers Resident Advisor
# GraphQL + the DuckDuckGo fallback + page fetches (web_sources.py); VINYL covers the
# record-shop adapters (vinyl_stores.py).
WEB_HTTP_TIMEOUT = float(os.environ.get("THECRATE_WEB_TIMEOUT", "8"))
VINYL_HTTP_TIMEOUT = float(os.environ.get("THECRATE_VINYL_TIMEOUT", "8"))

# Assistant chat-model tuning for the local LLM (Ollama).
# - NUM_CTX: the context window. The system prompt + 9 tool schemas already run ~3.7k
#   tokens, and Ollama's own default is only 4096, so the first tool result would push
#   the prompt out of the window and get it truncated/re-processed. We give headroom by
#   exporting OLLAMA_CONTEXT_LENGTH when we launch Ollama (assistant/ollama_client.py):
#   the OpenAI endpoint the agent uses ignores a per-request num_ctx, so the SERVER's
#   context length is the only lever. Applies only to a Crate-started Ollama — an
#   already-running server keeps its own window until restarted with this env.
# - KEEP_ALIVE: how long Ollama keeps the model resident after a turn (sent per request
#   via the OpenAI endpoint's extra_body). The default (5m) unloads it between questions,
#   so the next chat pays a cold reload — hold it warm.
# - TEMPERATURE: low on purpose — this assistant routes to deterministic tools and
#   returns structured data, so it wants precision over creativity (as the re-check does).
OLLAMA_NUM_CTX = int(os.environ.get("THECRATE_OLLAMA_NUM_CTX", "8192"))
OLLAMA_KEEP_ALIVE = os.environ.get("THECRATE_OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_TEMPERATURE = float(os.environ.get("THECRATE_OLLAMA_TEMPERATURE", "0.4"))

# ── Crates: genre profiles + active-crate persistence ─────────────────────────
# A crate is a LOGICAL collection (one row in the `crates` table); audio files
# all live in CRATE_DIR regardless of crate — membership is tracks.crate_id.
#
# Each genre profile supplies the BPM seed range used as the tempo prior while
# a crate is young (fewer analysed tracks than CRATE_PRIOR_MIN_TRACKS). Once a
# crate has enough tracks, its own median±MAD statistics take over — the prior
# becomes learned from the user's actual records instead of a genre stereotype.
GENRE_PROFILES = {
    "techno":      {"bpm_seed": (120.0, 152.0)},   # Mulero/Kastil territory.
    "hard_techno": {"bpm_seed": (140.0, 170.0)},
    "house":       {"bpm_seed": (118.0, 130.0)},
    "deep_house":  {"bpm_seed": (110.0, 126.0)},
    "electro":     {"bpm_seed": (118.0, 140.0)},
    "ambient":     {"bpm_seed": (60.0, 110.0)},
    # "other" is effectively prior-less: wide enough that folding never fires.
    "other":       {"bpm_seed": (50.0, 200.0)},
}
DEFAULT_GENRE = "techno"
# Name of the crate auto-created on first run (existing tracks are backfilled
# into it so the single-crate world migrates without any manual step).
DEFAULT_CRATE_NAME = "Vinyl Collection"
# Analysed tracks a crate needs before its own BPM stats replace the genre seed.
CRATE_PRIOR_MIN_TRACKS = 10

# ── Scoring / retrieval knobs ─────────────────────────────────────────────────
# Tonal-certainty gate for the harmonic modifier's LEGACY binary mode: either
# track below this → the Camelot penalty is skipped entirely (1.0 neutral).
# The continuous mode below replaces the hard gate but the constant is kept so
# the legacy behaviour stays reproducible.
KEY_STRENGTH_THRESHOLD = 0.4
# Feature toggle: weight the Camelot penalty by the JOINT tonal confidence
# (key_strength_1 × key_strength_2) instead of the binary gate above, so
# borderline-tonal pairs get a proportional partial penalty rather than
# all-or-nothing. Env kill-switch (no redeploy): THECRATE_HARMONIC_CONTINUOUS=0.
HARMONIC_CONTINUOUS_CONFIDENCE = os.environ.get(
    "THECRATE_HARMONIC_CONTINUOUS", "1") != "0"
# Floor for the density (spectral-complexity continuity) modifier. Every other
# modifier has one (bpm 0.5, energy 0.75, transition 0.85, emotional 0.6) so no
# single descriptor can veto a match the EffNet base endorses; without this the
# density modifier could reach 0.0 and annihilate the score outright.
DENSITY_MOD_FLOOR = 0.5
# Stage-1 breadth of the two-stage retrieval (HNSW top-K, then mix_score).
# score_candidates() doubles this window when the scored results can't prove
# the true winner is inside it, so K only sets the FIRST fetch size.
RETRIEVAL_K = 150

# ── Tonal (key) detection ─────────────────────────────────────────────────────
# Profile KeyExtractor correlates the HPCP against. The live path VOTES across
# EDM-tuned profiles (Faraldo/MTG): KEY_PROFILE is the primary the vote starts
# from (and the single-profile baseline the benchmark + the dormant
# _detect_key_simple use). Allowed: edma|edmm|bgate|braw|temperley|… — 'edma' is
# tuned for electronic dance music; Essentia's no-arg default is the also-EDM
# 'bgate', NOT a classical profile, so this is a lateral EDM-to-EDM choice to
# settle by benchmark, not a classical→EDM rescue.
KEY_PROFILE = 'edma'
# Extra profiles consulted ONLY when the primary's key_strength is below
# KEY_STRENGTH_THRESHOLD (the reading is shaky) — see analyze._detect_key_robust.
KEY_VOTE_FALLBACKS = ('edmm', 'bgate')
# Tonal band (Hz) for the DORMANT _isolate_tonal_band kick/air filter: techno
# fundamentals live ~80 Hz–2 kHz; below is sub-kick, above is percussive air.
KEY_TONAL_BAND = (80.0, 2000.0)
# File persisting the ACTIVE crate name between CLI invocations
# (`python crate.py use <name>` writes it; every import command reads it).
ACTIVE_CRATE_FILE = PROJECT_ROOT / ".cache" / "active_crate"

# Small JSON profile for the assistant — the user's stated physical location (so
# event/scene recommendations know "where are we") and assistant toggles (live
# re-check). Persisted like the active crate so it survives restarts.
PROFILE_FILE = PROJECT_ROOT / ".cache" / "assistant_profile.json"

