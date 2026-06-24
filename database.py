"""
The Crate — Data Persistence Layer
--------------------------------
The SINGLE source of truth for all data storage across The Crate.

Every other module (analysis.py, crate.py, listener.py) imports from here.
Nothing else touches Postgres directly — if a query isn't expressed as a
function in this file, it doesn't happen.

Storage stack:
    PostgreSQL 16 + pgvector   — relational tracks/sessions + vector similarity.
    psycopg2 (raw, no ORM)     — transparent SQL, minimal dependency surface.
    ThreadedConnectionPool     — 1..5 pooled connections, safe across threads.

Fail-loud philosophy:
    If Postgres is unreachable at import time we DO NOT silently degrade to a
    JSON file. We log CRITICAL, flip DB_AVAILABLE = False, and every public
    function raises DBUnavailableError. The user is meant to see this and start
    Docker (`docker compose up -d`) rather than discover days later that nothing
    was being persisted.

Typical bootstrap:
    docker compose up -d            # starts Postgres+pgvector (see docker-compose.yml)
    cp .env.example .env            # fill in credentials
    python -c "import database"     # auto-runs db_init() on first import
"""
import functools
import logging
import os
import re
import threading
import time

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════
# Dedicated child of the project's "thecrate" logger so DB chatter can be
# filtered independently (e.g. set thecrate.db to DEBUG without flooding the
# audio pipeline). We deliberately NEVER log embedding vectors — 1280+ floats
# per row would bury every other line — only their dimensionality.
logger = logging.getLogger("thecrate.db")

load_dotenv()  # Pull DB credentials from .env into os.environ. Never hardcode.


# ════════════════════════════════════════════════════════════
#  CONFIG  (all from environment — see .env.example)
# ════════════════════════════════════════════════════════════
# Mirrors the docker-compose.yml service env. DB_HOST/DB_PORT are client-side
# knobs (where to reach the container); the POSTGRES_* trio matches what the
# container initialises itself with.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "vinylid"),
    "user": os.getenv("POSTGRES_USER", "vinylid"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    """Read a positive-int env knob, clamped to [lo, hi]; bad/absent → default."""
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


# Pool sizing is a THECRATE_* knob (H4). FastAPI runs sync DB-bound routes in a
# threadpool; a pool smaller than that concurrency raises PoolError ("connection
# pool exhausted") and 500s requests. The default 20 covers api.py's threadpool
# cap (POOL_MAX - 4, set in its lifespan) plus headroom for the listener /
# analysis / MCP background threads that also borrow connections.
POOL_MIN = _int_env("THECRATE_DB_POOL_MIN", 1, 1, 100)
POOL_MAX = _int_env("THECRATE_DB_POOL_MAX", 20, POOL_MIN, 500)

# HNSW build/search tuning. Defaults are pgvector's; exposed here so they live
# next to the index DDL that consumes them rather than as magic numbers.
HNSW_M = 16              # Max links per node — higher = better recall, more RAM.
HNSW_EF_CONSTRUCTION = 64  # Candidate list size at build time — higher = better graph.


# ════════════════════════════════════════════════════════════
#  MODULE STATE  (connection pool + availability flag)
# ════════════════════════════════════════════════════════════
# These are mutated exactly once, by _connect() during the import-time init
# guard. After that they are read-only for the life of the process.
DB_AVAILABLE = False          # Flipped True only after a successful connect + db_init.
_pool = None                  # psycopg2.pool.ThreadedConnectionPool | None.
_init_lock = threading.Lock()  # Guards init/reconnect against concurrent callers.
_initialised = False           # True once the first connect has been attempted.
_last_connect_attempt = 0.0    # monotonic ts of the last attempt — throttles retries.
try:                           # Seconds to back off before retrying a down DB (H3).
    RECONNECT_THROTTLE_SEC = max(0.0, float(os.getenv("THECRATE_DB_RECONNECT_THROTTLE", "5")))
except (TypeError, ValueError):
    RECONNECT_THROTTLE_SEC = 5.0


class DBUnavailableError(RuntimeError):
    """Raised by every public operation when the database is unreachable.

    Carries a fixed, actionable message: the fix is almost always "start
    Docker". We raise rather than return None so callers can't accidentally
    treat a dead DB as an empty result set.
    """

    def __init__(self, detail: str = ""):
        msg = ("The Crate database is unavailable. Start it with "
               "`docker compose up -d` and check your .env credentials.")
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


# ════════════════════════════════════════════════════════════
#  CONNECTION MANAGEMENT
# ════════════════════════════════════════════════════════════
def _connect() -> None:
    """Build (or rebuild) the connection pool and run schema init if needed.

    Thread-safe via `_init_lock`. Returns immediately when a pool already exists.
    Otherwise it attempts a connect — but at most once every
    RECONNECT_THROTTLE_SEC, so a persistently-down Postgres is not hammered on
    every operation. That throttled retry is what lets the app recover on its own
    once Postgres comes back (H3) instead of needing a manual restart. On failure
    we log CRITICAL and leave DB_AVAILABLE = False so the module fails loud.
    """
    global _pool, DB_AVAILABLE, _initialised, _last_connect_attempt
    with _init_lock:                      # Serialise concurrent (re)connects.
        if _pool is not None:             # Already connected — nothing to do.
            return
        now = time.monotonic()
        if _initialised and (now - _last_connect_attempt) < RECONNECT_THROTTLE_SEC:
            return                        # Backed off after a recent failed attempt.
        _initialised = True
        _last_connect_attempt = now
        try:
            # ThreadedConnectionPool is the right pool here: The Crate's listener and
            # analysis pipeline can issue queries from different threads, and this
            # variant guards getconn/putconn with a lock (SimpleConnectionPool does not).
            _pool = psycopg2.pool.ThreadedConnectionPool(
                POOL_MIN, POOL_MAX, **DB_CONFIG)
            DB_AVAILABLE = True           # Provisionally up — db_init() confirms below.
            db_init()                     # Create extension/tables/index if missing.
            _log_versions()               # Confirm reachability + pgvector presence.
            logger.info("Connection pool ready (min=%d max=%d, db=%s@%s:%s)",
                        POOL_MIN, POOL_MAX, DB_CONFIG["dbname"],
                        DB_CONFIG["host"], DB_CONFIG["port"])
        except Exception as e:            # Wide net: any connect/DDL error → fail loud.
            DB_AVAILABLE = False
            _pool = None
            logger.critical(
                "Postgres unreachable — The Crate will not persist anything. "
                "Start Docker (`docker compose up -d`). Cause: %s", e)


def _conn_alive(conn) -> bool:
    """Liveness probe before a pooled connection is handed to a caller.

    A Postgres restart leaves every pooled socket broken but still sitting in the
    free list, and psycopg2 would keep dealing them out. We roll back first (which
    also clears any 'current transaction is aborted' state from a prior error) and
    run `SELECT 1`; only a connection that answers is considered usable.
    """
    if conn.closed:
        return False
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except psycopg2.Error:
        return False


def _get_live_conn():
    """Borrow a *live* connection, discarding any the server has killed (H3).

    Loops at most one full pool over: each dead connection is dropped with
    close=True (so the pool replaces it with a fresh one on the next getconn)
    until a live one is found. A hard connect failure becomes the standard
    DBUnavailableError so callers see one consistent, actionable error.
    """
    for _ in range(POOL_MAX + 1):
        try:
            conn = _pool.getconn()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            raise DBUnavailableError(str(e))
        if _conn_alive(conn):
            return conn
        _pool.putconn(conn, close=True)   # Dead → discard; next getconn makes a fresh one.
    raise DBUnavailableError("no live connection available")


class _transaction:
    """Context manager: borrow a connection, yield a cursor, commit/return.

    This is the single point of truth for transaction handling — every public
    operation funnels through it, so commit-on-success, rollback-on-error, and
    return-to-pool are written exactly ONCE rather than copy-pasted per query.

    Usage:
        with _transaction() as cur:
            cur.execute(...)            # Auto-commits on clean exit.

    Args:
        dict_rows: when True (default) rows come back as RealDictRow (dict-like)
            so callers get column-name access; set False for plain tuples.
    """

    def __init__(self, dict_rows: bool = True):
        self._dict_rows = dict_rows
        self._conn = None
        self._cur = None

    def __enter__(self):
        # Guard here too: callers may use the CM directly without a decorator.
        if not DB_AVAILABLE or _pool is None:
            _connect()                    # Lazy, throttled reconnect if Postgres came back.
        if not DB_AVAILABLE or _pool is None:
            raise DBUnavailableError()
        self._conn = _get_live_conn()     # Proven-alive connection (survives a PG restart).
        factory = psycopg2.extras.RealDictCursor if self._dict_rows else None
        self._cur = self._conn.cursor(cursor_factory=factory)
        return self._cur

    def __exit__(self, exc_type, exc, tb):
        broken = False
        try:
            if exc_type is None:
                self._conn.commit()       # Clean exit → persist.
            else:
                # A connection-level error means the socket is dead: don't roll back
                # on it (that just raises again) and flag it for disposal below.
                broken = (self._conn.closed != 0 or
                          issubclass(exc_type, (psycopg2.OperationalError,
                                                psycopg2.InterfaceError)))
                if not broken:
                    self._conn.rollback() # Recoverable error → undo partial work.
                logger.error("Transaction rolled back: %s", exc, exc_info=True)
        except psycopg2.Error:
            broken = True                 # commit/rollback itself failed → dead connection.
        finally:
            if self._cur is not None:
                try:
                    self._cur.close()
                except psycopg2.Error:
                    pass
            # Discard a dead connection instead of poisoning the pool with it (H3) —
            # otherwise the next borrower inherits the broken socket after a Postgres
            # restart and the app never recovers without a manual restart.
            _pool.putconn(self._conn, close=broken)
        return False                      # Never suppress exceptions.


def requires_db(fn):
    """Decorator: short-circuit a public op with DBUnavailableError when down.

    Keeps the availability check out of every function body (dedup) and
    guarantees a consistent, actionable failure everywhere.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not DB_AVAILABLE:
            raise DBUnavailableError()
        return fn(*args, **kwargs)
    return wrapper


# ════════════════════════════════════════════════════════════
#  SCHEMA INITIALISATION  (DDL)
# ════════════════════════════════════════════════════════════
# One big idempotent DDL string. Every statement is CREATE ... IF NOT EXISTS so
# db_init() is safe to run on every import. Schema notes inline.
_SCHEMA_SQL = """
-- pgvector must exist before any vector(...) column can be declared.
CREATE EXTENSION IF NOT EXISTS vector;

-- gen_random_uuid() lives in pgcrypto on older images; it is built-in from PG13+
-- but enabling the extension keeps the DEFAULT working on any 13/14/15/16 image.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── crates: a LOGICAL record box. Audio files all live in ./crate/ on disk; ──
-- membership is tracks.crate_id. name is the user-facing handle (UNIQUE so the
-- CLI can resolve by name); genre selects the BPM seed range used as a tempo
-- prior until the crate has enough analysed tracks to use its own statistics.
CREATE TABLE IF NOT EXISTS crates (
    crate_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    genre           TEXT NOT NULL DEFAULT 'techno',
    description     TEXT,
    bpm_seed_lo     FLOAT,
    bpm_seed_hi     FLOAT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
-- Migration for databases created before the description field existed.
ALTER TABLE crates ADD COLUMN IF NOT EXISTS description TEXT;

-- ── tracks: the canonical row per 120s crate excerpt. ──
-- crate_path is UNIQUE so re-scanning the crate is naturally idempotent
-- (insert_track / get_track_by_path key off it). analyzed_at IS NULL is our
-- "pending analysis" marker; features is the full TrackFeatures blob as JSONB
-- so the audio schema can evolve without a migration here.
CREATE TABLE IF NOT EXISTS tracks (
    track_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    crate_path      TEXT NOT NULL UNIQUE,
    filename        TEXT NOT NULL,
    duration_sec    FLOAT,
    added_at        TIMESTAMPTZ DEFAULT now(),
    analyzed_at     TIMESTAMPTZ,
    pipeline_level  SMALLINT DEFAULT 1,
    features        JSONB
);

-- ── embeddings_effnet: the 1280-D Discogs/EffNet vector — the workhorse. ──
-- Separate table (not a column on tracks) so we can keep MULTIPLE model
-- versions per track side-by-side and re-rank when the model is upgraded.
-- PK (track_id, model_version) enforces one vector per (track, model) and gives
-- upsert its ON CONFLICT target. ON DELETE CASCADE: deleting a track wipes it.
CREATE TABLE IF NOT EXISTS embeddings_effnet (
    track_id        UUID REFERENCES tracks ON DELETE CASCADE,
    embedding       vector(1280),
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (track_id, model_version)
);

-- ── embeddings_multimodal: 1408-D Google/ImageBind space — FUTURE USE. ──
-- Stubbed today (upsert logs a WARNING, no search implemented) but the table
-- exists so the door is visibly open. modalities records which inputs fed it.
CREATE TABLE IF NOT EXISTS embeddings_multimodal (
    track_id        UUID REFERENCES tracks ON DELETE CASCADE,
    embedding       vector(1408),
    modalities      TEXT[],
    model_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (track_id, model_name)
);

-- ── embeddings_text: 768-D semantic vector over track metadata. ──
-- source_text keeps the exact string that was embedded for debuggability.
CREATE TABLE IF NOT EXISTS embeddings_text (
    track_id        UUID REFERENCES tracks ON DELETE CASCADE,
    embedding       vector(768),
    source_text     TEXT,
    model_name      TEXT NOT NULL,
    PRIMARY KEY (track_id, model_name)
);

-- ── embeddings_genre_discogs400: 400-D Discogs genre style vector. ──
-- Each element is the softmax probability for one of 400 Discogs styles.
-- ANN search here finds genre-cohesive matches across the catalogue.
CREATE TABLE IF NOT EXISTS embeddings_genre_discogs400 (
    track_id        UUID REFERENCES tracks ON DELETE CASCADE,
    embedding       vector(400),
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (track_id, model_version)
);

-- ── embeddings_jamendo_moodtheme: 56-D MTG-Jamendo mood+theme vector. ──
-- Full sigmoid multi-label output — stores the whole distribution so mood-space
-- ANN search uses all 56 components, not just the 6 named display scalars.
CREATE TABLE IF NOT EXISTS embeddings_jamendo_moodtheme (
    track_id        UUID REFERENCES tracks ON DELETE CASCADE,
    embedding       vector(56),
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (track_id, model_version)
);

-- ── embeddings_jamendo_instrument: 40-D instrument presence vector. ──
-- Multi-label sigmoid (not softmax): each element is P(instrument present).
CREATE TABLE IF NOT EXISTS embeddings_jamendo_instrument (
    track_id        UUID REFERENCES tracks ON DELETE CASCADE,
    embedding       vector(40),
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (track_id, model_version)
);

-- ── fingerprints: Shazam-style landmark hashes (see fingerprint.py). ──
-- One row per landmark: `hash` packs (f_anchor, f_target, dt); t_offset is the
-- anchor's frame index inside the excerpt. No PK on purpose — the same hash can
-- legitimately recur within one track; lookups only ever filter on `hash`.
-- ON DELETE CASCADE: removing a track wipes its landmarks with it.
CREATE TABLE IF NOT EXISTS fingerprints (
    hash            BIGINT NOT NULL,
    track_id        UUID NOT NULL REFERENCES tracks ON DELETE CASCADE,
    t_offset        INTEGER NOT NULL
);
-- The matcher's whole cost is one index probe per query hash.
CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints (hash);
-- replace_fingerprints() deletes by track before re-inserting.
CREATE INDEX IF NOT EXISTS idx_fingerprints_track ON fingerprints (track_id);

-- ── mix_sessions: one DJ set. tracklist is a denormalised JSONB log written ──
-- at close_session(); session_tracks below is the normalised, queryable form.
CREATE TABLE IF NOT EXISTS mix_sessions (
    session_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at      TIMESTAMPTZ DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    tracklist       JSONB
);

-- ── session_tracks: normalised play log, one row per track played. ──
-- track_id is ON DELETE SET NULL (not CASCADE): purging a track from the crate
-- must NOT rewrite history — the set still happened, we just lose the link.
-- position orders the set; detected_by records how the track was identified.
CREATE TABLE IF NOT EXISTS session_tracks (
    session_id      UUID REFERENCES mix_sessions ON DELETE CASCADE,
    track_id        UUID REFERENCES tracks ON DELETE SET NULL,
    played_at       TIMESTAMPTZ NOT NULL,
    position        SMALLINT,
    detected_by     TEXT
);
-- Per-track user rating of the mix INTO this slot: 'good' | 'bad' | NULL (unrated).
-- Lives on the normalised row so get_session() rebuilds it live for saved sets too.
ALTER TABLE session_tracks ADD COLUMN IF NOT EXISTS rating TEXT;

-- Speeds up the "pending analysis" sweep the pipeline runs constantly.
CREATE INDEX IF NOT EXISTS idx_tracks_pending
    ON tracks (analyzed_at) WHERE analyzed_at IS NULL;

-- ── Multi-crate migration (idempotent on every init, safe on fresh installs). ──
-- ON DELETE SET NULL: deleting a crate must NOT delete its records — they become
-- orphans that ensure_default_crate() re-homes on the next init.
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS crate_id UUID REFERENCES crates ON DELETE SET NULL;
ALTER TABLE mix_sessions
    ADD COLUMN IF NOT EXISTS crate_id UUID REFERENCES crates ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_tracks_crate ON tracks (crate_id);

-- The DJ's "pull list": records pulled to the listening deck to dig mixes
-- around them (the /crate page's ON SPOT list; future recommendation focus).
-- A flag, not a table — the list is by definition a subset of the crate.
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS on_spot BOOLEAN NOT NULL DEFAULT FALSE;

-- The default crate (the master "all records" library) carries NO genre — it
-- holds every track regardless of style. genre is therefore nullable.
ALTER TABLE crates ALTER COLUMN genre DROP NOT NULL;

-- ── crate_tracks: MANY-TO-MANY user-crate membership. ──
-- A track can live in several user crates at once. The default crate (master
-- library) is implicit — it holds EVERY track and has NO rows here. tracks.
-- crate_id stays only as the ingest-time BPM-prior hint; this table is the sole
-- authority for "which user crates is this track in". CASCADE both ways: drop a
-- crate or a track and its memberships vanish.
CREATE TABLE IF NOT EXISTS crate_tracks (
    crate_id    UUID NOT NULL REFERENCES crates ON DELETE CASCADE,
    track_id    UUID NOT NULL REFERENCES tracks ON DELETE CASCADE,
    added_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (crate_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_crate_tracks_track ON crate_tracks (track_id);

-- ── artists + track_artists: structured artist entities (Phase 0). ──
-- Artist is otherwise only parsed from tracks.filename ("Artist - Title").
-- track_artists is many-to-many (a track can credit several artists, e.g.
-- "Obscure Shape, SHDW"). aliases/country/discogs_id fill later via enrichment.
CREATE TABLE IF NOT EXISTS artists (
    artist_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    aliases     TEXT[],
    country     TEXT,
    discogs_id  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS track_artists (
    track_id    UUID NOT NULL REFERENCES tracks ON DELETE CASCADE,
    artist_id   UUID NOT NULL REFERENCES artists ON DELETE CASCADE,
    role        TEXT DEFAULT 'primary',
    PRIMARY KEY (track_id, artist_id)
);
CREATE INDEX IF NOT EXISTS idx_track_artists_artist ON track_artists (artist_id);

-- ── embeddings_artist: an artist's sonic centroid, in EffNet space. ──
-- MEAN of the artist's tracks' 1280-D EffNet vectors (re-L2-normalised) — same
-- space as tracks/sessions, so "artists who sound like X" is one cosine ANN.
-- Zero new ML: it averages vectors already in embeddings_effnet.
CREATE TABLE IF NOT EXISTS embeddings_artist (
    artist_id     UUID REFERENCES artists ON DELETE CASCADE,
    embedding     vector(1280),
    model_version TEXT NOT NULL,
    n_tracks      SMALLINT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (artist_id, model_version)
);

-- ── labels (Phase 3b — Discogs enrichment). Mirrors artists exactly so a ──
-- label is a first-class entity with a sonic centroid: "labels that sound like
-- Ostgut Ton" is the SAME cosine ANN as artists/tracks/sessions. Populated when
-- a track is matched to a Discogs release (track_discogs.label).
CREATE TABLE IF NOT EXISTS labels (
    label_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    discogs_id  BIGINT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS track_labels (
    track_id    UUID NOT NULL REFERENCES tracks ON DELETE CASCADE,
    label_id    UUID NOT NULL REFERENCES labels ON DELETE CASCADE,
    PRIMARY KEY (track_id, label_id)
);
CREATE INDEX IF NOT EXISTS idx_track_labels_label ON track_labels (label_id);
CREATE TABLE IF NOT EXISTS embeddings_label (
    label_id      UUID REFERENCES labels ON DELETE CASCADE,
    embedding     vector(1280),
    model_version TEXT NOT NULL,
    n_tracks      SMALLINT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (label_id, model_version)
);

-- ── track_discogs: one row per track holding the matched Discogs release ──
-- metadata + cover. `status` drives the auto + confirm-doubtful flow:
-- matched (auto-accepted) | confirmed (user-picked) | doubtful (needs review,
-- `candidates` holds the shortlist) | unmatched | skipped (user dismissed).
CREATE TABLE IF NOT EXISTS track_discogs (
    track_id    UUID PRIMARY KEY REFERENCES tracks ON DELETE CASCADE,
    release_id  BIGINT,
    master_id   BIGINT,
    label       TEXT,
    catno       TEXT,
    year        INTEGER,
    country     TEXT,
    genres      TEXT[] DEFAULT '{}',
    styles      TEXT[] DEFAULT '{}',
    cover_url   TEXT,
    cover_path  TEXT,
    status      TEXT DEFAULT 'unmatched',
    confidence  REAL,
    candidates  JSONB DEFAULT '[]',
    matched_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_track_discogs_status ON track_discogs (status);

-- ── knowledge base (Phase 2 — RAG over music text). ──
-- kb_documents = a user-ingested source (bio, history, book excerpt, notes).
-- kb_chunks = its overlapping text chunks + 768-D embeddings (TEXT space, kept
-- entirely separate from the 1280-D AUDIO space). ONE collection (one ANN),
-- heterogeneous by design: `category` (open free text — dj, label, music-theory,
-- book, scene, gear…), `tags[]`, and a `meta` JSONB carry the document's type
-- and extras so new kinds of knowledge need no schema change. content_hash dedups.
CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title        TEXT NOT NULL,
    source_type  TEXT DEFAULT 'upload',
    source_url   TEXT,
    lang         TEXT,
    category     TEXT,
    tags         TEXT[] DEFAULT '{}',
    meta         JSONB DEFAULT '{}',
    content_hash TEXT UNIQUE,
    n_chunks     INTEGER DEFAULT 0,
    ingested_at  TIMESTAMPTZ DEFAULT now()
);
-- Heterogeneous-KB migration (idempotent; safe on the Phase 2 tables). ──
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS category TEXT;
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}';
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS meta JSONB DEFAULT '{}';
CREATE INDEX IF NOT EXISTS idx_kb_docs_category ON kb_documents (category);
CREATE INDEX IF NOT EXISTS idx_kb_docs_tags ON kb_documents USING gin (tags);

CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       UUID NOT NULL REFERENCES kb_documents ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    embedding    vector(768),
    model_name   TEXT NOT NULL,
    token_count  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks (doc_id);

-- ── reference web sources (assistant web scouting). ──
-- web_sources = a website the user registers on the Knowledge page for the agent to
-- SEARCH live, plus a MANDATORY topic saying what it is / what to look for there (so
-- the agent knows when to use it). web_cache = 768-D embeddings of what those
-- searches turn up (and a snapshot of each page at registration), in the SAME TEXT
-- space as kb_chunks but a SEPARATE table so the user's curated KB stays clean and
-- its capacity cap is independent. This is the semantic fallback when the live web
-- is unreachable; it is bounded (oldest evicted past config.WEB_CACHE_MAX_CHUNKS).
CREATE TABLE IF NOT EXISTS web_sources (
    source_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url         TEXT NOT NULL,
    topic       TEXT NOT NULL,
    note        TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS web_cache (
    cache_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID REFERENCES web_sources ON DELETE CASCADE,
    query       TEXT,
    title       TEXT,
    url         TEXT,
    text        TEXT NOT NULL,
    embedding   vector(768),
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_web_cache_source ON web_cache (source_id);

-- Saved Live Mode sessions carry a user-chosen UNIQUE name. NULL = not saved
-- yet (in progress, or abandoned without consent — purged on next start).
-- Postgres unique indexes admit multiple NULLs, so unsaved rows never collide.
ALTER TABLE mix_sessions
    ADD COLUMN IF NOT EXISTS name TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_mix_sessions_name ON mix_sessions (name);

-- ── embeddings_session: a saved set's sonic centroid, in EffNet space. ──
-- The MEAN of its tracks' 1280-D EffNet vectors (re-L2-normalised). Built at
-- save time from vectors that already exist in embeddings_effnet — zero model
-- inference, pure averaging. Lives in the SAME space as track embeddings, so a
-- future agent can ask "sessions like this session", "sessions like this
-- artist's sets", or even "tracks that fit this session's vibe" with one cosine
-- search. METADATA (tracklist, artists, crate, times) stays in mix_sessions /
-- session_tracks — this table is the vector only, joined back when needed.
-- n_tracks records how many vectors were pooled (centroid confidence).
CREATE TABLE IF NOT EXISTS embeddings_session (
    session_id      UUID REFERENCES mix_sessions ON DELETE CASCADE,
    embedding       vector(1280),
    model_version   TEXT NOT NULL,
    n_tracks        SMALLINT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (session_id, model_version)
);
"""

# HNSW indices built separately: they depend on tables existing AND can be slow
# on large datasets, so they live outside the table-creation block. All use
# vector_cosine_ops because the embeddings are L2-normalised (or compared via
# cosine distance) and we query with the <=> operator.
_HNSW_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_effnet_hnsw
    ON embeddings_effnet
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_genre_discogs400_hnsw
    ON embeddings_genre_discogs400
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_jamendo_moodtheme_hnsw
    ON embeddings_jamendo_moodtheme
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_jamendo_instrument_hnsw
    ON embeddings_jamendo_instrument
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_session_hnsw
    ON embeddings_session
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_artist_hnsw
    ON embeddings_artist
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_label_hnsw
    ON embeddings_label
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_kb_chunks_hnsw
    ON kb_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});

CREATE INDEX IF NOT EXISTS idx_web_cache_hnsw
    ON web_cache
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
"""


@requires_db
def db_init() -> None:
    """Create the pgvector extension, all tables, and all HNSW indices if missing.

    Idempotent — every statement is IF NOT EXISTS, so this runs on every import
    via the init guard with no ill effect. Logs each DDL phase at DEBUG.

    Raises:
        DBUnavailableError: if the DB is down.
    """
    with _transaction(dict_rows=False) as cur:
        logger.debug("DDL: creating extensions + tables (idempotent)")
        cur.execute(_SCHEMA_SQL)
        logger.debug("DDL: ensuring HNSW indices on all embedding tables")
        cur.execute(_HNSW_SQL)
    # Outside the DDL transaction: guarantee a default crate exists and adopt
    # any crate-less tracks into it (covers both fresh installs and the
    # single-crate -> multi-crate migration of an existing database).
    ensure_default_crate()
    logger.debug("Schema initialisation complete")


def _log_versions() -> None:
    """Log the Postgres server version + installed pgvector version at INFO.

    Called once after connect. Doubles as a reachability probe — if this query
    works, the DB is genuinely up, not just TCP-accepting.
    """
    with _transaction(dict_rows=False) as cur:
        cur.execute("SELECT version();")              # Full PG version banner.
        pg_version = cur.fetchone()[0].split(",")[0]  # Trim to "PostgreSQL X.Y".
        # extversion is NULL if pgvector somehow isn't installed in this DB.
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        vec_version = row[0] if row else "MISSING"
    logger.info("Connected: %s | pgvector %s", pg_version, vec_version)


@requires_db
def health_check() -> bool:
    """Verify the DB is reachable AND pgvector is installed.

    Returns:
        True if a trivial query succeeds and the 'vector' extension is present;
        False on any failure (logged at ERROR). Never raises — designed to be
        polled by a status endpoint / startup probe.
    """
    try:
        with _transaction(dict_rows=False) as cur:
            cur.execute("SELECT 1;")                  # Liveness.
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
            has_vector = cur.fetchone() is not None   # pgvector present?
        if not has_vector:
            logger.error("health_check: pgvector extension not installed")
        return has_vector
    except Exception as e:
        logger.error("health_check failed: %s", e, exc_info=True)
        return False


# ════════════════════════════════════════════════════════════
#  VECTOR SERIALISATION HELPER
# ════════════════════════════════════════════════════════════
# We talk to pgvector via its text representation '[f1,f2,...]' rather than
# pulling in the optional `pgvector` Python adapter — keeps the dependency list
# to psycopg2 + python-dotenv. Only the IN direction is needed: queries return
# track rows (not raw embeddings), so nothing ever parses a literal back out.
def _vec_to_literal(vector) -> str:
    """Format a numeric sequence as a pgvector text literal '[a,b,c]'.

    Args:
        vector: any iterable of numbers (list, tuple, numpy array).
    Returns:
        A string like '[0.1,0.2,0.3]' suitable for a `%s::vector` placeholder.
    """
    # repr(float(x)) keeps full precision without numpy's array formatting noise.
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _literal_to_vec(literal: str) -> list:
    """Parse a pgvector text literal '[a,b,c]' back into a list of floats."""
    s = literal.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [float(x) for x in s.split(",")] if s else []


# ════════════════════════════════════════════════════════════
#  CRATE OPERATIONS  (logical record boxes)
# ════════════════════════════════════════════════════════════
@requires_db
def create_crate(name: str, genre: str = None, bpm_seed: tuple = None,
                 description: str = None) -> str:
    """Create a crate (or return the existing one with that name).

    Args:
        name: user-facing crate name (unique, case-sensitive as given).
        genre: one of config.GENRE_PROFILES keys; defaults to config.DEFAULT_GENRE.
            Unknown genres are accepted but fall back to the 'other' seed range.
        bpm_seed: optional (lo, hi) override; when None the genre profile's
            seed range is stored.
        description: optional free-text note about the crate (era, vibe, source).
    Returns:
        The crate_id (UUID str).
    """
    import config as _cfg
    # genre is OPTIONAL: a genre-less crate (the master library) stores NULL
    # genre and NULL seeds — no genre stereotype, no BPM folding prior.
    if genre:
        genre = genre.strip().lower().replace(" ", "_")
        profile = _cfg.GENRE_PROFILES.get(genre, _cfg.GENRE_PROFILES["other"])
        lo, hi = bpm_seed if bpm_seed else profile["bpm_seed"]
    else:
        genre, (lo, hi) = None, (bpm_seed if bpm_seed else (None, None))
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO crates (name, genre, description, bpm_seed_lo, bpm_seed_hi)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
                SET description = COALESCE(EXCLUDED.description,
                                           crates.description)  -- touch so RETURNING fires
            RETURNING crate_id;
            """,
            (name, genre, description, lo, hi),
        )
        crate_id = str(cur.fetchone()["crate_id"])
    seed = f"{lo:.0f}-{hi:.0f}" if lo is not None else "none"
    logger.info("create_crate: '%s' genre=%s seed=%s -> %s",
                name, genre or "—", seed, crate_id)
    return crate_id


@requires_db
def rename_crate(crate_id: str, name: str) -> bool:
    """Rename a crate. Returns False if the name is taken (crates.name is UNIQUE),
    raises KeyError if the crate does not exist. The caller MUST refuse the default
    library first — its name is the is_default sentinel (see config.DEFAULT_CRATE_NAME)."""
    try:
        with _transaction() as cur:
            cur.execute("UPDATE crates SET name = %s WHERE crate_id = %s;",
                        (name, crate_id))
            if cur.rowcount == 0:
                raise KeyError(f"No crate with id {crate_id}")
    except psycopg2.errors.UniqueViolation:
        return False
    logger.info("rename_crate: %s -> '%s'", crate_id, name)
    return True


@requires_db
def get_crate(name_or_id: str) -> "dict | None":
    """Fetch one crate by name (exact, then case-insensitive) or by UUID."""
    with _transaction() as cur:
        cur.execute("SELECT * FROM crates WHERE name = %s;", (name_or_id,))
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT * FROM crates WHERE lower(name) = lower(%s);",
                        (name_or_id,))
            row = cur.fetchone()
        if row is None:
            try:
                cur.execute("SELECT * FROM crates WHERE crate_id = %s;",
                            (name_or_id,))
                row = cur.fetchone()
            except Exception:
                row = None   # not a UUID — fine, it was a name miss.
    return dict(row) if row else None


@requires_db
def delete_crate(crate_id: str) -> dict:
    """Delete a user crate. Its tracks are NEVER destroyed — only their
    membership in this crate is dropped (crate_tracks CASCADE); every record
    stays in the master library (Vinyl Collection) and any other user crate.
    The default crate is the safety net and cannot be deleted.

    Returns:
        {"deleted": bool, "rehomed": int, "reason": str|None}. deleted=False
        with a reason when the target is the default crate or does not exist.
        'rehomed' = how many memberships were dropped (informational).
    """
    import config as _cfg
    default_id = ensure_default_crate()
    row = get_crate(str(crate_id))
    if row is None:
        return {"deleted": False, "rehomed": 0, "reason": "no such crate"}
    if str(row["crate_id"]) == str(default_id):
        return {"deleted": False, "rehomed": 0,
                "reason": f"cannot delete the default crate '{_cfg.DEFAULT_CRATE_NAME}'"}
    with _transaction() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM crate_tracks WHERE crate_id = %s;",
                    (row["crate_id"],))
        dropped = cur.fetchone()["n"]
        # ON DELETE SET NULL re-homes any track whose ingest-origin was this crate
        # so the orphan adopter keeps them tidy; crate_tracks rows CASCADE away.
        cur.execute("DELETE FROM crates WHERE crate_id = %s;", (row["crate_id"],))
    logger.info("delete_crate: '%s' removed, %d membership(s) dropped (tracks kept)",
                row["name"], dropped)
    return {"deleted": True, "rehomed": dropped, "reason": None}


@requires_db
def list_crates() -> list:
    """All crates with per-crate track totals; the default (master) crate FIRST.

    The default crate is the master library, so its counts are the GLOBAL totals
    (every track, not just those physically filed under its crate_id).
    """
    import config as _cfg
    with _transaction() as cur:
        cur.execute(
            """
            SELECT c.*,
                   COUNT(t.track_id)                                        AS n_tracks,
                   COUNT(t.track_id) FILTER (WHERE t.analyzed_at IS NOT NULL) AS n_analyzed
              FROM crates c
              LEFT JOIN crate_tracks ct ON ct.crate_id = c.crate_id
              LEFT JOIN tracks t        ON t.track_id  = ct.track_id
             GROUP BY c.crate_id
             ORDER BY (c.name = %s) DESC, c.created_at ASC;
            """, (_cfg.DEFAULT_CRATE_NAME,)
        )
        rows = [dict(r) for r in cur.fetchall()]
        # Master crate reports the whole library, not just its own filed tracks.
        cur.execute("""SELECT COUNT(*) AS n,
                              COUNT(*) FILTER (WHERE analyzed_at IS NOT NULL) AS a
                         FROM tracks;""")
        tot = cur.fetchone()
    for r in rows:
        if r["name"] == _cfg.DEFAULT_CRATE_NAME:
            r["n_tracks"], r["n_analyzed"] = tot["n"], tot["a"]
    return rows


@requires_db
def ensure_default_crate() -> str:
    """Guarantee the default crate exists and adopt any crate-less tracks.

    Idempotent: called on every db_init. On a database that predates the
    multi-crate schema, this performs the whole migration in one pass —
    create the default crate, then backfill tracks WHERE crate_id IS NULL.

    Returns:
        The default crate's crate_id (UUID str).
    """
    import config as _cfg
    crate_id = create_crate(_cfg.DEFAULT_CRATE_NAME)        # genre-less master library
    with _transaction() as cur:
        # The default crate is the master library: no genre, no BPM seed. Force
        # it genre-less even if it predates this rule (create_crate's ON CONFLICT
        # only touches description).
        cur.execute("UPDATE crates SET genre = NULL, bpm_seed_lo = NULL, "
                    "bpm_seed_hi = NULL WHERE crate_id = %s;", (crate_id,))
        cur.execute("UPDATE tracks SET crate_id = %s WHERE crate_id IS NULL;",
                    (crate_id,))
        adopted = cur.rowcount
        # One-time migration to the many-to-many model: every track currently
        # filed under a NON-default crate becomes an explicit membership row.
        # Idempotent (ON CONFLICT) and a no-op once migrated.
        cur.execute(
            """
            INSERT INTO crate_tracks (crate_id, track_id)
            SELECT crate_id, track_id FROM tracks
             WHERE crate_id IS NOT NULL AND crate_id <> %s
            ON CONFLICT DO NOTHING;
            """, (crate_id,))
    if adopted:
        logger.info("ensure_default_crate: adopted %d crate-less track(s) into '%s'",
                    adopted, _cfg.DEFAULT_CRATE_NAME)
    return crate_id


# The default crate id is stable for a process lifetime; cache it so the
# master-view filter below stays cheap on hot paths (recognition / listing).
_DEFAULT_CRATE_ID = None


def default_crate_id() -> "str | None":
    """crate_id of the default 'master library' crate (config.DEFAULT_CRATE_NAME)."""
    global _DEFAULT_CRATE_ID
    if _DEFAULT_CRATE_ID is None and DB_AVAILABLE:
        import config as _cfg
        row = get_crate(_cfg.DEFAULT_CRATE_NAME)
        _DEFAULT_CRATE_ID = str(row["crate_id"]) if row else None
    return _DEFAULT_CRATE_ID


def _effective_crate_filter(crate_id: "str | None") -> "str | None":
    """Map a crate filter for TRACK queries to the master-view semantics.

    The default crate is the master library — it contains EVERY track regardless
    of which user crate it is filed in. So a request scoped to the default crate
    means 'all tracks' (no crate_id filter). Any other crate filters normally.
    """
    if crate_id is not None and str(crate_id) == (default_crate_id() or ""):
        return None
    return crate_id


def resolve_crate_id(crate: "str | None") -> "str | None":
    """Turn a CLI-style crate reference into a crate_id.

    Args:
        crate: a crate name, a crate_id, or None.
    Returns:
        crate_id for the named crate; for None, the ACTIVE crate (falling back
        to the default crate); None only when the DB is unavailable.
    Raises:
        KeyError: if a non-None name/id matches no crate (typos must be loud —
            silently importing into the wrong crate would be far worse).
    """
    if not DB_AVAILABLE:
        return None
    if crate:
        row = get_crate(str(crate))
        if row is None:
            raise KeyError(f"No crate named '{crate}' — create it first: "
                           f"python crate.py new-crate \"{crate}\"")
        return str(row["crate_id"])
    return active_crate_id()


def active_crate_id() -> "str | None":
    """The crate the CLI is currently 'in' (set via `crate.py use <name>`).

    Reads config.ACTIVE_CRATE_FILE; falls back to the default crate when the
    file is missing or names a crate that no longer exists.
    """
    if not DB_AVAILABLE:
        return None
    import config as _cfg
    try:
        name = _cfg.ACTIVE_CRATE_FILE.read_text().strip()
        if name:
            row = get_crate(name)
            if row:
                return str(row["crate_id"])
            logger.warning("active crate '%s' no longer exists — using default", name)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("could not read active-crate file: %s", e)
    return ensure_default_crate()


def set_active_crate(name: str) -> str:
    """Persist `name` as the active crate (validated against the DB).

    Returns the crate_id. Raises KeyError when no such crate exists.
    """
    import config as _cfg
    row = get_crate(name)
    if row is None:
        raise KeyError(f"No crate named '{name}'")
    _cfg.ACTIVE_CRATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _cfg.ACTIVE_CRATE_FILE.write_text(row["name"])
    logger.info("active crate -> '%s' (%s)", row["name"], row["crate_id"])
    return str(row["crate_id"])


@requires_db
def crate_bpm_stats(crate_id: str) -> tuple:
    """Median + MAD of the (corrected) BPM across a crate's analysed tracks.

    Returns:
        (median, mad, n) — (None, None, 0) when the crate has no analysed BPMs.
    """
    mclause, mparams = _member_clause(crate_id)   # default → all; user crate → its members
    member_and = f"AND {mclause}" if mclause else ""
    with _transaction(dict_rows=False) as cur:
        cur.execute(
            f"""
            WITH b AS (
                SELECT (features->>'bpm')::float AS bpm
                  FROM tracks
                 WHERE analyzed_at IS NOT NULL
                   AND features ? 'bpm'
                   AND (features->>'bpm') IS NOT NULL
                   {member_and}
            ),
            m AS (
                SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY bpm) AS med
                  FROM b
            )
            SELECT m.med,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(b.bpm - m.med)),
                   COUNT(*)
              FROM b, m
             GROUP BY m.med;
            """,
            mparams,
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return (None, None, 0)
    return (float(row[0]), float(row[1]), int(row[2]))


def crate_bpm_prior(crate_id: "str | None") -> "tuple | None":
    """The (lo, hi) BPM range a new track in this crate is expected to fall in.

    Once the crate has >= config.CRATE_PRIOR_MIN_TRACKS analysed tracks, the
    prior is learned: median ± max(1.5·MAD, 6 BPM) — the 6 BPM floor stops a
    very homogeneous crate (MAD ~ 0) from collapsing into a sliver that would
    fold legitimate neighbours. Before that, the crate's genre seed range
    applies. Returns None when no prior is determinable (DB down / no crate).
    """
    import config as _cfg
    if not DB_AVAILABLE or crate_id is None:
        return None
    try:
        med, mad, n = crate_bpm_stats(crate_id)
        if med is not None and n >= _cfg.CRATE_PRIOR_MIN_TRACKS:
            half = max(1.5 * (mad or 0.0), 6.0)
            return (med - half, med + half)
        row = None
        with _transaction() as cur:
            cur.execute("SELECT bpm_seed_lo, bpm_seed_hi FROM crates "
                        "WHERE crate_id = %s;", (crate_id,))
            row = cur.fetchone()
        if row and row["bpm_seed_lo"] is not None:
            return (float(row["bpm_seed_lo"]), float(row["bpm_seed_hi"]))
    except Exception as e:
        logger.warning("crate_bpm_prior failed for %s: %s", crate_id, e)
    return None


@requires_db
def insert_track(crate_path: str, filename: str, duration: float = None,
                 crate_id: str = None) -> str:
    """Insert a new (pending-analysis) track row.

    Args:
        crate_path: Path to the 120s excerpt in /crate. Must be unique — this is
            the natural key for re-scan idempotency.
        filename: Display filename.
        duration: Excerpt length in seconds (optional; may be filled at analysis).
        crate_id: owning crate; None resolves to the active/default crate.
    Returns:
        The generated track_id as a str (UUID).

    On a duplicate crate_path we DO NOT raise — we return the existing track_id
    so a re-scan is a no-op rather than a crash. Re-scans never MOVE a track
    between crates: the conflict branch only refreshes filename on purpose.
    """
    if crate_id is None:
        crate_id = active_crate_id()
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO tracks (crate_path, filename, duration_sec, crate_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (crate_path) DO UPDATE
                SET filename = EXCLUDED.filename      -- touch a column so RETURNING fires
            RETURNING track_id;
            """,
            (crate_path, filename, duration, crate_id),
        )
        track_id = str(cur.fetchone()["track_id"])
    logger.debug("insert_track: %s -> %s (crate=%s)", filename, track_id, crate_id)
    return track_id


@requires_db
def get_track(track_id: str) -> dict:
    """Fetch one track by id.

    Args:
        track_id: UUID of the track.
    Returns:
        The track row as a dict.
    Raises:
        KeyError: if no track with that id exists.
    """
    with _transaction() as cur:
        cur.execute("SELECT * FROM tracks WHERE track_id = %s;", (track_id,))
        row = cur.fetchone()
    logger.debug("get_track: %s -> %s", track_id, "hit" if row else "miss")
    if row is None:
        raise KeyError(f"No track with id {track_id}")
    return dict(row)


@requires_db
def get_track_by_path(crate_path: str):
    """Fetch one track by its crate_path.

    Args:
        crate_path: The unique path to the excerpt.
    Returns:
        The track row as a dict, or None if not found (callers branch on this to
        decide insert-vs-skip, so absence is a normal result, not an error).
    """
    with _transaction() as cur:
        cur.execute("SELECT * FROM tracks WHERE crate_path = %s;", (crate_path,))
        row = cur.fetchone()
    logger.debug("get_track_by_path: %s -> %s", crate_path, "hit" if row else "miss")
    return dict(row) if row else None


@requires_db
def update_track_features(track_id: str, features_dict: dict,
                          pipeline_level: int) -> None:
    """Attach analysis results to a track and stamp analyzed_at = now().

    Args:
        track_id: UUID of the track to update.
        features_dict: The full TrackFeatures payload (asdict()) → JSONB.
        pipeline_level: 1/2/3 level actually reached for this track.

    Setting analyzed_at flips the track out of the "pending" set. duration_sec
    is refreshed from features if present, so a track inserted with an unknown
    duration gets corrected here.
    """
    with _transaction() as cur:
        cur.execute(
            """
            UPDATE tracks
               SET features       = %s,
                   pipeline_level = %s,
                   duration_sec   = COALESCE(%s, duration_sec),
                   analyzed_at    = now()
             WHERE track_id = %s;
            """,
            # Json() adapts the dict to a JSONB-castable parameter.
            (psycopg2.extras.Json(features_dict), pipeline_level,
             features_dict.get("duration"), track_id),
        )
        affected = cur.rowcount
    logger.debug("update_track_features: %s level=%d (%d row)",
                 track_id, pipeline_level, affected)


@requires_db
def delete_track(track_id: str) -> None:
    """Delete a track. CASCADE removes all of its embedding rows.

    session_tracks references are SET NULL instead (history is preserved — see
    schema). Args: track_id — UUID to remove.
    """
    with _transaction() as cur:
        cur.execute("DELETE FROM tracks WHERE track_id = %s;", (track_id,))
        affected = cur.rowcount
    logger.debug("delete_track: %s (%d row, embeddings cascaded)", track_id, affected)


@requires_db
def list_tracks(analyzed_only: bool = False, crate_id: str = None) -> list:
    """List tracks, newest first.

    Args:
        analyzed_only: when True, return only tracks that have been analysed
            (analyzed_at IS NOT NULL).
        crate_id: when given, restrict to that crate; None = ALL crates (the
            cross-crate view live recognition relies on).
    Returns:
        list[dict] of track rows.
    """
    # Build the optional filters without string-concatenating user input.
    clauses, params = [], []
    if analyzed_only:
        clauses.append("analyzed_at IS NOT NULL")
    mclause, mparams = _member_clause(crate_id)    # default → all (master); user → crate_tracks
    if mclause:
        clauses.append(mclause); params += mparams
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _transaction() as cur:
        cur.execute(f"SELECT * FROM tracks {where} ORDER BY added_at DESC;", params)
        rows = cur.fetchall()
    logger.debug("list_tracks(analyzed_only=%s, crate=%s) -> %d rows",
                 analyzed_only, crate_id, len(rows))
    return [dict(r) for r in rows]


def _member_clause(crate_id: "str | None", alias: str = "") -> tuple:
    """WHERE-clause fragment + params restricting tracks to a crate's members.

    Many-to-many: the default crate (master) means ALL tracks (no filter); any
    user crate means the tracks listed in crate_tracks for it. `alias` is the
    tracks-table alias used in the surrounding query ('t' or '').
    """
    cid = _effective_crate_filter(crate_id)       # default → None (master = all)
    if cid is None:
        return ("", [])
    col = f"{alias}.track_id" if alias else "track_id"
    return (f"{col} IN (SELECT track_id FROM crate_tracks WHERE crate_id = %s)", [cid])


@requires_db
def add_tracks_to_crate(crate_id: str, track_ids: list) -> int:
    """Add tracks to a USER crate (membership; a track can be in several crates).

    Idempotent per (crate, track). The default crate is the master library — it
    holds everything implicitly, so adding to it is a no-op.
    """
    if not track_ids or str(crate_id) == (default_crate_id() or ""):
        return 0
    with _transaction() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO crate_tracks (crate_id, track_id) VALUES %s "
            "ON CONFLICT DO NOTHING;",
            [(crate_id, t) for t in track_ids])
        added = cur.rowcount
    logger.info("add_tracks_to_crate: %d track(s) -> crate %s", added, crate_id)
    return added


@requires_db
def remove_track_from_crate(track_id: str, crate_id: str) -> bool:
    """Remove a track from ONE user crate (drops the membership; the track stays
    in the master library and any other user crates it belongs to)."""
    with _transaction() as cur:
        cur.execute("DELETE FROM crate_tracks WHERE crate_id = %s AND track_id = %s;",
                    (crate_id, track_id))
        removed = cur.rowcount > 0
    logger.debug("remove_track_from_crate: %s from %s (removed=%s)",
                 track_id, crate_id, removed)
    return removed


@requires_db
def track_crate_ids(track_id: str) -> list:
    """The user-crate ids a track is a member of (master is implicit, not listed)."""
    with _transaction() as cur:
        cur.execute("SELECT crate_id FROM crate_tracks WHERE track_id = %s;", (track_id,))
        return [str(r["crate_id"]) for r in cur.fetchall()]


@requires_db
def set_track_on_spot(track_id: str, on_spot: bool) -> bool:
    """Flag/unflag a track as ON SPOT (the crate page's pull list).

    Returns True when a row was updated, False when the track does not exist.
    """
    with _transaction() as cur:
        cur.execute("UPDATE tracks SET on_spot = %s WHERE track_id = %s;",
                    (on_spot, track_id))
        updated = cur.rowcount > 0
    logger.debug("set_track_on_spot: %s -> %s (updated=%s)",
                 track_id, on_spot, updated)
    return updated


@requires_db
def rename_track(track_id: str, filename: str) -> bool:
    """Update a track's display filename ('Artist - Title [EP].ext').

    Only the metadata label changes — the crate excerpt (UUID.wav) and every
    analysis/embedding row stay keyed to the same track_id. Because all lists
    read this column, the edit shows up everywhere the track appears.
    """
    with _transaction() as cur:
        cur.execute("UPDATE tracks SET filename = %s WHERE track_id = %s;",
                    (filename, track_id))
        updated = cur.rowcount > 0
    logger.debug("rename_track: %s -> '%s' (updated=%s)", track_id, filename, updated)
    return updated


@requires_db
def list_tracks_below_level(target_level: int, crate_id: str = None) -> list:
    """Return analyzed tracks whose pipeline_level is below target_level.

    Used by crate.upgrade_pipeline() to find tracks that need a re-analysis
    now that more models are available. Only returns tracks that have already
    been analyzed (analyzed_at IS NOT NULL) — pending tracks are handled by
    analyze_pending() instead.

    Args:
        target_level: tracks with pipeline_level < this value are returned.
        crate_id: when given, restrict to that crate; None = all crates.
    Returns:
        list[dict] of track rows, same shape as list_tracks().
    """
    clauses = ["analyzed_at IS NOT NULL", "pipeline_level < %s"]
    params: list = [target_level]
    if crate_id is not None:
        clauses.append("crate_id = %s")
        params.append(crate_id)
    where = "WHERE " + " AND ".join(clauses)
    with _transaction() as cur:
        cur.execute(f"SELECT * FROM tracks {where} ORDER BY added_at DESC;", params)
        rows = cur.fetchall()
    logger.debug("list_tracks_below_level(<%d, crate=%s) -> %d rows",
                 target_level, crate_id, len(rows))
    return [dict(r) for r in rows]


@requires_db
def count_tracks(crate_id: str = None) -> tuple:
    """Count tracks by analysis state, optionally within one crate.

    Returns:
        (total, analyzed, pending) — a 3-tuple of ints. pending = total - analyzed,
        computed in SQL so the three numbers are always internally consistent.
    """
    mclause, params = _member_clause(crate_id)     # default → all (master); user → crate_tracks
    where = f"WHERE {mclause}" if mclause else ""
    with _transaction(dict_rows=False) as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)                                        AS total,
                   COUNT(*) FILTER (WHERE analyzed_at IS NOT NULL) AS analyzed
              FROM tracks {where};
            """, params
        )
        total, analyzed = cur.fetchone()
    pending = total - analyzed
    logger.debug("count_tracks -> total=%d analyzed=%d pending=%d",
                 total, analyzed, pending)
    return (total, analyzed, pending)


# ════════════════════════════════════════════════════════════
#  EMBEDDING OPERATIONS
# ════════════════════════════════════════════════════════════
@requires_db
def upsert_effnet_embedding(track_id: str, vector, model_version: str) -> None:
    """Insert or replace the EffNet (1280-D) embedding for a track+model.

    Args:
        track_id: UUID the embedding belongs to.
        vector: 1280-element numeric sequence (L2-normalised upstream).
        model_version: e.g. 'discogs-effnet-bs64-1'. Part of the PK, so the same
            track can hold vectors from several model versions simultaneously.

    ON CONFLICT updates the vector + created_at — re-analysing a track refreshes
    its embedding in place rather than erroring on the PK.
    """
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO embeddings_effnet (track_id, embedding, model_version)
            VALUES (%s, %s::vector, %s)
            ON CONFLICT (track_id, model_version) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    created_at = now();
            """,
            (track_id, _vec_to_literal(vector), model_version),
        )
    # NOTE: vector intentionally NOT logged — only its size.
    logger.debug("upsert_effnet_embedding: %s [%s] dim=%d",
                 track_id, model_version, len(vector))


@requires_db
def find_similar_effnet(query_vector, n: int = 5, exclude_track_id: str = None,
                        crate_id: str = None) -> list:
    """Nearest-neighbour search over EffNet embeddings via cosine distance.

    Uses pgvector's `<=>` cosine-distance operator, which the HNSW index
    (vector_cosine_ops) accelerates. Results are joined back to tracks so the
    caller gets full metadata, not bare ids.

    Args:
        query_vector: 1280-D query embedding.
        n: number of neighbours to return.
        exclude_track_id: optional track_id to omit (e.g. the "now playing" track
            so it never recommends itself).
        crate_id: optional crate to restrict results to. Recommendations pass it;
            live RECOGNITION deliberately does not (you can play any record you
            own regardless of which crate it is filed in).
    Returns:
        list[dict], each row = the track's columns plus 'cosine_distance'
        (0.0 = identical direction, 2.0 = opposite), ordered nearest-first.
    """
    # Parameterise the optional filters rather than splicing SQL.
    mclause, mparams = _member_clause(crate_id, alias="t")   # default → all; user → crate_tracks
    exclude_clause = "AND e.track_id <> %s" if exclude_track_id else ""
    crate_clause = f"AND {mclause}" if mclause else ""
    params = [_vec_to_literal(query_vector)]
    if exclude_track_id:
        params.append(exclude_track_id)
    params += mparams
    params.append(n)

    with _transaction() as cur:
        cur.execute(
            f"""
            SELECT t.*,
                   e.model_version,
                   e.embedding <=> %s::vector AS cosine_distance
              FROM embeddings_effnet e
              JOIN tracks t ON t.track_id = e.track_id
             WHERE TRUE {exclude_clause} {crate_clause}
             ORDER BY cosine_distance ASC          -- nearest first
             LIMIT %s;
            """,
            # The query literal is referenced twice in spirit but bound once here;
            # psycopg2 fills %s positionally, so params order matches the SQL.
            params,
        )
        rows = cur.fetchall()
    logger.debug("find_similar_effnet: n=%d exclude=%s -> %d results",
                 n, exclude_track_id, len(rows))
    return [dict(r) for r in rows]


@requires_db
def upsert_genre_discogs400_embedding(track_id: str, vector, model_version: str) -> None:
    """Insert or replace the 400-D Discogs genre style vector for a track+model.

    Args:
        track_id: UUID the embedding belongs to.
        vector: 400-element numeric sequence (softmax genre probabilities).
        model_version: e.g. 'genre_discogs400-discogs-effnet-1'. Part of the PK.
    """
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO embeddings_genre_discogs400 (track_id, embedding, model_version)
            VALUES (%s, %s::vector, %s)
            ON CONFLICT (track_id, model_version) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    created_at = now();
            """,
            (track_id, _vec_to_literal(vector), model_version),
        )
    logger.debug("upsert_genre_discogs400_embedding: %s [%s] dim=%d",
                 track_id, model_version, len(vector))


@requires_db
def find_similar_genre_discogs400(query_vector, n: int = 5,
                                   exclude_track_id: str = None) -> list:
    """Nearest-neighbour search over the 400-D genre style space.

    Args:
        query_vector: 400-D query vector.
        n: number of neighbours to return.
        exclude_track_id: optional track to omit from results.
    Returns:
        list[dict] with track columns + 'cosine_distance', nearest first.
    """
    exclude_clause = "AND e.track_id <> %s" if exclude_track_id else ""
    params = [_vec_to_literal(query_vector)]
    if exclude_track_id:
        params.append(exclude_track_id)
    params.append(n)
    with _transaction() as cur:
        cur.execute(
            f"""
            SELECT t.*,
                   e.model_version,
                   e.embedding <=> %s::vector AS cosine_distance
              FROM embeddings_genre_discogs400 e
              JOIN tracks t ON t.track_id = e.track_id
             WHERE TRUE {exclude_clause}
             ORDER BY cosine_distance ASC
             LIMIT %s;
            """,
            params,
        )
        rows = cur.fetchall()
    logger.debug("find_similar_genre_discogs400: n=%d -> %d results", n, len(rows))
    return [dict(r) for r in rows]


@requires_db
def upsert_jamendo_moodtheme_embedding(track_id: str, vector, model_version: str) -> None:
    """Insert or replace the 56-D MTG-Jamendo mood+theme vector for a track+model.

    Args:
        track_id: UUID the embedding belongs to.
        vector: 56-element numeric sequence (multi-label sigmoid probabilities).
        model_version: model identifier. Part of the PK alongside track_id.
    """
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO embeddings_jamendo_moodtheme (track_id, embedding, model_version)
            VALUES (%s, %s::vector, %s)
            ON CONFLICT (track_id, model_version) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    created_at = now();
            """,
            (track_id, _vec_to_literal(vector), model_version),
        )
    logger.debug("upsert_jamendo_moodtheme_embedding: %s [%s] dim=%d",
                 track_id, model_version, len(vector))


@requires_db
def find_similar_jamendo_moodtheme(query_vector, n: int = 5,
                                    exclude_track_id: str = None) -> list:
    """Nearest-neighbour search over the 56-D mood+theme space.

    Args:
        query_vector: 56-D query vector.
        n: number of neighbours to return.
        exclude_track_id: optional track to omit from results.
    Returns:
        list[dict] with track columns + 'cosine_distance', nearest first.
    """
    exclude_clause = "AND e.track_id <> %s" if exclude_track_id else ""
    params = [_vec_to_literal(query_vector)]
    if exclude_track_id:
        params.append(exclude_track_id)
    params.append(n)
    with _transaction() as cur:
        cur.execute(
            f"""
            SELECT t.*,
                   e.model_version,
                   e.embedding <=> %s::vector AS cosine_distance
              FROM embeddings_jamendo_moodtheme e
              JOIN tracks t ON t.track_id = e.track_id
             WHERE TRUE {exclude_clause}
             ORDER BY cosine_distance ASC
             LIMIT %s;
            """,
            params,
        )
        rows = cur.fetchall()
    logger.debug("find_similar_jamendo_moodtheme: n=%d -> %d results", n, len(rows))
    return [dict(r) for r in rows]


@requires_db
def upsert_jamendo_instrument_embedding(track_id: str, vector, model_version: str) -> None:
    """Insert or replace the 40-D MTG-Jamendo instrument vector for a track+model.

    Args:
        track_id: UUID the embedding belongs to.
        vector: 40-element numeric sequence (multi-label sigmoid, per-instrument presence).
        model_version: model identifier. Part of the PK alongside track_id.
    """
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO embeddings_jamendo_instrument (track_id, embedding, model_version)
            VALUES (%s, %s::vector, %s)
            ON CONFLICT (track_id, model_version) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    created_at = now();
            """,
            (track_id, _vec_to_literal(vector), model_version),
        )
    logger.debug("upsert_jamendo_instrument_embedding: %s [%s] dim=%d",
                 track_id, model_version, len(vector))


@requires_db
def upsert_multimodal_embedding(track_id: str, vector, modalities: list,
                                model_name: str) -> None:
    """STUB: store a multimodal (1408-D) embedding. Search NOT yet implemented.

    The row is persisted so no data is lost, but we log a WARNING on every call
    to make it unmistakable that nothing consumes these vectors yet. When the
    multimodal recommender lands, add a find_similar_multimodal() mirroring the
    EffNet search and remove this warning.

    Args:
        track_id: UUID the embedding belongs to.
        vector: 1408-element numeric sequence (Google/ImageBind space).
        modalities: which inputs fed it, e.g. ['audio','image','text'].
        model_name: model identifier (part of the PK alongside track_id).
    """
    logger.warning(
        "upsert_multimodal_embedding: multimodal SEARCH is not yet implemented — "
        "storing the %d-D vector for track %s (model=%s) but nothing reads it yet.",
        len(vector), track_id, model_name)
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO embeddings_multimodal (track_id, embedding, modalities, model_name)
            VALUES (%s, %s::vector, %s, %s)
            ON CONFLICT (track_id, model_name) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    modalities = EXCLUDED.modalities,
                    created_at = now();
            """,
            (track_id, _vec_to_literal(vector), modalities, model_name),
        )
    logger.debug("upsert_multimodal_embedding: %s [%s] modalities=%s",
                 track_id, model_name, modalities)


@requires_db
def upsert_text_embedding(track_id: str, vector, source_text: str,
                          model_name: str) -> None:
    """Insert or replace a text (768-D) embedding for semantic metadata search.

    Args:
        track_id: UUID the embedding belongs to.
        vector: 768-element numeric sequence.
        source_text: the exact string that was embedded (kept for debugging).
        model_name: model identifier (part of the PK alongside track_id).
    """
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO embeddings_text (track_id, embedding, source_text, model_name)
            VALUES (%s, %s::vector, %s, %s)
            ON CONFLICT (track_id, model_name) DO UPDATE
                SET embedding   = EXCLUDED.embedding,
                    source_text = EXCLUDED.source_text;
            """,
            (track_id, _vec_to_literal(vector), source_text, model_name),
        )
    logger.debug("upsert_text_embedding: %s [%s] dim=%d (src %d chars)",
                 track_id, model_name, len(vector), len(source_text or ""))


# ════════════════════════════════════════════════════════════
#  FINGERPRINT OPERATIONS  (Shazam-style landmark hashes)
# ════════════════════════════════════════════════════════════
@requires_db
def replace_fingerprints(track_id: str, hashes: list) -> int:
    """Store a track's landmark hashes, replacing any previous set. Idempotent.

    DELETE-then-INSERT inside one transaction so a re-analysis (or `upgrade`)
    can never leave a track with two generations of hashes mixed together.
    Bulk insert via execute_values — a 120 s excerpt yields ~10-20k rows.

    Args:
        track_id: DB primary key the landmarks belong to.
        hashes: [(hash_int, t_offset_frame), ...] from fingerprint.extract_hashes.
    Returns:
        Number of landmarks stored.
    """
    with _transaction(dict_rows=False) as cur:
        cur.execute("DELETE FROM fingerprints WHERE track_id = %s;", (track_id,))
        if hashes:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO fingerprints (hash, track_id, t_offset) VALUES %s",
                [(h, track_id, t) for h, t in hashes],
                page_size=5000,
            )
    logger.debug("replace_fingerprints: %s -> %d landmarks", track_id, len(hashes))
    return len(hashes)


@requires_db
def match_fingerprints(query_hashes: list, top: int = 3) -> list:
    """Constellation match: which crate track aligns with this live snippet?

    The whole Shazam vote runs in ONE SQL statement: unnest the query landmarks,
    join on hash (one index probe each), then GROUP BY (track, t_track - t_query)
    — a true match piles votes onto a single offset bin, noise scatters across
    bins. Rows come back joined to tracks so the caller gets metadata directly.

    Args:
        query_hashes: [(hash_int, t_offset_frame), ...] from the live snippet.
        top: number of (track, offset) candidates to return, best first.
    Returns:
        list[dict]: track columns + 'votes' (aligned matches — the strength) and
        'offset_frames' (where the snippet sits inside the excerpt). Empty when
        nothing matches at all.
    """
    if not query_hashes:
        return []
    q_hash = [int(h) for h, _ in query_hashes]
    q_t = [int(t) for _, t in query_hashes]
    with _transaction() as cur:
        cur.execute(
            """
            SELECT t.*, v.votes, v.delta AS offset_frames
              FROM (
                    SELECT f.track_id,
                           f.t_offset - q.t_query AS delta,
                           COUNT(*)               AS votes
                      FROM unnest(%s::bigint[], %s::int[]) AS q(hash, t_query)
                      JOIN fingerprints f ON f.hash = q.hash
                     GROUP BY f.track_id, delta
                     ORDER BY votes DESC
                     LIMIT %s
                   ) v
              JOIN tracks t ON t.track_id = v.track_id;
            """,
            (q_hash, q_t, top),
        )
        rows = cur.fetchall()
    rows = sorted([dict(r) for r in rows], key=lambda r: -r["votes"])
    logger.debug("match_fingerprints: %d query landmarks -> %d candidates (best=%s)",
                 len(query_hashes), len(rows),
                 rows[0]["votes"] if rows else "none")
    return rows


@requires_db
def tracks_without_fingerprints() -> list:
    """Analyzed tracks that have no landmarks yet — the backfill worklist."""
    with _transaction() as cur:
        cur.execute(
            """
            SELECT t.* FROM tracks t
             WHERE t.analyzed_at IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM fingerprints f
                                WHERE f.track_id = t.track_id)
             ORDER BY t.added_at DESC;
            """
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
#  SESSION OPERATIONS
# ════════════════════════════════════════════════════════════
@requires_db
def create_session(crate_id: str = None) -> str:
    """Open a new mix session, optionally tagged with the crate it was played from.

    Recording the crate makes the session log a richer future training set
    (style learning can condition on which record box the DJ brought).

    Returns:
        The generated session_id (UUID str). started_at defaults to now().
    """
    with _transaction() as cur:
        cur.execute(
            "INSERT INTO mix_sessions (crate_id) VALUES (%s) RETURNING session_id;",
            (crate_id,))
        session_id = str(cur.fetchone()["session_id"])
    logger.debug("create_session -> %s (crate=%s)", session_id, crate_id)
    return session_id


@requires_db
def log_track_played(session_id: str, track_id: str,
                     detected_by: str = "manual") -> int:
    """Append a track to a session's play log and return its position.

    Position is computed server-side as (current row count for the session) + 1,
    so concurrent appends within one transaction can't collide on a stale count.
    played_at is stamped now().

    Args:
        session_id: the open session.
        track_id: the track that was played.
        detected_by: 'fingerprint' | 'manual' — how the track was identified.
    Returns:
        The 1-based position of this track in the set.
    """
    with _transaction() as cur:
        # Derive the next position atomically from the existing rows.
        cur.execute(
            "SELECT COUNT(*) AS c FROM session_tracks WHERE session_id = %s;",
            (session_id,))
        position = cur.fetchone()["c"] + 1
        cur.execute(
            """
            INSERT INTO session_tracks (session_id, track_id, played_at, position, detected_by)
            VALUES (%s, %s, now(), %s, %s);
            """,
            (session_id, track_id, position, detected_by),
        )
    logger.debug("log_track_played: session=%s track=%s pos=%d by=%s",
                 session_id, track_id, position, detected_by)
    return position


def _session_tracklist(cur, session_id: str) -> list:
    """Return a session's ordered tracklist (shared by close/get).

    Internal helper — assumes `cur` is an open dict cursor inside a transaction.
    Centralises the play-log query so close_session() and get_session() can't
    drift apart. Joins tracks for filename/path; LEFT JOIN so a since-deleted
    track (track_id SET NULL) still appears in history with null metadata.
    """
    cur.execute(
        """
        SELECT st.position,
               st.track_id,
               st.played_at,
               st.detected_by,
               st.rating,
               t.filename,
               t.crate_path
          FROM session_tracks st
          LEFT JOIN tracks t ON t.track_id = st.track_id
         WHERE st.session_id = %s
         ORDER BY st.position ASC;
        """,
        (session_id,),
    )
    return [dict(r) for r in cur.fetchall()]


@requires_db
def set_session_track_rating(session_id: str, position: int,
                             rating: "str | None") -> bool:
    """Flag a played track in a session as a good/bad mix (or clear it).

    rating ∈ {'good', 'bad', None}. Keyed by (session_id, position) — position is
    the set order, unique within a session — so it tags the exact slot even when the
    underlying track was later deleted from the crate. Returns True if a row updated.
    """
    with _transaction() as cur:
        cur.execute(
            "UPDATE session_tracks SET rating = %s WHERE session_id = %s AND position = %s;",
            (rating, session_id, position))
        updated = cur.rowcount > 0
    logger.debug("set_session_track_rating: %s pos=%s -> %s (updated=%s)",
                 session_id, position, rating, updated)
    return updated


@requires_db
def close_session(session_id: str) -> list:
    """Close a session: stamp ended_at, snapshot the tracklist into JSONB, return it.

    The normalised session_tracks rows remain the source of truth; we also write
    a denormalised JSONB snapshot onto mix_sessions.tracklist so a finished set
    can be read in a single row without the join.

    Args:
        session_id: the session to close.
    Returns:
        The full ordered tracklist as list[dict].
    """
    with _transaction() as cur:
        tracklist = _session_tracklist(cur, session_id)
        cur.execute(
            "UPDATE mix_sessions SET ended_at = now(), tracklist = %s WHERE session_id = %s;",
            # default=str so datetime/UUID values serialise cleanly into JSONB.
            (psycopg2.extras.Json(tracklist, dumps=_json_dumps), session_id),
        )
    logger.debug("close_session: %s -> %d tracks", session_id, len(tracklist))
    return tracklist


@requires_db
def get_session(session_id: str) -> dict:
    """Fetch a session plus its live, ordered tracklist.

    Args:
        session_id: the session to fetch.
    Returns:
        The mix_sessions row as a dict with an added 'tracklist' key (rebuilt
        live from session_tracks, so it reflects the current state for an
        open session, not just the close-time snapshot).
    Raises:
        KeyError: if the session does not exist.
    """
    with _transaction() as cur:
        cur.execute("SELECT * FROM mix_sessions WHERE session_id = %s;", (session_id,))
        row = cur.fetchone()
        session = None
        if row is not None:
            session = dict(row)
            session["tracklist"] = _session_tracklist(cur, session_id)
    # Raise AFTER the read transaction has committed cleanly. A missing session is
    # an EXPECTED lookup miss (the API maps it to 404), not a transaction failure:
    # raising inside the context manager would trip its rollback path and emit an
    # ERROR-level "Transaction rolled back" log WITH a full traceback for every
    # poll of a non-existent/closed session — drowning real errors in noise.
    if session is None:
        raise KeyError(f"No session with id {session_id}")
    logger.debug("get_session: %s -> %d tracks", session_id, len(session["tracklist"]))
    return session


@requires_db
def save_session(session_id: str, name: str) -> bool:
    """Name a session (user consented to keep it) and close it.

    Returns False when the name is already taken (unique index) — the caller
    surfaces that as a validation error so the user picks another name.
    """
    try:
        with _transaction() as cur:
            cur.execute("UPDATE mix_sessions SET name = %s WHERE session_id = %s;",
                        (name, session_id))
            if cur.rowcount == 0:
                raise KeyError(f"No session with id {session_id}")
    except psycopg2.errors.UniqueViolation:
        return False
    close_session(session_id)
    logger.info("save_session: %s -> '%s'", session_id, name)
    return True


@requires_db
def rename_session(session_id: str, name: str) -> bool:
    """Rename an already-saved session. Returns False if the name is taken (the
    UNIQUE index), raises KeyError if the session does not exist. Unlike
    save_session it does NOT re-close or touch the tracklist — a pure rename."""
    try:
        with _transaction() as cur:
            cur.execute("UPDATE mix_sessions SET name = %s WHERE session_id = %s;",
                        (name, session_id))
            if cur.rowcount == 0:
                raise KeyError(f"No session with id {session_id}")
    except psycopg2.errors.UniqueViolation:
        return False
    logger.info("rename_session: %s -> '%s'", session_id, name)
    return True


@requires_db
def session_track_vectors(session_id: str, model_version: str) -> list:
    """The EffNet vectors of a session's identified tracks, for centroid pooling.

    DISTINCT ON (track_id): a track replayed within the set contributes once
    (the centroid is about which records the set is built FROM, not airtime).
    Returns raw float lists — the caller mean-pools them.
    """
    with _transaction() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (e.track_id) e.embedding::text AS vec
              FROM session_tracks st
              JOIN embeddings_effnet e ON e.track_id = st.track_id
             WHERE st.session_id = %s AND e.model_version = %s;
            """,
            (session_id, model_version))
        rows = cur.fetchall()
    return [_literal_to_vec(r["vec"]) for r in rows]


@requires_db
def upsert_session_embedding(session_id: str, vector, model_version: str,
                             n_tracks: int) -> None:
    """Store a session's centroid vector (one per session+model)."""
    with _transaction() as cur:
        cur.execute(
            """
            INSERT INTO embeddings_session
                   (session_id, embedding, model_version, n_tracks)
            VALUES (%s, %s::vector, %s, %s)
            ON CONFLICT (session_id, model_version) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    n_tracks   = EXCLUDED.n_tracks,
                    created_at = now();
            """,
            (session_id, _vec_to_literal(vector), model_version, n_tracks))
    logger.debug("upsert_session_embedding: %s [%s] pooled %d tracks",
                 session_id, model_version, n_tracks)


@requires_db
def find_similar_sessions(query_vector, n: int = 5,
                          exclude_session_id: str = None) -> list:
    """ANN over session centroids — 'sets that resemble this one'.

    Returns each match's metadata (name, times, crate, n_tracks, tracklist)
    alongside its cosine_distance, so an agent gets the vector hit AND what the
    set actually contains in one call.
    """
    exclude = "AND es.session_id <> %s" if exclude_session_id else ""
    params = [_vec_to_literal(query_vector)]
    if exclude_session_id:
        params.append(exclude_session_id)
    params.append(n)
    with _transaction() as cur:
        cur.execute(
            f"""
            SELECT s.session_id, s.name, s.started_at, s.ended_at,
                   s.tracklist, es.n_tracks,
                   c.name AS crate_name,
                   es.embedding <=> %s::vector AS cosine_distance
              FROM embeddings_session es
              JOIN mix_sessions s USING (session_id)
              LEFT JOIN crates c USING (crate_id)
             WHERE s.name IS NOT NULL {exclude}
             ORDER BY cosine_distance ASC
             LIMIT %s;
            """,
            params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
#  ARTIST ENTITIES  (Phase 0 — structured artists + centroids)
# ════════════════════════════════════════════════════════════
def _parse_artist_names(filename: str) -> list:
    """Artist name(s) from a 'Artist - Title [EP].ext' filename.

    Splits multi-artist credits on ', ' / ' & ' / ' x ' / ' vs ' / ' feat '.
    Returns [] when there is no ' - ' separator (no parseable artist).
    """
    name = re.sub(r"\.(wav|mp3|flac|aiff?)$", "", filename or "", flags=re.I).strip()
    name = re.sub(r"\[.*?\]", "", name).strip()        # drop the [EP] tag
    i = name.find(" - ")
    if i < 0:
        return []
    field = name[:i].strip()
    parts = re.split(r"\s*,\s*|\s+&\s+|\s+x\s+|\s+vs\.?\s+|\s+feat\.?\s+|\s+ft\.?\s+",
                     field, flags=re.I)
    return [p.strip() for p in parts if p.strip() and p.strip() != "—"]


@requires_db
def upsert_artist(name: str) -> str:
    """Insert an artist by name (idempotent on UNIQUE name). Returns artist_id."""
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO artists (name) VALUES (%s)
               ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
               RETURNING artist_id;""", (name,))
        return str(cur.fetchone()["artist_id"])


@requires_db
def link_track_artist(track_id: str, artist_id: str, role: str = "primary") -> None:
    """Link a track to an artist (many-to-many; idempotent)."""
    with _transaction() as cur:
        cur.execute(
            "INSERT INTO track_artists (track_id, artist_id, role) VALUES (%s,%s,%s) "
            "ON CONFLICT DO NOTHING;", (track_id, artist_id, role))


@requires_db
def relink_track_artists(track_id: str, filename: str) -> list:
    """Re-derive a track's artist links from a (possibly edited) filename.

    Clears the old links and re-creates them from the new name. Returns the set
    of affected artist_ids (old ∪ new) so the caller can refresh their centroids.
    """
    affected = set()
    with _transaction() as cur:
        cur.execute("SELECT artist_id FROM track_artists WHERE track_id = %s;", (track_id,))
        affected.update(str(r["artist_id"]) for r in cur.fetchall())
        cur.execute("DELETE FROM track_artists WHERE track_id = %s;", (track_id,))
    for nm in _parse_artist_names(filename):
        aid = upsert_artist(nm)
        link_track_artist(track_id, aid)
        affected.add(aid)
    return list(affected)


# Same multi-artist separators as _parse_artist_names, but CAPTURING so a rewrite
# can rejoin the artist field with the exact separators it found (', ' / ' & ' / …).
_ARTIST_SEP_CAP = re.compile(
    r"(\s*,\s*|\s+&\s+|\s+x\s+|\s+vs\.?\s+|\s+feat\.?\s+|\s+ft\.?\s+)", re.I)


def _rewrite_artist_in_filename(filename: str, old_name: str, new_name: str) -> str:
    """Swap ONE artist token inside the 'Artist - Title [EP].ext' label, leaving
    the title, EP tag and extension untouched. Only the artist field (before the
    first ' - ') is touched, so a multi-artist credit keeps its other names and
    separators — renaming 'A' in 'A & B - X' yields 'NEW & B - X'."""
    i = (filename or "").find(" - ")
    if i < 0:                                   # not 'Artist - Title' shaped → leave as is
        return filename
    seg, rest = filename[:i], filename[i:]
    old = old_name.strip().lower()
    parts = _ARTIST_SEP_CAP.split(seg)          # tokens at even idx, separators at odd
    changed = False
    for k in range(0, len(parts), 2):
        if parts[k].strip().lower() == old:
            parts[k] = parts[k].replace(parts[k].strip(), new_name.strip())
            changed = True
    return ("".join(parts) + rest) if changed else filename


@requires_db
def rename_artist(old: str, new: str) -> "dict | None":
    """Rename an artist GLOBALLY: the artists row AND every linked track's filename
    label, so the change appears in every list and in similar_artists — not just on
    the one track that was clicked.

    If `new` already names another artist the two are MERGED (the existing one
    survives, links are re-pointed, the old row is deleted) — so 'Surgon' → 'Surgeon'
    folds a typo into the real entity. The whole thing is one transaction. The
    caller refreshes the survivor's audio centroid (analyze.persist_artist_embedding)
    because a merge grows its track set; that derived step is kept out of here so
    database.py never imports analyze (same split as relink_track_artists).

    Returns {"artist_id", "old_name", "new_name", "affected_tracks", "merged"} or
    None when `old` is unknown or `new` is blank.
    """
    new_name = (new or "").strip()
    src = get_artist(old)
    if src is None or not new_name:
        return None
    old_id, old_name = str(src["artist_id"]), src["name"]
    with _transaction() as cur:
        # Every track credited to the old artist → rewrite its filename label.
        cur.execute(
            "SELECT t.track_id, t.filename FROM tracks t "
            "JOIN track_artists ta ON ta.track_id = t.track_id "
            "WHERE ta.artist_id = %s;", (old_id,))
        tracks = cur.fetchall()
        for t in tracks:
            new_fn = _rewrite_artist_in_filename(t["filename"], old_name, new_name)
            if new_fn != t["filename"]:
                cur.execute("UPDATE tracks SET filename = %s WHERE track_id = %s;",
                            (new_fn, t["track_id"]))
        # Does the target name already exist as a DIFFERENT artist? → merge into it.
        cur.execute("SELECT artist_id FROM artists WHERE name = %s;", (new_name,))
        dst = cur.fetchone()
        merged = bool(dst and str(dst["artist_id"]) != old_id)
        if merged:
            survivor = str(dst["artist_id"])
            # Re-point the old artist's links to the survivor (skip dups), then drop
            # the old row — CASCADE clears any leftover links and its centroid row.
            cur.execute(
                "INSERT INTO track_artists (track_id, artist_id, role) "
                "SELECT track_id, %s, role FROM track_artists WHERE artist_id = %s "
                "ON CONFLICT DO NOTHING;", (survivor, old_id))
            cur.execute("DELETE FROM artists WHERE artist_id = %s;", (old_id,))
        else:
            survivor = old_id
            cur.execute("UPDATE artists SET name = %s WHERE artist_id = %s;",
                        (new_name, old_id))
    logger.info("rename_artist: '%s' -> '%s' (%d tracks, merged=%s)",
                old_name, new_name, len(tracks), merged)
    return {"artist_id": survivor, "old_name": old_name, "new_name": new_name,
            "affected_tracks": len(tracks), "merged": merged}


@requires_db
def backfill_artists() -> dict:
    """Parse every track's filename → upsert artists → link them. Idempotent.

    Returns {"tracks": n_with_artist, "artists": distinct, "links": total}.
    """
    with _transaction() as cur:
        cur.execute("SELECT track_id, filename FROM tracks;")
        rows = cur.fetchall()
    n_tracks, n_links, seen = 0, 0, set()
    for r in rows:
        names = _parse_artist_names(r["filename"])
        if not names:
            continue
        n_tracks += 1
        for nm in names:
            aid = upsert_artist(nm)
            seen.add(aid)
            link_track_artist(str(r["track_id"]), aid)
            n_links += 1
    logger.info("backfill_artists: %d tracks, %d artists, %d links",
                n_tracks, len(seen), n_links)
    return {"tracks": n_tracks, "artists": len(seen), "links": n_links}


@requires_db
def get_artist(name_or_id: str) -> "dict | None":
    """Fetch one artist by name or UUID."""
    with _transaction() as cur:
        cur.execute("SELECT * FROM artists WHERE name = %s;", (name_or_id,))
        row = cur.fetchone()
        if row is None:
            try:
                cur.execute("SELECT * FROM artists WHERE artist_id = %s;", (name_or_id,))
                row = cur.fetchone()
            except Exception:
                row = None
    return dict(row) if row else None


@requires_db
def list_artists() -> list:
    """All artists with their track counts, alphabetical."""
    with _transaction() as cur:
        cur.execute(
            """SELECT a.artist_id, a.name, COUNT(ta.track_id) AS n_tracks
                 FROM artists a LEFT JOIN track_artists ta USING (artist_id)
                GROUP BY a.artist_id ORDER BY a.name;""")
        return [dict(r) for r in cur.fetchall()]


@requires_db
def artist_track_vectors(artist_id: str, model_version: str) -> list:
    """EffNet vectors of an artist's tracks, for centroid pooling (mean-pool)."""
    with _transaction() as cur:
        cur.execute(
            """SELECT DISTINCT ON (e.track_id) e.embedding::text AS vec
                 FROM track_artists ta
                 JOIN embeddings_effnet e ON e.track_id = ta.track_id
                WHERE ta.artist_id = %s AND e.model_version = %s;""",
            (artist_id, model_version))
        rows = cur.fetchall()
    return [_literal_to_vec(r["vec"]) for r in rows]


@requires_db
def upsert_artist_embedding(artist_id: str, vector, model_version: str,
                            n_tracks: int) -> None:
    """Store an artist's centroid vector (one per artist+model)."""
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO embeddings_artist
                      (artist_id, embedding, model_version, n_tracks)
               VALUES (%s, %s::vector, %s, %s)
               ON CONFLICT (artist_id, model_version) DO UPDATE
                   SET embedding  = EXCLUDED.embedding,
                       n_tracks   = EXCLUDED.n_tracks,
                       created_at = now();""",
            (artist_id, _vec_to_literal(vector), model_version, n_tracks))


@requires_db
def find_similar_artists(query_vector, n: int = 5,
                         exclude_artist_id: str = None) -> list:
    """ANN over artist centroids — 'artists who sound like this'.

    Returns {artist_id, name, n_tracks, cosine_distance} nearest-first.
    """
    exclude = "AND ea.artist_id <> %s" if exclude_artist_id else ""
    params = [_vec_to_literal(query_vector)]
    if exclude_artist_id:
        params.append(exclude_artist_id)
    params.append(n)
    with _transaction() as cur:
        cur.execute(
            f"""SELECT a.artist_id, a.name, ea.n_tracks,
                       ea.embedding <=> %s::vector AS cosine_distance
                  FROM embeddings_artist ea
                  JOIN artists a USING (artist_id)
                 WHERE TRUE {exclude}
                 ORDER BY cosine_distance ASC
                 LIMIT %s;""", params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── helpers for the AI assistant tools (Phase 1) ──────────────────────────────
@requires_db
def get_track_embedding(track_id: str, model_version: str) -> "list | None":
    """One track's EffNet vector (for 'tracks like this track')."""
    with _transaction() as cur:
        cur.execute("SELECT embedding::text AS v FROM embeddings_effnet "
                    "WHERE track_id = %s AND model_version = %s;", (track_id, model_version))
        row = cur.fetchone()
    return _literal_to_vec(row["v"]) if row else None


@requires_db
def get_artist_embedding(artist_id: str, model_version: str) -> "list | None":
    """An artist's stored centroid vector (for 'tracks/artists like this artist')."""
    with _transaction() as cur:
        cur.execute("SELECT embedding::text AS v FROM embeddings_artist "
                    "WHERE artist_id = %s AND model_version = %s;", (artist_id, model_version))
        row = cur.fetchone()
    return _literal_to_vec(row["v"]) if row else None


# ── labels (Phase 3b) — mirror the artist helpers exactly ─────────────────────
@requires_db
def upsert_label(name: str, discogs_id=None) -> str:
    """Insert a label by name (idempotent on UNIQUE name). Returns label_id."""
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO labels (name, discogs_id) VALUES (%s, %s)
               ON CONFLICT (name) DO UPDATE
                   SET discogs_id = COALESCE(EXCLUDED.discogs_id, labels.discogs_id)
               RETURNING label_id;""", (name, discogs_id))
        return str(cur.fetchone()["label_id"])


@requires_db
def link_track_label(track_id: str, label_id: str) -> None:
    """Link a track to a label (many-to-many; idempotent)."""
    with _transaction() as cur:
        cur.execute("INSERT INTO track_labels (track_id, label_id) VALUES (%s,%s) "
                    "ON CONFLICT DO NOTHING;", (track_id, label_id))


@requires_db
def relink_track_label(track_id: str, label_name: str, discogs_id=None) -> list:
    """Point a track at exactly one label (clears old links). Returns affected
    label_ids (old ∪ new) so the caller can refresh their centroids."""
    affected = set()
    with _transaction() as cur:
        cur.execute("SELECT label_id FROM track_labels WHERE track_id = %s;", (track_id,))
        affected.update(str(r["label_id"]) for r in cur.fetchall())
        cur.execute("DELETE FROM track_labels WHERE track_id = %s;", (track_id,))
    if label_name:
        lid = upsert_label(label_name, discogs_id)
        link_track_label(track_id, lid)
        affected.add(lid)
    return list(affected)


@requires_db
def get_label(name_or_id: str) -> "dict | None":
    """Fetch one label by name or UUID."""
    with _transaction() as cur:
        cur.execute("SELECT * FROM labels WHERE name = %s;", (name_or_id,))
        row = cur.fetchone()
        if row is None:
            try:
                cur.execute("SELECT * FROM labels WHERE label_id = %s;", (name_or_id,))
                row = cur.fetchone()
            except Exception:
                row = None
    return dict(row) if row else None


@requires_db
def list_labels() -> list:
    """All labels with their track counts, alphabetical."""
    with _transaction() as cur:
        cur.execute(
            """SELECT l.label_id, l.name, COUNT(tl.track_id) AS n_tracks
                 FROM labels l LEFT JOIN track_labels tl USING (label_id)
                GROUP BY l.label_id ORDER BY l.name;""")
        return [dict(r) for r in cur.fetchall()]


@requires_db
def label_track_vectors(label_id: str, model_version: str) -> list:
    """EffNet vectors of a label's tracks, for centroid pooling (mean-pool)."""
    with _transaction() as cur:
        cur.execute(
            """SELECT DISTINCT ON (e.track_id) e.embedding::text AS vec
                 FROM track_labels tl
                 JOIN embeddings_effnet e ON e.track_id = tl.track_id
                WHERE tl.label_id = %s AND e.model_version = %s;""",
            (label_id, model_version))
        rows = cur.fetchall()
    return [_literal_to_vec(r["vec"]) for r in rows]


@requires_db
def upsert_label_embedding(label_id: str, vector, model_version: str,
                           n_tracks: int) -> None:
    """Store a label's centroid vector (one per label+model)."""
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO embeddings_label
                      (label_id, embedding, model_version, n_tracks)
               VALUES (%s, %s::vector, %s, %s)
               ON CONFLICT (label_id, model_version) DO UPDATE
                   SET embedding  = EXCLUDED.embedding,
                       n_tracks   = EXCLUDED.n_tracks,
                       created_at = now();""",
            (label_id, _vec_to_literal(vector), model_version, n_tracks))


@requires_db
def get_label_embedding(label_id: str, model_version: str) -> "list | None":
    """A label's stored centroid vector."""
    with _transaction() as cur:
        cur.execute("SELECT embedding::text AS v FROM embeddings_label "
                    "WHERE label_id = %s AND model_version = %s;", (label_id, model_version))
        row = cur.fetchone()
    return _literal_to_vec(row["v"]) if row else None


@requires_db
def find_similar_labels(query_vector, n: int = 5, exclude_label_id: str = None) -> list:
    """ANN over label centroids — 'labels that sound like this'.
    Returns {label_id, name, n_tracks, cosine_distance} nearest-first."""
    exclude = "AND el.label_id <> %s" if exclude_label_id else ""
    params = [_vec_to_literal(query_vector)]
    if exclude_label_id:
        params.append(exclude_label_id)
    params.append(n)
    with _transaction() as cur:
        cur.execute(
            f"""SELECT l.label_id, l.name, el.n_tracks,
                       el.embedding <=> %s::vector AS cosine_distance
                  FROM embeddings_label el
                  JOIN labels l USING (label_id)
                 WHERE TRUE {exclude}
                 ORDER BY cosine_distance ASC
                 LIMIT %s;""", params)
        return [dict(r) for r in cur.fetchall()]


# ── track_discogs: the enrichment record + auto/confirm-doubtful queue ────────
@requires_db
def upsert_track_discogs(track_id: str, **fields) -> None:
    """Insert/replace a track's Discogs enrichment row. Pass any of: release_id,
    master_id, label, catno, year, country, genres, styles, cover_url,
    cover_path, status, confidence, candidates."""
    cols = ["release_id", "master_id", "label", "catno", "year", "country",
            "genres", "styles", "cover_url", "cover_path", "status",
            "confidence", "candidates"]
    vals = {c: fields.get(c) for c in cols}
    if vals["candidates"] is not None:
        vals["candidates"] = _json_dumps(vals["candidates"])
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO track_discogs
                   (track_id, release_id, master_id, label, catno, year, country,
                    genres, styles, cover_url, cover_path, status, confidence,
                    candidates, matched_at)
               VALUES (%(track_id)s, %(release_id)s, %(master_id)s, %(label)s,
                       %(catno)s, %(year)s, %(country)s,
                       COALESCE(%(genres)s::text[], '{}'), COALESCE(%(styles)s::text[], '{}'),
                       %(cover_url)s, %(cover_path)s,
                       COALESCE(%(status)s, 'unmatched'), %(confidence)s,
                       COALESCE(%(candidates)s::jsonb, '[]'), now())
               ON CONFLICT (track_id) DO UPDATE SET
                       release_id = EXCLUDED.release_id,
                       master_id  = EXCLUDED.master_id,
                       label      = EXCLUDED.label,
                       catno      = EXCLUDED.catno,
                       year       = EXCLUDED.year,
                       country    = EXCLUDED.country,
                       genres     = EXCLUDED.genres,
                       styles     = EXCLUDED.styles,
                       cover_url  = EXCLUDED.cover_url,
                       cover_path = EXCLUDED.cover_path,
                       status     = EXCLUDED.status,
                       confidence = EXCLUDED.confidence,
                       candidates = EXCLUDED.candidates,
                       matched_at = now();""",
            {"track_id": track_id, **vals})


@requires_db
def set_track_label(track_id: str, label: str) -> None:
    """Set ONLY the Discogs label for a track (the inline-edited Label column),
    preserving any other enrichment (cover, year, styles). Creates a minimal row
    when the track has none yet. Use upsert_track_discogs for a full enrichment."""
    label = (label or "").strip() or None
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO track_discogs (track_id, label, status)
                   VALUES (%s, %s, 'manual')
               ON CONFLICT (track_id) DO UPDATE SET label = EXCLUDED.label;""",
            (track_id, label))


@requires_db
def set_track_cover(track_id: str, cover_url: str, cover_path: str) -> None:
    """Set ONLY the cover fields (the auto cover re-search after an inline edit),
    preserving the rest of the enrichment row. Creates a minimal row if none yet."""
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO track_discogs (track_id, cover_url, cover_path, status)
                   VALUES (%s, %s, %s, 'matched')
               ON CONFLICT (track_id) DO UPDATE SET
                   cover_url = EXCLUDED.cover_url, cover_path = EXCLUDED.cover_path;""",
            (track_id, cover_url, cover_path))


@requires_db
def get_track_discogs(track_id: str) -> "dict | None":
    """One track's Discogs enrichment row, or None."""
    with _transaction() as cur:
        cur.execute("SELECT * FROM track_discogs WHERE track_id = %s;", (track_id,))
        row = cur.fetchone()
    return dict(row) if row else None


@requires_db
def track_discogs_map(track_ids: list) -> dict:
    """Bulk-fetch enrichment for many tracks at once (avoids N+1 in listings).
    Returns {track_id: {label, year, styles, status, has_cover}}."""
    if not track_ids:
        return {}
    with _transaction() as cur:
        cur.execute(
            """SELECT track_id, label, year, styles, status,
                      (cover_path IS NOT NULL) AS has_cover
                 FROM track_discogs WHERE track_id = ANY(%s::uuid[]);""",
            (list(track_ids),))
        return {str(r["track_id"]): {"label": r["label"], "year": r["year"],
                                     "styles": r["styles"], "status": r["status"],
                                     "has_cover": r["has_cover"]}
                for r in cur.fetchall()}


@requires_db
def set_track_discogs_status(track_id: str, status: str) -> bool:
    """Update only the status (e.g. user 'skipped' a doubtful match)."""
    with _transaction() as cur:
        cur.execute("UPDATE track_discogs SET status = %s WHERE track_id = %s;",
                    (status, track_id))
        return cur.rowcount > 0


@requires_db
def discogs_queue(status: str = "doubtful") -> list:
    """Tracks awaiting review (default the doubtful ones), with their candidates
    and filename so the UI can render the confirm queue."""
    with _transaction() as cur:
        cur.execute(
            """SELECT td.*, t.filename
                 FROM track_discogs td JOIN tracks t USING (track_id)
                WHERE td.status = %s
                ORDER BY td.confidence DESC NULLS LAST;""", (status,))
        return [dict(r) for r in cur.fetchall()]


@requires_db
def tracks_pending_discogs() -> list:
    """Analyzed tracks with no Discogs row yet (or still unmatched) — the work
    list for a batch enrich. Returns {track_id, filename}."""
    with _transaction() as cur:
        cur.execute(
            """SELECT t.track_id, t.filename
                 FROM tracks t
                 LEFT JOIN track_discogs td USING (track_id)
                WHERE t.analyzed_at IS NOT NULL
                  AND (td.track_id IS NULL OR td.status = 'unmatched')
                ORDER BY t.added_at;""")
        return [dict(r) for r in cur.fetchall()]


@requires_db
def find_track_by_query(text: str) -> "dict | None":
    """Best-effort resolve a track by a fuzzy filename match (ILIKE)."""
    with _transaction() as cur:
        cur.execute("SELECT * FROM tracks WHERE filename ILIKE %s "
                    "ORDER BY length(filename) ASC LIMIT 1;", (f"%{text}%",))
        row = cur.fetchone()
    return dict(row) if row else None


@requires_db
def track_camelots(track_ids: list) -> dict:
    """Bulk map track_id -> Camelot key (features->>'camelot'), in one query.

    Used to stamp the harmonic key onto session tracklists, whose denormalised
    JSONB snapshot stores only filename/time/method — so the key shows in the
    session view (and stays current) without an N+1 lookup per row.
    """
    ids = [str(t) for t in track_ids if t]
    if not ids:
        return {}
    with _transaction() as cur:
        cur.execute(
            "SELECT track_id, features->>'camelot' AS camelot "
            "FROM tracks WHERE track_id = ANY(%s::uuid[]);", (ids,))
        return {str(r["track_id"]): r["camelot"] for r in cur.fetchall()}


@requires_db
def search_tracks(artist: str = None, bpm_min: float = None, bpm_max: float = None,
                  camelot: str = None, on_spot: bool = None, limit: int = 20) -> list:
    """Structured metadata search over the catalogue (the assistant's Tool 2).

    All filters optional and AND-combined. Returns track rows with their parsed
    features — the LLM formats them.
    """
    clauses, params = [], []
    if artist:
        clauses.append("t.track_id IN (SELECT ta.track_id FROM track_artists ta "
                       "JOIN artists a USING (artist_id) WHERE a.name ILIKE %s)")
        params.append(f"%{artist}%")
    if bpm_min is not None:
        clauses.append("(t.features->>'bpm')::float >= %s"); params.append(bpm_min)
    if bpm_max is not None:
        clauses.append("(t.features->>'bpm')::float <= %s"); params.append(bpm_max)
    if camelot:
        clauses.append("t.features->>'camelot' = %s"); params.append(camelot)
    if on_spot is not None:
        clauses.append("t.on_spot = %s"); params.append(on_spot)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(min(limit, 50))
    with _transaction() as cur:
        cur.execute(f"SELECT t.* FROM tracks t {where} "
                    f"ORDER BY (t.features->>'bpm')::float NULLS LAST LIMIT %s;", params)
        return [dict(r) for r in cur.fetchall()]


@requires_db
def track_artist_names(track_id: str) -> list:
    """The artist name(s) credited on a track."""
    with _transaction() as cur:
        cur.execute("SELECT a.name FROM track_artists ta JOIN artists a USING (artist_id) "
                    "WHERE ta.track_id = %s ORDER BY ta.role;", (track_id,))
        return [r["name"] for r in cur.fetchall()]


@requires_db
def delete_session(session_id: str) -> None:
    """Drop a session and (via CASCADE) its play log — the user declined to save."""
    with _transaction() as cur:
        cur.execute("DELETE FROM mix_sessions WHERE session_id = %s;", (session_id,))
    logger.info("delete_session: %s", session_id)


@requires_db
def purge_unnamed_sessions() -> int:
    """Delete every unsaved (name IS NULL) session — crash/abandon leftovers.

    Called when a new Live Mode session starts: anything unnamed by then was
    never consented to, so it must not accumulate.
    """
    with _transaction() as cur:
        cur.execute("DELETE FROM mix_sessions WHERE name IS NULL;")
        n = cur.rowcount
    if n:
        logger.info("purge_unnamed_sessions: removed %d abandoned session(s)", n)
    return n


@requires_db
def list_sessions(crate_id: str = None) -> list:
    """Saved sessions (name IS NOT NULL), newest first, with track counts.

    crate_id scopes the list to sessions played FROM that crate — the per-crate
    SESSIONS view. None returns every saved session (the global fallback).
    """
    crate_clause = "AND s.crate_id = %s" if crate_id else ""
    params = [crate_id] if crate_id else []
    with _transaction() as cur:
        cur.execute(
            f"""
            SELECT s.session_id, s.name, s.started_at, s.ended_at, s.crate_id,
                   c.name AS crate_name,
                   COUNT(st.track_id) AS n_tracks
              FROM mix_sessions s
              LEFT JOIN crates c USING (crate_id)
              LEFT JOIN session_tracks st USING (session_id)
             WHERE s.name IS NOT NULL {crate_clause}
             GROUP BY s.session_id, c.name
             ORDER BY s.started_at DESC;
            """, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


@requires_db
def crate_affinity(track_id: str, crate_id: str = None) -> list:
    """Cosine similarity of every crate track's EffNet embedding vs `track_id`.

    One SQL pass — the 'Afinidad' column of the Live Mode reference list.
    DISTINCT ON keeps one row per track when several model versions coexist.

    Returns:
        list of {track_id, affinity} with affinity in [0, 1] (1 = identical
        direction), unordered; the UI sorts client-side.
    """
    mclause, mparams = _member_clause(crate_id, alias="t")   # default → all; user → crate_tracks
    crate_clause = f"AND {mclause}" if mclause else ""
    params = [track_id] + mparams
    with _transaction() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (t.track_id)
                   t.track_id,
                   1 - (e.embedding <=> q.embedding) AS affinity
              FROM embeddings_effnet e
              JOIN tracks t USING (track_id),
                   (SELECT embedding FROM embeddings_effnet
                     WHERE track_id = %s LIMIT 1) q
             WHERE TRUE {crate_clause}
             ORDER BY t.track_id;
            """,
            params)
        rows = cur.fetchall()
    return [{"track_id": str(r["track_id"]), "affinity": float(r["affinity"])}
            for r in rows]


def _json_dumps(obj) -> str:
    """json.dumps with default=str — coerces datetime/UUID into JSONB-safe strings."""
    import json
    return json.dumps(obj, default=str)


# ── knowledge base ops (Phase 2 — RAG) ───────────────────────────────────────
@requires_db
def kb_document_by_hash(content_hash: str) -> "dict | None":
    """Look up a previously ingested document by content hash (dedup guard)."""
    with _transaction() as cur:
        cur.execute("SELECT * FROM kb_documents WHERE content_hash = %s;",
                    (content_hash,))
        row = cur.fetchone()
    return dict(row) if row else None


@requires_db
def insert_kb_document(title: str, chunks: list, model_name: str,
                       source_type: str = "upload", source_url: str = None,
                       lang: str = None, content_hash: str = None,
                       category: str = None, tags: list = None,
                       meta: dict = None) -> str:
    """Insert a document and all its embedded chunks in one transaction.

    `chunks` is a list of {text, embedding, token_count}. `category`/`tags`/`meta`
    carry the heterogeneous classification. Returns the new doc_id. The whole
    thing is atomic: a document never half-lands in the index.
    """
    with _transaction() as cur:
        cur.execute(
            """INSERT INTO kb_documents
                      (title, source_type, source_url, lang, category, tags, meta,
                       content_hash, n_chunks)
               VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
               RETURNING doc_id;""",
            (title, source_type, source_url, lang, category, tags or [],
             _json_dumps(meta or {}), content_hash, len(chunks)))
        doc_id = cur.fetchone()["doc_id"]
        for i, ch in enumerate(chunks):
            cur.execute(
                """INSERT INTO kb_chunks
                          (doc_id, chunk_index, text, embedding, model_name, token_count)
                   VALUES (%s, %s, %s, %s::vector, %s, %s);""",
                (doc_id, i, ch["text"], _vec_to_literal(ch["embedding"]),
                 model_name, ch.get("token_count")))
    return str(doc_id)


@requires_db
def update_kb_document(doc_id: str, title: str = None, category: str = None,
                       tags: list = None) -> bool:
    """Edit a document's curation fields (the 'editable' half of auto+editable).
    Only the provided fields change. Returns True if the document exists."""
    sets, params = [], []
    if title is not None:
        sets.append("title = %s"); params.append(title)
    if category is not None:
        sets.append("category = %s"); params.append(category or None)
    if tags is not None:
        sets.append("tags = %s"); params.append(tags)
    if not sets:
        return True
    params.append(doc_id)
    with _transaction() as cur:
        cur.execute(f"UPDATE kb_documents SET {', '.join(sets)} WHERE doc_id = %s;",
                    params)
        return cur.rowcount > 0


@requires_db
def search_kb_chunks(query_vector, n: int = 5, max_distance: float = None,
                     category: str = None, tags: list = None) -> list:
    """ANN over text chunks — the retrieval half of RAG.

    Returns {chunk_id, doc_id, title, category, chunk_index, text,
    cosine_distance} nearest-first. `max_distance` drops weak matches (a cheap
    hallucination guard); `category`/`tags` optionally scope retrieval to one
    kind of knowledge while still using the single shared collection.
    """
    qv = _vec_to_literal(query_vector)
    where, params = [], [qv]                       # %s #1 = SELECT distance
    if max_distance is not None:
        where.append("(kc.embedding <=> %s::vector) <= %s")
        params.extend([qv, max_distance])
    if category:
        where.append("d.category = %s"); params.append(category)
    if tags:
        where.append("d.tags && %s"); params.append(tags)     # array overlap
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(n)
    with _transaction() as cur:
        cur.execute(
            f"""SELECT kc.chunk_id, kc.doc_id, d.title, d.category, kc.chunk_index,
                       kc.text, kc.embedding <=> %s::vector AS cosine_distance
                  FROM kb_chunks kc
                  JOIN kb_documents d USING (doc_id)
                  {clause}
                 ORDER BY cosine_distance ASC
                 LIMIT %s;""", params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


@requires_db
def list_kb_documents(category: str = None) -> list:
    """All ingested documents, newest first, for the knowledge manager UI.
    Optionally filtered to one category."""
    where = "WHERE category = %s" if category else ""
    params = [category] if category else []
    with _transaction() as cur:
        cur.execute(
            f"""SELECT doc_id, title, source_type, source_url, lang, category, tags,
                      n_chunks, ingested_at
                 FROM kb_documents
                 {where}
                ORDER BY ingested_at DESC;""", params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


@requires_db
def kb_categories() -> list:
    """Distinct categories present, with document counts — drives the UI filter
    and tells the assistant what kinds of knowledge exist. Newest-used first."""
    with _transaction() as cur:
        cur.execute(
            """SELECT category, count(*) AS n
                 FROM kb_documents
                WHERE category IS NOT NULL AND category <> ''
                GROUP BY category
                ORDER BY n DESC, category ASC;""")
        rows = cur.fetchall()
    return [dict(r) for r in rows]


@requires_db
def delete_kb_document(doc_id: str) -> bool:
    """Remove a document and its chunks (ON DELETE CASCADE). True if it existed."""
    with _transaction() as cur:
        cur.execute("DELETE FROM kb_documents WHERE doc_id = %s;", (doc_id,))
        return cur.rowcount > 0


@requires_db
def kb_stats() -> dict:
    """Document + chunk counts for the assistant status panel."""
    with _transaction() as cur:
        cur.execute("SELECT count(*) AS docs FROM kb_documents;")
        docs = cur.fetchone()["docs"]
        cur.execute("SELECT count(*) AS chunks FROM kb_chunks;")
        chunks = cur.fetchone()["chunks"]
    return {"documents": docs, "chunks": chunks}


# ── reference web sources + their search cache (assistant web scouting) ─────────
@requires_db
def insert_web_source(url: str, topic: str, note: str = None) -> str:
    """Register a website the assistant may search live. Returns the source_id."""
    with _transaction() as cur:
        cur.execute("INSERT INTO web_sources (url, topic, note) "
                    "VALUES (%s, %s, %s) RETURNING source_id;", (url, topic, note))
        return str(cur.fetchone()["source_id"])


@requires_db
def list_web_sources() -> list:
    """Registered reference sources, newest first (for the Knowledge UI + the tool)."""
    with _transaction() as cur:
        cur.execute("SELECT source_id, url, topic, note, created_at "
                    "FROM web_sources ORDER BY created_at DESC;")
        return [dict(r) for r in cur.fetchall()]


@requires_db
def delete_web_source(source_id: str) -> bool:
    """Remove a source and its cached rows (ON DELETE CASCADE). True if it existed."""
    with _transaction() as cur:
        cur.execute("DELETE FROM web_sources WHERE source_id = %s;", (source_id,))
        return cur.rowcount > 0


@requires_db
def insert_web_cache(rows: list) -> int:
    """Store embedded web snippets / page snapshots. Each row is {source_id?, query?,
    title?, url?, text, embedding}. Returns how many were inserted."""
    if not rows:
        return 0
    with _transaction() as cur:
        for r in rows:
            cur.execute(
                "INSERT INTO web_cache (source_id, query, title, url, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s::vector);",
                (r.get("source_id"), r.get("query"), r.get("title"), r.get("url"),
                 r["text"], _vec_to_literal(r["embedding"])))
    return len(rows)


@requires_db
def search_web_cache(query_vector, n: int = 5, max_distance: float = None) -> list:
    """ANN over the web cache — the semantic fallback when the live web is down.
    Returns {cache_id, source_id, query, title, url, text, cosine_distance}."""
    qv = _vec_to_literal(query_vector)
    where, params = [], [qv]
    if max_distance is not None:
        where.append("(embedding <=> %s::vector) <= %s"); params.extend([qv, max_distance])
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(n)
    with _transaction() as cur:
        cur.execute(
            f"""SELECT cache_id, source_id, query, title, url, text,
                       embedding <=> %s::vector AS cosine_distance
                  FROM web_cache
                  {clause}
                 ORDER BY cosine_distance ASC
                 LIMIT %s;""", params)
        return [dict(r) for r in cur.fetchall()]


@requires_db
def evict_web_cache(cap: int) -> int:
    """Bound the cache: delete everything older than the newest `cap` rows. Returns
    how many were evicted (keeps the rudimentary cache from growing without limit)."""
    with _transaction() as cur:
        cur.execute(
            """DELETE FROM web_cache WHERE cache_id IN (
                   SELECT cache_id FROM web_cache ORDER BY created_at DESC OFFSET %s
               );""", (max(0, cap),))
        return cur.rowcount


@requires_db
def web_cache_count() -> int:
    with _transaction() as cur:
        cur.execute("SELECT count(*) AS n FROM web_cache;")
        return cur.fetchone()["n"]


# ════════════════════════════════════════════════════════════
#  IMPORT-TIME INIT GUARD
# ════════════════════════════════════════════════════════════
# Connect + db_init() run exactly once, the first time this module is imported
# anywhere in the project. On failure DB_AVAILABLE stays False and the CRITICAL
# log already told the user to start Docker — imports still succeed so tooling
# (linters, --help) doesn't explode just because the DB is down.
_connect()


# ════════════════════════════════════════════════════════════
#  REFACTOR NOTES
# ════════════════════════════════════════════════════════════
# Structure (top → bottom): logging/config → module state + DBUnavailableError →
# connection management → schema DDL/init/health → vector helpers → track CRUD →
# embedding ops → session ops → import-time init guard.
#
# Deduplication performed:
#   • `_transaction` context manager is the ONE place transaction lifecycle
#     (borrow → cursor → commit/rollback → return-to-pool) is written. Every
#     public op uses it; no function hand-rolls getconn/putconn or commit.
#   • `requires_db` decorator centralises the DB_AVAILABLE short-circuit so the
#     fail-loud check isn't copy-pasted into ~20 function bodies.
#   • `_vec_to_literal` isolates the pgvector text protocol so no SQL string
#     elsewhere has to know the '[...]' wire format.
#   • `_session_tracklist` is shared by close_session() and get_session() so the
#     play-log query can never diverge between the two.
#   • All upserts follow one ON CONFLICT (PK) DO UPDATE shape for idempotent
#     re-analysis; insert_track uses the same trick on its UNIQUE crate_path.
#
# Deliberate choices:
#   • No ORM, no pgvector Python adapter — raw psycopg2 + a tiny text codec keeps
#     dependencies at psycopg2 + python-dotenv and the SQL fully visible.
#   • Vectors are never logged (size only) per the logging spec.
#   • analyzed_at IS NULL is the single "pending" marker (+ partial index).
