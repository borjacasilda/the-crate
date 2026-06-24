"""
The Crate MCP Server
==================
A Model Context Protocol server that exposes The Crate's music-analysis engine
(analyze.py) to any MCP-compatible AI agent. It lets an agent reason about a
music collection of ANY genre: extract per-track musical features (tempo, key,
energy, timbre, mood/emotional scores, mix points), find and rank compatible
tracks, evaluate transitions with explanations, build setlists, and track a
live session — all as structured JSON, not formatted console text.

The server imports analyze.py as a module and keeps the Essentia models warm in
one long-lived process, so an agent's repeated calls are far cheaper than
shelling out to the CLI per request. Track features are cached in-process keyed
by the resolved absolute path, which is also the `track_id` an agent passes
between tools (analyze_track → get_recommendations → compare_tracks → …).


## TOOL DESIGN DECISIONS

The guiding rule: expose what an AGENT needs to *reason* about music, not what a
CLI user needs to *see*. Agents want structured data + short interpretive
strings; they do not want print output or 1280-float vectors. Every analyze.py
symbol is classified below.

Feature extraction & caching
  extract_features ............ EXPOSE (tool: analyze_track / analyze_folder)
                                via an in-process cache keyed by resolved path.
  embed_effnet ................ INTERNAL — used only inside find_similar.
  TrackFeatures (dataclass) ... INTERNAL — never returned raw; _feature_dict()
                                filters it (drops energy_curve/mfcc/bark/effnet
                                /genre/jamendo vectors unless verbose=True).

Scoring & similarity (the engine)
  mix_score ................... INTERNAL — wrapped by recommend/compare/setlist.
  sample_by_score ............. INTERNAL — temperature sampling for recommend/setlist.
  bpm_compatibility / bpm_delta INTERNAL — feed BPM interpretation strings.
  cosine_sim / timbre_compatibility .. INTERNAL.
  track_energy / energy_compatibility / energy_direction /
  density_continuity .......... INTERNAL — energy summaries & modifiers.
  emotional_vector_similarity . INTERNAL — wrapped by compare_emotional.
  key_relationship_label ...... INTERNAL — harmonic interpretation.
  mix_tip ..................... INTERNAL — wrapped by get_mix_technique / transitions.
  to_camelot / camelot_energy_direction .. INTERNAL.

Config / model registry
  ModelManager ................ INTERNAL + EXPOSE as a resource/tool (model_status,
                                thecrate://models/status) for pipeline level + missing models.
  ModifierStrengths / MODE_CONFIG / ENERGY_TARGETS / MODIFIER_NAMES /
  PERFECT_MIX_THRESHOLD ....... INTERNAL — drive mode/energy/temperature params.
  persist_embeddings / _db_persist .. INTERNAL — writing to the crate is crate.py's
                                job, not the agent's.

CLI / display (all OMIT — print formatted text, useless to an agent)
  cmd_analyze, cmd_scan, cmd_mixpoints, cmd_next, cmd_compare, cmd_setlist,
  cmd_download, print_pipeline_banner, build_parser, main, _format_modifiers,
  _format_breakdown, _print_picks.

Private DB/cache helpers reused by tools (INTERNAL)
  _get_or_analyze, _db_lookup, _hydrate, _load_library, _ensure_strengths,
  _model_version.

Library reads (recommend/search/overview/list) DO go through analyze._load_library,
which reads the analysed crate from PostgreSQL — that is the collection the agent
reasons over. Only the SESSION tools are forbidden from touching database.py:
they are ephemeral, in-process memory for a single agent conversation.


## RETURN-SHAPE CONTRACT (applies to every tool)
  * Always a JSON-serialisable dict; never a TrackFeatures dataclass.
  * Errors never raise to the agent — they return {"success": false, "error": "..."}
    with an actionable message.
  * Score-bearing returns also carry interpretive strings (bpm, harmonic, energy,
    mix tip, emotional character, pipeline-level meaning).
  * Default returns omit big internal arrays; verbose=True adds the full feature set.
  * track_id == the resolved absolute path (the cache key), so tools compose.
"""
import argparse
import asyncio
import dataclasses
import functools
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from pydantic import Field

from mcp.server.fastmcp import FastMCP

import analyze
import database

logger = logging.getLogger("thecrate.mcp")

# ════════════════════════════════════════════════════════════
#  CONSTANTS & SHARED STATE
# ════════════════════════════════════════════════════════════
SERVER_NAME = "thecrate"
VERSION = "1.0.0"
SERVER_DESCRIPTION = (
    "The Crate — a music-analysis engine for AI agents. Extracts tempo, key/Camelot, "
    "energy, timbre, mood and emotional features from audio files; finds and ranks "
    "compatible tracks (by sound, or by harmonic key + tempo); explains why two tracks do "
    "or don't mix; judges how far to trust a detected key; detects when two files are the "
    "same record at different turntable speeds; builds setlists; and tracks a live play "
    "session. Works for any genre. Call analyze_track first to get a track_id (the file's "
    "absolute path), then pass it to the recommendation, comparison, key, mix-point and "
    "emotional tools. Scores come with plain-language "
    "interpretations. Library-wide tools require the PostgreSQL crate to be running."
)

# In-process feature cache: resolved absolute path -> analyze.TrackFeatures.
# This is the agent-facing cache and the source of `track_id` composability.
_FEATURE_CACHE: "dict[str, analyze.TrackFeatures]" = {}

# Ephemeral, in-memory session memory (NOT the database — see module docstring).
# {"current": session_id|None, "sessions": {sid: {...}}}
SESSION_STATE: dict = {"current": None, "sessions": {}}

# Essentia model instances (analyze.ModelManager._instances) are NOT safe to call
# concurrently, so all CPU-bound analysis runs on a SINGLE worker — calls serialise
# rather than racing a shared TF graph. The event loop stays free meanwhile.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thecrate-analysis")

mcp = FastMCP(SERVER_NAME, instructions=SERVER_DESCRIPTION)


# ════════════════════════════════════════════════════════════
#  CORE HELPERS
# ════════════════════════════════════════════════════════════
async def _run(fn, *args, **kwargs):
    """Run a blocking analyze.py/DB call off the event loop on the single worker.

    Essentia and psycopg2 are synchronous and CPU/IO-bound; awaiting them here
    keeps MCP responsive while guaranteeing analysis never runs two-at-a-time.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, functools.partial(fn, *args, **kwargs))


def _error(msg: str) -> dict:
    """Standard agent-readable error envelope (never raises to the agent)."""
    return {"success": False, "error": msg}


def _require_db():
    """Return an error dict when the library DB is down, else None.

    Single-track analysis works without it; only library-wide operations need it.
    """
    if not database.DB_AVAILABLE:
        return _error(
            "The track library database is unavailable, so library-wide operations "
            "can't run. Start it with `docker compose up -d` and retry. Single-file "
            "analysis (analyze_track) still works without it."
        )
    return None


def _now() -> str:
    """UTC ISO-8601 timestamp for session logs."""
    return datetime.now(timezone.utc).isoformat()


# Mood/characterisation fields, in display order, that _feature_dict surfaces when present.
_MOOD_FIELDS = (
    "mood_aggressive", "danceability_nn",
    "mood_electronic", "mood_sad", "mood_relaxed", "mood_happy", "mood_party",
    "jamendo_dark", "jamendo_groovy", "jamendo_meditative",
    "jamendo_energetic", "jamendo_heavy", "jamendo_space",
    "voice_instrumental", "tonal", "approachability", "engagement", "timbre_bright",
)

_LEVEL_MEANING = {
    1: "Level 1 — classic DSP only (tempo, key, energy, timbre). No ML.",
    2: "Level 2 — + EffNet 1280-D embedding (true 'musical world' similarity) + TempoCNN.",
    3: "Level 3 — + neural mood-aggressive and danceability.",
    4: "Level 4 — + full emotional fingerprint (5 moods + Jamendo mood/theme axes).",
    5: "Level 5 — + genre, voice/tonal, approachability, engagement, timbre, instruments.",
}


def _feature_dict(f: "analyze.TrackFeatures", track_id: str, verbose: bool = False) -> dict:
    """Filter a TrackFeatures into an agent-readable dict.

    Always includes core identity/tempo/key/energy fields; adds mood scores and
    emotional-vector presence when the track reached the ML levels. Big internal
    arrays (curves, MFCC/Bark, EffNet/genre/Jamendo vectors) are excluded unless
    verbose=True, in which case the complete dataclass is attached under "full".

    Args:
        f:        the analysed TrackFeatures record.
        track_id: the resolved absolute path used as the agent-facing id.
        verbose:  attach the full, unfiltered feature set.
    Returns:
        A JSON-serialisable dict.
    """
    d = {
        "track_id": track_id,
        "path": track_id,
        "filename": Path(track_id).name,
        "duration_sec": round(f.duration, 2),
        "bpm": round(f.bpm, 2),
        "bpm_confidence": round(f.bpm_confidence, 3),
        "key": f.key,
        "scale": f.scale,
        "camelot": f.camelot,
        "key_strength": round(f.key_strength, 3),
        "key_agreement": round(f.agreement, 3),   # 1.0 = key profiles unanimous; 0.0 = pre-vote row.
        "pipeline_level": f.pipeline_level,
        "energy": round(analyze.track_energy(f), 4),
        "intro_end_sec": round(f.intro_end, 1),
        "outro_start_sec": round(f.outro_start, 1),
        "has_emotional_vector": f.emotional_vector is not None,
        "emotional_vector_dims": len(f.emotional_vector) if f.emotional_vector else 0,
    }
    moods = {name: round(v, 3) for name in _MOOD_FIELDS
             if (v := getattr(f, name, None)) is not None}
    if moods:
        d["mood_scores"] = moods
    if verbose:
        d["full"] = dataclasses.asdict(f)
    return d


def _camelot_neighbors(camelot: str) -> dict:
    """The harmonically compatible Camelot codes for mixing into `camelot`.

    The Camelot wheel is laid out in perfect fifths, so the classic DJ rule is:
    same code (no clash), the relative major/minor (toggle A/B — a mood shift), and
    the two adjacent numbers on the same letter (±1 fifth — an energy lift/relax).
    Numbers wrap 12↔1. Returns {} for an atonal/unparseable code.
    """
    c = (camelot or "").upper().strip()
    if len(c) < 2 or c[-1] not in ("A", "B") or not c[:-1].isdigit():
        return {}
    num, letter = int(c[:-1]), c[-1]
    if not 1 <= num <= 12:
        return {}
    other = "B" if letter == "A" else "A"
    return {
        "same_key": f"{num}{letter}",
        "relative": f"{num}{other}",                 # major/minor swap — mood shift
        "adjacent_up": f"{num % 12 + 1}{letter}",    # +1 fifth — energy lift
        "adjacent_down": f"{(num - 2) % 12 + 1}{letter}",  # -1 fifth — relax
    }


def _key_interp(f: "analyze.TrackFeatures") -> str:
    """Key read that also reports the multi-profile vote trust (agreement)."""
    base = (f"{f.key} {f.scale} → Camelot {f.camelot}; "
            + ("clear tonal centre" if f.key_strength >= 0.5
               else "weak tonal centre — effectively atonal, mixes harmonically with anything"))
    if f.agreement >= 0.999:
        base += "; key profiles unanimous (high trust)"
    elif f.agreement > 0:
        base += f"; key profiles split ({f.agreement:.0%} agreement) — treat the key as tentative"
    return base


def _interpretations(f: "analyze.TrackFeatures") -> dict:
    """Short plain-language reads of a single track for agent context."""
    out = {
        "pipeline_level": _LEVEL_MEANING.get(f.pipeline_level, "unknown"),
        "tempo": f"{f.bpm:.0f} BPM "
                 + ("(high-confidence)" if f.bpm_confidence >= 0.7 else "(low-confidence; treat as approximate)"),
        "key": _key_interp(f),
        "energy": _energy_level_text(analyze.track_energy(f)),
    }
    if f.pipeline_level >= 3:
        out["emotional_character"] = _emotional_character(f)
    if f.intro_end and f.bpm > 0:
        intro_bars = f.intro_end * f.bpm / (60 * 4)
        out["mixability"] = (f"~{intro_bars:.0f}-bar mixable intro; "
                             + ("long, easy to blend into" if intro_bars >= 16 else "short intro — cue carefully"))
    return out


def _energy_level_text(e: float) -> str:
    if e == 0:
        return "energy unavailable"
    if e < 0.1:
        return f"low/sparse energy ({e:.3f}) — opener or breakdown territory"
    if e < 0.25:
        return f"moderate energy ({e:.3f}) — mid-set groove"
    return f"high energy ({e:.3f}) — peak-time driving"


def _emotional_character(f: "analyze.TrackFeatures") -> str:
    """Summarise the dominant emotional axes that are present."""
    labels = {
        "mood_aggressive": "aggressive", "mood_party": "party", "mood_relaxed": "relaxed",
        "mood_sad": "melancholic", "mood_happy": "happy", "mood_electronic": "electronic",
        "jamendo_dark": "dark", "jamendo_groovy": "groovy", "jamendo_meditative": "hypnotic",
        "jamendo_energetic": "energetic", "jamendo_heavy": "heavy", "jamendo_space": "spacious",
    }
    present = [(labels[k], getattr(f, k)) for k in labels if getattr(f, k, None) is not None]
    strong = sorted([(lbl, v) for lbl, v in present if v >= 0.5], key=lambda x: x[1], reverse=True)
    if not strong:
        return "even emotional profile — no single axis dominates"
    return "predominantly " + ", ".join(f"{lbl} ({v:.2f})" for lbl, v in strong[:4])


def _bpm_interp(a, b) -> str:
    d = analyze.bpm_delta(a.bpm, b.bpm)
    comp = analyze.bpm_compatibility(a.bpm, b.bpm)
    if abs(d) <= 4:
        q = "very close — beatmatchable with minimal pitch"
    elif abs(d) <= 8:
        q = "pushable within a turntable's ±8% pitch range, but you'll ride the fader"
    else:
        q = "a wide tempo gap — pitch hard, or bridge with a percussive loop"
    return f"Δ{d:+.1f} BPM ({a.bpm:.0f}→{b.bpm:.0f}); {q}. compatibility {comp:.2f}/1.0"


def _harmonic_interp(key_rel: str) -> str:
    table = {
        "Same key": "same key — nothing fights; overlap freely",
        "Adjacent": "one step on the Camelot wheel — smooth, slight lift",
        "Relative (mood shift)": "relative major/minor — emotional colour change; mask it over a breakdown",
        "Dissonant": "keys don't agree — usable as tension, but not a clean blend",
        "Unknown": "at least one track is atonal — harmonically neutral, mixes with anything",
    }
    return table.get(key_rel, key_rel)


def _energy_interp(dir_label: str, pct: float) -> str:
    base = {"build": "energy rises", "flat": "energy holds level", "drop": "energy falls",
            "unknown": "energy direction unknown"}.get(dir_label, dir_label)
    return f"{base} ({pct:+.0f}%) — " + (
        "ideal forward motion" if dir_label in ("build", "flat")
        else "a drop can deflate the floor; hard-cut rather than long-blend" if dir_label == "drop"
        else "")


def _overall_interp(s: dict) -> str:
    t = s["total"]
    if t >= analyze.PERFECT_MIX_THRESHOLD:
        return f"strong pairing ({t:.2f}) — flagged a perfect mix"
    if t >= 0.6:
        return f"solid pairing ({t:.2f}) — workable with attention to the flagged modifiers"
    if t >= 0.35:
        return f"weak pairing ({t:.2f}) — they live in different sonic worlds"
    return f"poor pairing ({t:.2f}) — the EffNet/timbre base similarity is low"


def _track_features(track_id: str):
    """Resolve a track_id to (abs_path, TrackFeatures), analysing on a cache miss.

    Cache order: in-process cache → PostgreSQL (crate hit, hydrated) → fresh
    extraction. Returns (abs_path, None) when no file exists at the path.
    """
    abs_path = str(Path(track_id).resolve())
    cached = _FEATURE_CACHE.get(abs_path)
    if cached is not None:
        return abs_path, cached
    _, f = analyze._db_lookup(abs_path)        # crate hit → hydrate from DB.
    if f is None:
        if not Path(abs_path).exists():
            return abs_path, None
        f = analyze.extract_features(track_id)  # expensive — runs once, then cached.
    _FEATURE_CACHE[abs_path] = f
    return abs_path, f


def _row_brief(row: dict) -> dict:
    """Compact metadata for a `tracks` DB row (no curves/vectors)."""
    feat = row.get("features") or {}
    return {
        "track_id": str(Path(row["crate_path"]).resolve()),
        "filename": row.get("filename"),
        "bpm": round(feat.get("bpm", 0.0), 1),
        "camelot": feat.get("camelot", "?"),
        "key": feat.get("key"),
        "scale": feat.get("scale"),
        "pipeline_level": row.get("pipeline_level"),
    }


def _resolve_strengths(mode: str, energy: str):
    """Mode preset + energy-direction target, ready for analyze.mix_score."""
    strengths = analyze._ensure_strengths(mode)
    strengths.energy_target = analyze.ENERGY_TARGETS.get(energy, 0.0)
    return strengths


def _pick_dict(current, path: str, f, score: dict) -> dict:
    """A scored candidate: filtered features + breakdown + pairwise interpretation."""
    bpm_d = analyze.bpm_delta(current.bpm, f.bpm)
    key_rel = analyze.key_relationship_label(current.camelot, f.camelot)
    _, dir_label, energy_pct = analyze.energy_direction(current, f)
    return {
        **_feature_dict(f, str(Path(path).resolve())),
        "score": round(score["total"], 3),
        "perfect_mix": score["total"] >= analyze.PERFECT_MIX_THRESHOLD,
        "base_similarity": round(score["effnet_base"], 3),
        "base_source": score["timbre_source"],
        "modifiers": {k: round(score[k], 3) for k in analyze.MODIFIER_NAMES},
        "bpm_delta": round(bpm_d, 2),
        "key_relationship": key_rel,
        "energy_direction": dir_label,
        "energy_change_pct": round(energy_pct, 1),
        "mix_tip": analyze.mix_tip(score, key_rel, bpm_d, dir_label),
    }


# ════════════════════════════════════════════════════════════
#  CATEGORY 1 — TRACK ANALYSIS
# ════════════════════════════════════════════════════════════
def _analyze_impl(path: str, verbose: bool) -> dict:
    abs_path, f = _track_features(path)
    if f is None:
        return _error(f"No audio file found at '{path}'. Pass an absolute or working-directory "
                      f"path to a .wav/.mp3/.flac file.")
    out = _feature_dict(f, abs_path, verbose)
    out["interpretations"] = _interpretations(f)
    return out


@mcp.tool()
async def analyze_track(
    path: Annotated[str, Field(description="Path to a .wav/.mp3/.flac audio file (absolute, or relative to the server's working directory).")],
    verbose: Annotated[bool, Field(description="Include the full feature set (energy curve, MFCC/Bark, EffNet & genre/mood vectors). Default False returns only agent-readable insights.")] = False,
) -> dict:
    """Extract musical features from an audio file: BPM + confidence, key/Camelot, energy,
    timbre, mix points, and (if the ML models are installed) mood and emotional scores.

    Call this FIRST for any track before comparing, recommending, or describing it. It
    returns a `track_id` (the file's resolved absolute path) to pass to the other tools.
    Expensive on first call (~5–15 s for the full ML pipeline); results are cached in the
    server process, so repeated calls on the same file are instant. Returns interpretive
    strings alongside the raw numbers."""
    try:
        return await _run(_analyze_impl, path, verbose)
    except Exception as e:
        return _error(f"Analysis failed for '{path}': {e}")


def _analyze_folder_impl(folder: str, verbose: bool) -> dict:
    base = Path(folder)
    if not base.is_dir():
        return _error(f"'{folder}' is not a directory.")
    files = sorted(base.glob("*.wav")) + sorted(base.glob("*.mp3")) + sorted(base.glob("*.flac"))
    if not files:
        return _error(f"No .wav/.mp3/.flac files in '{folder}'.")
    tracks, failed = [], []
    for w in files:
        try:
            _, f = _track_features(str(w))
            if f is None:
                failed.append({"file": w.name, "error": "could not read"})
            else:
                tracks.append(_feature_dict(f, str(w.resolve()), verbose))
        except Exception as e:
            failed.append({"file": w.name, "error": str(e)})
    return {"folder": str(base.resolve()), "analyzed": len(tracks),
            "failed": failed, "tracks": tracks}


@mcp.tool()
async def analyze_folder(
    folder: Annotated[str, Field(description="Directory containing audio files (non-recursive).")],
    verbose: Annotated[bool, Field(description="Include the full feature set per track.")] = False,
) -> dict:
    """Analyse every .wav/.mp3/.flac in a folder and return their features as a list.

    Use to bulk-load a collection an agent will then reason over. Each file is analysed
    once and cached; tracks that fail to read are reported under `failed` rather than
    aborting the batch. For a large library, prefer the library_* tools which read the
    already-indexed crate from the database."""
    try:
        return await _run(_analyze_folder_impl, folder, verbose)
    except Exception as e:
        return _error(f"Folder analysis failed: {e}")


def _cached_impl(track_id: str, verbose: bool) -> dict:
    abs_path = str(Path(track_id).resolve())
    f = _FEATURE_CACHE.get(abs_path)
    source = "memory"
    if f is None:
        _, f = analyze._db_lookup(abs_path)
        source = "database"
    if f is None:
        return _error("This track hasn't been analysed yet. Call analyze_track(path) first.")
    out = _feature_dict(f, abs_path, verbose)
    out["cache_source"] = source
    out["interpretations"] = _interpretations(f)
    return out


@mcp.tool()
async def get_cached_features(
    track_id: Annotated[str, Field(description="A track_id (resolved path) returned by analyze_track.")],
    verbose: Annotated[bool, Field(description="Include the full feature set.")] = False,
) -> dict:
    """Return a track's already-computed features WITHOUT re-analysing it.

    Use when you have a track_id and want its data again cheaply. Errors if the track was
    never analysed — call analyze_track first in that case."""
    try:
        return await _run(_cached_impl, track_id, verbose)
    except Exception as e:
        return _error(f"Lookup failed: {e}")


def _is_cached_impl(track_id: str) -> dict:
    abs_path = str(Path(track_id).resolve())
    if abs_path in _FEATURE_CACHE:
        return {"track_id": abs_path, "cached": True, "source": "memory"}
    _, f = analyze._db_lookup(abs_path)
    if f is not None:
        return {"track_id": abs_path, "cached": True, "source": "database"}
    return {"track_id": abs_path, "cached": False, "source": None}


@mcp.tool()
async def is_track_cached(
    track_id: Annotated[str, Field(description="A track path / track_id to check.")],
) -> dict:
    """Check whether a track is already analysed (in process memory or the crate DB).

    Cheap — use to decide whether analyze_track will be instant or will pay the full
    extraction cost."""
    try:
        return await _run(_is_cached_impl, track_id)
    except Exception as e:
        return _error(f"Check failed: {e}")


def _harmonic_mixing_text(camelot: str) -> str:
    n = _camelot_neighbors(camelot)
    if not n:
        return "atonal/unknown key — harmonically neutral, blends with anything"
    return (f"clean with {n['same_key']}, lift to {n['adjacent_up']}, relax to "
            f"{n['adjacent_down']}, or shift mood to the relative {n['relative']}")


def _key_analysis_impl(track_id: str) -> dict:
    abs_path, f = _track_features(track_id)
    if f is None:
        return _error(f"No audio file at '{track_id}'.")
    cents = analyze._tuning_cents(f.tuning_frequency)
    off_pitch = abs(cents) > 20
    if f.agreement >= 0.999:
        trust = "all key profiles agreed — high confidence in the detected key"
    elif f.agreement > 0:
        trust = (f"{f.agreement:.0%} of key profiles agreed — moderate confidence; "
                 "treat the key as tentative")
    else:
        trust = ("no multi-profile vote on record (re-analyse with the current engine "
                 "for a key-trust score)")
    return {
        "track_id": abs_path,
        "key": f.key, "scale": f.scale, "camelot": f.camelot,
        "key_strength": round(f.key_strength, 3),
        "key_agreement": round(f.agreement, 3),
        "tuning_frequency_hz": round(f.tuning_frequency, 2) if f.tuning_frequency else None,
        "tuning_cents_off_440": round(cents, 1),
        "vinyl_speed_suspect": off_pitch,
        "compatible_camelot": _camelot_neighbors(f.camelot),
        "interpretation": {
            "tonal_certainty": ("clear tonal centre" if f.key_strength >= 0.5
                                else "weak/atonal — mixes harmonically with anything"),
            "vote_trust": trust,
            "tuning": ("≈concert pitch (A=440), standard tuning" if not off_pitch
                       else f"{cents:+.0f} cents off A=440 — likely a vinyl rip at an off platter "
                            "speed; the key was tuning-corrected before detection"),
            "harmonic_mixing": _harmonic_mixing_text(f.camelot),
        },
    }


@mcp.tool()
async def get_key_analysis(
    track_id: Annotated[str, Field(description="track_id (from analyze_track) whose tonal/harmonic report you want.")],
) -> dict:
    """Focused tonal + harmonic-mixing report for a track.

    Surfaces the key/Camelot, the tonal certainty (key_strength) AND the multi-profile vote
    trust (key_agreement — how strongly independent key profiles concurred, which catches
    confidently-wrong detections that key_strength alone misses), the reference tuning in Hz
    with its cents offset from A=440 (a vinyl rip at an off platter speed reads sharp/flat,
    flagged as vinyl_speed_suspect), and the Camelot-compatible codes to mix into (same /
    adjacent ±a fifth / relative). Use for harmonic mixing and to judge how far to trust the
    detected key."""
    try:
        return await _run(_key_analysis_impl, track_id)
    except Exception as e:
        return _error(f"Key analysis failed: {e}")


# ════════════════════════════════════════════════════════════
#  CATEGORY 2 — SIMILARITY & RECOMMENDATION
# ════════════════════════════════════════════════════════════
def _recommend_impl(track_id, mode, n, energy, temperature, exclude_ids) -> dict:
    err = _require_db()
    if err:
        return err
    abs_path, current = _track_features(track_id)
    if current is None:
        return _error(f"No audio file at '{track_id}'.")
    library = analyze._load_library(exclude_path=track_id)
    excluded = {str(Path(t).resolve()) for t in (exclude_ids or [])}
    cand = [(p, f) for p, f in library if str(Path(p).resolve()) not in excluded]
    if not cand:
        return _error("No other analysed tracks in the library to recommend. Index the crate "
                      "(crate.py / analyze_folder) first.")
    strengths = _resolve_strengths(mode, energy)
    scored = [(p, f, analyze.mix_score(current, f, mode=mode, strengths=strengths)) for p, f in cand]
    picks = analyze.sample_by_score(scored, n, temperature)
    return {
        "now_playing": _feature_dict(current, abs_path),
        "mode": mode, "energy_direction": energy, "temperature": temperature,
        "count": len(picks),
        "recommendations": [_pick_dict(current, p, f, s) for p, f, s in picks],
    }


@mcp.tool()
async def get_recommendations(
    track_id: Annotated[str, Field(description="track_id of the currently-playing track (from analyze_track).")],
    mode: Annotated[str, Field(description="Scoring preset: 'safe' (strict BPM/key), 'balanced' (default), or 'creative' (vibe-led, mood-contrast).")] = "balanced",
    n: Annotated[int, Field(description="How many next-track picks to return.")] = 5,
    energy: Annotated[str, Field(description="Desired energy direction for the next track: 'up', 'stable' (default), or 'down'.")] = "stable",
    temperature: Annotated[float, Field(description="0.0 = the single best picks (deterministic); higher (e.g. 0.7) samples for more adventurous variety.")] = 0.0,
    exclude: Annotated[list[str], Field(description="track_ids to exclude (e.g. already played this set).")] = [],
) -> dict:
    """Rank the best next tracks to mix into the given track, from the indexed library.

    The score is an immutable EffNet 'musical world' similarity multiplied by DJ-tunable
    modifiers (bpm, key, energy direction, transition window, mood, emotional, density).
    Use after analyze_track to answer 'what should I play next?'. Each pick carries its
    score breakdown, BPM delta, key relationship, energy direction and a one-line mix tip.
    Requires the library database."""
    try:
        return await _run(_recommend_impl, track_id, mode, n, energy, temperature, exclude)
    except Exception as e:
        return _error(f"Recommendation failed: {e}")


def _compare_impl(a_id, b_id, mode) -> dict:
    abs_a, a = _track_features(a_id)
    abs_b, b = _track_features(b_id)
    if a is None or b is None:
        return _error("One or both tracks could not be found/analysed.")
    s = analyze.mix_score(a, b, mode=mode)
    bpm_d = analyze.bpm_delta(a.bpm, b.bpm)
    key_rel = analyze.key_relationship_label(a.camelot, b.camelot)
    _, dir_label, energy_pct = analyze.energy_direction(a, b)
    return {
        "track_a": _feature_dict(a, abs_a),
        "track_b": _feature_dict(b, abs_b),
        "mode": mode,
        "total_score": round(s["total"], 3),
        "perfect_mix": s["total"] >= analyze.PERFECT_MIX_THRESHOLD,
        "base_similarity": round(s["effnet_base"], 3),
        "base_source": s["timbre_source"],
        "modifiers": {k: round(s[k], 3) for k in analyze.MODIFIER_NAMES},
        "bpm_delta": round(bpm_d, 2),
        "key_relationship": key_rel,
        "energy_direction": dir_label,
        "energy_change_pct": round(energy_pct, 1),
        "mix_tip": analyze.mix_tip(s, key_rel, bpm_d, dir_label),
        "interpretations": {
            "bpm": _bpm_interp(a, b),
            "harmonic": _harmonic_interp(key_rel),
            "energy": _energy_interp(dir_label, energy_pct),
            "overall": _overall_interp(s),
        },
    }


@mcp.tool()
async def compare_tracks(
    track_id_a: Annotated[str, Field(description="track_id of the outgoing / first track.")],
    track_id_b: Annotated[str, Field(description="track_id of the incoming / second track.")],
    mode: Annotated[str, Field(description="Scoring preset: 'safe' | 'balanced' | 'creative'.")] = "balanced",
) -> dict:
    """Full compatibility breakdown between two specific tracks, with reasoning.

    Returns the total score, the immutable base similarity, every modifier value, the BPM
    delta, key relationship, energy direction, a mix tip, and plain-language
    interpretations of each. Use to answer 'do these two work together, and why?'."""
    try:
        return await _run(_compare_impl, track_id_a, track_id_b, mode)
    except Exception as e:
        return _error(f"Comparison failed: {e}")


def _vinyl_offset_impl(a_id: str, b_id: str) -> dict:
    import math
    abs_a, a = _track_features(a_id)
    abs_b, b = _track_features(b_id)
    if a is None or b is None:
        return _error("One or both tracks could not be found/analysed.")
    if not (a.tuning_frequency > 0 and b.tuning_frequency > 0 and a.bpm > 0 and b.bpm > 0):
        return _error("Need tempo + reference tuning on BOTH tracks (analyse them with the "
                      "current engine first).")
    tuning_ratio = b.tuning_frequency / a.tuning_frequency
    bpm_ratio = b.bpm / a.bpm
    cents = 1200.0 * math.log2(tuning_ratio)
    speed_pct = (tuning_ratio - 1.0) * 100.0
    proportional = abs(bpm_ratio - tuning_ratio) < 0.005
    same_record = (0.003 < abs(tuning_ratio - 1.0) < 0.03) and proportional
    if same_record:
        interp = (f"B is almost certainly the SAME recording as A, played ~{abs(speed_pct):.2f}% "
                  f"{'faster' if speed_pct > 0 else 'slower'}: tempo and pitch shift PROPORTIONALLY, "
                  f"which is a platter-speed difference (e.g. a vinyl rip vs the digital master), not "
                  f"a different track. Pitch B by {-speed_pct:+.2f}% to align them.")
    else:
        interp = ("Tempo and tuning do NOT shift proportionally — these read as different recordings "
                  "(or any speed difference is negligible).")
    return {
        "track_a": Path(abs_a).name, "track_b": Path(abs_b).name,
        "bpm_a": round(a.bpm, 2), "bpm_b": round(b.bpm, 2),
        "tuning_a_hz": round(a.tuning_frequency, 2), "tuning_b_hz": round(b.tuning_frequency, 2),
        "bpm_ratio": round(bpm_ratio, 4), "tuning_ratio": round(tuning_ratio, 4),
        "speed_offset_pct": round(speed_pct, 2), "cents": round(cents, 1),
        "same_record_at_different_speed": same_record,
        "interpretation": interp,
    }


@mcp.tool()
async def detect_vinyl_speed_offset(
    track_id_a: Annotated[str, Field(description="track_id of the reference version (e.g. the digital master).")],
    track_id_b: Annotated[str, Field(description="track_id of the other version (e.g. a vinyl rip).")],
) -> dict:
    """Tell whether two files are the SAME recording at a different platter speed.

    Answers the vinyl-vs-digital question: a turntable running off-nominal scales BOTH tempo
    and pitch by the same factor, so when A and B differ a small, PROPORTIONAL amount in BPM
    AND reference tuning, it is one record at two speeds — not two tracks. Returns the speed
    offset (% and cents), the BPM/tuning ratios, a same_record_at_different_speed verdict, and
    how much to pitch B to align. Needs tempo + tuning on both (any pipeline level)."""
    try:
        return await _run(_vinyl_offset_impl, track_id_a, track_id_b)
    except Exception as e:
        return _error(f"Vinyl-speed comparison failed: {e}")


def _find_similar_impl(track_id, n) -> dict:
    err = _require_db()
    if err:
        return err
    abs_path, f = _track_features(track_id)
    if f is None:
        return _error(f"No audio file at '{track_id}'.")
    if not f.effnet_embedding:
        return _error("This track has no EffNet embedding (pipeline level < 2). Install "
                      "essentia-tensorflow and run `python analyze.py download`, then re-analyze.")
    rows = database.find_similar_effnet(f.effnet_embedding, n=n + 1)
    similar = []
    for r in rows:
        rp = str(Path(r["crate_path"]).resolve())
        if rp == abs_path:
            continue
        dist = float(r.get("cosine_distance", 2.0))
        similar.append({**_row_brief(r),
                        "cosine_distance": round(dist, 4),
                        "similarity": round(max(0.0, 1.0 - dist / 2.0), 3)})
        if len(similar) >= n:
            break
    return {
        "query": abs_path,
        "count": len(similar),
        "similar": similar,
        "interpretation": ("Ranked by EffNet 'musical world' distance: cosine_distance 0 = "
                           "identical direction, 2 = opposite. `similarity` is a 0–1 convenience "
                           "score. This is pure semantic/timbral closeness, independent of BPM or key."),
    }


@mcp.tool()
async def find_similar(
    track_id: Annotated[str, Field(description="track_id whose sonic neighbours you want.")],
    n: Annotated[int, Field(description="Number of nearest neighbours to return.")] = 5,
) -> dict:
    """Semantic search: find the library tracks that sound most like this one (EffNet vector).

    This is similarity in 'musical world' space — atmosphere, production, scene — NOT tempo
    or key. Use for 'find me more like this'. Requires the library database and a track
    analysed at pipeline level ≥ 2."""
    try:
        return await _run(_find_similar_impl, track_id, n)
    except Exception as e:
        return _error(f"Similarity search failed: {e}")


def _search_impl(bpm_min, bpm_max, camelot, key, scale, min_level, limit) -> dict:
    err = _require_db()
    if err:
        return err
    lib = analyze._load_library()
    res = []
    for p, f in lib:
        if bpm_min is not None and f.bpm < bpm_min:
            continue
        if bpm_max is not None and f.bpm > bpm_max:
            continue
        if camelot and f.camelot.upper() != camelot.upper():
            continue
        if key and f.key != key:
            continue
        if scale and f.scale != scale:
            continue
        if min_level and f.pipeline_level < min_level:
            continue
        res.append(_feature_dict(f, str(Path(p).resolve())))
    return {
        "count": len(res),
        "filters": {"bpm_min": bpm_min, "bpm_max": bpm_max, "camelot": camelot,
                    "key": key, "scale": scale, "min_pipeline_level": min_level},
        "tracks": res[:limit],
    }


@mcp.tool()
async def search_tracks(
    bpm_min: Annotated[Optional[float], Field(description="Minimum BPM (inclusive).")] = None,
    bpm_max: Annotated[Optional[float], Field(description="Maximum BPM (inclusive).")] = None,
    camelot: Annotated[Optional[str], Field(description="Exact Camelot code, e.g. '8A'.")] = None,
    key: Annotated[Optional[str], Field(description="Pitch class, e.g. 'A' (use with scale).")] = None,
    scale: Annotated[Optional[str], Field(description="'major' or 'minor'.")] = None,
    min_pipeline_level: Annotated[Optional[int], Field(description="Only tracks analysed at this ML level or higher (1–5).")] = None,
    limit: Annotated[int, Field(description="Max tracks to return.")] = 50,
) -> dict:
    """Filter the indexed library by hard criteria (BPM range, key/Camelot, min ML level).

    Use for precise lookups like 'tracks in A minor between 128 and 132 BPM'. All filters
    are optional and ANDed. Requires the library database."""
    try:
        return await _run(_search_impl, bpm_min, bpm_max, camelot, key, scale, min_pipeline_level, limit)
    except Exception as e:
        return _error(f"Search failed: {e}")


def _harmonic_matches_impl(track_id, bpm_tolerance, include_relative, limit) -> dict:
    err = _require_db()
    if err:
        return err
    abs_path, current = _track_features(track_id)
    if current is None:
        return _error(f"No audio file at '{track_id}'.")
    neighbors = _camelot_neighbors(current.camelot)
    if not neighbors:
        return _error(f"Track '{Path(abs_path).name}' has no detected key (atonal/unanalysed) — "
                      "nothing to harmonically match against.")
    compatible = {neighbors["same_key"], neighbors["adjacent_up"], neighbors["adjacent_down"]}
    if include_relative:
        compatible.add(neighbors["relative"])
    matches = []
    for p, f in analyze._load_library(exclude_path=track_id):
        if f.camelot not in compatible:
            continue
        bpm_d = analyze.bpm_delta(current.bpm, f.bpm)
        if abs(bpm_d) > bpm_tolerance:
            continue
        d = _feature_dict(f, str(Path(p).resolve()))
        d["key_relationship"] = analyze.key_relationship_label(current.camelot, f.camelot)
        d["bpm_delta"] = round(bpm_d, 2)
        matches.append(d)
    rank = {"Same key": 0, "Adjacent": 1, "Relative (mood shift)": 2}
    matches.sort(key=lambda x: (rank.get(x["key_relationship"], 3), abs(x["bpm_delta"])))
    return {
        "query": abs_path, "current_camelot": current.camelot,
        "current_bpm": round(current.bpm, 1), "bpm_tolerance": bpm_tolerance,
        "compatible_camelot": sorted(compatible), "count": len(matches),
        "matches": matches[:limit],
        "interpretation": ("Library tracks in a harmonically compatible Camelot key "
                           + ("(same/adjacent/relative)" if include_relative else "(same/adjacent)")
                           + f" and within ±{bpm_tolerance} BPM, ranked by harmonic closeness then "
                           "tempo proximity. Pure key+tempo (works at any pipeline level), independent "
                           "of the EffNet 'sounds-like' similarity get_recommendations uses."),
    }


@mcp.tool()
async def find_harmonic_matches(
    track_id: Annotated[str, Field(description="track_id whose harmonically-mixable partners you want.")],
    bpm_tolerance: Annotated[float, Field(description="Max |BPM difference| to allow (beatmatchable range).")] = 6.0,
    include_relative: Annotated[bool, Field(description="Also include the relative major/minor (a mood shift), not just same/adjacent keys.")] = True,
    limit: Annotated[int, Field(description="Max matches to return.")] = 25,
) -> dict:
    """Find library tracks that mix HARMONICALLY with this one (Camelot wheel + tempo).

    The classic harmonic-mixing query: tracks whose key is the same, one step adjacent (a
    fifth), or the relative major/minor, AND within a beatmatchable BPM window — ranked by
    harmonic closeness then tempo proximity. Pure key+tempo, so it works even for level-1
    tracks; complementary to get_recommendations (EffNet 'sounds-like'). Requires the library
    database."""
    try:
        return await _run(_harmonic_matches_impl, track_id, bpm_tolerance, include_relative, limit)
    except Exception as e:
        return _error(f"Harmonic match search failed: {e}")


# ════════════════════════════════════════════════════════════
#  CATEGORY 3 — MIX INTELLIGENCE
# ════════════════════════════════════════════════════════════
def _mix_points_impl(track_id) -> dict:
    abs_path, f = _track_features(track_id)
    if f is None:
        return _error(f"No audio file at '{track_id}'.")
    outro_len = max(0.0, f.duration - f.outro_start)
    bars_per_sec = f.bpm / (60 * 4) if f.bpm > 0 else 0.0
    intro_bars = f.intro_end * bars_per_sec
    outro_bars = outro_len * bars_per_sec
    return {
        "track_id": abs_path,
        "duration_sec": round(f.duration, 1),
        "bpm": round(f.bpm, 1),
        "intro": {"end_sec": round(f.intro_end, 1), "bars": round(intro_bars, 1),
                  "phrases_32bar": round(intro_bars / 32, 2)},
        "outro": {"start_sec": round(f.outro_start, 1), "length_sec": round(outro_len, 1),
                  "bars": round(outro_bars, 1), "phrases_32bar": round(outro_bars / 32, 2)},
        "interpretation": (
            f"Mix in over the first {intro_bars:.0f} bars; mix out over the last "
            f"{outro_bars:.0f} bars. " + ("Generous windows — easy long blends." if min(intro_bars, outro_bars) >= 16
                                          else "Tight window(s) — cue carefully, consider a shorter blend.")
            if f.bpm > 0 else "BPM unknown — mix points given in seconds only."),
    }


@mcp.tool()
async def get_mix_points(
    track_id: Annotated[str, Field(description="track_id to get intro/outro mix zones for.")],
) -> dict:
    """Return a track's mixable intro and outro zones, in BOTH seconds and bars (and 32-bar phrases).

    Use to plan where to bring a track in and out. Bars are computed at the track's detected
    BPM in 4/4. Falls back to seconds-only when BPM is unknown."""
    try:
        return await _run(_mix_points_impl, track_id)
    except Exception as e:
        return _error(f"Mix-point detection failed: {e}")


def _setlist_impl(seed, length, mode, energy, temperature) -> dict:
    err = _require_db()
    if err:
        return err
    abs_seed, current = _track_features(seed)
    if current is None:
        return _error(f"No audio file at '{seed}'.")
    library = analyze._load_library(exclude_path=seed)
    if not library:
        return _error("The library has no other analysed tracks to build a set from.")
    strengths = _resolve_strengths(mode, energy)
    setlist = [(abs_seed, current)]
    used = {abs_seed}
    transitions = []
    for _ in range(max(0, length - 1)):
        cand = [(p, f) for p, f in library if str(Path(p).resolve()) not in used]
        if not cand:
            break
        prev = setlist[-1][1]
        scored = [(p, f, analyze.mix_score(prev, f, mode=mode, strengths=strengths)) for p, f in cand]
        for i, (p, f, s) in enumerate(scored):     # nudge toward rising density (energy arc).
            bonus = 0.1 if f.spectral_complexity > prev.spectral_complexity else 0.0
            scored[i] = (p, f, {**s, "total": s["total"] + bonus})
        best = analyze.sample_by_score(scored, 1, temperature)
        if not best:
            break
        bp, bf, bs = best[0]
        rp = str(Path(bp).resolve())
        setlist.append((rp, bf))
        used.add(rp)
        key_rel = analyze.key_relationship_label(prev.camelot, bf.camelot)
        bpm_d = analyze.bpm_delta(prev.bpm, bf.bpm)
        _, dir_label, _ = analyze.energy_direction(prev, bf)
        transitions.append({
            "from": Path(setlist[-2][0]).name, "to": Path(rp).name,
            "score": round(bs["total"], 3), "key_relationship": key_rel,
            "bpm_delta": round(bpm_d, 2), "energy_direction": dir_label,
            "mix_tip": analyze.mix_tip(bs, key_rel, bpm_d, dir_label),
            "needs_attention": bs["total"] < 0.6 or dir_label == "drop",
        })
    return {
        "seed": abs_seed, "requested_length": length, "actual_length": len(setlist),
        "mode": mode, "energy_direction": energy, "temperature": temperature,
        "tracklist": [_feature_dict(f, p) for p, f in setlist],
        "transitions": transitions,
        "note": ("Greedy/local — each step picks the best next track against the previous one "
                 "with no look-ahead. Treat as a strong first draft, not a global optimum."),
    }


@mcp.tool()
async def build_setlist(
    seed_track_id: Annotated[str, Field(description="track_id of the opening track.")],
    length: Annotated[int, Field(description="Total tracks in the set, including the seed.")] = 8,
    mode: Annotated[str, Field(description="Scoring preset: 'safe' | 'balanced' | 'creative'.")] = "balanced",
    energy: Annotated[str, Field(description="Energy direction to favour at each step: 'up' (default for a building set), 'stable', or 'down'.")] = "up",
    temperature: Annotated[float, Field(description="0.0 = deterministic best chain; higher samples for a different (still strong) set each run.")] = 0.0,
) -> dict:
    """Build a setlist by greedily chaining the best next track from a seed.

    Returns the ordered tracklist plus a transition analysis (score, key relationship, BPM
    delta, energy direction, mix tip, and a needs_attention flag for weak or
    energy-dropping joins). Greedy and local — a good draft, not a proven optimum.
    Requires the library database."""
    try:
        return await _run(_setlist_impl, seed_track_id, length, mode, energy, temperature)
    except Exception as e:
        return _error(f"Setlist build failed: {e}")


@mcp.tool()
async def evaluate_transition(
    track_id_a: Annotated[str, Field(description="track_id of the outgoing track.")],
    track_id_b: Annotated[str, Field(description="track_id of the incoming track.")],
    mode: Annotated[str, Field(description="Scoring preset: 'safe' | 'balanced' | 'creative'.")] = "balanced",
) -> dict:
    """Judge a specific A→B transition with full reasoning and a verdict.

    Like compare_tracks, but framed as a go/no-go: returns the same breakdown plus a
    `verdict` ('perfect' | 'good' | 'workable' | 'weak') so an agent can decide whether to
    recommend the move."""
    try:
        res = await _run(_compare_impl, track_id_a, track_id_b, mode)
        if res.get("success") is False:
            return res
        t = res["total_score"]
        res["verdict"] = ("perfect" if t >= analyze.PERFECT_MIX_THRESHOLD
                          else "good" if t >= 0.6 else "workable" if t >= 0.35 else "weak")
        return res
    except Exception as e:
        return _error(f"Transition evaluation failed: {e}")


def _technique_impl(a_id, b_id, mode) -> dict:
    abs_a, a = _track_features(a_id)
    abs_b, b = _track_features(b_id)
    if a is None or b is None:
        return _error("One or both tracks could not be found/analysed.")
    s = analyze.mix_score(a, b, mode=mode)
    bpm_d = analyze.bpm_delta(a.bpm, b.bpm)
    key_rel = analyze.key_relationship_label(a.camelot, b.camelot)
    _, dir_label, _ = analyze.energy_direction(a, b)
    return {
        "track_a": Path(abs_a).name, "track_b": Path(abs_b).name, "mode": mode,
        "technique": analyze.mix_tip(s, key_rel, bpm_d, dir_label),
        "key_relationship": key_rel,
        "bpm_delta": round(bpm_d, 2),
        "energy_direction": dir_label,
        "reasoning": (f"{_harmonic_interp(key_rel)}; {_bpm_interp(a, b)}; "
                      f"{_energy_interp(dir_label, 0.0).split(' — ')[0]}."),
    }


@mcp.tool()
async def get_mix_technique(
    track_id_a: Annotated[str, Field(description="track_id of the outgoing track.")],
    track_id_b: Annotated[str, Field(description="track_id of the incoming track.")],
    mode: Annotated[str, Field(description="Scoring preset: 'safe' | 'balanced' | 'creative'.")] = "balanced",
) -> dict:
    """Recommend HOW to mix A into B — the concrete hand technique, plus why.

    Returns a one-line instruction (e.g. 'EQ swap on bar 16…', 'long blend, keep kicks
    aligned', 'hard cut at end of phrase') chosen from the pair's key relationship, tempo
    gap and energy move, with a short rationale."""
    try:
        return await _run(_technique_impl, track_id_a, track_id_b, mode)
    except Exception as e:
        return _error(f"Mix-technique lookup failed: {e}")


# ════════════════════════════════════════════════════════════
#  CATEGORY 4 — EMOTIONAL & MOOD ANALYSIS
# ════════════════════════════════════════════════════════════
def _emotional_profile_impl(track_id) -> dict:
    abs_path, f = _track_features(track_id)
    if f is None:
        return _error(f"No audio file at '{track_id}'.")
    if f.pipeline_level < 3:
        return _error(f"This track is pipeline level {f.pipeline_level}; emotional/mood scores "
                      f"need level ≥ 3 (install essentia-tensorflow + download models, then re-analyze).")
    scores = {name: round(v, 3) for name in _MOOD_FIELDS
              if (v := getattr(f, name, None)) is not None}
    return {
        "track_id": abs_path,
        "pipeline_level": f.pipeline_level,
        "mood_scores": scores,
        "emotional_vector": {"available": f.emotional_vector is not None,
                             "dimensions": len(f.emotional_vector) if f.emotional_vector else 0},
        "character": _emotional_character(f),
    }


@mcp.tool()
async def get_emotional_profile(
    track_id: Annotated[str, Field(description="track_id whose emotional/mood profile you want.")],
) -> dict:
    """Return the full emotional/mood profile of a track (every available mood score).

    Includes mood-aggressive, danceability, the 5 extended moods, the 6 named Jamendo
    mood/theme axes (dark/groovy/meditative/energetic/heavy/space), and level-5 audio
    character (voice/tonal/timbre/approachability/engagement), plus a one-line character
    summary and the emotional-vector dimension count. Needs pipeline level ≥ 3."""
    try:
        return await _run(_emotional_profile_impl, track_id)
    except Exception as e:
        return _error(f"Emotional profile failed: {e}")


def _compare_emotional_impl(a_id, b_id) -> dict:
    abs_a, a = _track_features(a_id)
    abs_b, b = _track_features(b_id)
    if a is None or b is None:
        return _error("One or both tracks could not be found/analysed.")
    sim = analyze.emotional_vector_similarity(a, b)
    if sim is None:
        return _error("Not enough shared emotional components (need both tracks at pipeline "
                      "level ≥ 3/4 with overlapping mood scores).")
    shared = {}
    for name in _MOOD_FIELDS:
        va, vb = getattr(a, name, None), getattr(b, name, None)
        if va is not None and vb is not None:
            shared[name] = {"a": round(va, 3), "b": round(vb, 3), "delta": round(abs(va - vb), 3)}
    verdict = ("near-identical emotional register" if sim >= 0.9
               else "similar emotional register" if sim >= 0.7
               else "noticeably different register" if sim >= 0.4
               else "opposite emotional registers")
    return {
        "track_a": Path(abs_a).name, "track_b": Path(abs_b).name,
        "emotional_similarity": round(sim, 3),
        "shared_components": shared,
        "interpretation": f"{verdict} (cosine {sim:.2f} over {len(shared)} shared mood axes). "
                          "High = coherent mood match; low = a deliberate emotional contrast.",
    }


@mcp.tool()
async def compare_emotional(
    track_id_a: Annotated[str, Field(description="track_id of the first track.")],
    track_id_b: Annotated[str, Field(description="track_id of the second track.")],
) -> dict:
    """Compare the emotional fingerprints of two tracks (cosine over shared mood axes).

    Returns an overall emotional-similarity score, the per-axis values and deltas, and a
    verdict from 'near-identical' to 'opposite registers'. Use to keep a mood coherent — or
    to find a deliberate contrast. Both tracks need pipeline level ≥ 3."""
    try:
        return await _run(_compare_emotional_impl, track_id_a, track_id_b)
    except Exception as e:
        return _error(f"Emotional comparison failed: {e}")


# Emotional criterion -> TrackFeatures field.
_EMOTION_FIELDS = {
    "aggressive": "mood_aggressive", "party": "mood_party", "danceable": "danceability_nn",
    "dark": "jamendo_dark", "groovy": "jamendo_groovy", "meditative": "jamendo_meditative",
    "energetic": "jamendo_energetic", "heavy": "jamendo_heavy", "space": "jamendo_space",
}


def _find_by_emotion_impl(criteria: dict, limit: int) -> dict:
    err = _require_db()
    if err:
        return err
    active = {k: v for k, v in criteria.items() if v is not None}
    if not active:
        return _error("Provide at least one emotional minimum, e.g. dark=0.5 or groovy=0.6.")
    lib = analyze._load_library()
    matches = []
    for p, f in lib:
        score = 0.0
        ok = True
        for crit, minv in active.items():
            field = _EMOTION_FIELDS[crit]
            val = getattr(f, field, None)
            if val is None or val < minv:
                ok = False
                break
            score += val
        if ok:
            d = _feature_dict(f, str(Path(p).resolve()))
            d["match_strength"] = round(score / len(active), 3)
            matches.append(d)
    matches.sort(key=lambda x: x["match_strength"], reverse=True)
    return {"criteria": active, "count": len(matches), "tracks": matches[:limit]}


@mcp.tool()
async def find_by_emotion(
    aggressive: Annotated[Optional[float], Field(description="Min mood-aggressive [0–1].")] = None,
    dark: Annotated[Optional[float], Field(description="Min darkness [0–1].")] = None,
    groovy: Annotated[Optional[float], Field(description="Min groove [0–1].")] = None,
    energetic: Annotated[Optional[float], Field(description="Min raw energy [0–1].")] = None,
    heavy: Annotated[Optional[float], Field(description="Min heaviness/intensity [0–1].")] = None,
    meditative: Annotated[Optional[float], Field(description="Min hypnotic/meditative [0–1].")] = None,
    space: Annotated[Optional[float], Field(description="Min spatial/atmospheric [0–1].")] = None,
    party: Annotated[Optional[float], Field(description="Min party/dancefloor energy [0–1].")] = None,
    danceable: Annotated[Optional[float], Field(description="Min neural danceability [0–1].")] = None,
    limit: Annotated[int, Field(description="Max tracks to return.")] = 25,
) -> dict:
    """Find library tracks matching emotional criteria (e.g. dark AND hypnotic AND ~heavy).

    Every supplied minimum is ANDed; results are sorted by how strongly they match. Use for
    vibe-led queries like 'find me something dark and hypnotic'. Only tracks at pipeline
    level ≥ 4 carry these axes. Requires the library database."""
    criteria = {"aggressive": aggressive, "dark": dark, "groovy": groovy, "energetic": energetic,
                "heavy": heavy, "meditative": meditative, "space": space, "party": party,
                "danceable": danceable}
    try:
        return await _run(_find_by_emotion_impl, criteria, limit)
    except Exception as e:
        return _error(f"Emotional search failed: {e}")


# ════════════════════════════════════════════════════════════
#  CATEGORY 5 — LIBRARY MANAGEMENT
# ════════════════════════════════════════════════════════════
def _overview_impl() -> dict:
    err = _require_db()
    if err:
        return err
    lib = analyze._load_library()
    if not lib:
        return {"track_count": 0, "note": "Library is empty — index the crate first."}
    bpms = [f.bpm for _, f in lib if f.bpm > 0]
    levels, keys, bpm_buckets = {}, {}, {}
    mood_acc, mood_n = {}, {}
    for _, f in lib:
        levels[f.pipeline_level] = levels.get(f.pipeline_level, 0) + 1
        keys[f.camelot] = keys.get(f.camelot, 0) + 1
        if f.bpm > 0:
            b = f"{int(f.bpm // 10 * 10)}-{int(f.bpm // 10 * 10) + 9}"
            bpm_buckets[b] = bpm_buckets.get(b, 0) + 1
        for name in _MOOD_FIELDS:
            v = getattr(f, name, None)
            if v is not None:
                mood_acc[name] = mood_acc.get(name, 0.0) + v
                mood_n[name] = mood_n.get(name, 0) + 1
    return {
        "track_count": len(lib),
        "pipeline_levels": dict(sorted(levels.items())),
        "bpm_range": {"min": round(min(bpms), 1), "max": round(max(bpms), 1),
                      "mean": round(sum(bpms) / len(bpms), 1)} if bpms else None,
        "bpm_histogram": dict(sorted(bpm_buckets.items())),
        "key_distribution": dict(sorted(keys.items(), key=lambda kv: kv[1], reverse=True)),
        "mood_averages": {k: round(mood_acc[k] / mood_n[k], 3) for k in sorted(mood_acc)},
    }


@mcp.tool()
async def library_overview() -> dict:
    """High-level stats for the whole indexed library.

    Track count, ML pipeline-level distribution, BPM range + 10-BPM histogram, Camelot key
    distribution, and average mood scores where available. Use to understand the shape of
    the collection before querying it. Requires the library database."""
    try:
        return await _run(_overview_impl)
    except Exception as e:
        return _error(f"Overview failed: {e}")


def _list_tracks_impl(limit, offset) -> dict:
    err = _require_db()
    if err:
        return err
    rows = database.list_tracks(analyzed_only=True)
    page = rows[offset:offset + limit]
    return {"total": len(rows), "offset": offset, "limit": limit,
            "tracks": [_row_brief(r) for r in page]}


@mcp.tool()
async def list_tracks(
    limit: Annotated[int, Field(description="Page size.")] = 50,
    offset: Annotated[int, Field(description="Offset for pagination.")] = 0,
) -> dict:
    """List indexed tracks with basic metadata (filename, BPM, Camelot, key, level), paginated.

    No curves or vectors. Use to enumerate the collection; page with limit/offset on large
    libraries. Requires the library database."""
    try:
        return await _run(_list_tracks_impl, limit, offset)
    except Exception as e:
        return _error(f"Listing failed: {e}")


def _model_status_impl() -> dict:
    reg = analyze.ModelManager.REGISTRY
    present = [n for n in reg if analyze.ModelManager.path(n).exists()]
    missing = [n for n in reg if not analyze.ModelManager.path(n).exists()]
    level = analyze.ModelManager.pipeline_level() if analyze.TF_AVAILABLE else 1
    return {
        "tensorflow_available": analyze.TF_AVAILABLE,
        "pipeline_level": level,
        "max_pipeline_level": 5,
        "level_meaning": _LEVEL_MEANING.get(level),
        "models_present": present,
        "models_missing": missing,
        "download_hint": ("Run `python analyze.py download` to fetch the missing models."
                          if (missing or not analyze.TF_AVAILABLE) else "All models present."),
    }


@mcp.tool()
async def model_status() -> dict:
    """Report which Essentia models are installed and the achievable pipeline level.

    Tells you whether ML features (embeddings, mood, emotional, genre) are available, lists
    any missing model files, and gives the download command. Call this if mood/similarity
    tools report a too-low pipeline level."""
    try:
        return await _run(_model_status_impl)
    except Exception as e:
        return _error(f"Model status failed: {e}")


@mcp.tool()
async def clear_track_cache(
    track_id: Annotated[str, Field(description="track_id (resolved path) to drop from the in-process cache.")],
) -> dict:
    """Drop a track from the SERVER's in-process feature cache so the next analyze_track re-extracts it.

    Use after re-encoding or replacing a file. This does NOT delete anything from the crate
    database (that is crate.py's job) — it only forces fresh analysis in this process."""
    abs_path = str(Path(track_id).resolve())
    existed = _FEATURE_CACHE.pop(abs_path, None) is not None
    return {"track_id": abs_path, "cleared": existed,
            "note": "In-process cache only; the crate database row (if any) is untouched."}


# ════════════════════════════════════════════════════════════
#  CATEGORY 6 — SESSION CONTEXT (in-memory, ephemeral)
# ════════════════════════════════════════════════════════════
def _session_view(s: dict) -> dict:
    """Public view of a session dict."""
    return {"session_id": s["session_id"], "started_at": s["started_at"],
            "ended_at": s["ended_at"], "track_count": len(s["tracks"]), "tracklist": s["tracks"]}


@mcp.tool()
async def start_session() -> dict:
    """Start a new in-memory play session and return its session_id.

    Ephemeral, single-conversation memory — it is NOT written to the database. Use it to
    track what's been played so recommendations can avoid repeats. Pass the session_id to
    the other session tools."""
    sid = uuid.uuid4().hex
    SESSION_STATE["sessions"][sid] = {"session_id": sid, "started_at": _now(),
                                      "ended_at": None, "tracks": []}
    SESSION_STATE["current"] = sid
    return _session_view(SESSION_STATE["sessions"][sid])


@mcp.tool()
async def log_played(
    session_id: Annotated[str, Field(description="session_id from start_session.")],
    track_id: Annotated[str, Field(description="track_id that was just played.")],
) -> dict:
    """Log a track as played in a session, with a timestamp and position.

    Appends to the in-memory tracklist. Use after each track goes out so the session
    reflects the real set."""
    s = SESSION_STATE["sessions"].get(session_id)
    if not s:
        return _error(f"No active session '{session_id}'. Call start_session first.")
    if s["ended_at"]:
        return _error(f"Session '{session_id}' has already ended.")
    abs_path = str(Path(track_id).resolve())
    entry = {"position": len(s["tracks"]) + 1, "track_id": abs_path,
             "filename": Path(abs_path).name, "played_at": _now()}
    s["tracks"].append(entry)
    return {"session_id": session_id, "logged": entry, "track_count": len(s["tracks"])}


@mcp.tool()
async def get_session(
    session_id: Annotated[str, Field(description="session_id from start_session.")],
) -> dict:
    """Return a session's current ordered tracklist and metadata."""
    s = SESSION_STATE["sessions"].get(session_id)
    if not s:
        return _error(f"No session '{session_id}'.")
    return _session_view(s)


@mcp.tool()
async def recommend_avoiding_session(
    session_id: Annotated[str, Field(description="session_id whose played tracks should be excluded.")],
    track_id: Annotated[str, Field(description="track_id of the currently-playing track.")],
    mode: Annotated[str, Field(description="Scoring preset: 'safe' | 'balanced' | 'creative'.")] = "balanced",
    n: Annotated[int, Field(description="How many picks to return.")] = 5,
    energy: Annotated[str, Field(description="Energy direction: 'up' | 'stable' | 'down'.")] = "stable",
    temperature: Annotated[float, Field(description="0.0 deterministic; higher = more variety.")] = 0.0,
) -> dict:
    """Recommend next tracks that have NOT been played in this session.

    Same engine as get_recommendations, but it excludes every track already logged in the
    session so you never suggest a repeat mid-set. Requires the library database."""
    s = SESSION_STATE["sessions"].get(session_id)
    if not s:
        return _error(f"No active session '{session_id}'. Call start_session first.")
    exclude = [t["track_id"] for t in s["tracks"]]
    try:
        return await _run(_recommend_impl, track_id, mode, n, energy, temperature, exclude)
    except Exception as e:
        return _error(f"Recommendation failed: {e}")


@mcp.tool()
async def end_session(
    session_id: Annotated[str, Field(description="session_id to close.")],
) -> dict:
    """End a session and return its complete, timestamped tracklist as structured data.

    Marks the session ended (and clears it as 'current'). The returned tracklist is the
    permanent record of the set for this conversation."""
    s = SESSION_STATE["sessions"].get(session_id)
    if not s:
        return _error(f"No session '{session_id}'.")
    s["ended_at"] = _now()
    if SESSION_STATE["current"] == session_id:
        SESSION_STATE["current"] = None
    return _session_view(s)


# ════════════════════════════════════════════════════════════
#  RESOURCES  (read-only data endpoints)
# ════════════════════════════════════════════════════════════
def _json(obj) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)


@mcp.resource("thecrate://library/summary")
async def res_library_summary() -> str:
    """Library statistics: track count, BPM histogram, key distribution, pipeline-level
    breakdown, and average mood scores. Read this to understand the collection."""
    return _json(await _run(_overview_impl))


@mcp.resource("thecrate://library/tracks")
async def res_library_tracks() -> str:
    """Full list of indexed tracks with basic metadata (no curves or vectors)."""
    return _json(await _run(_list_tracks_impl, 100000, 0))


@mcp.resource("thecrate://models/status")
async def res_models_status() -> str:
    """Which Essentia models are installed, the achievable pipeline level, the missing
    model names, and how to download them."""
    return _json(await _run(_model_status_impl))


@mcp.resource("thecrate://session/current")
async def res_session_current() -> str:
    """The current (most recently started, still-open) session's tracklist, or an empty
    tracklist if no session is active."""
    sid = SESSION_STATE["current"]
    s = SESSION_STATE["sessions"].get(sid) if sid else None
    return _json(_session_view(s) if s else {"session_id": None, "tracklist": []})


# ════════════════════════════════════════════════════════════
#  PROMPTS  (reusable agent instructions)
# ════════════════════════════════════════════════════════════
@mcp.prompt()
def analyze_and_recommend(
    track_path: Annotated[str, Field(description="Path to the audio file to analyse.")],
    mode: Annotated[str, Field(description="Scoring mode: 'safe' | 'balanced' | 'creative'.")] = "balanced",
    genre_context: Annotated[str, Field(description="Optional free text about the genre/scene/setting, to frame the explanation.")] = "",
) -> str:
    """Analyse a track, fetch its top-5 next-track recommendations, and explain the best pick."""
    ctx = f"\nGenre/context to frame your explanation: {genre_context}" if genre_context else ""
    return (
        f"Analyse the track at '{track_path}' using analyze_track, then call "
        f"get_recommendations on its track_id with mode='{mode}', n=5.\n\n"
        f"Then, in plain language:\n"
        f"1. Summarise the track (BPM, key/Camelot, energy, emotional character).\n"
        f"2. Present the 5 recommendations briefly, with their scores.\n"
        f"3. Explain WHY the top pick is the best transition — reference its key relationship, "
        f"BPM delta, energy direction and the mix tip — and give concrete mixing advice.{ctx}"
    )


@mcp.prompt()
def build_set(
    seed_track: Annotated[str, Field(description="Path/track_id of the opening track.")],
    length: Annotated[int, Field(description="Number of tracks in the set.")] = 8,
    energy_arc: Annotated[str, Field(description="'build' | 'peak' | 'journey' | 'cool_down'.")] = "build",
    mode: Annotated[str, Field(description="Scoring mode: 'safe' | 'balanced' | 'creative'.")] = "balanced",
) -> str:
    """Construct a full DJ set from a seed track and reason about its energy arc."""
    energy_map = {"build": "up", "peak": "up", "journey": "stable", "cool_down": "down"}
    energy = energy_map.get(energy_arc, "up")
    return (
        f"Build a {length}-track set from the seed '{seed_track}'.\n\n"
        f"1. Call analyze_track on the seed if needed, then build_setlist(seed_track_id, "
        f"length={length}, mode='{mode}', energy='{energy}').\n"
        f"2. Read the returned tracklist and transitions.\n"
        f"3. Assess whether the set delivers a '{energy_arc}' energy arc across its length, "
        f"referencing each track's energy and the transition scores.\n"
        f"4. Flag every transition whose `needs_attention` is true (weak score or an energy "
        f"drop) and suggest a fix — a different track, or a specific mixing technique.\n"
        f"5. Give the final ordered tracklist with one-line mixing notes per transition."
    )


@mcp.prompt()
def describe_track(
    track_path: Annotated[str, Field(description="Path to the audio file to describe.")],
) -> str:
    """Produce a rich natural-language description of a track's musical character (any genre)."""
    return (
        f"Analyse the track at '{track_path}' with analyze_track (and get_emotional_profile "
        f"if it reaches pipeline level ≥ 3), then write a rich, natural-language description "
        f"of its musical character. Cover: tempo and feel; key and tonality; energy and "
        f"dynamics; timbre/texture; emotional register; the best moment in a set to play it; "
        f"and the kinds of genres, scenes or artists it would sit well alongside. Write for a "
        f"musician or DJ of ANY genre — do not assume electronic music. Ground every claim in "
        f"the analysed features."
    )


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
def main() -> None:
    """Run the MCP server over the chosen transport (stdio default, or SSE/HTTP)."""
    parser = argparse.ArgumentParser(
        prog="mcp_server.py",
        description="The Crate MCP server — exposes the music-analysis engine to AI agents.")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"],
                        default="stdio",
                        help="stdio (default, for Claude Desktop) or sse/streamable-http (remote agents).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for sse/http transports.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for sse/http transports.")
    args = parser.parse_args()

    if args.transport in ("sse", "streamable-http"):
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info("The Crate MCP %s on %s://%s:%d", VERSION, args.transport, args.host, args.port)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════
## DEPLOYMENT NOTES
# ════════════════════════════════════════════════════════════
#
# ── Requirements ─────────────────────────────────────────────────────────────
#   uv add mcp uvicorn anyio        # mcp = SDK; uvicorn/anyio back the SSE/HTTP server
#   (analyze.py's own deps — essentia-tensorflow, psycopg2, numpy — must already be
#    installed; the server imports analyze, which loads them.)
#
# ── Claude Desktop (local stdio) ─────────────────────────────────────────────
#   Add to claude_desktop_config.json → "mcpServers":
#     {
#       "mcpServers": {
#         "thecrate": {
#           "command": "uv",
#           "args": ["run", "python", "mcp_server.py", "--transport", "stdio"],
#           "cwd": "/Users/5_prcntr/Documents/master IA/embeddings2"
#         }
#       }
#     }
#   (Or point "command" at the venv python directly. Restart Claude Desktop to load it.)
#
# ── Remote agents (Claude.ai / custom) via SSE ───────────────────────────────
#   uv run python mcp_server.py --transport sse --host 0.0.0.0 --port 8000
#   The SSE endpoint is served at  http://<host>:8000/sse  (messages POSTed to /messages).
#   Put it behind TLS + auth before exposing it publicly — there is no auth layer here.
#   `--transport streamable-http` serves the newer Streamable-HTTP transport instead.
#
# ── Testing without an agent (SSE mode) ──────────────────────────────────────
#   # 1. open the event stream (keep it running):
#   curl -N http://127.0.0.1:8000/sse
#   #    → it prints an "endpoint" event with a /messages/?session_id=... URL.
#   # 2. in another shell, POST JSON-RPC to that URL. List tools:
#   curl -X POST 'http://127.0.0.1:8000/messages/?session_id=<ID>' \
#        -H 'content-type: application/json' \
#        -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
#   # 3. call a tool:
#   curl -X POST 'http://127.0.0.1:8000/messages/?session_id=<ID>' \
#        -H 'content-type: application/json' \
#        -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
#             "params":{"name":"model_status","arguments":{}}}'
#   The official `mcp` CLI / Inspector (`mcp dev mcp_server.py`) is the friendlier path.
#
# ── Stateful vs stateless tools ──────────────────────────────────────────────
#   STATEFUL  (mutate the in-process SESSION_STATE dict, lost on restart):
#       start_session, log_played, get_session, recommend_avoiding_session,
#       end_session, and the thecrate://session/current resource.
#   STATELESS (pure functions of their args + the DB/cache): every other tool and
#       resource. The _FEATURE_CACHE is a transparent performance cache, not agent
#       state — clearing it only forces re-analysis.
#
# ── Thread-safety / concurrency ──────────────────────────────────────────────
#   analyze.ModelManager._instances is a CLASS-LEVEL cache of Essentia TF algorithm
#   instances, and Essentia algorithm objects are NOT safe to invoke concurrently.
#   All CPU-bound analysis is therefore funnelled through a SINGLE-worker
#   ThreadPoolExecutor (_EXECUTOR, max_workers=1): tool calls that touch Essentia or
#   the DB execute one-at-a-time, so two requests can never drive the same TF graph
#   simultaneously. The asyncio event loop stays free to accept calls meanwhile, but
#   heavy work is effectively serialised — correct, at the cost of no analysis
#   parallelism. SESSION_STATE / _FEATURE_CACHE are mutated only from coroutine bodies
#   on the single event-loop thread (not inside the executor), so those dict updates
#   are race-free. The PostgreSQL pool (database.py ThreadedConnectionPool) is itself
#   thread-safe. If you ever raise max_workers, add a lock around ModelManager.get().
#
# ── Tool / resource / prompt inventory ───────────────────────────────────────
#   Tools (27): analyze_track, analyze_folder, get_cached_features, is_track_cached,
#     get_key_analysis, get_recommendations, compare_tracks, detect_vinyl_speed_offset,
#     find_similar, search_tracks, find_harmonic_matches, get_mix_points, build_setlist,
#     evaluate_transition, get_mix_technique, get_emotional_profile, compare_emotional,
#     find_by_emotion, library_overview, list_tracks, model_status, clear_track_cache,
#     start_session, log_played, get_session, recommend_avoiding_session, end_session.
#   Resources (4): thecrate://library/summary, thecrate://library/tracks,
#     thecrate://models/status, thecrate://session/current.
#   Prompts (3): analyze_and_recommend, build_set, describe_track.
# ─────────────────────────────────────────────────────────────────────────────
