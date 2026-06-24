"""
Oscar Mulero Mix Engine — The Crate
---------------------------------
Techno mix analysis and recommendation system built on Essentia.

Five-level graceful degradation:
    Level 1 (classic):   RhythmExtractor2013, KeyExtractor, MFCC, spectral features.
    Level 2 (+ML):       Level 1 + EffNet embeddings + TempoCNN.
    Level 3 (full):      Level 2 + mood_aggressive + neural danceability.
    Level 4 (emotional): Level 3 + 5 extended moods + Jamendo 56-D mood/theme vector.
    Level 5 (extended):  Level 4 + genre style (400-D) + voice/instrumental + tonal +
                         approachability + engagement + timbre + instrument vector (40-D).

Pipeline:
    1. INDEX  — extract audio features per track and persist them to PostgreSQL.
    2. SCORE  — compare a "now playing" track against every cached candidate.
    3. SUGGEST — return the top picks with BPM/Camelot/energy/timbre breakdown.

CLI:
    uv run python analyze.py analyze <wav>
    uv run python analyze.py scan <folder>
    uv run python analyze.py next <wav> [--mode safe|balanced|creative]
    uv run python analyze.py compare <wav1> <wav2> [--mode ...]
    uv run python analyze.py setlist <wav> [--length N]
    uv run python analyze.py mixpoints <wav>
    uv run python analyze.py download          # pre-download all models for offline use
"""
import argparse
import logging
import sys
import urllib.request

import database
from config import (SAMPLE_RATE, ML_SAMPLE_RATE,   # Sample rates: single source of truth.
                    DENSITY_MOD_FLOOR, RETRIEVAL_K,  # Scoring/retrieval knobs.
                    KEY_STRENGTH_THRESHOLD, HARMONIC_CONTINUOUS_CONFIDENCE,
                    KEY_PROFILE, KEY_VOTE_FALLBACKS, KEY_TONAL_BAND)  # Tonal detection.
from dataclasses import dataclass, fields, replace
from pathlib import Path

import essentia.standard as es
import numpy as np

# essentia-tensorflow is optional. If unavailable, all ML algorithms below
# will be missing from es.* — we detect that at startup and downgrade.
try:
    _ = es.TensorflowPredictEffnetDiscogs                # Triggers attribute probe.
    TF_AVAILABLE = True
except AttributeError:
    TF_AVAILABLE = False

logger = logging.getLogger("thecrate")
logging.basicConfig(level=logging.INFO, format="[The Crate] %(message)s")

# SAMPLE_RATE (44100, CD/vinyl rips) and ML_SAMPLE_RATE (16000, EffNet/TempoCNN/
# mood/danceability inputs) are imported from config.py above — defined ONCE there.
MODELS_DIR = Path(__file__).parent / "models"       # Where downloaded .pb files live.
PERFECT_MIX_THRESHOLD = 0.85                        # Score above which a pairing is flagged ★.

# ── MTG-Jamendo label indices ─────────────────────────────────────────────────
# Verified from the official model metadata JSON:
#   https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/
#   mtg_jamendo_moodtheme-discogs-effnet-1.json
# The classifier outputs a 56-D probability vector; these are the indices of the
# six labels most relevant to techno/electronic DJ use. Any change to the model
# version may shift these indices — re-verify against the metadata JSON.
_JAMENDO_DARK_IDX       = 11   # "dark"       — shadow, darkness
_JAMENDO_GROOVY_IDX     = 25   # "groovy"     — rhythmic pull, groove
_JAMENDO_MEDITATIVE_IDX = 32   # "meditative" — hypnotic, meditative
_JAMENDO_ENERGETIC_IDX  = 18   # "energetic"  — raw energy
_JAMENDO_HEAVY_IDX      = 27   # "heavy"      — weight, intensity
_JAMENDO_SPACE_IDX      = 49   # "space"      — spatial, atmospheric

# Fields that must be populated for a track to reach pipeline Level 4.
_LEVEL4_FIELDS = (
    'mood_electronic', 'mood_sad', 'mood_relaxed',
    'mood_happy', 'mood_party', 'jamendo_dark',
)

# Fields that must be populated for a track to reach pipeline Level 5.
_LEVEL5_FIELDS = (
    'voice_instrumental', 'tonal', 'approachability',
    'engagement', 'timbre_bright', 'genre_discogs400',
)

# Canonical order for emotional_vector assembly (see _build_emotional_vector).
# Order is fixed here and must never change — two vectors are only comparable
# if they share the same field-to-index mapping.
_EMOTIONAL_VECTOR_ORDER = (
    'mood_aggressive', 'mood_electronic', 'mood_sad', 'mood_relaxed',
    'mood_happy', 'mood_party', 'danceability_nn',
    'jamendo_dark', 'jamendo_groovy', 'jamendo_meditative',
    'jamendo_energetic', 'jamendo_heavy', 'jamendo_space',
)

# ── NOTE FOR database.py MAINTAINER ──────────────────────────────────────────
# TrackFeatures now includes 11 new optional mood fields and an emotional_vector
# list (variable length, 4–13 elements depending on pipeline level). These are
# stored verbatim in the features JSONB column — no schema migration needed.
# If a future version adds a dedicated embeddings_emotional table (analogous to
# embeddings_effnet), emotional_vector is the source data. Suggested pgvector
# column: vector(13) for Level 4 tracks with all components present.
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════
#  CAMELOT WHEEL — (key, mode) → Camelot code
# ════════════════════════════════════════════════════════════
# Pitch-class index used to look up Camelot codes. Enharmonic spellings (C#/Db, etc.)
# collapse to the same index so Essentia's output never falls outside the table.
KEY_INDEX = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
             'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

# Camelot wheel lookup: B = major, A = minor. Used to translate musical key
# into the harmonic-mixing notation DJs reason about.
CAMELOT = {
    (0,'major'):'8B', (1,'major'):'3B', (2,'major'):'10B', (3,'major'):'5B',
    (4,'major'):'12B',(5,'major'):'7B', (6,'major'):'2B',  (7,'major'):'9B',
    (8,'major'):'4B', (9,'major'):'11B',(10,'major'):'6B', (11,'major'):'1B',
    (0,'minor'):'5A', (1,'minor'):'12A',(2,'minor'):'7A',  (3,'minor'):'2A',
    (4,'minor'):'9A', (5,'minor'):'4A', (6,'minor'):'11A', (7,'minor'):'6A',
    (8,'minor'):'1A', (9,'minor'):'8A', (10,'minor'):'3A', (11,'minor'):'10A',
}

def to_camelot(key: str, scale: str) -> str:
    """Map an Essentia (key, scale) pair to its Camelot code.

    Args:
        key:   Pitch class as returned by KeyExtractor ('C', 'F#', 'Bb', …).
        scale: 'major' or 'minor'.
    Returns:
        Camelot code (e.g. '8B') or '?' if the pair is unknown.
    """
    return CAMELOT.get((KEY_INDEX.get(key, 0), scale), '?')


def camelot_energy_direction(c1: str, c2: str) -> int:
    """Signed energy direction of a single-step Camelot move c1→c2.

    On the Camelot wheel one step clockwise is a perfect fifth up (+7 semitones) —
    the classic harmonic 'energy boost', heard as a brightness lift. One step
    anticlockwise is a fifth down, a slight relax. This is the DIRECTION the old
    `key_relationship_label` "Energy boost (+7)" branch tried (and failed) to
    capture: a folded distance can't tell up from down, so it lives here as a
    signed value the energy modifier consumes.

    Returns:
        +1  one step up the wheel (same letter) — energy lift,
        -1  one step down the wheel — relax,
         0  any other move, atonal, or unknown ('?').
    """
    if not c1 or not c2 or '?' in (c1, c2) or c1[-1] != c2[-1]:
        return 0
    try:
        step = (int(c2[:-1]) - int(c1[:-1])) % 12        # 1..11 clockwise distance.
    except ValueError:
        return 0
    return 1 if step == 1 else -1 if step == 11 else 0


# ════════════════════════════════════════════════════════════
#  MODEL MANAGER — lazy load, registry, level detection
# ════════════════════════════════════════════════════════════
# All TF graphs live under ./models/. Filenames + sub-paths match the layout on
# https://essentia.upf.edu/models.html (see corrections at end of file).
# Loading a .pb is expensive (~1–3 s) — instances are cached on first use and
# reused across every track in the run.
class ModelManager:
    """Registry, auto-downloader, and lazy loader for Essentia TF pretrained models."""

    BASE_URL = "https://essentia.upf.edu/models/"

    # Registry format: (filename, sub_path, output_node)
    # output_node: tensor name for TensorflowPredict2D; None = use Essentia default.
    # All paths verified live via HTTP HEAD, 2026-06-04/2026-06-07.
    REGISTRY = {
        # ── Feature extractor ────────────────────────────────────────────────
        'effnet':             ('discogs-effnet-bs64-1.pb',
                               'feature-extractors/discogs-effnet/', None),
        # ── Tempo ────────────────────────────────────────────────────────────
        'tempocnn':           ('deeptemp-k16-3.pb',
                               'tempo/tempocnn/', None),
        # ── Level 3: primary mood + danceability ─────────────────────────────
        # NOTE: previously at 'classifiers/' (wrong — 404). Correct: 'classification-heads/'.
        'mood_aggressive':    ('mood_aggressive-discogs-effnet-1.pb',
                               'classification-heads/mood_aggressive/', None),
        'danceability':       ('danceability-discogs-effnet-1.pb',
                               'classification-heads/danceability/', None),
        # ── Level 4: extended emotional fingerprint ───────────────────────────
        'mood_electronic':    ('mood_electronic-discogs-effnet-1.pb',
                               'classification-heads/mood_electronic/', None),
        'mood_sad':           ('mood_sad-discogs-effnet-1.pb',
                               'classification-heads/mood_sad/', None),
        'mood_relaxed':       ('mood_relaxed-discogs-effnet-1.pb',
                               'classification-heads/mood_relaxed/', None),
        'mood_happy':         ('mood_happy-discogs-effnet-1.pb',
                               'classification-heads/mood_happy/', None),
        'mood_party':         ('mood_party-discogs-effnet-1.pb',
                               'classification-heads/mood_party/', None),
        # 56-D multi-label mood+theme; full vector stored in embeddings table.
        'jamendo_moodtheme':  ('mtg_jamendo_moodtheme-discogs-effnet-1.pb',
                               'classification-heads/mtg_jamendo_moodtheme/', None),
        # ── Level 5: genre style + audio characterisation ────────────────────
        # output nodes confirmed from each model's JSON metadata file.
        'genre_discogs400':   ('genre_discogs400-discogs-effnet-1.pb',
                               'classification-heads/genre_discogs400/',
                               'PartitionedCall:0'),
        'voice_instrumental': ('voice_instrumental-discogs-effnet-1.pb',
                               'classification-heads/voice_instrumental/',
                               'model/Softmax'),
        'tonal_atonal':       ('tonal_atonal-discogs-effnet-1.pb',
                               'classification-heads/tonal_atonal/',
                               'model/Softmax'),
        'approachability':    ('approachability_regression-discogs-effnet-1.pb',
                               'classification-heads/approachability/',
                               'model/Identity'),
        'engagement':         ('engagement_regression-discogs-effnet-1.pb',
                               'classification-heads/engagement/',
                               'model/Identity'),
        'timbre':             ('timbre-discogs-effnet-1.pb',
                               'classification-heads/timbre/',
                               'model/Softmax'),
        'jamendo_instrument': ('mtg_jamendo_instrument-discogs-effnet-1.pb',
                               'classification-heads/mtg_jamendo_instrument/',
                               'model/Sigmoid'),
    }

    _instances: dict = {}
    _metadata_cache: dict = {}  # name → parsed JSON metadata dict (classes, schema, …)

    @classmethod
    def path(cls, name: str) -> Path:
        return MODELS_DIR / cls.REGISTRY[name][0]

    @classmethod
    def url(cls, name: str) -> str:
        entry = cls.REGISTRY[name]
        return cls.BASE_URL + entry[1] + entry[0]

    @classmethod
    def _metadata(cls, name: str) -> dict:
        """Load and cache the model's JSON metadata file, or {} if absent/unreadable."""
        if name in cls._metadata_cache:
            return cls._metadata_cache[name]
        import json as _json
        meta: dict = {}
        json_path = cls.path(name).with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path) as fh:
                    meta = _json.load(fh)
            except Exception:
                meta = {}
        cls._metadata_cache[name] = meta
        return meta

    @classmethod
    def labels(cls, name: str) -> list:
        """Return class labels from the model's JSON metadata file, or [] if absent."""
        return cls._metadata(name).get('classes', [])

    @classmethod
    def class_index(cls, name: str, label: str, default: int = 0) -> int:
        """Index of `label` in the model's class list — the column to read from preds.

        Class orderings differ PER MODEL (e.g. mood_sad is ['non_sad','sad'] but
        mood_happy is ['happy','non_happy']), so the correct probability column
        must be looked up by name rather than assumed. Falls back to `default`
        when the metadata is missing or the label isn't present, which preserves
        behaviour on an offline install with no JSON alongside the .pb.
        """
        labels = cls.labels(name)
        try:
            return labels.index(label)
        except (ValueError, AttributeError):
            return default

    @classmethod
    def output_node(cls, name: str) -> "str | None":
        """Resolve the TF output tensor for a TensorflowPredict2D model.

        Priority: an explicit REGISTRY override (3rd tuple element) wins; else the
        'predictions' output declared in the model's JSON metadata; else None
        (Essentia's default). The JSON lookup fixes the 2-class Softmax heads
        whose REGISTRY override is None — without it, Essentia defaults to a
        'model/Sigmoid' node that those graphs don't expose, so the model fails
        to load and the whole track silently degrades a pipeline level.
        """
        override = cls.REGISTRY[name][2]
        if override is not None:
            return override
        for out in (cls._metadata(name).get('schema') or {}).get('outputs') or []:
            if out.get('output_purpose') == 'predictions' and out.get('name'):
                return out['name']
        return None

    # Essentia's default TensorflowPredict2D input tensor; only graphs that differ
    # from it (SavedModel exports) need an explicit `input=` override.
    _DEFAULT_INPUT_NODE = 'model/Placeholder'

    @classmethod
    def input_node(cls, name: str) -> "str | None":
        """Resolve a non-default input tensor for TensorflowPredict2D, or None.

        Most classifier heads take Essentia's default 'model/Placeholder', but
        genre_discogs400 (a SavedModel export) exposes a different input name
        ('serving_default_model_Placeholder'). Returns the metadata input node
        only when it differs from the default, so callers pass `input=` only when
        a graph actually needs it.
        """
        ins = (cls._metadata(name).get('schema') or {}).get('inputs') or []
        first = ins[0].get('name') if ins else None
        return first if first and first != cls._DEFAULT_INPUT_NODE else None

    @classmethod
    def download_all(cls) -> None:
        """Pre-download every registered model for offline use.

        Skips models that are already on disk. Safe to call repeatedly.
        """
        if not TF_AVAILABLE:
            print("⚠️  essentia-tensorflow not installed — models won't load even after download.")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name in cls.REGISTRY:
            dest = cls.path(name)
            if dest.exists():
                print(f"  ✓  {name} already present ({dest.name})")
                continue
            try:
                _download_model(name, cls.url(name), dest)
            except Exception as e:
                print(f"  ✗  {name} download failed ({cls.url(name)}): {e}")

    @classmethod
    def has(cls, name: str) -> bool:
        """True if TF is installed AND the .pb is on disk."""
        return TF_AVAILABLE and cls.path(name).exists()

    # Model sets required for each ML level — explicit lists beat "all in REGISTRY"
    # because REGISTRY may grow without changing level semantics.
    _L3_MODELS = ('mood_aggressive', 'danceability')
    _L4_MODELS = ('mood_electronic', 'mood_sad', 'mood_relaxed',
                  'mood_happy', 'mood_party', 'jamendo_moodtheme')
    _L5_MODELS = ('genre_discogs400', 'voice_instrumental', 'tonal_atonal',
                  'approachability', 'engagement', 'timbre', 'jamendo_instrument')

    @classmethod
    def pipeline_level(cls) -> int:
        """Highest level we can run end-to-end based on available models.

        Returns:
            1 — classic DSP only (no TF or no EffNet).
            2 — EffNet embedding + TempoCNN available.
            3 — Level 2 + mood_aggressive + danceability.
            4 — Level 3 + all six extended mood/theme classifiers.
            5 — Level 4 + genre (400-D) + voice/tonal/approachability/engagement/timbre/instruments.
        """
        if not cls.has('effnet'):
            return 1
        if all(cls.has(n) for n in cls._L3_MODELS) and \
                all(cls.has(n) for n in cls._L4_MODELS) and \
                all(cls.has(n) for n in cls._L5_MODELS):
            return 5
        if all(cls.has(n) for n in cls._L3_MODELS) and \
                all(cls.has(n) for n in cls._L4_MODELS):
            return 4
        if all(cls.has(n) for n in cls._L3_MODELS):
            return 3
        return 2

    @classmethod
    def get(cls, name: str):
        """Return a cached algorithm instance, downloading the .pb on first use.

        If TF is not installed returns None immediately — callers degrade gracefully.
        Any load/graph error also returns None with a warning rather than crashing.
        """
        if name in cls._instances:
            return cls._instances[name]
        if not TF_AVAILABLE:
            return None
        # Auto-download the model file if it is missing — no manual step required.
        dest = cls.path(name)
        if not dest.exists():
            try:
                _download_model(name, cls.url(name), dest)
            except Exception as e:
                # The URL is in the message on purpose: a 404 here is almost always a
                # moved path on essentia.upf.edu, and the exact URL is the diagnosis.
                logger.warning("Auto-download failed for '%s' (%s): %s",
                               name, cls.url(name), e)
                return None
        try:
            graph = str(dest)
            output_node = cls.output_node(name)  # REGISTRY override → JSON 'predictions' → default
            if name == 'effnet':
                inst = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=graph, output="PartitionedCall:1")
            elif name == 'tempocnn':
                inst = es.TempoCNN(graphFilename=graph)
            elif output_node is not None:
                tf_kwargs = {'graphFilename': graph, 'output': output_node}
                input_node = cls.input_node(name)   # only set for SavedModel-style graphs
                if input_node is not None:
                    tf_kwargs['input'] = input_node
                inst = es.TensorflowPredict2D(**tf_kwargs)
            else:
                inst = es.TensorflowPredict2D(graphFilename=graph)
            cls._instances[name] = inst
            return inst
        except Exception as e:
            logger.warning("Failed to load model '%s': %s", name, e)
            return None


def print_pipeline_banner() -> None:
    """Print the current pipeline level at startup."""
    desc = {
        1: "classic DSP only (install `essentia-tensorflow` for ML)",
        2: "classic + EffNet + TempoCNN",
        3: "classic + EffNet + TempoCNN + mood + danceability",
        4: "full — Level 3 + emotional fingerprint (5 moods + Jamendo themes)",
        5: "maximum — Level 4 + genre (400-D) + voice/tonal/approachability/engagement/timbre/instruments",
    }
    level = ModelManager.pipeline_level() if TF_AVAILABLE else 1
    print(f"[The Crate] Pipeline level: {level}/5 — {desc[level]}")


# ════════════════════════════════════════════════════════════
#  PER-TRACK FEATURE RECORD
# ════════════════════════════════════════════════════════════
# Design choice: give EVERY field a sensible default rather than reordering.
# Reason — _hydrate() iterates dataclass fields with v.get(name, default) so
# tracks analysed at older pipeline levels rehydrate cleanly from PostgreSQL JSONB.
# It also means callers like extract_features() still use kwargs unchanged.
@dataclass
class TrackFeatures:
    """All persisted features for a single track. Stored as JSONB in the tracks table."""
    path: str = ""                                   # Absolute path on disk.
    duration: float = 0.0                            # Seconds.
    bpm: float = 0.0                                 # Tempo (multifeature estimator).
    bpm_confidence: float = 0.0                      # 0..1 — multifeature self-confidence.
    key: str = "C"                                   # Pitch class, e.g. 'A'.
    scale: str = "major"                             # 'major' or 'minor'.
    camelot: str = "?"                               # Camelot code, e.g. '8A'.
    key_strength: float = 0.0                        # Tonal certainty (0..1).
    danceability: float = 0.0                        # Essentia danceability metric (DSP).
    onset_rate: float = 0.0                          # Onsets per second.
    loudness: float = 0.0                            # Integrated loudness (linear).
    replay_gain: float = 0.0                         # dB gain to normalise track to reference.
    dynamic_complexity: float = 0.0                  # Loudness variability across the track.
    spectral_centroid: float = 0.0                   # Mean centroid in Hz (brightness).
    spectral_complexity: float = 0.0                 # Mean spectral peakiness (density).
    spectral_flux: float = 0.0                       # Mean frame-to-frame spectral change.
    spectral_rolloff: float = 0.0                    # Mean 85%-energy roll-off in Hz.
    zcr: float = 0.0                                 # Zero-crossing rate (rough texture cue).
    mfcc_mean: list = None                           # 13-D mean MFCC — timbral fingerprint.
    bark_mean: list = None                           # 27 Bark bands — tonal/spectral profile.
    energy_curve: list = None                        # RMS per 1 s frame.
    complexity_curve: list = None                    # Spectral complexity per 1 s frame.
    tuning_frequency: float = 0.0                    # Hz reference (nominally 440.0).
    intro_end: float = 0.0                           # Seconds — last sample of mixable intro.
    outro_start: float = 0.0                         # Seconds — first sample of mixable outro.
    # ── Optional ML enrichment (level ≥ 2). None means "not computed". ──
    effnet_embedding: list = None                    # 1280-D EffNet/Discogs mean embedding.
    bpm_cnn: float = None                            # TempoCNN BPM (second opinion; always computed at L2+).
    bpm_raw: float = None                            # BPM as the DSP chain measured it, before any fold.
    bpm_fold_ratio: float = None                     # Multiplier applied by the crate prior (2, 0.5, 1.5, 2/3).
    bpm_suspect: bool = False                        # Raw BPM outside prior AND no candidate fit — review me.
    mood_aggressive: float = None                    # Mood-aggressive probability (0..1).
    danceability_nn: float = None                    # Neural danceability probability (0..1).
    # ── Level 4: extended emotional fingerprint ──────────────────────────────
    mood_electronic: float = None   # Electronic/synthetic character (0..1).
    mood_sad: float = None          # Melancholy / introspection (0..1).
    mood_relaxed: float = None      # Spaciousness vs tension (0..1).
    mood_happy: float = None        # Positivity — near-zero for most techno (0..1).
    mood_party: float = None        # Dancefloor/party energy angle (0..1).
    jamendo_dark: float = None      # Darkness / shadow — Jamendo label 11 (0..1).
    jamendo_groovy: float = None    # Groove / rhythmic pull — label 25 (0..1).
    jamendo_meditative: float = None  # Hypnotic / meditative — label 32 (0..1).
    jamendo_energetic: float = None   # Raw energy — label 18 (0..1).
    jamendo_heavy: float = None     # Weight / intensity — label 27 (0..1).
    jamendo_space: float = None     # Spatial / atmospheric — label 49 (0..1).
    # Full 56-D sigmoid vector from MTG-Jamendo moodtheme; stored here so _db_persist
    # can upsert it to embeddings_jamendo_moodtheme without a second model pass.
    jamendo_moodtheme_vector: list = None
    # Assembled vector of all non-None mood scores in canonical order
    # (see _EMOTIONAL_VECTOR_ORDER). Stored for fast retrieval; comparisons use
    # emotional_vector_similarity() which recomputes the intersection per pair.
    emotional_vector: list = None
    # ── Level 5: extended audio characterisation ─────────────────────────────
    voice_instrumental: float = None   # P(instrumental) — 0 = voice-heavy, 1 = purely instrumental.
    tonal: float = None                # P(tonal) — 0 = atonal/noise/percussion, 1 = pitched/melodic.
    approachability: float = None      # Mainstream appeal: 0 = very niche, 1 = very accessible.
    engagement: float = None           # Listening mode: 0 = background/ambient, 1 = demands attention.
    timbre_bright: float = None        # Timbre brightness: 0 = dark/heavy, 1 = bright/airy.
    genre_discogs400: list = None      # 400-D Discogs genre softmax vector (also in embeddings table).
    jamendo_instrument: list = None    # 40-D instrument presence vector (multi-label sigmoid).
    # ── Tonal-detection trust (Level 1; multi-profile vote) ──────────────────
    agreement: float = 0.0   # Fraction of key profiles that concurred on (key, scale);
                             # 1.0 = unanimous/confident, 0.0 = a pre-vote (legacy) row.
    pipeline_level: int = 1                          # Level actually reached for THIS track.


# ════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════

# Metrical-octave fold candidates, in preference order after the identity.
# 2 and 1/2 are the classic half/double errors; 3/2 and 2/3 are the dotted/
# triplet locks (the Kastil case: DSP reads 88 for a track whose tactus is 132).
BPM_FOLD_RATIOS = (2.0, 0.5, 1.5, 2.0 / 3.0)


def _fold_bpm(raw: float, cnn: "float | None", lo: float, hi: float) -> tuple:
    """Fold a raw BPM into the crate's plausible range via metrical ratios.

    Tempo estimators do not mis-measure periodicity — they pick the wrong
    METRICAL LEVEL, so the true tempo is almost always raw x {2, 1/2, 3/2, 2/3}.
    Given a prior range (the crate's learned median±MAD or its genre seed),
    candidates inside the range are generated and the one closest to the
    TempoCNN second opinion wins (falling back to the range midpoint when no
    CNN reading exists).

    Args:
        raw: BPM as measured by the DSP chain.
        cnn: TempoCNN estimate, or None.
        lo, hi: plausible range from the crate prior.
    Returns:
        (bpm, fold_ratio, suspect):
          bpm         the value to use (== raw when no fold applies).
          fold_ratio  the multiplier applied, or None when untouched.
          suspect     True when raw is outside the prior and NO candidate fits —
                      the value is kept raw but flagged for human review.
    """
    if lo <= raw <= hi:
        return raw, None, False                  # Already plausible — never touch it.
    candidates = [(raw * r, r) for r in BPM_FOLD_RATIOS if lo <= raw * r <= hi]
    if not candidates:
        return raw, None, True                   # Genuinely odd — flag, don't guess.
    target = cnn if cnn else (lo + hi) / 2.0
    bpm, ratio = min(candidates, key=lambda c: abs(c[0] - target))
    return bpm, ratio, False


# Frame-level analysis grid: one sample per second of audio, analysed over
# 2048-sample (~46 ms @ 44.1 kHz) Hann windows. Shared by the frame-curve,
# tuning, and mix-point helpers so the time axis is defined once.
_HOP_SECONDS = 1.0
_FRAME_SIZE = 2048


def _extract_frame_curves(audio) -> dict:
    """Frame-level energy + spectral analysis over the whole track (1 s hop).

    Walks the signal in _FRAME_SIZE windows and accumulates per-frame
    descriptors, then collapses them into the curves and time-averaged
    fingerprints the record needs.

    Args:
        audio: mono float32 signal at SAMPLE_RATE.
    Returns:
        dict with the per-frame curves (energy_curve, complexity_curve) plus the
        time-averaged spectral scalars and MFCC/Bark fingerprints — keys match
        the TrackFeatures fields they populate.
    """
    hop_samples = int(_HOP_SECONDS * SAMPLE_RATE)
    frame_size = _FRAME_SIZE

    energy_curve, complexity_curve = [], []
    mfcc_acc, bark_acc = [], []
    centroid_acc, flux_acc, rolloff_acc, zcr_acc = [], [], [], []

    # Instantiate algorithms once and reuse across frames (faster + lower allocation overhead).
    spectrum_alg = es.Spectrum()
    window_alg   = es.Windowing(type='hann')
    centroid_alg = es.Centroid(range=SAMPLE_RATE/2)  # Centroid normalised against Nyquist.
    rolloff_alg  = es.RollOff()
    flux_alg     = es.Flux()
    complexity_alg = es.SpectralComplexity()
    mfcc_alg     = es.MFCC()
    bark_alg     = es.BarkBands()
    zcr_alg      = es.ZeroCrossingRate()

    for start in range(0, len(audio) - frame_size, hop_samples):
        frame = audio[start:start + frame_size]
        rms = float(np.sqrt(np.mean(frame**2)))      # Per-frame RMS = loudness envelope.
        energy_curve.append(rms)

        spec = spectrum_alg(window_alg(frame))       # Hann-windowed magnitude spectrum.
        complexity_curve.append(float(complexity_alg(spec)))
        centroid_acc.append(float(centroid_alg(spec)))
        rolloff_acc.append(float(rolloff_alg(spec)))
        flux_acc.append(float(flux_alg(spec)))
        zcr_acc.append(float(zcr_alg(frame)))

        _, mfcc = mfcc_alg(spec)                     # MFCC returns (bands, coeffs); keep the 13 coeffs.
        mfcc_acc.append(mfcc)
        bark_acc.append(bark_alg(spec))

    # Collapse frame-level features into single fingerprints by averaging across time.
    return {
        'energy_curve': energy_curve,
        'complexity_curve': complexity_curve,
        'mfcc_mean': np.mean(mfcc_acc, axis=0).tolist(),
        'bark_mean': np.mean(bark_acc, axis=0).tolist(),
        'spectral_centroid': float(np.mean(centroid_acc)),
        'spectral_complexity': float(np.mean(complexity_curve)),
        'spectral_flux': float(np.mean(flux_acc)),
        'spectral_rolloff': float(np.mean(rolloff_acc)),
        'zcr': float(np.mean(zcr_acc)),
    }


def _estimate_tuning_frequency(audio) -> float:
    """Reference tuning (Hz) from spectral peaks; 440 Hz fallback on any error.

    TuningFrequency takes (frequencies, magnitudes) from SpectralPeaks, NOT raw
    audio. Peaks are derived from spectrum frames sampled every 4 s across the
    track; any failure falls back to standard 440 Hz (a non-critical field).
    """
    hop_samples = int(_HOP_SECONDS * SAMPLE_RATE)
    frame_size = _FRAME_SIZE
    try:
        spectrum_alg = es.Spectrum()
        window_alg   = es.Windowing(type='hann')
        peaks_alg    = es.SpectralPeaks()
        all_peak_f, all_peak_m = [], []
        for spec_frame in [spectrum_alg(window_alg(audio[s:s + frame_size]))
                           for s in range(0, len(audio) - frame_size, hop_samples * 4)]:
            pf, pm = peaks_alg(spec_frame)
            if len(pf) > 0:
                all_peak_f.extend(pf.tolist())
                all_peak_m.extend(pm.tolist())
        if all_peak_f:
            return float(
                es.TuningFrequency()(
                    np.array(all_peak_f, dtype=np.float32),
                    np.array(all_peak_m, dtype=np.float32))[0])
        return 440.0
    except Exception:
        return 440.0  # standard tuning; non-critical field


# ── Tonal detection (key) ─────────────────────────────────────────────────────
# extract_features measures the reference tuning FIRST and feeds it here, because a
# vinyl rip captured at a platter speed off nominal is pitched sharp/flat as a
# whole: the semitone bins KeyExtractor correlates the HPCP against shift with it,
# so without the correction the detected tonic can drift up to a semitone. Detection
# then VOTES across EDM-tuned profiles (Faraldo/MTG) so a single profile's
# confident-but-wrong reading — common in techno, where the kick injects a strong
# spurious tonal peak — can be outvoted, and the agreement fraction is persisted as
# a trust signal INDEPENDENT of key_strength (which can be high AND wrong).
def _key_extract_one(audio, profile: str, tuning_frequency: float = 440.0) -> tuple:
    """One KeyExtractor pass with an explicit profile and tuning reference.

    Thin wrapper over es.KeyExtractor so the voting path and the dormant
    single-profile path share exactly one Essentia call site.

    Args:
        audio: mono float32 signal at SAMPLE_RATE.
        profile: KeyExtractor profileType (e.g. 'edma', 'edmm', 'bgate').
        tuning_frequency: reference tuning in Hz; aligns the HPCP semitone bins to
            the track's real tuning so an off-speed vinyl rip is not mis-keyed.
    Returns:
        (key, scale, key_strength).
    """
    key, scale, strength = es.KeyExtractor(
        profileType=profile, tuningFrequency=float(tuning_frequency or 440.0))(audio)
    return key, scale, float(strength)


def _detect_key_robust(audio, primary: str = None, fallbacks: tuple = None,
                       tuning_frequency: float = 440.0) -> tuple:
    """Multi-profile key detection with a confidence-gated vote.

    Runs the `primary` profile first. When its key_strength clears
    KEY_STRENGTH_THRESHOLD the track is solidly tonal and we trust it outright
    (agreement = 1.0, no extra passes — the common, cheap path). Otherwise the
    reading is shaky — exactly the C-major/E-minor failure mode — so the
    `fallbacks` are also run and every profile casts one vote for its (key, scale);
    the most-voted pair wins, ties broken by summed key_strength. This rescues
    cases where one profile fixes the wrong tonic with middling confidence while
    the EDM profiles concur on the right one.

    Args:
        audio: mono float32 signal at SAMPLE_RATE.
        primary: first profile to try (defaults to config.KEY_PROFILE).
        fallbacks: profiles consulted only on low confidence
            (defaults to config.KEY_VOTE_FALLBACKS).
        tuning_frequency: reference tuning fed to every pass (see _key_extract_one).
    Returns:
        (key, scale, key_strength, agreement) where agreement is the fraction of
        profiles that landed on the winning (key, scale) — a vote-based trust
        signal independent of key_strength, persisted for the harmonic modifier.
    """
    primary = primary or KEY_PROFILE
    fallbacks = KEY_VOTE_FALLBACKS if fallbacks is None else fallbacks
    key, scale, strength = _key_extract_one(audio, primary, tuning_frequency)
    if strength >= KEY_STRENGTH_THRESHOLD:
        return key, scale, strength, 1.0          # Confident: trust it, skip the vote.
    # Shaky: gather one vote per profile and pick the consensus (key, scale).
    votes = [(key, scale, strength)]
    for prof in fallbacks:
        try:
            votes.append(_key_extract_one(audio, prof, tuning_frequency))
        except Exception:
            continue                              # A bad profile never sinks the vote.
    tally = {}                                    # (key, scale) -> [count, summed_strength]
    for k, sc, st in votes:
        slot = tally.setdefault((k, sc), [0, 0.0])
        slot[0] += 1
        slot[1] += st
    (win_key, win_scale), (count, strength_sum) = max(
        tally.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return win_key, win_scale, strength_sum / count, count / len(votes)


# ╔══ FOR IF IT'S NEEDED: IMPROVE ANALYZE IN HARMONY/CAMELOT ══════════════════╗
# The two helpers below are BUILT but deliberately NOT wired into extract_features
# (the live path uses _detect_key_robust above). They are kept ready to drop in if
# the benchmark (ab_tests/benchmark_key.py, which already exercises both as
# comparison arms) shows they help:
#   • _detect_key_simple  — the lightweight single-profile alternative to voting.
#   • _isolate_tonal_band — strips the techno kick/hats before detection.
# To activate either, call it from extract_features' Tonality block (and re-analyse).
def _detect_key_simple(audio, profile: str = None,
                       tuning_frequency: float = 440.0) -> tuple:
    """[DORMANT] Single-profile key detection — the cheap alternative to the vote.

    One KeyExtractor pass with the configured EDM profile and the measured tuning.
    Returns the same 4-tuple shape as _detect_key_robust (agreement fixed at 1.0 —
    a lone profile has nothing to disagree with) so it can replace the voting call
    in extract_features verbatim if voting proves unnecessary or too slow.

    Args:
        audio: mono float32 signal at SAMPLE_RATE.
        profile: KeyExtractor profileType (defaults to config.KEY_PROFILE).
        tuning_frequency: reference tuning fed to KeyExtractor.
    Returns:
        (key, scale, key_strength, agreement=1.0).
    """
    key, scale, strength = _key_extract_one(audio, profile or KEY_PROFILE,
                                            tuning_frequency)
    return key, scale, strength, 1.0


def _isolate_tonal_band(audio):
    """[DORMANT] Restrict audio to the tonal band before key detection.

    Techno's kick (<~80 Hz) and hi-hat/cymbal air (>~2 kHz) inject broadband
    energy that contaminates the HPCP and pulls the detected tonic off. Band-
    limiting to roughly 80 Hz–2 kHz — where the bassline/chord fundamentals of
    techno live — lets KeyExtractor correlate against mostly pitched content.

    Returns a NEW filtered buffer for tonal detection ONLY; never feed it to
    _extract_frame_curves (energy/mixpoints must see the full spectrum). Falls back
    to the untouched signal on any filter error.

    Args:
        audio: mono float32 signal at SAMPLE_RATE.
    Returns:
        A band-limited copy of `audio`.
    """
    lo, hi = KEY_TONAL_BAND
    try:
        hp = es.HighPass(cutoffFrequency=float(lo), sampleRate=SAMPLE_RATE)
        lp = es.LowPass(cutoffFrequency=float(hi), sampleRate=SAMPLE_RATE)
        return lp(hp(audio))
    except Exception:
        return audio
# ╚════════════════════════════════════════════════════════════════════════════╝


def _tuning_cents(tuning_frequency: float) -> float:
    """Tuning offset from concert pitch (A=440) in cents (100 cents = 1 semitone).

    A vinyl rip captured off nominal platter speed reads sharp/flat as a whole;
    >±20 cents is a practical "this rip may be speed-shifted" threshold. Returns
    0.0 when the tuning is unknown/non-positive.

    Args:
        tuning_frequency: reference tuning in Hz from _estimate_tuning_frequency.
    Returns:
        Signed cents relative to 440 Hz.
    """
    if not tuning_frequency or tuning_frequency <= 0:
        return 0.0
    return 1200.0 * float(np.log2(tuning_frequency / 440.0))


def _detect_mix_points(complexity_curve: list, duration: float) -> tuple:
    """Locate mixable intro_end / outro_start from the spectral-complexity curve.

    Percussive intros/outros have low spectral complexity (few simultaneous
    tones). The intro ends where complexity first rises above the bottom-40%
    threshold; the outro starts at the last frame still above it.

    Returns:
        (intro_end, outro_start) in seconds.
    """
    comp_arr = np.array(complexity_curve)
    if len(comp_arr) == 0:
        return 0.0, duration
    threshold = np.percentile(comp_arr, 40)                   # Bottom 40% = "sparse" zones.
    low_complexity = comp_arr < threshold
    # intro_end: first frame whose complexity exceeds the threshold; default 32 s.
    intro_end = next((i for i, low in enumerate(low_complexity) if not low), 32)
    intro_end = min(intro_end, len(comp_arr) - 1) * _HOP_SECONDS
    # outro_start: scan backwards for the last "dense" frame; everything after is outro.
    outro_start_idx = next((i for i in range(len(comp_arr) - 1, 0, -1)
                            if not low_complexity[i]), len(comp_arr) - 32)
    outro_start = outro_start_idx * _HOP_SECONDS
    return intro_end, outro_start


def _enrich_level4(record: TrackFeatures, embeddings, audio_path: str) -> None:
    """Level 4: extended emotional fingerprint over the shared EffNet embeddings.

    Runs five mood classifiers + the 56-D Jamendo mood/theme head on the SAME
    `embeddings` array (no new audio load). Each model is wrapped individually so
    one failure never blocks the others. Promotes record.pipeline_level to 4 only
    if every essential Level-4 field landed. Mutates `record` in place.
    """
    print("  🎭 Emotional fingerprint (Level 4)...")
    # (model_key, record_field, positive_class_label). The positive class is NOT
    # always column 0 — its index is resolved by name from each model's metadata
    # (see ModelManager.class_index).
    for mood_name, field_name, pos_label in [
        ('mood_electronic', 'mood_electronic', 'electronic'),
        ('mood_sad',        'mood_sad',        'sad'),
        ('mood_relaxed',    'mood_relaxed',    'relaxed'),
        ('mood_happy',      'mood_happy',      'happy'),
        ('mood_party',      'mood_party',      'party'),
    ]:
        model = ModelManager.get(mood_name)
        if model is not None:
            try:
                preds = model(embeddings)      # shape (T, 2)
                idx = ModelManager.class_index(mood_name, pos_label)
                setattr(record, field_name, float(np.mean(preds[:, idx])))
            except Exception as ex:
                logger.warning("%s failed on %s: %s",
                               mood_name, Path(audio_path).name, ex)

    # MTG-Jamendo multi-label: full 56-D vector + 6 named display scalars.
    jamendo_model = ModelManager.get('jamendo_moodtheme')
    if jamendo_model is not None:
        try:
            preds      = jamendo_model(embeddings)  # shape (T, 56)
            mean_preds = np.mean(preds, axis=0)     # (56,) mean per label
            record.jamendo_moodtheme_vector = mean_preds.tolist()
            record.jamendo_dark       = float(mean_preds[_JAMENDO_DARK_IDX])
            record.jamendo_groovy     = float(mean_preds[_JAMENDO_GROOVY_IDX])
            record.jamendo_meditative = float(mean_preds[_JAMENDO_MEDITATIVE_IDX])
            record.jamendo_energetic  = float(mean_preds[_JAMENDO_ENERGETIC_IDX])
            record.jamendo_heavy      = float(mean_preds[_JAMENDO_HEAVY_IDX])
            record.jamendo_space      = float(mean_preds[_JAMENDO_SPACE_IDX])
        except Exception as ex:
            logger.warning("jamendo_moodtheme failed on %s: %s",
                           Path(audio_path).name, ex)

    # Assemble and cache the emotional vector for fast retrieval.
    record.emotional_vector = _build_emotional_vector(record)

    # Level 4 only if all six extended fields landed.
    if all(getattr(record, f) is not None for f in _LEVEL4_FIELDS):
        record.pipeline_level = 4


def _enrich_level5(record: TrackFeatures, embeddings, audio_path: str) -> None:
    """Level 5: genre style + extended audio characterisation.

    All models run on the same `embeddings` — no additional audio decode. Each
    failure is isolated so one bad model can't block the others. Promotes
    record.pipeline_level to 5 only if every essential Level-5 field landed.
    Mutates `record` in place.
    """
    print("  🧬 Extended characterisation (Level 5)...")

    # 400-D Discogs genre style vector (softmax over 400 styles).
    genre_model = ModelManager.get('genre_discogs400')
    if genre_model is not None:
        try:
            preds = genre_model(embeddings)          # shape (T, 400)
            record.genre_discogs400 = np.mean(preds, axis=0).tolist()
        except Exception as ex:
            logger.warning("genre_discogs400 failed on %s: %s",
                           Path(audio_path).name, ex)

    # Voice/instrumental: P(instrumental). classes are
    # ['instrumental','voice'] so the positive column is 0, not 1.
    vi_model = ModelManager.get('voice_instrumental')
    if vi_model is not None:
        try:
            preds = vi_model(embeddings)             # shape (T, 2)
            idx = ModelManager.class_index('voice_instrumental', 'instrumental')
            record.voice_instrumental = float(np.mean(preds[:, idx]))
        except Exception as ex:
            logger.warning("voice_instrumental failed on %s: %s",
                           Path(audio_path).name, ex)

    # Tonal/atonal: P(tonal/pitched). classes are ['atonal','tonal'].
    ta_model = ModelManager.get('tonal_atonal')
    if ta_model is not None:
        try:
            preds = ta_model(embeddings)             # shape (T, 2)
            idx = ModelManager.class_index('tonal_atonal', 'tonal')
            record.tonal = float(np.mean(preds[:, idx]))
        except Exception as ex:
            logger.warning("tonal_atonal failed on %s: %s",
                           Path(audio_path).name, ex)

    # Approachability regression: single-value output (T, 1).
    app_model = ModelManager.get('approachability')
    if app_model is not None:
        try:
            preds = app_model(embeddings)            # shape (T, 1)
            record.approachability = float(np.mean(preds))
        except Exception as ex:
            logger.warning("approachability failed on %s: %s",
                           Path(audio_path).name, ex)

    # Engagement regression: single-value output (T, 1).
    eng_model = ModelManager.get('engagement')
    if eng_model is not None:
        try:
            preds = eng_model(embeddings)            # shape (T, 1)
            record.engagement = float(np.mean(preds))
        except Exception as ex:
            logger.warning("engagement failed on %s: %s",
                           Path(audio_path).name, ex)

    # Timbre brightness: P(bright). classes are ['bright','dark']
    # so the positive column is 0, not 1.
    timbre_model = ModelManager.get('timbre')
    if timbre_model is not None:
        try:
            preds = timbre_model(embeddings)         # shape (T, 2)
            idx = ModelManager.class_index('timbre', 'bright')
            record.timbre_bright = float(np.mean(preds[:, idx]))
        except Exception as ex:
            logger.warning("timbre failed on %s: %s",
                           Path(audio_path).name, ex)

    # 40-D instrument presence vector (multi-label sigmoid).
    inst_model = ModelManager.get('jamendo_instrument')
    if inst_model is not None:
        try:
            preds = inst_model(embeddings)           # shape (T, 40)
            record.jamendo_instrument = np.mean(preds, axis=0).tolist()
        except Exception as ex:
            logger.warning("jamendo_instrument failed on %s: %s",
                           Path(audio_path).name, ex)

    # Level 5 only if all essential scalar fields landed.
    if all(getattr(record, f) is not None for f in _LEVEL5_FIELDS):
        record.pipeline_level = 5


def _enrich_with_ml(record: TrackFeatures, audio_path: str) -> None:
    """Best-effort ML enrichment (Levels 2–5). Mutates `record` in place.

    Gated on TF_AVAILABLE. Decodes the audio once at 16 kHz, computes the EffNet
    embedding (Level 2) and its derived classifiers (TempoCNN, mood, dance →
    Level 3), then hands the shared embeddings to _enrich_level4 / _enrich_level5.
    Every model is wrapped so any single failure leaves the field None and the
    record valid at whatever level completed — a model issue never kills a track.
    """
    if not TF_AVAILABLE:
        return
    # get() auto-downloads the .pb on first call; returns None on any failure.
    # Gating on get() rather than has() means download happens transparently here.
    embed_model = ModelManager.get('effnet')
    if embed_model is None:
        return
    try:
        print("  🤖 ML enrichment (16 kHz)...")
        audio_16k = es.MonoLoader(
            filename=audio_path, sampleRate=ML_SAMPLE_RATE, resampleQuality=4)()

        # (a) EffNet — L2-normalised 1280-D embedding.
        embeddings = embed_model(audio_16k)
        emb  = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(emb)
        record.effnet_embedding = (emb / norm).tolist() if norm > 0 else emb.tolist()
        record.pipeline_level   = 2

        # (b) TempoCNN — ALWAYS computed (it is cheap on the already-loaded
        # 16 kHz audio). Two reasons: (1) _effective_bpm() prefers it up to
        # confidence < 0.7, so gating at < 0.4 left a 0.4-0.7 gap where the
        # rescue could never fire; (2) the crate BPM prior uses it as the
        # tie-breaker between fold candidates. NOTE: octave errors fool the
        # DSP chain at HIGH confidence (a stable wrong metrical level reads
        # as "confident"), so confidence alone can never gate this.
        tempo_model = ModelManager.get('tempocnn')
        if tempo_model is not None:
            try:
                g_bpm, _, _ = tempo_model(audio_16k)
                record.bpm_cnn = float(g_bpm)
            except Exception as e:
                logger.warning("TempoCNN failed on %s: %s", Path(audio_path).name, e)

        # (c) mood_aggressive — classifier head over EffNet embeddings.
        mood_model = ModelManager.get('mood_aggressive')
        if mood_model is not None:
            try:
                preds = mood_model(embeddings)
                idx = ModelManager.class_index('mood_aggressive', 'aggressive')
                record.mood_aggressive = float(np.mean(preds[:, idx]))
            except Exception as e:
                logger.warning("mood_aggressive failed on %s: %s", Path(audio_path).name, e)

        # (d) neural danceability — same architecture as mood.
        dance_model = ModelManager.get('danceability')
        if dance_model is not None:
            try:
                preds = dance_model(embeddings)
                idx = ModelManager.class_index('danceability', 'danceable')
                record.danceability_nn = float(np.mean(preds[:, idx]))
            except Exception as e:
                logger.warning("danceability NN failed on %s: %s", Path(audio_path).name, e)

        if record.mood_aggressive is not None and record.danceability_nn is not None:
            record.pipeline_level = 3

        # Levels 4 and 5 reuse the same embeddings; each gates on the prior level.
        if record.pipeline_level >= 3:
            _enrich_level4(record, embeddings, audio_path)
        if record.pipeline_level >= 4:
            _enrich_level5(record, embeddings, audio_path)

    except Exception as e:
        logger.warning("ML enrichment failed on %s: %s", Path(audio_path).name, e)


def _apply_bpm_prior(record: TrackFeatures, bpm_prior: tuple, audio_path: str) -> None:
    """Fold metrical-octave BPM errors into the crate's plausible range.

    Runs AFTER the ML block so the TempoCNN second opinion can break ties.
    bpm_raw always keeps the unfolded measurement (transparency + re-foldable
    later if the crate prior shifts). Mutates `record` in place.
    """
    record.bpm_raw = record.bpm
    if not bpm_prior:
        return
    lo, hi = float(bpm_prior[0]), float(bpm_prior[1])
    folded, ratio, suspect = _fold_bpm(record.bpm, record.bpm_cnn, lo, hi)
    record.bpm_suspect = suspect
    if ratio is not None:
        logger.info("bpm fold: %s %.1f -> %.1f (x%.3g, prior %.0f-%.0f, cnn=%s)",
                    Path(audio_path).name, record.bpm, folded, ratio, lo, hi,
                    f"{record.bpm_cnn:.1f}" if record.bpm_cnn else "n/a")
        record.bpm = folded
        record.bpm_fold_ratio = ratio
    elif suspect:
        logger.warning("bpm suspect: %s %.1f outside prior %.0f-%.0f and no "
                       "fold candidate fits — kept raw, flagged for review",
                       Path(audio_path).name, record.bpm, lo, hi)


def extract_features(audio_path: str, bpm_prior: tuple = None) -> TrackFeatures:
    """Run the full Essentia analysis chain on one audio file.

    Args:
        audio_path: Path to a .wav/.mp3/.flac decodable by MonoLoader.
        bpm_prior: optional (lo, hi) plausible-BPM range from the owning crate
            (learned stats or genre seed). When given, a raw BPM outside the
            range is folded by metrical ratios — see _fold_bpm(). bpm_raw always
            preserves the unfolded measurement.
    Returns:
        A fully populated TrackFeatures record. Expensive — call once per
        track and persist via the cache layer.
    """
    print(f"  📂 Loading {Path(audio_path).name}...")
    audio = es.MonoLoader(filename=audio_path, sampleRate=SAMPLE_RATE)()  # Decode to mono float32.
    duration = len(audio) / SAMPLE_RATE

    # ── Rhythm ──
    print("  🥁 Analysing rhythm...")
    # RhythmExtractor2013 returns (bpm, beats, confidence, _, _); multifeature is the most robust mode.
    bpm, _, bpm_conf, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
    onset_rate = es.OnsetRate()(audio)[1]           # OnsetRate returns (onsets, rate) — we want rate.
    dance = es.Danceability()(audio)[0]             # Danceability returns (score, dfa) — we want score.

    # ── Tonality ──
    # Reference tuning is measured FIRST and fed to key detection: an off-speed
    # vinyl rip is pitched as a whole, so aligning KeyExtractor's bins to the real
    # tuning keeps the tonic from drifting. Detection then votes across EDM profiles
    # and records how strongly they agreed (see _detect_key_robust).
    print("  🎼 Detecting key...")
    tuning_frequency = _estimate_tuning_frequency(audio)
    key, scale, key_strength, key_agreement = _detect_key_robust(
        audio, tuning_frequency=tuning_frequency)
    camelot = to_camelot(key, scale)

    # ── Loudness ──
    print("  🔊 Measuring loudness...")
    loudness_val = float(es.Loudness()(audio))
    replay = float(es.ReplayGain()(audio))
    dyn_comp, _ = es.DynamicComplexity()(audio)

    # ── Frame-level energy + spectral curves and mix points ──
    print("  📈 Computing energy curves...")
    curves = _extract_frame_curves(audio)
    print("  🎯 Detecting mix points...")
    intro_end, outro_start = _detect_mix_points(curves['complexity_curve'], duration)

    # Build the classic record up-front. ML fields are filled in below if available.
    record = TrackFeatures(
        path=audio_path,
        duration=duration,
        bpm=float(bpm),
        bpm_confidence=float(bpm_conf),
        key=key, scale=scale, camelot=camelot,
        key_strength=float(key_strength),
        agreement=float(key_agreement),
        danceability=float(dance),
        onset_rate=float(onset_rate),
        loudness=loudness_val,
        replay_gain=replay,
        dynamic_complexity=float(dyn_comp),
        spectral_centroid=curves['spectral_centroid'],
        spectral_complexity=curves['spectral_complexity'],
        spectral_flux=curves['spectral_flux'],
        spectral_rolloff=curves['spectral_rolloff'],
        zcr=curves['zcr'],
        mfcc_mean=curves['mfcc_mean'],
        bark_mean=curves['bark_mean'],
        energy_curve=curves['energy_curve'],
        complexity_curve=curves['complexity_curve'],
        tuning_frequency=tuning_frequency,
        intro_end=intro_end,
        outro_start=outro_start,
        pipeline_level=1,
    )

    # ── ML enrichment (Levels 2–5) + crate BPM-prior folding. ──
    # Both mutate `record` in place; the BPM fold runs AFTER ML so TempoCNN's
    # second opinion can break fold ties (see _apply_bpm_prior).
    _enrich_with_ml(record, audio_path)
    _apply_bpm_prior(record, bpm_prior, audio_path)

    return record


def embed_effnet(audio_16k) -> "list | None":
    """L2-normalised 1280-D EffNet embedding for an in-memory 16 kHz mono array.

    The same vector extract_features() stores, but computed from a buffer instead
    of a file — so a live snippet (listener.py) lands in the exact same embedding
    space as the crate's stored excerpts and can be matched against them with
    find_similar_effnet(). Returns None when the EffNet model is unavailable
    (essentia-tensorflow missing, or the .pb failed to download/load).

    Args:
        audio_16k: mono float32 samples at ML_SAMPLE_RATE (16 kHz).
    Returns:
        A 1280-element unit-norm list, or None.
    """
    model = ModelManager.get('effnet')
    if model is None:
        return None
    frames = model(audio_16k)                 # (T, 1280) per-frame embeddings.
    emb = np.mean(frames, axis=0)             # collapse time to one vector.
    norm = float(np.linalg.norm(emb))
    return (emb / norm).tolist() if norm > 0 else emb.tolist()


# ════════════════════════════════════════════════════════════
#  PAIRWISE COMPATIBILITY METRICS
# ════════════════════════════════════════════════════════════
def bpm_compatibility(b1: float, b2: float) -> float:
    """Score how close two BPMs are, with vinyl half/double folding.

    Args:
        b1, b2: BPM of the two tracks.
    Returns:
        1.0 for ±4 BPM, linearly decreasing to 0.5 at ±8, to 0.0 at ±16.
        Always picks the closest match among (b2, b2*2, b2/2) — accounts for
        the half/double-time ambiguity common in techno (and vinyl pitching).
    """
    if b1 == 0 or b2 == 0:
        return 0.0
    delta = min(abs(b1 - b2), abs(b1 - b2 * 2), abs(b1 - b2 / 2))
    if delta <= 4:
        return 1.0
    if delta <= 8:
        return 1.0 - 0.5 * (delta - 4) / 4               # 1.0 → 0.5 over (4..8].
    return max(0.0, 0.5 - 0.5 * (delta - 8) / 8)         # 0.5 → 0.0 over (8..16].


def bpm_delta(b1: float, b2: float) -> float:
    """Signed BPM delta accounting for half/double-time aliasing.

    Args:
        b1: BPM of the currently playing track.
        b2: BPM of the candidate track.
    Returns:
        The smallest-magnitude delta among (b2 - b1, 2*b2 - b1, b2/2 - b1).
        Used to display "+1.2 BPM" hints to the DJ.
    """
    candidates = [b2 - b1, b2 * 2 - b1, b2 / 2 - b1]
    return min(candidates, key=abs)


def cosine_sim(v1: list, v2: list) -> float:
    """Cosine similarity between two equal-length vectors. Returns 0.0 if either is zero-norm."""
    a, b = np.array(v1), np.array(v2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def timbre_compatibility(t1: TrackFeatures, t2: TrackFeatures) -> float:
    """Timbral similarity via cosine of the 13-D MFCC fingerprints (level-1 base)."""
    return cosine_sim(t1.mfcc_mean, t2.mfcc_mean)


def track_energy(t: TrackFeatures) -> float:
    """Mean RMS energy across the track (single scalar)."""
    return float(np.mean(t.energy_curve)) if t.energy_curve else 0.0


def energy_compatibility(t1: TrackFeatures, t2: TrackFeatures,
                         target: float = 0.0) -> float:
    """Score the energy move t1→t2 against a desired DIRECTION.

    `target` is the relative energy change the DJ wants next (0.0 = flat/stable,
    >0 = build/'up', <0 = relax/'down'). The reward band sits AROUND the target,
    keeping the original Oscar-Mulero asymmetry — undershooting the target is
    penalised harder than overshooting it, since a collapsing floor is the
    expensive mistake. With target=0.0 this reproduces the old flat-is-ideal
    behaviour. A harmonic step up a fifth (Camelot +1) reads as a small
    brightness lift and a step down as a slight relax, so it nudges the perceived
    energy delta — that is how the recovered 'energy boost' move now contributes.

    Returns 1.0 inside the band, falling to 0.0 as the move diverges from target.
    """
    e1, e2 = track_energy(t1), track_energy(t2)
    if e1 == 0 or e2 == 0:
        return 0.0
    delta = (e2 - e1) / max(e1, e2)                      # Relative change vs the louder of the two.
    delta += 0.05 * camelot_energy_direction(t1.camelot, t2.camelot)  # Harmonic brightness nudge.
    rel = delta - target                                 # Deviation from the wanted direction.
    if -0.05 <= rel <= 0.15:
        return 1.0                                       # Sweet spot, centred on target.
    if rel < -0.05:
        return max(0.0, 1.0 - (abs(rel) - 0.05) / 0.25)   # Under-shoot falls off over 25%.
    return max(0.0, 1.0 - (rel - 0.15) / 0.45)            # Over-shoot falls off over 45%.


def density_continuity(t1: TrackFeatures, t2: TrackFeatures) -> float:
    """Penalise jumps in spectral complexity (track "thickness")."""
    diff = abs(t1.spectral_complexity - t2.spectral_complexity) / \
        max(t1.spectral_complexity, t2.spectral_complexity, 1e-6)
    return max(0.0, 1.0 - diff)


# ════════════════════════════════════════════════════════════
#  EMOTIONAL FINGERPRINT  (Level 4 — assembled from individual classifiers)
# ════════════════════════════════════════════════════════════

def _build_emotional_vector(t: "TrackFeatures") -> "list | None":
    """Assemble the emotional fingerprint vector from individual mood scores.

    Iterates _EMOTIONAL_VECTOR_ORDER and collects every non-None value in
    that fixed order, so the resulting vector is always positionally consistent
    for a given set of available components. Two tracks' vectors are only
    comparable when they share the same components — use
    emotional_vector_similarity() for pair-wise comparison rather than
    comparing stored vectors directly.

    Args:
        t: A TrackFeatures record, typically after Level 3 or 4 enrichment.

    Returns:
        A list of floats (4–13 elements) when enough components are available,
        or None when fewer than 4 mood scores are present (insufficient for
        meaningful cosine similarity).
    """
    values = [getattr(t, field) for field in _EMOTIONAL_VECTOR_ORDER
              if getattr(t, field) is not None]
    return values if len(values) >= 4 else None


def emotional_vector_similarity(t1: "TrackFeatures",
                                 t2: "TrackFeatures") -> "float | None":
    """Cosine similarity between two tracks' emotional fingerprints.

    Builds aligned sub-vectors using only the components that are non-None
    in BOTH tracks, so Level-3 and Level-4 tracks can be compared without
    padding or error — they share the mood_aggressive + danceability_nn
    components at minimum once both reach Level 3.

    Args:
        t1: First TrackFeatures record.
        t2: Second TrackFeatures record.

    Returns:
        Cosine similarity in [0.0, 1.0], or None when fewer than 4
        components are shared (not enough dimensions for reliable similarity).
        A score of 1.0 means identical emotional profile; 0.0 means orthogonal.
    """
    # Collect the fields that are non-None in BOTH tracks — the intersection.
    shared = [f for f in _EMOTIONAL_VECTOR_ORDER
              if getattr(t1, f) is not None and getattr(t2, f) is not None]
    if len(shared) < 4:
        return None                         # Not enough dimensions to be meaningful.
    v1 = np.array([getattr(t1, f) for f in shared], dtype=np.float64)
    v2 = np.array([getattr(t2, f) for f in shared], dtype=np.float64)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return None
    # Clamp to [0, 1]: mood scores are all non-negative probabilities, so the
    # dot product is always ≥ 0, but floating-point noise can push it past 1.0.
    return float(np.clip(np.dot(v1, v2) / denom, 0.0, 1.0))


# ════════════════════════════════════════════════════════════
#  TWO-STAGE MIX SCORE  (EffNet backbone + DJ-controllable modifiers)
# ════════════════════════════════════════════════════════════
# Scoring philosophy (the inverse of the old weighted sum):
#
#   total = effnet_base × bpm × harmonic × energy × transition × mood × density
#
# Stage 1 — the EffNet cosine similarity is the IMMUTABLE semantic base. It can
#   never be weighted down, disabled, or overridden. When embeddings are missing
#   (level 1) it degrades to the MFCC cosine, which is equally immutable.
# Stage 2 — every DSP descriptor is a multiplicative modifier in (0, 1]. Each
#   carries a DJ-controllable `strength`: 0.0 disables it (always 1.0, no effect),
#   1.0 applies its raw penalty, >1.0 amplifies it (tighter matches only).
#
# Because modifiers multiply a ≤1 base by ≤1 factors, turning them all off
# collapses the ranking to pure EffNet cosine — the defining guarantee.
@dataclass
class ModifierStrengths:
    """Per-modifier intensity. 0.0=disabled, 1.0=full, 1.5=amplified."""
    bpm:        float = 1.0
    harmonic:   float = 1.0
    energy:     float = 1.0
    transition: float = 1.0
    mood:       float = 1.0
    emotional:  float = 1.0  # emotional vector similarity (Level 4 only)
    density:    float = 1.0
    mood_mode:  str   = 'similarity'  # 'similarity' | 'contrast'
    # Wanted energy DIRECTION for the next track (relative change): 0.0 = stable,
    # >0 = build/'up', <0 = relax/'down'. Reshapes the energy modifier's sweet
    # spot; the `up`/`stable`/`down` presets below are just named points on it.
    energy_target: float = 0.0


# Named energy-direction presets → an energy_target value. Continuous control is
# available directly via ModifierStrengths.energy_target (e.g. from a future API).
ENERGY_TARGETS = {'down': -0.30, 'stable': 0.0, 'up': 0.30}


# Per-mode default modifier strengths. CLI flags override these per invocation.
MODE_CONFIG = {
    'safe': {
        # Conservative: all modifiers active and amplified.
        # emotional=0.0: not enough data yet to trust for safe/no-surprise sets.
        'default_strengths': ModifierStrengths(
            bpm=1.5, harmonic=1.5, energy=1.0,
            transition=1.0, mood=0.0, emotional=0.0, density=1.0,
            mood_mode='similarity'
        ),
    },
    'balanced': {
        # All modifiers at natural strength; emotional contributes but doesn't dominate.
        'default_strengths': ModifierStrengths(
            bpm=1.0, harmonic=1.0, energy=1.0,
            transition=1.0, mood=1.0, emotional=0.5, density=1.0,
            mood_mode='similarity'
        ),
    },
    'creative': {
        # BPM/harmonic/transition nearly off — EffNet vibe dominates.
        # emotional=1.5 amplified: emotional coherence/contrast is the
        # primary creative signal when Level 4 data is available.
        'default_strengths': ModifierStrengths(
            bpm=0.3, harmonic=0.3, energy=0.8,
            transition=0.0, mood=1.0, emotional=1.5, density=0.5,
            mood_mode='contrast'
        ),
    },
}

# Camelot relationship → harmonic modifier value. Floored at 0.70 so even a
# dissonant pairing dampens rather than annihilates the semantic base.
HARMONIC_MOD_MAP = {
    'Same key': 1.0, 'Adjacent': 0.95, 'Relative (mood shift)': 0.90,
    'Dissonant': 0.70, 'Unknown': 0.70,
}

# Names of all DJ-controllable modifiers, in display order.
# 'emotional' sits between 'mood' and 'density': both are Level 3+/4 signals
# and grouping them together keeps the breakdown line readable.
MODIFIER_NAMES = ('bpm', 'harmonic', 'energy', 'transition', 'mood', 'emotional', 'density')


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return min(max(x, lo), hi)


def _apply_strength(mod_raw: float, strength: float) -> float:
    """Scale a raw modifier by its DJ strength.

    strength 0.0 → 1.0 (disabled), 1.0 → mod_raw (full), >1.0 amplifies the
    penalty (clamped at 0.0 so it never goes negative).
    """
    return _clamp(1.0 - strength * (1.0 - mod_raw), 0.0, 1.0)


def _effective_bpm(t: TrackFeatures) -> float:
    """BPM to match on: prefer TempoCNN when the DSP estimate is low-confidence."""
    if t.bpm_confidence < 0.7 and t.bpm_cnn:
        return t.bpm_cnn
    return t.bpm


def _bpm_mod_raw(t1: TrackFeatures, t2: TrackFeatures) -> float:
    """1.0 at ±4 BPM, falling to a 0.5 floor at ±16 (Technics SL-1200 range)."""
    return 0.5 + 0.5 * bpm_compatibility(_effective_bpm(t1), _effective_bpm(t2))


def _harmonic_mod_raw(t1: TrackFeatures, t2: TrackFeatures) -> float:
    """Camelot-relationship penalty, weighted by joint tonal confidence.

    A detected key is only as trustworthy as KeyExtractor's certainty, so the
    Camelot penalty is blended toward neutral by confidence = ks1 × ks2:
    two solidly tonal tracks apply (nearly) the full penalty, a tonal/atonal
    pair barely any, and borderline pairs a proportional partial one — no
    cliff at an arbitrary threshold. Range stays [0.70, 1.0]: full neutrality
    at zero confidence, the raw HARMONIC_MOD_MAP value at full confidence.

    Legacy mode (HARMONIC_CONTINUOUS_CONFIDENCE off): the original binary
    gate — either track under KEY_STRENGTH_THRESHOLD → exactly 1.0, otherwise
    the full Camelot penalty.
    """
    camelot_mod = HARMONIC_MOD_MAP.get(
        key_relationship_label(t1.camelot, t2.camelot), 0.70)
    if not HARMONIC_CONTINUOUS_CONFIDENCE:
        if min(t1.key_strength, t2.key_strength) < KEY_STRENGTH_THRESHOLD:
            return 1.0                                   # Atonal — harmonically neutral.
        return camelot_mod
    confidence = _clamp(t1.key_strength * t2.key_strength, 0.0, 1.0)
    # When BOTH tracks carry a multi-profile agreement score (re-analysed under the
    # voting path; 0.0 = a pre-vote row), let weak consensus pull the penalty toward
    # neutral too: a key the profiles disputed should not impose a full Camelot
    # penalty even if each profile was internally confident. Absent the signal on
    # either track, behaviour is unchanged (the legacy key_strength product).
    if t1.agreement > 0.0 and t2.agreement > 0.0:
        confidence *= t1.agreement * t2.agreement
    return 1.0 - confidence * (1.0 - camelot_mod)


def _energy_mod_raw(t1: TrackFeatures, t2: TrackFeatures, target: float = 0.0) -> float:
    """1.0 when the move matches the wanted direction, down to a 0.75 floor otherwise."""
    return 0.75 + 0.25 * energy_compatibility(t1, t2, target)


def _transition_mod_raw(t1: TrackFeatures, t2: TrackFeatures) -> tuple:
    """Mixable-window penalty. Returns (mod_raw, overlap_bars).

    Overlap = shorter of t1's outro and t2's intro, expressed in t1's bars.
    1.0 if ≥16 bars of headroom, else 0.85. Neutral (1.0, 0.0) when BPM unknown.
    """
    if t1.bpm <= 0:
        return 1.0, 0.0
    overlap_seconds = max(0.0, min(t1.duration - t1.outro_start, t2.intro_end))
    bars = overlap_seconds * t1.bpm / (60 * 4)
    return (1.0 if bars >= 16 else 0.85), bars


def _mood_mod_raw(t1: TrackFeatures, t2: TrackFeatures, mood_mode: str) -> tuple:
    """Mood-aggressive penalty. Returns (mod_raw, available).

    'similarity' rewards matching energy; 'contrast' rewards opposite moods.
    Neutral (1.0) whenever either track lacks a mood_aggressive score.
    """
    if t1.mood_aggressive is None or t2.mood_aggressive is None:
        return 1.0, False
    delta = abs(t1.mood_aggressive - t2.mood_aggressive)
    if mood_mode == 'contrast':
        return 0.5 + 0.5 * delta, True
    return 1.0 - 0.5 * delta, True


def _density_mod_raw(t1: TrackFeatures, t2: TrackFeatures) -> float:
    """Density-continuity penalty, floored at DENSITY_MOD_FLOOR.

    density_continuity() alone reaches 0.0 on an extreme thickness jump, which
    would annihilate the EffNet base — the only modifier without a floor. The
    floor keeps it a dampener like the rest (bpm 0.5, energy 0.75,
    transition 0.85, emotional 0.6): a full mismatch costs half the score,
    never all of it.
    """
    return DENSITY_MOD_FLOOR + (1.0 - DENSITY_MOD_FLOOR) * density_continuity(t1, t2)


def _emotional_mod_raw(t1: "TrackFeatures", t2: "TrackFeatures",
                       mood_mode: str) -> tuple:
    """Emotional-vector similarity penalty. Returns (mod_raw, available).

    Uses emotional_vector_similarity() to compare the two tracks' full
    emotional fingerprints across all shared mood components. Falls back
    gracefully to neutral (1.0) when fewer than 4 components are shared
    (Level 1–2 tracks, or only partial Level 3 data).

    Args:
        t1, t2:    TrackFeatures records for the two tracks being compared.
        mood_mode: 'similarity' rewards matching emotional registers;
                   'contrast' rewards opposite ones (creative mode).

    Returns:
        (mod_raw, available) where available=False means the modifier
        contributes a neutral 1.0 (no data) and available=True means
        the score reflects real emotional comparison.
    """
    sim = emotional_vector_similarity(t1, t2)
    if sim is None:
        return 1.0, False                   # Not enough shared components — neutral.
    # Floor at 0.6: even a fully mismatched emotional register should not
    # annihilate the EffNet vibe base score. The 0.4 range gives meaningful
    # differentiation without catastrophic penalties.
    if mood_mode == 'contrast':
        return 0.6 + 0.4 * (1.0 - sim), True   # Reward divergence.
    return 0.6 + 0.4 * sim, True               # Reward coherence.


def mix_score(t1: TrackFeatures, t2: TrackFeatures, mode: str = 'balanced',
              strengths: ModifierStrengths = None) -> dict:
    """Two-stage compatibility score: immutable EffNet base × DJ modifiers.

    Args:
        t1:        currently playing track.
        t2:        candidate next track.
        mode:      'safe' | 'balanced' | 'creative'. Unknown modes → 'balanced'.
        strengths: explicit per-modifier strengths. When None, the mode's
                   default_strengths are used (a copy, never mutated).

    Returns:
        dict with the immutable 'effnet_base', the seven EFFECTIVE modifier values
        ('bpm','harmonic','energy','transition','mood','emotional','density'),
        the multiplied 'total', plus 'timbre_source', 'mode', 'mood_mode',
        'modifier_strengths', and display helpers ('transition_raw',
        'overlap_bars', 'mood_available', 'emotional_available', and a
        'timbre' alias of the base for the unchanged mix_tip()).
    """
    strengths = _ensure_strengths(mode, strengths)

    # ── Stage 1: immutable semantic base (EffNet cosine; MFCC cosine fallback). ──
    if t1.effnet_embedding is not None and t2.effnet_embedding is not None:
        effnet_base = cosine_sim(t1.effnet_embedding, t2.effnet_embedding)
        timbre_source = 'effnet'
    else:
        effnet_base = timbre_compatibility(t1, t2)
        timbre_source = 'mfcc'
    # Floor at 0: a negative cosine (tracks pointing "away" in feature space) would
    # invert the multiplicative penalties below, so treat it as zero similarity.
    effnet_base = max(0.0, effnet_base)

    # ── Stage 2: DJ-controllable multiplicative modifiers. ──
    bpm_raw  = _bpm_mod_raw(t1, t2)
    harm_raw = _harmonic_mod_raw(t1, t2)
    eng_raw  = _energy_mod_raw(t1, t2, strengths.energy_target)
    trans_raw, overlap_bars   = _transition_mod_raw(t1, t2)
    mood_raw, mood_available   = _mood_mod_raw(t1, t2, strengths.mood_mode)
    emot_raw, emot_available   = _emotional_mod_raw(t1, t2, strengths.mood_mode)
    dens_raw = _density_mod_raw(t1, t2)

    bpm_eff   = _apply_strength(bpm_raw,  strengths.bpm)
    harm_eff  = _apply_strength(harm_raw, strengths.harmonic)
    eng_eff   = _apply_strength(eng_raw,  strengths.energy)
    trans_eff = _apply_strength(trans_raw, strengths.transition)
    mood_eff  = _apply_strength(mood_raw, strengths.mood)
    emot_eff  = _apply_strength(emot_raw, strengths.emotional)
    dens_eff  = _apply_strength(dens_raw, strengths.density)

    total = (effnet_base * bpm_eff * harm_eff * eng_eff
             * trans_eff * mood_eff * emot_eff * dens_eff)

    return {
        'total': total,
        'effnet_base': effnet_base,
        'timbre': effnet_base,            # Alias so the unchanged mix_tip() still reads a base.
        'bpm': bpm_eff, 'harmonic': harm_eff, 'energy': eng_eff,
        'transition': trans_eff, 'mood': mood_eff,
        'emotional': emot_eff, 'density': dens_eff,
        'transition_raw': trans_raw, 'overlap_bars': overlap_bars,
        'mood_available': mood_available,
        'emotional_available': emot_available,
        'timbre_source': timbre_source,
        'mode': mode,
        'mood_mode': strengths.mood_mode,
        'modifier_strengths': strengths,
    }


# ════════════════════════════════════════════════════════════
#  HUMAN-READABLE LABELS AND MIX TIPS
# ════════════════════════════════════════════════════════════
def key_relationship_label(c1: str, c2: str) -> str:
    """Convert a Camelot pair into a DJ-readable relationship label (clash axis).

    Returns one of: 'Same key', 'Adjacent', 'Relative (mood shift)', 'Dissonant',
    'Unknown'. This is the direction-AGNOSTIC clash read the harmonic modifier
    uses; the energy DIRECTION of an adjacent move (the old, unreachable
    'Energy boost' branch) lives in camelot_energy_direction() instead.
    """
    if not c1 or not c2 or '?' in (c1, c2):
        return "Unknown"
    if c1 == c2:
        return "Same key"
    n1, l1 = int(c1[:-1]), c1[-1]
    n2, l2 = int(c2[:-1]), c2[-1]
    diff = min(abs(n1 - n2), 12 - abs(n1 - n2))          # Shortest distance on the 12-tick wheel.
    if l1 == l2 and diff == 1:
        return "Adjacent"
    if n1 == n2 and l1 != l2:
        return "Relative (mood shift)"
    return "Dissonant"


def energy_direction(t1: TrackFeatures, t2: TrackFeatures) -> tuple:
    """Describe how energy moves from t1 to t2.

    Returns:
        (arrow, label, delta_pct) — e.g. ('↑', 'build', 12.4).
        Thresholds: ±10% defines the flat band.
    """
    e1, e2 = track_energy(t1), track_energy(t2)
    if e1 == 0:
        return ("?", "unknown", 0.0)
    delta_pct = 100.0 * (e2 - e1) / e1
    if delta_pct > 10:
        return ("↑", "build", delta_pct)
    if delta_pct < -10:
        return ("↓", "drop", delta_pct)
    return ("→", "flat", delta_pct)


def mix_tip(scores: dict, key_rel: str, bpm_d: float, energy_dir: str) -> str:
    """Return a one-line, Spanish-language mix instruction.

    Dispatches on the dominant signal in `scores` / `key_rel` / `bpm_d` —
    earlier branches win, so the order encodes priority (harmonic > adjacent >
    relative > BPM drift > timbre twin > energy drop > default).

    Args:
        scores:     dict produced by mix_score().
        key_rel:    label from key_relationship_label().
        bpm_d:      signed BPM delta from bpm_delta().
        energy_dir: 'build' | 'flat' | 'drop' | 'unknown'.
    """
    if scores['harmonic'] >= 1.0 and abs(bpm_d) <= 2:
        return "Long blend — overlap 32 bars, keep the kicks aligned."
    if "Adjacent" in key_rel:
        return "EQ swap on bar 16: cut the outgoing mids, let the new bass take over."
    if "Relative" in key_rel:
        return "Mood shift — drop it over a breakdown to mask the mode change."
    if abs(bpm_d) > 4:
        return f"Pitch {'+' if bpm_d > 0 else ''}{bpm_d:.0f} BPM or bridge with a percussive loop."
    if scores['timbre'] > 0.9:
        return "Timbral twin — long overlap with an HPF on the outgoing."
    if energy_dir == "drop":
        return "Energy drop — hard cut at end of phrase, avoid a long blend."
    return "HPF the outgoing, let the new kick breathe on its own, then bring elements in."


# ════════════════════════════════════════════════════════════
#  POSTGRESQL PERSISTENCE LAYER
# ════════════════════════════════════════════════════════════
def _hydrate(v: dict) -> TrackFeatures:
    """Rehydrate a features JSONB dict from PostgreSQL into a TrackFeatures record.

    Tolerates missing keys: every field falls back to its dataclass default,
    so tracks analysed at an older pipeline level load cleanly alongside new ones.
    """
    kwargs = {}
    for f in fields(TrackFeatures):
        if f.name in v:
            kwargs[f.name] = v[f.name]
    return TrackFeatures(**kwargs)


def _db_lookup(abs_path: str) -> "tuple[str | None, TrackFeatures | None]":
    """Look up a fully-analysed track in PostgreSQL by its absolute path.

    Returns:
        (track_id, TrackFeatures) on hit, (None, None) on miss or DB unavailable.
    """
    if not database.DB_AVAILABLE:
        return None, None
    try:
        row = database.get_track_by_path(abs_path)
        if row is None or row.get("analyzed_at") is None or not row.get("features"):
            return None, None
        return str(row["track_id"]), _hydrate(row["features"])
    except Exception as e:
        logger.warning("DB lookup failed for %s: %s", Path(abs_path).name, e)
        return None, None


def _model_version(name: str) -> str:
    """DB model_version string for a registered model = its .pb filename stem.

    e.g. 'effnet' → 'discogs-effnet-bs64-1'. Keys every embedding row so a model
    upgrade never silently overwrites vectors produced by the previous version.
    """
    return Path(ModelManager.REGISTRY[name][0]).stem


def persist_embeddings(track_id: str, features: dict) -> None:
    """Upsert every embedding vector present in `features` to its pgvector table.

    The single fan-out from a freshly-analysed track to the per-model embedding
    tables, so crate.py (the sole ingest authority) persists the full Level 2–5
    vector set in one call rather than hand-rolling each upsert (and forgetting
    the Level 4/5 ones, as the original crate path did). Best-effort per vector:
    a DB error on one is logged, never raised, so one bad upsert can't abort a batch.

    Args:
        track_id: the database primary key the vectors belong to.
        features: a TrackFeatures asdict()-style mapping (the same blob stored in
            tracks.features); only the keys that are present and non-empty persist.
    """
    if not database.DB_AVAILABLE:
        return
    for field, model_key, upsert in (
        ("effnet_embedding",         "effnet",             database.upsert_effnet_embedding),
        ("genre_discogs400",         "genre_discogs400",   database.upsert_genre_discogs400_embedding),
        ("jamendo_moodtheme_vector", "jamendo_moodtheme",  database.upsert_jamendo_moodtheme_embedding),
        ("jamendo_instrument",       "jamendo_instrument", database.upsert_jamendo_instrument_embedding),
    ):
        vec = features.get(field)
        if vec:
            try:
                upsert(track_id, vec, _model_version(model_key))
            except Exception as e:
                logger.warning("persist %s embedding failed for %s: %s", model_key, track_id, e)


def persist_session_embedding(session_id: str) -> "int | None":
    """Compute and store a saved session's EffNet centroid.

    The session embedding is the MEAN of its tracks' 1280-D EffNet vectors,
    re-L2-normalised so it sits on the same unit sphere as the track vectors it
    summarises (cosine search stays meaningful across both). No model inference:
    it averages vectors already in embeddings_effnet, so the cost is one small
    SQL read plus a numpy mean — safe to call inline on the save request.

    Args:
        session_id: the just-saved session.
    Returns:
        The number of track vectors pooled, or None when none were available
        (e.g. every track was de-indexed) — the session simply has no centroid.
    """
    if not database.DB_AVAILABLE:
        return None
    model_version = _model_version("effnet")
    vecs = database.session_track_vectors(session_id, model_version)
    if not vecs:
        logger.info("session %s has no EffNet vectors to pool — no centroid", session_id)
        return None
    centroid = np.mean(np.array(vecs, dtype=np.float64), axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    database.upsert_session_embedding(
        session_id, centroid.tolist(), model_version, len(vecs))
    logger.info("session %s centroid stored (pooled %d tracks)", session_id, len(vecs))
    return len(vecs)


def persist_artist_embedding(artist_id: str) -> "int | None":
    """Compute and store an artist's EffNet centroid (mean of their tracks).

    Same trick as sessions: the MEAN of the artist's tracks' 1280-D EffNet
    vectors, re-L2-normalised onto the unit sphere, so 'artists who sound like
    X' is one cosine ANN in the same space as tracks. No model inference — it
    averages vectors already in embeddings_effnet.

    Returns the number of track vectors pooled, or None when the artist has no
    analysed tracks yet (no centroid).
    """
    if not database.DB_AVAILABLE:
        return None
    model_version = _model_version("effnet")
    vecs = database.artist_track_vectors(artist_id, model_version)
    if not vecs:
        return None
    centroid = np.mean(np.array(vecs, dtype=np.float64), axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    database.upsert_artist_embedding(
        artist_id, centroid.tolist(), model_version, len(vecs))
    logger.info("artist %s centroid stored (pooled %d tracks)", artist_id, len(vecs))
    return len(vecs)


def backfill_all_artist_embeddings() -> int:
    """(Re)compute every artist's centroid. Returns how many got one."""
    if not database.DB_AVAILABLE:
        return 0
    done = 0
    for a in database.list_artists():
        if persist_artist_embedding(str(a["artist_id"])):
            done += 1
    logger.info("backfill_all_artist_embeddings: %d artist centroids", done)
    return done


def persist_label_embedding(label_id: str) -> "int | None":
    """Compute and store a label's EffNet centroid (mean of its tracks).

    Identical mechanism to artists/sessions: mean of the label's tracks' 1280-D
    EffNet vectors, re-L2-normalised — so 'labels that sound like X' is one cosine
    ANN in the same space. No model inference. Returns tracks pooled, or None when
    the label has no analysed tracks yet."""
    if not database.DB_AVAILABLE:
        return None
    model_version = _model_version("effnet")
    vecs = database.label_track_vectors(label_id, model_version)
    if not vecs:
        return None
    centroid = np.mean(np.array(vecs, dtype=np.float64), axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    database.upsert_label_embedding(label_id, centroid.tolist(), model_version, len(vecs))
    logger.info("label %s centroid stored (pooled %d tracks)", label_id, len(vecs))
    return len(vecs)


def backfill_all_label_embeddings() -> int:
    """(Re)compute every label's centroid. Returns how many got one."""
    if not database.DB_AVAILABLE:
        return 0
    done = 0
    for l in database.list_labels():
        if persist_label_embedding(str(l["label_id"])):
            done += 1
    logger.info("backfill_all_label_embeddings: %d label centroids", done)
    return done


def _get_or_analyze(path: str) -> TrackFeatures:
    """Return TrackFeatures for a path: DB hit → hydrate instantly, miss → extract transiently.

    The crate (crate.py) is the single source of truth for STORED tracks, so this
    never inserts: it only reads. A crate excerpt already analysed is hydrated
    straight from PostgreSQL by its crate_path; any other file (an ad-hoc
    `analyze`/`compare`/`mixpoints` on a track not yet in the crate) is analysed
    in-memory for display only and not written back — that keeps a single
    crate_path convention in the tracks table (all rows point at ./crate/<id>.wav).
    """
    abs_path = str(Path(path).resolve())
    _, f = _db_lookup(abs_path)
    if f is not None:
        return f
    return extract_features(path)


def _load_library(exclude_path: str = None, crate: str = "__active__") -> list:
    """Load analysed tracks from PostgreSQL into memory as [(path, TrackFeatures)].

    Called once at the start of next/setlist commands so scoring never hits the
    DB per candidate — all comparisons run against the in-memory list.

    Args:
        exclude_path: Optional file path to omit (the currently-playing track).
        crate: "__active__" (default) scopes recommendations to the active crate
            — in a single-crate world this equals the old all-tracks behaviour.
            Pass an explicit crate name/id to scope elsewhere, or None to load
            EVERY crate (the cross-crate view).
    Returns:
        List of (crate_path, TrackFeatures) tuples, newest-first.
    """
    if not database.DB_AVAILABLE:
        print("\n⚠️  Database unavailable — start Docker: docker compose up -d\n")
        return []
    try:
        if crate == "__active__":
            crate_id = database.active_crate_id()
        elif crate is None:
            crate_id = None
        else:
            crate_id = database.resolve_crate_id(crate)
        rows = database.list_tracks(analyzed_only=True, crate_id=crate_id)
    except Exception as e:
        logger.warning("Failed to load library from DB: %s", e)
        return []
    exclude_abs = str(Path(exclude_path).resolve()) if exclude_path else None
    result = []
    for row in rows:
        if not row.get("features"):
            continue
        p = row["crate_path"]
        if exclude_abs and str(Path(p).resolve()) == exclude_abs:
            continue
        try:
            result.append((p, _hydrate(row["features"])))
        except Exception as e:
            logger.warning("Failed to hydrate track %s: %s", p, e)
    return result


# RETRIEVAL_K (config.py) is the stage-1 breadth of the two-stage retrieval
# below. Modifiers <= 1.0 only guarantee total <= own effnet_base — NOT that the
# winner lives inside the top-K by base: when the whole window gets penalised, a
# candidate just outside it with a slightly lower base but no penalties wins.
# score_candidates() closes that hole by expanding the window until the bound
# proves the winner is inside; K only sets the first fetch size.


def _load_candidates(current: "TrackFeatures", exclude_paths=None,
                     crate: str = "__active__", k: int = RETRIEVAL_K) -> list:
    """Two-stage retrieval, stage 1: the top-k EffNet neighbours of `current`.

    Replaces the O(N) full-library load on every recommendation with an
    O(log N + k) HNSW lookup — recommendation latency stays constant no matter
    how large the crate grows. Stage 2 (mix_score over what this returns) is
    unchanged, so ranking semantics are identical to the full scan within the
    retrieved set.

    Falls back to _load_library() whenever stage 1 cannot run (the current
    track has no embedding — Level 1 analysis — or the ANN query fails), so
    behaviour degrades to the old full scan rather than to an empty slate.

    Args:
        current: the playing/seed track whose neighbours we want.
        exclude_paths: iterable of file paths to omit (the seed itself, or a
            setlist's already-used tracks).
        crate: same semantics as _load_library ("__active__" | name/id | None).
        k: stage-1 breadth (net of exclusions).
    Returns:
        List of (crate_path, TrackFeatures), nearest-first.
    """
    exclude = {str(Path(p).resolve()) for p in (exclude_paths or [])}

    if current.effnet_embedding is None or not database.DB_AVAILABLE:
        lib = _load_library(crate=crate)
        return [(p, f) for p, f in lib if str(Path(p).resolve()) not in exclude]

    try:
        if crate == "__active__":
            crate_id = database.active_crate_id()
        elif crate is None:
            crate_id = None
        else:
            crate_id = database.resolve_crate_id(crate)
        # Over-fetch by the exclusion count so k NET candidates survive the filter.
        rows = database.find_similar_effnet(
            current.effnet_embedding, n=k + len(exclude), crate_id=crate_id)
    except Exception as e:
        logger.warning("ANN candidate retrieval failed (%s) — falling back to "
                       "full library scan", e)
        lib = _load_library(crate=crate)
        return [(p, f) for p, f in lib if str(Path(p).resolve()) not in exclude]

    result = []
    for row in rows:
        if not row.get("features"):
            continue
        p = row["crate_path"]
        if str(Path(p).resolve()) in exclude:
            continue
        try:
            result.append((p, _hydrate(row["features"])))
        except Exception as e:
            logger.warning("Failed to hydrate candidate %s: %s", p, e)
        if len(result) >= k:
            break
    return result


def score_candidates(current: "TrackFeatures", mode: str = 'balanced',
                     strengths: "ModifierStrengths" = None, exclude_paths=None,
                     crate: str = "__active__", k: int = RETRIEVAL_K) -> list:
    """Retrieve and mix_score candidates with an exact-winner safeguard.

    Truncating at the top-k by EffNet base is NOT safe on its own: modifiers
    <= 1.0 bound each total by its own base, but a candidate outside the window
    (base slightly below the k-th) that takes no penalties can still beat an
    entirely penalised window. The safeguard uses the bound the other way:
    every unfetched track has base <= the smallest base fetched (nearest-first),
    so once best_total >= that frontier the true winner is provably inside the
    scored set — otherwise the window doubles and re-scores. Exactness is
    modulo HNSW recall (the ANN index itself is approximate).

    Args:
        current: the playing/seed track.
        mode / strengths: scoring configuration, as in mix_score(). The bound
            depends on the active penalties, so callers comparing several
            strength profiles (e.g. energy up vs down) must call once per profile.
        exclude_paths / crate / k: as in _load_candidates(); k is the FIRST
            window size only.
    Returns:
        List of (crate_path, TrackFeatures, scores_dict), nearest-first by base
        (unsorted by total — sample_by_score() handles ranking/sampling).
    """
    strengths = _ensure_strengths(mode, strengths)
    n, prev_count = k, -1
    while True:
        cands = _load_candidates(current, exclude_paths=exclude_paths,
                                 crate=crate, k=n)
        scored = [(p, f, mix_score(current, f, mode=mode, strengths=strengths))
                  for p, f in cands]
        # Window can't grow further (empty, exhausted crate, or the full-scan
        # fallback already returned everything) — what we have is exact.
        if not scored or len(scored) <= prev_count or len(scored) != n:
            return scored
        best_total = max(s['total'] for _, _, s in scored)
        frontier = scored[-1][2]['effnet_base']   # Smallest base in the window.
        if best_total >= frontier:
            return scored                          # No unseen track can win.
        prev_count = len(scored)
        n *= 2
        logger.debug("score_candidates: best total %.3f < frontier base %.3f — "
                     "expanding window to %d", best_total, frontier, n)


def _download_model(name: str, url: str, dest: Path) -> None:
    """Stream-download one .pb (plus its JSON class-label metadata) with progress.

    Downloads are ATOMIC: each file lands first at a `.part` sibling and is renamed
    onto its final name only after the transfer completes. An interrupted download
    (Ctrl-C, network drop, kill) therefore never leaves a truncated `.pb` at `dest`
    — which `get()`/`download_all()` would see via `dest.exists()` and never
    re-fetch, silently degrading the pipeline a level forever — only a stray
    `.part` that the next attempt overwrites.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇  Downloading {name} …")

    def _progress(blocks, block_size, total):
        if total <= 0:
            return
        pct     = min(100, int(100 * blocks * block_size / total))
        mb_done = blocks * block_size / 1_048_576
        mb_tot  = total / 1_048_576
        sys.stdout.write(f"\r     {pct:3d}%  ({mb_done:.1f} / {mb_tot:.1f} MB)")
        sys.stdout.flush()

    part = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, part, reporthook=_progress)
        part.replace(dest)            # Atomic on the same filesystem.
    except BaseException:             # incl. KeyboardInterrupt — clean up the stub.
        part.unlink(missing_ok=True)
        raise
    print()

    # JSON metadata (class labels, output shapes, etc.) — best-effort, but LOUD on
    # failure: labels()/class_index()/output_node() silently degrade without it.
    json_dest = dest.with_suffix('.json')
    if not json_dest.exists():
        json_url = url.replace('.pb', '.json')
        json_part = json_dest.with_name(json_dest.name + ".part")
        try:
            urllib.request.urlretrieve(json_url, json_part)
            json_part.replace(json_dest)
        except Exception as e:
            json_part.unlink(missing_ok=True)
            logger.warning("Metadata download failed for '%s' (%s): %s — "
                           "class labels/IO nodes will fall back to defaults.",
                           name, json_url, e)


# ════════════════════════════════════════════════════════════
#  CLI COMMANDS
# ════════════════════════════════════════════════════════════
def cmd_analyze(path: str):
    """Print a one-track feature summary (analyses on cache miss).

    Args:
        path: Absolute or relative path to an audio file.
    """
    f = _get_or_analyze(path)
    sep = "=" * 60
    print(f"\n{sep}\n  🎧  {Path(path).name}\n{sep}")
    print(f"  Duration:             {f.duration/60:.2f} min")
    print(f"  BPM:                  {f.bpm:.2f}  (confidence {f.bpm_confidence:.2f})")
    print(f"  Key:                  {f.key} {f.scale}  →  Camelot {f.camelot}")
    print(f"  Key strength:         {f.key_strength:.2f}")
    if f.agreement:
        print(f"  Key agreement:        {f.agreement:.0%}  (multi-profile vote)")
    cents = _tuning_cents(f.tuning_frequency)
    if f.tuning_frequency > 0 and abs(cents) > 20:
        print(f"  Tuning:               {f.tuning_frequency:.1f} Hz  "
              f"({cents:+.0f} cents — possible vinyl speed offset)")
    print(f"  Onset rate:           {f.onset_rate:.2f} /s")
    print(f"  Danceability:         {f.danceability:.2f}")
    print(f"  Loudness:             {f.loudness:.0f}")
    print(f"  ReplayGain:           {f.replay_gain:.2f} dB")
    print(f"  Dynamic complexity:   {f.dynamic_complexity:.2f}")
    print(f"  Spectral centroid:    {f.spectral_centroid:.0f} Hz")
    print(f"  Spectral complexity:  {f.spectral_complexity:.2f}")
    print(f"  Roll-off:             {f.spectral_rolloff:.0f} Hz")
    print(f"  Intro ends at:        {f.intro_end:.1f} s")
    print(f"  Outro starts at:      {f.outro_start:.1f} s  ({f.duration - f.outro_start:.1f} s of outro)")
    # ── ML extras — only show fields that were actually computed. ──
    print(f"  Pipeline level:       {f.pipeline_level}/5")
    if f.bpm_cnn is not None:
        print(f"  BPM (TempoCNN):       {f.bpm_cnn:.2f}")
    if f.mood_aggressive is not None:
        print(f"  Mood aggressive:      {f.mood_aggressive:.2f}")
    if f.danceability_nn is not None:
        print(f"  Danceability (NN):    {f.danceability_nn:.2f}")
    if f.effnet_embedding is not None:
        print(f"  EffNet embedding:     {len(f.effnet_embedding)}-D vector")
    # ── Level 4: full emotional fingerprint with 20-char bar graph ──
    if f.pipeline_level >= 4:
        print(f"  ── Emotional fingerprint ────────────────────────")
        for fname, label in [
            ('mood_electronic',   'Electronic'),
            ('mood_sad',          'Sad'),
            ('mood_relaxed',      'Relaxed'),
            ('mood_happy',        'Happy'),
            ('mood_party',        'Party'),
            ('jamendo_dark',      'Dark [Jamendo]'),
            ('jamendo_groovy',    'Groovy [Jamendo]'),
            ('jamendo_meditative','Meditative [J.]'),
            ('jamendo_energetic', 'Energetic [J.]'),
            ('jamendo_heavy',     'Heavy [Jamendo]'),
            ('jamendo_space',     'Space [Jamendo]'),
        ]:
            val = getattr(f, fname)
            if val is not None:
                bar = '█' * int(val * 20)     # 20-char bar for quick visual scan
                print(f"  {label:<22} {val:.2f}  {bar}")
        if f.emotional_vector is not None:
            print(f"  Emotional vector:     {len(f.emotional_vector)}-D")

    # ── Level 5: genre, voice/tonal/timbre/approachability/engagement ──
    if f.pipeline_level >= 5:
        print(f"  ── Extended characterisation ────────────────────")
        for fname, label in [
            ('voice_instrumental', 'Instrumental'),
            ('tonal',              'Tonal'),
            ('timbre_bright',      'Timbre bright'),
            ('approachability',    'Approachability'),
            ('engagement',         'Engagement'),
        ]:
            val = getattr(f, fname)
            if val is not None:
                bar = '█' * int(val * 20)
                print(f"  {label:<22} {val:.2f}  {bar}")

        # Top-5 genre labels from the 400-D Discogs style vector.
        if f.genre_discogs400 is not None:
            labels = ModelManager.labels('genre_discogs400')
            vec = f.genre_discogs400
            top5 = sorted(range(len(vec)), key=lambda i: vec[i], reverse=True)[:5]
            print(f"  ── Top genres (Discogs 400) ─────────────────────")
            for idx in top5:
                lbl = labels[idx] if idx < len(labels) else f"genre_{idx}"
                bar = '█' * int(vec[idx] * 20)
                print(f"  {lbl:<28} {vec[idx]:.2f}  {bar}")

        # Top-5 instrument labels from the 40-D Jamendo instrument vector.
        if f.jamendo_instrument is not None:
            labels = ModelManager.labels('jamendo_instrument')
            vec = f.jamendo_instrument
            top5 = sorted(range(len(vec)), key=lambda i: vec[i], reverse=True)[:5]
            print(f"  ── Top instruments (Jamendo 40) ─────────────────")
            for idx in top5:
                lbl = labels[idx] if idx < len(labels) else f"instrument_{idx}"
                bar = '█' * int(vec[idx] * 20)
                print(f"  {lbl:<28} {vec[idx]:.2f}  {bar}")

    print(sep + "\n")


def cmd_scan(folder: str):
    """Batch-import a folder into the crate, then analyse it — single ingest path.

    Ingestion + standardisation + persistence are owned by crate.py (every stored
    track is a ./crate/<id>.wav with a crate_path-keyed row), so scanning delegates
    to it rather than inserting raw source-file paths itself. This keeps ONE
    crate_path convention in the tracks table. crate.add_from_folder() standardises
    + dedups (re-scanning the same folder is a no-op); analyze_pending() then runs
    the full feature pipeline on everything still unanalysed.

    Args:
        folder: Path to a directory of audio files (non-recursive).
    """
    # Late import: crate.py imports analyze at module load, so importing it here
    # (not at top) avoids a circular import while still routing through the one
    # ingest authority.
    import crate
    print(f"\n🔍 Importing {folder} into the crate...\n")
    crate.add_from_folder(folder)
    crate.analyze_pending()

    # Pipeline-level summary from DB — health check for ML coverage across the library.
    library = _load_library()
    print(f"\n✅ {len(library)} tracks in library.\n")
    levels = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for _, v in library:
        levels[v.pipeline_level] = levels.get(v.pipeline_level, 0) + 1
    print("📊 Pipeline-level breakdown:")
    print(f"     Level 1 (classic):               {levels[1]}")
    print(f"     Level 2 (+EffNet/Tempo):          {levels[2]}")
    print(f"     Level 3 (mood + danceability):    {levels[3]}")
    print(f"     Level 4 (full emotional):         {levels[4]}")
    print(f"     Level 5 (genre + extended audio): {levels[5]}")
    print()


def cmd_mixpoints(path: str):
    """Report intro/outro mix points for a single track, in seconds and bars."""
    f = _get_or_analyze(path)
    print(f"\n🎯  MIX POINTS — {Path(path).name}\n")
    print(f"  ▶  Optimal entry:   0:00 → {f.intro_end:.1f}s  (intro zone)")
    print(f"  ⏏  Optimal exit:    {f.outro_start:.1f}s → {f.duration:.1f}s  (outro zone)")
    print(f"\n  Mixable intro length: {f.intro_end:.1f}s")
    print(f"  Mixable outro length: {f.duration - f.outro_start:.1f}s")

    # Convert seconds to bars assuming 4/4 (4 beats per bar). Useful for DJ phrasing.
    if f.bpm > 0:
        bars_per_sec = f.bpm / (60 * 4)
        intro_bars = f.intro_end * bars_per_sec
        outro_bars = (f.duration - f.outro_start) * bars_per_sec
        print(f"\n  📏 In bars (at {f.bpm:.0f} BPM):")
        print(f"     Intro ≈ {intro_bars:.0f} bars  (~{intro_bars/32:.1f} 32-bar phrases)")
        print(f"     Outro ≈ {outro_bars:.0f} bars  (~{outro_bars/32:.1f} 32-bar phrases)")
    print()


# (header label, breakdown abbreviation) for each modifier — one dict, two uses.
_MOD_LABELS = {
    'bpm':        ('bpm',        'bpm'),
    'harmonic':   ('harm',       'harm'),
    'energy':     ('energy',     'eng'),
    'transition': ('transition', 'trans'),
    'mood':       ('mood',       'mood'),
    'emotional':  ('emot',       'emot'),  # neutral when no Level 4 data
    'density':    ('density',    'density'),
}


def _format_modifiers(strengths: ModifierStrengths) -> str:
    """One-line summary of the active modifier configuration for command headers.

    Example: 'Modifiers: bpm ×1.0 · harm ×0.0 [off] · energy ×1.0 · …'
    Tags: [off] when strength=0.0, [amplified] when >1.0, [contrast] on mood.
    """
    parts = []
    for m in MODIFIER_NAMES:
        val = getattr(strengths, m)
        tag = ""
        if val == 0.0:
            tag = " [off]"
        elif val > 1.0:
            tag = " [amplified]"
        if m == 'mood' and strengths.mood_mode == 'contrast':
            tag += " [contrast]"
        parts.append(f"{_MOD_LABELS[m][0]} ×{val:.1f}{tag}")
    return "Modifiers: " + " · ".join(parts)


def _format_breakdown(score: dict, strengths: ModifierStrengths) -> str:
    """Per-result score breakdown: the immutable base then each effective modifier.

    Example: 'base(EffNet) 0.87 | ×0.95 bpm · ×1.00 harm · [off] eng · …'
    A disabled modifier (strength 0.0) shows '[off]' instead of a multiplier.
    """
    base_tag = 'EffNet' if score['timbre_source'] == 'effnet' else 'MFCC'
    parts = []
    for m in MODIFIER_NAMES:
        label = _MOD_LABELS[m][1]
        if getattr(strengths, m) == 0.0:
            parts.append(f"[off] {label}")
        else:
            parts.append(f"×{score[m]:.2f} {label}")
    return f"base({base_tag}) {score['effnet_base']:.2f} | " + " · ".join(parts)


def _ensure_strengths(mode: str, strengths: ModifierStrengths = None) -> ModifierStrengths:
    """Return strengths unchanged if provided, else a fresh copy of the mode preset."""
    if strengths is not None:
        return strengths
    return replace(MODE_CONFIG.get(mode, MODE_CONFIG['balanced'])['default_strengths'])


def sample_by_score(scored: list, n: int, temperature: float = 0.0) -> list:
    """Select `n` of the `scored` candidates by their mix_score 'total'.

    `scored` is a list of (key, value, score_dict) tuples — the shape cmd_next
    and listener._recommend build. temperature 0.0 returns the deterministic
    top-n (pure exploit); temperature > 0 samples without replacement from a
    softmax over the totals, so a higher temperature surfaces more adventurous
    picks (explore) — the LLM-sampling analogue the DJ dials for variety.
    Returned items are always ordered best-first for display.
    """
    ranked = sorted(scored, key=lambda x: x[2]['total'], reverse=True)
    if temperature <= 0.0 or len(ranked) <= n:
        return ranked[:n]
    totals = np.array([s[2]['total'] for s in ranked], dtype=np.float64)
    logits = totals / temperature
    logits -= logits.max()                       # Stabilise exp() against overflow.
    probs = np.exp(logits)
    probs /= probs.sum()
    chosen = np.random.default_rng().choice(len(ranked), size=n, replace=False, p=probs)
    return [ranked[i] for i in sorted(chosen)]   # ranked is best-first → keep that order.


def _print_picks(current: TrackFeatures, picks: list,
                 strengths: ModifierStrengths, start: int = 1) -> None:
    """Print a ranked block of (path, features, score) recommendations."""
    for i, (k, v, s) in enumerate(picks, start):
        bpm_d   = bpm_delta(current.bpm, v.bpm)
        key_rel = key_relationship_label(current.camelot, v.camelot)
        arrow, dir_label, energy_pct = energy_direction(current, v)
        tip     = mix_tip(s, key_rel, bpm_d, dir_label)

        header = f"  {i}. {Path(k).name}"
        if s['total'] >= PERFECT_MIX_THRESHOLD:               # Flag ★ when score ≥ threshold.
            header += f"   ★ PERFECT MIX"
        # Warn on a tight mixable window, but only while the transition modifier is active.
        if strengths.transition > 0 and s['transition_raw'] < 0.86:
            header += f"   ⚠ short mix window (~{s['overlap_bars']:.0f} bars)"
        print(f"\n{header}")
        timb_tag = 'EffNet' if s['timbre_source'] == 'effnet' else 'MFCC'
        print(f"     Total {s['total']:.2f}")
        print(f"     {_format_breakdown(s, strengths)}")
        bpm_str = f"{v.bpm:.0f} BPM  (Δ {bpm_d:+.1f})"
        key_str = f"{current.camelot} → {v.camelot}  ({key_rel})"
        eng_str = f"{arrow} {dir_label} ({energy_pct:+.0f}%)"
        print(f"     BPM:    {bpm_str:<22}  Key:    {key_str}")
        print(f"     Energy: {eng_str:<22}  Timbre: {s['effnet_base']:.2f} ({timb_tag})")
        print(f"     ▶  {tip}")


def cmd_next(path: str, mode: str = 'balanced', strengths: ModifierStrengths = None,
             temperature: float = 0.0, energy: str = None):
    """Print the next-track recommendations to mix into `path`.

    Args:
        path: Path of the currently playing track. Must be in the cache (or it
            will be analysed on the fly). Other tracks must already be indexed
            via `cmd_scan` for there to be any candidates.
        mode: 'safe' | 'balanced' | 'creative' — see MODE_CONFIG.
        strengths: explicit modifier strengths; defaults to the mode preset.
        temperature: 0.0 = deterministic top picks; >0 samples for variety.
        energy: None / 'up' / 'stable' / 'down' (set on `strengths` upstream),
            or 'both' to print an energy-up AND an energy-down slate side by side.
    """
    strengths = _ensure_strengths(mode, strengths)
    current = _get_or_analyze(path)
    # Cheap emptiness probe (n=1) — each slate below retrieves and scores its
    # own window because the safeguard bound depends on the active strengths.
    if not _load_candidates(current, exclude_paths=[path], k=1):
        print("\n⚠️  No other tracks in the library. Scan a folder first:")
        print("   uv run python analyze.py scan <folder>\n")
        return

    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  🎧  NOW PLAYING — {Path(path).name}")
    print(f"{sep}")
    print(f"  {current.bpm:.0f} BPM · {current.camelot} ({current.key} {current.scale})"
          f" · energy {track_energy(current):.3f}")
    # Active scoring configuration — printed so the DJ always knows the exact setup.
    print(f"  Mode: {mode}  ·  temperature {temperature:.1f}")
    print(f"  {_format_modifiers(strengths)}\n")

    def _slate(strs: ModifierStrengths, n: int) -> list:
        scored = score_candidates(current, mode=mode, strengths=strs,
                                  exclude_paths=[path])
        return sample_by_score(scored, n, temperature)

    if energy == 'both':
        # Two short slates: one that lifts the floor, one that eases it back.
        for title, tgt in (("⏫  ENERGY UP", ENERGY_TARGETS['up']),
                           ("⏬  ENERGY DOWN", ENERGY_TARGETS['down'])):
            strs = replace(strengths, energy_target=tgt)
            print(f"🎚️  {title}")
            print("─" * 64)
            _print_picks(current, _slate(strs, 3), strs)
            print()
    else:
        print(f"🎚️  TOP 5 RECOMMENDATIONS")
        print("─" * 64)
        _print_picks(current, _slate(strengths, 5), strengths)
        print()


def cmd_compare(path1: str, path2: str, mode: str = 'balanced',
                strengths: ModifierStrengths = None):
    """Full sub-score breakdown between exactly two tracks.

    Args:
        path1: Outgoing / current track.
        path2: Incoming / candidate track.
        mode:  'safe' | 'balanced' | 'creative'.
        strengths: explicit modifier strengths; defaults to the mode preset.
    """
    strengths = _ensure_strengths(mode, strengths)
    a = _get_or_analyze(path1)
    b = _get_or_analyze(path2)
    s = mix_score(a, b, mode=mode, strengths=strengths)

    bpm_d   = bpm_delta(a.bpm, b.bpm)
    key_rel = key_relationship_label(a.camelot, b.camelot)
    arrow, dir_label, energy_pct = energy_direction(a, b)
    tip     = mix_tip(s, key_rel, bpm_d, dir_label)
    level   = min(a.pipeline_level, b.pipeline_level)    # Pair is limited by the weaker record.

    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  🎚️   COMPARE  (mode: {mode}, pair pipeline level: {level}/5)")
    print(f"{sep}")
    print(f"  {_format_modifiers(strengths)}")
    print(f"  A: {Path(path1).name}")
    print(f"     {a.bpm:.1f} BPM · {a.camelot} ({a.key} {a.scale}) · energy {track_energy(a):.3f}")
    print(f"  B: {Path(path2).name}")
    print(f"     {b.bpm:.1f} BPM · {b.camelot} ({b.key} {b.scale}) · energy {track_energy(b):.3f}")
    print("─" * 64)
    print(f"  Total score:   {s['total']:.3f}")
    print(f"  {_format_breakdown(s, strengths)}")
    print(f"  Harmonic rel:  {key_rel}")
    print(f"  BPM delta:     {bpm_d:+.2f} BPM")
    print(f"  Energy dir:    {arrow} {dir_label} ({energy_pct:+.1f}%)")
    if s['mood_available']:
        print(f"  Mood match:    {s['mood']:.3f}   "
              f"(|Δ aggressive| = {abs(a.mood_aggressive - b.mood_aggressive):.2f}, "
              f"mode: {s['mood_mode']})")
    if s['emotional_available']:
        print(f"  Emot. match:   {s['emotional']:.3f}  "
              f"(emotional vector similarity, mode: {s['mood_mode']})")
    # Tuning delta + vinyl-speed reconciliation. A platter running off-nominal
    # scales tempo AND pitch by the SAME factor, so when two versions of a track
    # differ a small, PROPORTIONAL amount in both BPM and tuning it is almost
    # certainly one record at two speeds — not two different tracks.
    if a.tuning_frequency > 0 and b.tuning_frequency > 0 and a.bpm > 0 and b.bpm > 0:
        tuning_ratio = b.tuning_frequency / a.tuning_frequency
        bpm_ratio = b.bpm / a.bpm
        cents = 1200.0 * np.log2(tuning_ratio)
        speed_pct = (tuning_ratio - 1.0) * 100.0
        proportional = abs(bpm_ratio - tuning_ratio) < 0.005      # within 0.5%
        if 0.003 < abs(tuning_ratio - 1.0) < 0.03 and proportional:
            print(f"  Vinyl speed:   {speed_pct:+.2f}% (B vs A) — same track at a "
                  f"different platter speed ({cents:+.0f} cents)")
        elif abs(a.tuning_frequency - b.tuning_frequency) > 2:
            print(f"  Tuning delta:  {abs(a.tuning_frequency - b.tuning_frequency):.1f} Hz")
    print("─" * 64)
    print(f"  ▶  {tip}")
    print(sep + "\n")


def cmd_setlist(path: str, length: int = 8, mode: str = 'balanced',
                strengths: ModifierStrengths = None, temperature: float = 0.0):
    """Greedy setlist builder — chain the highest-scoring next track repeatedly.

    Args:
        path:   Seed track (opening track of the set).
        length: Total tracks including the seed. Default 8.
        mode:   Scoring preset to use throughout the chain.
        strengths: explicit modifier strengths; defaults to the mode preset.
        temperature: 0.0 picks the best next track each step; >0 samples it, so a
            re-run yields a different (still strong) set.

    Strategy: at each step pick the best mix_score against the *previous* track,
    with a +0.1 bonus when spectral complexity rises (favours an energy arc).
    No look-ahead — this is intentionally simple, fast, and locally optimal.
    """
    strengths = _ensure_strengths(mode, strengths)
    current = _get_or_analyze(path)

    print(f"\n🎛️   Suggested SETLIST (progressive energy, mode={mode}):")
    print(f"  {_format_modifiers(strengths)}\n")
    setlist = [(path, current)]
    used = {str(Path(path).resolve())}

    for _ in range(length - 1):
        # Safeguarded retrieval PER STEP: the chain head changes every
        # iteration, so each step scores the neighbours of the CURRENT head
        # (excluding everything already in the set). The exactness guarantee
        # applies to the mix_score totals; the +0.1 complexity bonus below is
        # a deliberate heuristic layered on top, outside the bound.
        scored = score_candidates(setlist[-1][1], mode=mode,
                                  strengths=strengths, exclude_paths=used)
        if not scored:
            if len(setlist) == 1:
                print("\n⚠️  You need more tracks in the library.\n")
                return
            break
        # Reward rising complexity to push the set forward instead of plateauing.
        for i, (k, v, s) in enumerate(scored):
            energy_bonus = 0.1 if v.spectral_complexity > setlist[-1][1].spectral_complexity else 0.0
            scored[i] = (k, v, {**s, 'total': s['total'] + energy_bonus})
        best = sample_by_score(scored, 1, temperature)   # temperature 0 → the single best.
        if best:
            best_k, best_v, _ = best[0]
            setlist.append((best_k, best_v))
            used.add(str(Path(best_k).resolve()))

    for i, (k, v) in enumerate(setlist, 1):
        marker = "🎤" if i == 1 else " ↓ "
        print(f"  {marker} [{i:02d}] {v.bpm:.0f} BPM · {v.camelot} · {Path(k).name}")
    print()


def cmd_verify():
    """Check every registered model URL (HEAD request) without downloading.

    Catches moved/renamed paths on essentia.upf.edu BEFORE they surface as a
    silent pipeline-level degradation mid-analysis. Checks the .pb and its
    .json metadata companion for all registry entries; also reports which
    files are already on disk. Exits non-zero if any URL fails, so it can be
    used as a pre-flight check in scripts.
    """
    import urllib.error
    total, failures = len(ModelManager.REGISTRY), 0
    print(f"\n🔎  Verifying {total} model URLs on essentia.upf.edu (HEAD, no download)…\n")
    for name in ModelManager.REGISTRY:
        on_disk = "on disk" if ModelManager.path(name).exists() else "not downloaded"
        for url in (ModelManager.url(name), ModelManager.url(name).replace('.pb', '.json')):
            kind = "pb  " if url.endswith('.pb') else "json"
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=15):
                    pass
                print(f"  ✓  {kind}  {name:<20s} ({on_disk})")
            except Exception as e:
                failures += 1
                print(f"  ✗  {kind}  {name:<20s} FAILED: {e}\n         {url}")
    if failures:
        print(f"\n⚠️   {failures} URL(s) unreachable — fix REGISTRY paths before analysing.\n")
        sys.exit(1)
    print(f"\n✅  All {total} models reachable.\n")


def cmd_download():
    """Pre-download all registered models to disk for offline use.

    Models are downloaded once to ./models/ and reused forever. Running this
    before a gig ensures the pipeline works with no internet connection.
    Existing files are skipped — safe to run repeatedly.
    """
    total = len(ModelManager.REGISTRY)
    print(f"\n⬇   Pre-downloading all {total} The Crate models for offline use…\n")
    ModelManager.download_all()
    present = sum(1 for n in ModelManager.REGISTRY if ModelManager.path(n).exists())
    print(f"\n✅  {present}/{total} models on disk  →  pipeline level {ModelManager.pipeline_level()}/5\n")


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════



def _add_modifier_flags(sp: argparse.ArgumentParser) -> None:
    """Attach the shared modifier-control flags to a subparser (next/compare/setlist).

    For each modifier: a `--no-X` toggle (strength → 0.0) and a `--X-strength F`
    override [0.0–1.5]. Plus `--mood-contrast` to flip mood scoring to 'contrast'.
    """
    for m in MODIFIER_NAMES:
        sp.add_argument(f"--no-{m}", action="store_true",
                        help=f"disable the {m} modifier (strength → 0.0)")
        sp.add_argument(f"--{m}-strength", type=float, default=None, metavar="F",
                        help=f"override {m} strength [0.0–1.5]")
    sp.add_argument("--mood-contrast", action="store_true",
                    help="reward opposite moods (mood_mode → 'contrast')")
    sp.add_argument("--energy", choices=["up", "stable", "down", "both"], default="stable",
                    help="wanted energy direction next ('both' = show up+down slates)")
    sp.add_argument("--temperature", type=float, default=0.0, metavar="T",
                    help="recommendation variety: 0 = best picks, higher = more adventurous")


def _resolve_strengths(mode: str, args: argparse.Namespace) -> ModifierStrengths:
    """Resolve effective modifier strengths for one invocation.

    Order (per spec): start from the mode's default_strengths (a copy), apply any
    --no-X (→0.0), then any --X-strength overrides (clamped to [0.0, 1.5]), then
    --mood-contrast. getattr with defaults keeps this safe when args lacks the
    flags (e.g. the no-subcommand `next` default-path workflow).
    """
    s = replace(MODE_CONFIG.get(mode, MODE_CONFIG['balanced'])['default_strengths'])
    for m in MODIFIER_NAMES:
        if getattr(args, f"no_{m}", False):
            setattr(s, m, 0.0)
    for m in MODIFIER_NAMES:
        override = getattr(args, f"{m}_strength", None)
        if override is not None:
            setattr(s, m, _clamp(override, 0.0, 1.5))
    if getattr(args, "mood_contrast", False):
        s.mood_mode = 'contrast'
    # Energy direction: 'up'/'stable'/'down' map to a target; 'both' is handled in
    # cmd_next (two slates) and leaves the target at the mode default here.
    energy = getattr(args, "energy", "stable")
    if energy in ENERGY_TARGETS:
        s.energy_target = ENERGY_TARGETS[energy]
    return s


def build_parser() -> argparse.ArgumentParser:
    """Configure the argparse CLI. Subcommands mirror the original cmd_* functions."""
    p = argparse.ArgumentParser(
        prog="analyze.py",
        description="The Crate — Essentia mix engine (classic + optional ML).",
    )
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("scan", help="Index every audio file in a folder.")
    sp.add_argument("folder", type=str)

    sp = sub.add_parser("analyze", help="Print a single track's features.")
    sp.add_argument("file", type=str)

    sp = sub.add_parser("next", help="Top-5 next-track picks for a current track.")
    sp.add_argument("file", type=str)
    sp.add_argument("--mode", choices=list(MODE_CONFIG.keys()), default="balanced")
    _add_modifier_flags(sp)

    sp = sub.add_parser("compare", help="Full sub-score breakdown between two tracks.")
    sp.add_argument("file1", type=str)
    sp.add_argument("file2", type=str)
    sp.add_argument("--mode", choices=list(MODE_CONFIG.keys()), default="balanced")
    _add_modifier_flags(sp)

    sp = sub.add_parser("mixpoints", help="Intro/outro mix-point detection.")
    sp.add_argument("file", type=str)

    sp = sub.add_parser("setlist", help="Greedy setlist builder.")
    sp.add_argument("file", type=str)
    sp.add_argument("--length", type=int, default=8)
    sp.add_argument("--mode", choices=list(MODE_CONFIG.keys()), default="balanced")
    _add_modifier_flags(sp)

    sub.add_parser("download", help="Pre-download all models for offline use.")
    sub.add_parser("verify", help="Check all model URLs are reachable (no download).")

    return p


def main():
    print_pipeline_banner()
    args = build_parser().parse_args()
    if args.command == "scan":
        cmd_scan(args.folder)
    elif args.command == "analyze":
        cmd_analyze(args.file)
    elif args.command == "next":
        cmd_next(args.file, mode=args.mode, strengths=_resolve_strengths(args.mode, args),
                 temperature=getattr(args, "temperature", 0.0),
                 energy=getattr(args, "energy", None))
    elif args.command == "compare":
        cmd_compare(args.file1, args.file2, mode=args.mode,
                    strengths=_resolve_strengths(args.mode, args))
    elif args.command == "mixpoints":
        cmd_mixpoints(args.file)
    elif args.command == "setlist":
        cmd_setlist(args.file, length=args.length, mode=args.mode,
                    strengths=_resolve_strengths(args.mode, args),
                    temperature=getattr(args, "temperature", 0.0))
    elif args.command == "download":
        cmd_download()
    elif args.command == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════
## REFACTOR NOTES — Level 4 emotional fingerprint addition
# ════════════════════════════════════════════════════════════
#
# ── NEW REGISTRY ENTRIES ──────────────────────────────────────────────────────
# All verified live via HTTP HEAD on 2026-06-04.
# Sub-path corrected for ALL classifier heads: 'classifiers/' → 'classification-heads/'
# (the old path gave 404 for mood_aggressive and danceability too — now fixed).
#
#   Model key             Filename                                     Status
#   mood_aggressive       mood_aggressive-discogs-effnet-1.pb          ✓ verified
#   danceability          danceability-discogs-effnet-1.pb             ✓ verified (path fixed)
#   mood_electronic       mood_electronic-discogs-effnet-1.pb          ✓ verified
#   mood_sad              mood_sad-discogs-effnet-1.pb                 ✓ verified
#   mood_relaxed          mood_relaxed-discogs-effnet-1.pb             ✓ verified
#   mood_happy            mood_happy-discogs-effnet-1.pb               ✓ verified
#   mood_party            mood_party-discogs-effnet-1.pb               ✓ verified
#   jamendo_moodtheme     mtg_jamendo_moodtheme-discogs-effnet-1.pb    ✓ verified (2.6 MB, 56 labels)
#
# ── NEW TrackFeatures FIELDS ──────────────────────────────────────────────────
#   mood_electronic    float | None  — electronic/synthetic character
#   mood_sad           float | None  — melancholy / introspection
#   mood_relaxed       float | None  — spaciousness vs tension
#   mood_happy         float | None  — positivity (near-zero for most techno)
#   mood_party         float | None  — dancefloor/party energy angle
#   jamendo_dark       float | None  — label 11: darkness/shadow
#   jamendo_groovy     float | None  — label 25: rhythmic pull
#   jamendo_meditative float | None  — label 32: hypnotic/meditative
#   jamendo_energetic  float | None  — label 18: raw energy
#   jamendo_heavy      float | None  — label 27: weight/intensity
#   jamendo_space      float | None  — label 49: spatial/atmospheric
#   emotional_vector   list | None   — assembled fingerprint (4–13 floats)
#
# ── EMOTIONAL_VECTOR CANONICAL ORDER ─────────────────────────────────────────
# Defined in _EMOTIONAL_VECTOR_ORDER (module-level constant). Order chosen so
# the most universal components (present from Level 3) come first, Jamendo
# components (Level 4 only) come last. This ensures Level-3 tracks share a
# meaningful 2-component intersection with each other (mood_aggressive +
# danceability_nn), and Level-4 tracks can share up to 13 components.
#
#   [0]  mood_aggressive      [1]  mood_electronic    [2]  mood_sad
#   [3]  mood_relaxed         [4]  mood_happy         [5]  mood_party
#   [6]  danceability_nn      [7]  jamendo_dark       [8]  jamendo_groovy
#   [9]  jamendo_meditative   [10] jamendo_energetic  [11] jamendo_heavy
#   [12] jamendo_space
#
# ── JAMENDO LABEL INDICES — VERIFIED 2026-06-07 ──────────────────────────────
# All six _JAMENDO_*_IDX constants were checked against the model's 'classes'
# list in the official JSON metadata and confirmed correct:
#   dark=11, energetic=18, groovy=25, heavy=27, meditative=32, space=49.
#   https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/
#   mtg_jamendo_moodtheme-discogs-effnet-1.json
# If the model is upgraded to a new version, re-fetch this JSON and re-verify
# the label list at the 'classes' key. The constants are _JAMENDO_*_IDX at the
# top of this file. (Other heads no longer hardcode indices — ModelManager.
# class_index() resolves the positive-class column by name from each JSON.)
#
# ── HOW TO DISABLE LEVEL 4 IF MODELS SLOW DOWN ANALYSIS ─────────────────────
# Set the environment variable THECRATE_MAX_LEVEL=3 before running. To respect
# it, add this near the top of the ML enrichment block in extract_features():
#
#   import os
#   _MAX_LEVEL = int(os.getenv('THECRATE_MAX_LEVEL', '4'))
#
# Then gate the Level 4 block with:
#   if record.pipeline_level >= 3 and _MAX_LEVEL >= 4:
#       print("  🎭 Emotional fingerprint (Level 4)...")
#       ...
#
# Alternatively, pass --no-emotional on any CLI command to set emotional
# strength to 0.0, which disables the modifier without skipping extraction.
# ─────────────────────────────────────────────────────────────────────────────
