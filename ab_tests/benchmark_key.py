#!/usr/bin/env python3
"""
ab_tests/benchmark_key.py — measure tonal-detection accuracy across profiles.

The only honest way to "ensure the detected key is correct" is to score it against
keys we trust. This loads a hand-labelled ground-truth set (see key_ground_truth.csv)
and runs every track through a set of detection ARMS — the old default, each EDM
profile with vinyl-tuning correction, the live multi-profile VOTE, and the dormant
band-isolation pre-filter — then reports two metrics the Camelot modifier cares about
differently:

  • EXACT     predicted Camelot == truth (what matters for 8B vs 9A).
  • COMPATIBLE predicted is Same/Adjacent/Relative to truth (a cheaper error).

Run:  uv run python ab_tests/benchmark_key.py [path/to/ground_truth.csv]

No DB, no new test framework — it reuses analyze.py's own detection helpers, so the
'VOTE' arm is byte-for-byte the live path (analyze._detect_key_robust).
"""
import csv
import re
import sys
from pathlib import Path

# Run from anywhere: make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import essentia.standard as es  # noqa: E402

from analyze import (to_camelot, key_relationship_label, _estimate_tuning_frequency,  # noqa: E402
                     _key_extract_one, _detect_key_robust, _isolate_tonal_band)
from config import SAMPLE_RATE  # noqa: E402

# Relationships counted as a "compatible" (forgivable) miss — anything a DJ could
# still blend. An exact match is the strict metric.
COMPATIBLE = {"Same key", "Adjacent", "Relative (mood shift)"}

# Each arm maps loaded audio + measured tuning → (key, scale). The first arm
# reproduces the OLD production behaviour (no-arg KeyExtractor == bgate @ 440 Hz)
# so every later arm reads as a delta against it.
ARMS = [
    ("bgate (old default, 440Hz)", lambda a, tf: _key_extract_one(a, "bgate", 440.0)[:2]),
    ("bgate + tuning",             lambda a, tf: _key_extract_one(a, "bgate", tf)[:2]),
    ("edma + tuning",              lambda a, tf: _key_extract_one(a, "edma", tf)[:2]),
    ("edmm + tuning",              lambda a, tf: _key_extract_one(a, "edmm", tf)[:2]),
    ("VOTE + tuning (LIVE)",       lambda a, tf: _detect_key_robust(a, tuning_frequency=tf)[:2]),
    ("VOTE + band + tuning",       lambda a, tf: _detect_key_robust(
        _isolate_tonal_band(a), tuning_frequency=tf)[:2]),
]

_NOTE_RE = re.compile(r"^([A-Ga-g][#b]?)\s*(maj|major|min|minor|m)?$")
_CAMELOT_RE = re.compile(r"^(1[0-2]|[1-9])[ABab]$")


def truth_to_camelot(s: str) -> "str | None":
    """Parse a ground-truth cell into a Camelot code, or None if unparseable.

    Accepts Camelot directly ('9A', '8b') or a musical name ('E minor', 'Em',
    'F#m', 'C major', 'Ab maj'); enharmonic spellings collapse via KEY_INDEX.
    """
    s = (s or "").strip()
    if _CAMELOT_RE.match(s):
        return s.upper()
    m = _NOTE_RE.match(s)
    if not m:
        return None
    note = m.group(1)[0].upper() + m.group(1)[1:]          # 'f#' → 'F#'
    mode = "minor" if (m.group(2) or "major").lower().startswith("m") and \
        (m.group(2) or "").lower() != "major" else "major"
    cam = to_camelot(note, mode)
    return cam if cam != "?" else None


def load_rows(csv_path: Path) -> list:
    """Read (path, truth_camelot) pairs, skipping comments/blanks/bad rows."""
    rows = []
    with csv_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 2 or parts[0].strip().lower() == "path":
                continue
            path, truth = parts[0].strip(), parts[1].strip()
            cam = truth_to_camelot(truth)
            if cam is None:
                print(f"  ⚠️  skipping unparseable key {truth!r} for {path}")
                continue
            rows.append((path, truth, cam))
    return rows


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent / "key_ground_truth.csv"
    if not csv_path.exists():
        print(f"No ground-truth file at {csv_path}")
        return 1
    rows = load_rows(csv_path)
    if not rows:
        print(f"No usable rows in {csv_path} — add tracks with trusted keys "
              "(see the header comments).")
        return 1

    print(f"\nBenchmarking {len(rows)} track(s) across {len(ARMS)} arms…\n")
    tallies = {label: {"exact": 0, "compat": 0, "n": 0} for label, _ in ARMS}
    for path, truth_raw, truth_cam in rows:
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  missing file, skipping: {path}")
            continue
        try:
            audio = es.MonoLoader(filename=str(p), sampleRate=SAMPLE_RATE)()
        except Exception as e:
            print(f"  ⚠️  could not load {path}: {e}")
            continue
        tuning = _estimate_tuning_frequency(audio)
        print(f"• {p.name}  (truth {truth_cam} / {truth_raw}, tuning {tuning:.1f} Hz)")
        for label, fn in ARMS:
            try:
                key, scale = fn(audio, tuning)
                pred = to_camelot(key, scale)
            except Exception as e:
                print(f"    {label:28} ERROR {e}")
                continue
            exact = pred == truth_cam
            compat = exact or key_relationship_label(pred, truth_cam) in COMPATIBLE
            t = tallies[label]
            t["n"] += 1
            t["exact"] += int(exact)
            t["compat"] += int(compat)
            mark = "✓" if exact else ("~" if compat else "✗")
            print(f"    {label:28} {pred:>3}  {mark}")
        print()

    print("=" * 64)
    print(f"  {'ARM':28} {'EXACT':>12} {'COMPATIBLE':>14}")
    print("-" * 64)
    for label, _ in ARMS:
        t = tallies[label]
        if not t["n"]:
            continue
        ex = 100.0 * t["exact"] / t["n"]
        co = 100.0 * t["compat"] / t["n"]
        print(f"  {label:28} {t['exact']:>3}/{t['n']:<3} {ex:5.1f}%  "
              f"{t['compat']:>3}/{t['n']:<3} {co:5.1f}%")
    print("=" * 64)
    print("  EXACT = tonic+mode correct (8B vs 9A).  COMPATIBLE = +relative/adjacent.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
