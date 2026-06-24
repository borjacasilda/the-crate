#!/usr/bin/env python3
"""
Environment sanity check for The Crate.

Run:  uv run python scripts/verify.py

Each check is independent and never aborts the others; the script exits non-zero
if any check fails, so it doubles as a CI smoke test. It only READS — it starts
nothing and changes nothing.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hush TensorFlow import noise
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_passed = 0
_total = 0


def check(name, fn):
    """Run one check; print ok/--, count the result."""
    global _passed, _total
    _total += 1
    try:
        detail = fn()
        print(f"  ok  {name}" + (f" — {detail}" if detail else ""))
        _passed += 1
    except Exception as e:                            # any failure is just a failed check
        print(f"  --  {name}: {e}")


def _python_version():
    if sys.version_info < (3, 11):
        raise RuntimeError(f"Python 3.11+ required, found {sys.version.split()[0]}")
    return f"Python {sys.version.split()[0]}"


def _env_file():
    if not (ROOT / ".env").exists():
        raise RuntimeError("missing — create it with `cp .env.example .env`")
    return ".env present"


def _core_deps():
    import fastapi, numpy, psycopg2  # noqa: F401
    return "fastapi, numpy, psycopg2 import"


def _essentia():
    import essentia  # noqa: F401  (heavy import; just confirms the audio engine is installed)
    return "audio engine available"


def _database():
    import database
    if not database.DB_AVAILABLE:
        raise RuntimeError("not reachable — start it with `docker compose up -d`")
    return "PostgreSQL + pgvector reachable"


if __name__ == "__main__":
    print("\nThe Crate — environment check\n" + "─" * 34)
    check("Python version", _python_version)
    check(".env file", _env_file)
    check("Core dependencies", _core_deps)
    check("Essentia (audio analysis)", _essentia)
    check("Database (PostgreSQL + pgvector)", _database)
    print("─" * 34)
    print(f"{_passed}/{_total} checks passed\n")
    sys.exit(0 if _passed == _total else 1)
