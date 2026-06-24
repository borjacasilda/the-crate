"""
assistant/profile.py — small persistent profile for the assistant.

Two things the agent needs to remember across turns and restarts:
  • location — where the user physically is ("Madrid", "Ibiza this weekend"), so
    location-dependent recommendations (events, what's on) know "where are we".
    Set by the user in chat (the agent calls set_user_location) or from the UI.
  • recheck — the Live-Mode toggle: when on, the agent re-checks / confirms the
    live recommendations; when off it only answers what the user asks.

Stored as JSON in config.PROFILE_FILE (.cache), mirroring the active-crate cache.
Best-effort: any read error falls back to defaults so the assistant never breaks
because of a missing/corrupt profile file.
"""
import json

import config

_DEFAULTS = {"location": None, "recheck": False}


def _read() -> dict:
    try:
        data = json.loads(config.PROFILE_FILE.read_text())
        return {**_DEFAULTS, **(data if isinstance(data, dict) else {})}
    except Exception:
        return dict(_DEFAULTS)


def _write(data: dict) -> None:
    config.PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.PROFILE_FILE.write_text(json.dumps(data))


def get() -> dict:
    """The whole profile (location + recheck)."""
    return _read()


def get_location() -> "str | None":
    return _read().get("location")


def set_location(location: str) -> "str | None":
    """Persist the user's physical location (or clear it with an empty value)."""
    data = _read()
    data["location"] = (location or "").strip() or None
    _write(data)
    return data["location"]


def get_recheck() -> bool:
    return bool(_read().get("recheck"))


def set_recheck(on: bool) -> bool:
    data = _read()
    data["recheck"] = bool(on)
    _write(data)
    return data["recheck"]
