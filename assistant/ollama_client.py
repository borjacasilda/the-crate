"""
assistant/ollama_client.py — thin Ollama management over its REST API.

Chat/generation goes through PydanticAI's OpenAI-compatible model (Ollama
exposes /v1); this module only handles the operational side: is Ollama up, what
is pulled, and pulling a new model (streamed progress). Everything degrades
gracefully when Ollama is not running so the rest of the app never breaks.
"""
import os
import shutil
import subprocess
import time

import httpx

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OPENAI_BASE = OLLAMA_HOST.rstrip("/") + "/v1"

# Short-TTL cache for the installed-model list. A single chat turn resolves the active
# model, then checks has_model, then (for KB/web tools) re-checks before embedding — a
# handful of /api/tags hits within one second. The cache collapses them to one round
# trip while staying fresh enough to notice a just-pulled model. THECRATE_OLLAMA_TAGS_TTL.
_TAGS_TTL = float(os.environ.get("THECRATE_OLLAMA_TAGS_TTL", "5.0"))
_tags_cache = None                              # (expires_monotonic, frozenset) | None


def is_up(timeout: float = 1.5) -> bool:
    """True when the Ollama server answers."""
    try:
        return httpx.get(f"{OLLAMA_HOST}/api/version", timeout=timeout).status_code == 200
    except Exception:
        return False


def _launch_server() -> None:
    """Best-effort start of the local Ollama SERVER (headless) when it is down.

    Starts ONLY the `ollama serve` daemon via the CLI — deliberately NOT
    `open -a Ollama`, which would launch the desktop app and pop its chat-window /
    model-picker GUI. This keeps the server-only behaviour: no app, no window. If
    the `ollama` CLI isn't on PATH we stay down and the caller surfaces the
    friendly 'start Ollama' message. The daemon is detached so it outlives this
    request and serves the rest of the session.
    """
    exe = shutil.which("ollama")
    if not exe:
        return
    # Raise the default context window for every model this server loads. Ollama's
    # OpenAI-compatible endpoint (used by the chat agent) ignores a per-request num_ctx,
    # so OLLAMA_CONTEXT_LENGTH on the server is the only way to give the big system+tools
    # prompt (~3.7k tokens) headroom instead of being truncated at the 4096 default. NOTE:
    # this only takes effect when WE start Ollama; an already-running Ollama keeps its own
    # context length (restart it, or set this env there, to apply the larger window).
    import config
    env = {**os.environ, "OLLAMA_CONTEXT_LENGTH": str(config.OLLAMA_NUM_CTX)}
    try:
        subprocess.Popen([exe, "serve"], start_new_session=True, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def ensure_up(timeout: float = 15.0) -> bool:
    """Reachability check that STARTS Ollama on demand when it is down.

    Returns immediately True if already up; otherwise tries to launch the server
    and polls until `timeout`. This is what makes the assistant work on a request
    even if the user quit Ollama between requests. BLOCKING (it waits for the
    server to come up) — callers on the event loop must run it in a thread.
    """
    if is_up():
        return True
    _launch_server()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_up(timeout=1.0):
            return True
        time.sleep(0.6)
    return False


def _invalidate_tags() -> None:
    """Drop the installed-model cache (called after a pull adds a model)."""
    global _tags_cache
    _tags_cache = None


def installed_models(timeout: float = 3.0, use_cache: bool = True) -> set:
    """Set of model tags already pulled (e.g. {'qwen3:8b'}). Empty if down.

    Cached for `_TAGS_TTL` seconds to collapse the repeated /api/tags lookups a single
    request makes (model resolution + has_model + per-tool embed checks). A transient
    failure returns an empty set WITHOUT caching it, so the next call retries instead of
    being stuck 'empty'. Pass use_cache=False to force a live probe.
    """
    global _tags_cache
    if use_cache and _tags_cache and time.monotonic() < _tags_cache[0]:
        return _tags_cache[1]
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        r.raise_for_status()
        out = set()
        for m in r.json().get("models", []):
            name = m.get("name", "")
            out.add(name)
            if ":latest" in name:               # 'qwen3:8b:latest' never happens, but
                out.add(name.replace(":latest", ""))
        _tags_cache = (time.monotonic() + _TAGS_TTL, out)
        return out
    except Exception:
        return set()


def has_model(tag: str) -> bool:
    inst = installed_models()
    return tag in inst or f"{tag}:latest" in inst


def pull_stream(tag: str):
    """Yield human-readable progress lines while Ollama pulls `tag`.

    A generator so the API can stream it to the browser. Each yield is a short
    status string; the final yield is 'done' or 'error: …'.
    """
    try:
        with httpx.stream("POST", f"{OLLAMA_HOST}/api/pull",
                          json={"model": tag}, timeout=None) as r:
            if r.status_code != 200:
                yield f"error: Ollama returned {r.status_code}"
                return
            import json as _json
            last = None
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    evt = _json.loads(line)
                except Exception:
                    continue
                if evt.get("error"):
                    yield f"error: {evt['error']}"
                    return
                status = evt.get("status", "")
                total, done = evt.get("total"), evt.get("completed")
                if total and done:
                    pct = int(100 * done / total)
                    msg = f"{status} {pct}%"
                else:
                    msg = status
                if msg and msg != last:
                    last = msg
                    yield msg
            _invalidate_tags()                   # a new model is now installed
            yield "done"
    except Exception as e:
        yield f"error: {e}"
