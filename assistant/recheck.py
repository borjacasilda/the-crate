"""
assistant/recheck.py — Live-Mode "agent re-check" (the ASSISTANT dashboard toggle).

When the user turns re-check ON, the agent confirms the live recommendations on
each new lock: a single focused LLM pass that says, in a sentence or two, whether
the picks are solid harmonic/energy choices to mix into next, which is strongest,
and any risk. When OFF none of this runs (the agent only answers chat). One-shot
Ollama chat — out of process, never touches the Essentia executor, and degrades
gracefully (returns a note/error, never raises) so Live Mode is never blocked.
"""
import re

import httpx

from assistant import models as registry
from assistant import ollama_client

_PROMPT = (
    "You are a techno DJ's mixing assistant. The track now playing and the "
    "candidate next tracks are given with BPM, Camelot key and energy direction. "
    "In ONE or TWO short sentences: confirm whether these are solid harmonic and "
    "energy choices to mix into next, name the strongest pick, and flag any risk "
    "(a big BPM jump or a clashing key). Be concrete and brief. No preamble."
)


def confirm(track: dict, recs: list, timeout: float = 120.0) -> dict:
    """Confirm the live picks with the local model. Returns {"note": text} or
    {"error": reason} — never raises, so the live loop / UI is never blocked."""
    if not recs:
        return {"note": "No recommendations to review yet."}
    if not ollama_client.is_up():
        return {"error": "Ollama is not running — start it to enable re-check."}
    from assistant import agent
    # Re-check fires on every live lock, so latency matters most here: prefer the
    # instant (no-thinking) instruct build when it is pulled, even if the user
    # picked a thinking model for chat. Fall back to the active chat model.
    inst = ollama_client.installed_models()
    model = registry.fastest_chat_model(inst) or agent.active_model()
    if not ollama_client.has_model(model):
        return {"error": f"Model '{model}' is not installed."}

    now_playing = (f"{track.get('filename', '?')} "
                   f"({track.get('bpm', '?')} BPM, {track.get('camelot', '?')})")
    lines = [
        f"- {r.get('filename', '?')} | {r.get('bpm', '?')} BPM "
        f"(Δ{r.get('bpm_delta', '?')}) | {r.get('camelot', '?')} "
        f"({r.get('key_relationship', '?')}) | energy "
        f"{r.get('energy_direction', '?')} {r.get('energy_pct', '?')}%"
        for r in recs
    ]
    user = f"NOW PLAYING: {now_playing}\nCANDIDATES:\n" + "\n".join(lines)
    try:
        # Reasoning models (qwen3) route their <think> to a separate `thinking`
        # field, leaving `content` clean — so DON'T cap num_predict (that would
        # spend the whole budget on thinking and return empty) and read content.
        # keep_alive holds the model warm between locks so only the first is cold.
        r = httpx.post(
            f"{ollama_client.OLLAMA_HOST}/api/chat", timeout=timeout,
            json={"model": model, "stream": False, "keep_alive": "10m",
                  "messages": [{"role": "system", "content": _PROMPT},
                               {"role": "user", "content": user}],
                  "options": {"temperature": 0.3}})
        r.raise_for_status()
        msg = (r.json().get("message") or {}).get("content", "")
        # Safety net if a model inlines its reasoning as <think>…</think>.
        msg = re.sub(r"<think>.*?</think>", "", msg, flags=re.S).strip()
        return {"note": msg or "No comment."}
    except Exception as e:
        return {"error": f"re-check failed: {e}"}
