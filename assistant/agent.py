"""
assistant/agent.py — the single PydanticAI tool-calling agent.

One agent, the four tools, a domain system prompt. The LLM's choice of tool IS
the routing (no separate router). Model-agnostic: it talks to Ollama via the
OpenAI-compatible endpoint, so swapping to a cloud model later is a one-liner.
Built lazily and cached per model tag, so importing this module never needs
Ollama to be running.
"""
import asyncio
import functools

from assistant import models as registry
from assistant import ollama_client, scope, tools

# Active chat model for this process; settable from the API (model picker).
_active_model = None


def active_model() -> str:
    """The chat model tag in use. Defaults to the RAM-recommended model, but if
    that isn't pulled yet falls back to the largest INSTALLED chat model so the
    assistant works out of the box with whatever the user has downloaded."""
    if _active_model:
        return _active_model
    rec = registry.default_chat_model()
    try:
        inst = ollama_client.installed_models()
    except Exception:
        inst = set()
    if rec in inst or f"{rec}:latest" in inst:
        return rec
    ram = {m["tag"]: m["ram_gb"] for m in registry.MODELS}
    installed_chat = [m["tag"] for m in registry.MODELS if m["role"] == "chat"
                      and (m["tag"] in inst or f"{m['tag']}:latest" in inst)]
    return max(installed_chat, key=lambda t: ram[t]) if installed_chat else rec


def set_active_model(tag: str) -> None:
    global _active_model
    _active_model = tag


@functools.lru_cache(maxsize=6)
def _build_agent(model_tag: str):
    import config
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    # Ollama tuning sent on every turn. `temperature` is a standard OpenAI param (Ollama
    # maps it to options.temperature). `keep_alive` is Ollama-specific and rides in
    # `extra_body` — verified honoured by Ollama's OpenAI-compatible endpoint as a
    # top-level field — so the model stays resident between questions (no cold reload).
    # num_ctx is NOT settable here: Ollama's OpenAI endpoint ignores a per-request
    # `options` object (verified — it stays at the server default), so the context window
    # is raised server-side via OLLAMA_CONTEXT_LENGTH when we launch Ollama (see
    # ollama_client._launch_server / config.OLLAMA_NUM_CTX).
    settings = OpenAIChatModelSettings(
        temperature=config.OLLAMA_TEMPERATURE,
        extra_body={"keep_alive": config.OLLAMA_KEEP_ALIVE},
    )
    model = OpenAIChatModel(
        model_tag,
        provider=OpenAIProvider(base_url=ollama_client.OPENAI_BASE, api_key="ollama"),
        settings=settings,
    )
    return Agent(
        model,
        system_prompt=scope.SYSTEM_PROMPT,
        tools=[tools.audio_similarity, tools.similar_artists, tools.similar_labels,
               tools.metadata_search, tools.kb_rag_search,
               tools.music_web_search, tools.set_user_location],
        retries=2,
    )


def _compose(message: str, context: dict = None) -> str:
    """Prepend live runtime context so the agent always knows WHEN it is and
    WHERE the user is — plus any page context (a viewed track/crate). This is how
    the modest local model gets "time + location" awareness without spending a
    tool call: it is handed the facts every turn."""
    from datetime import datetime
    from assistant import profile

    bits = []
    now = datetime.now()
    bits.append(f"[context: now is {now:%A %Y-%m-%d %H:%M} (the user's local time)]")
    loc = profile.get_location()
    if loc:
        bits.append(f"[context: the user's current location is {loc} — use it for "
                    f"events/what's-on unless they say otherwise]")
    else:
        bits.append("[context: the user's location is UNKNOWN — for any "
                    "location-dependent answer, ask where they are first]")
    if context:
        if context.get("track_id"):
            bits.append(f"[context: the user is currently viewing track_id="
                        f"{context['track_id']}]")
        if context.get("crate_id"):
            bits.append(f"[context: current crate_id={context['crate_id']}]")
    return "\n".join(bits) + "\n\n" + message


def _preflight(model_tag: str = None) -> str:
    """Make sure Ollama is up (starting it on demand) and resolve the chat model.

    Returns the model tag to use, or raises RuntimeError('ollama-down' |
    'model-missing:<tag>'). Synchronous and potentially slow (it may wait for the
    server to boot), so run_stream calls it off the event loop.
    """
    if not ollama_client.ensure_up():            # auto-starts Ollama if it was quit
        raise RuntimeError("ollama-down")
    tag = model_tag or active_model()            # resolve AFTER it's up, so /api/tags is live
    if not ollama_client.has_model(tag):
        raise RuntimeError(f"model-missing:{tag}")
    return tag


async def run_stream(message: str, context: dict = None, model_tag: str = None):
    """Async-generate the assistant's answer as text deltas.

    Raises RuntimeError when Ollama can't be reached/started or the model isn't
    pulled — the API turns that into a friendly, actionable message. The blocking
    preflight (which may boot Ollama) runs in a thread so it never stalls the event
    loop while other requests are served.
    """
    tag = await asyncio.to_thread(_preflight, model_tag)
    agent = _build_agent(tag)
    async with agent.run_stream(_compose(message, context)) as result:
        async for delta in result.stream_text(delta=True):
            yield delta
