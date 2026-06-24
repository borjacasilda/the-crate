"""
assistant/models.py — local LLM catalog + per-RAM suitability.

The user can run any model Ollama offers; this module just advises which ones
fit comfortably in the machine's RAM and picks a sensible default. The audio
analysis stack (Essentia/TF) and Postgres share the same memory, so we reserve
headroom before judging a model's footprint.
"""
import os

# Approx RAM footprint of a Q4_K_M quant (weights + a working context), in GB.
# Chat models do the tool-calling + answering; the embed model powers RAG.
# `instant`=True marks a no-thinking (instruct) build: qwen3 "thinking" models
# reason on every turn, which is too slow for live use (the re-check + KB gate),
# whereas the instruct build answers and tool-calls just as well in a fraction of
# the time — so we prefer it for the default and for latency-critical paths.
MODELS = [
    {"tag": "qwen3:4b-instruct", "label": "Qwen 3 4B Instruct", "params": "4B", "ram_gb": 3.0,
     "role": "chat", "instant": True,
     "note": "Recommended: no thinking step, so chat, the KB gate and live re-check "
             "are near-instant — and light enough to run beside live audio analysis."},
    {"tag": "qwen3:4b",   "label": "Qwen 3 4B",   "params": "4B",   "ram_gb": 3.0,  "role": "chat",
     "note": "Thinking build — capable but reasons every turn, so noticeably slower."},
    {"tag": "qwen3:8b",   "label": "Qwen 3 8B",   "params": "8B",   "ram_gb": 5.5,  "role": "chat",
     "note": "Strong tool-calling and prose, but the thinking step makes it slow for live use."},
    {"tag": "qwen3:14b",  "label": "Qwen 3 14B",  "params": "14B",  "ram_gb": 9.5,  "role": "chat",
     "note": "More capable prose; tight on 16 GB — close other apps."},
    {"tag": "qwen3:32b",  "label": "Qwen 3 32B",  "params": "32B",  "ram_gb": 20.0, "role": "chat",
     "note": "Best reasoning/storytelling; needs 32 GB+ to run smoothly."},
    {"tag": "nomic-embed-text", "label": "nomic-embed-text", "params": "0.1B", "ram_gb": 0.4,
     "role": "embed", "note": "768-D text embeddings for the knowledge base (Phase 2)."},
]

# RAM the OS + browser + Postgres + (lazy) TensorFlow models can claim before
# the LLM gets any — judged conservatively so 'comfortable' really is.
RESERVED_GB = 5.0


def system_ram_gb() -> float:
    """Total physical RAM in GB (macOS/Linux via sysconf)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return 16.0


def verdict(model_ram_gb: float, system_ram: float = None) -> str:
    """'comfortable' | 'tight' | 'heavy' for a model on this machine."""
    sys_ram = system_ram if system_ram is not None else system_ram_gb()
    usable = max(1.0, sys_ram - RESERVED_GB)
    if model_ram_gb <= usable * 0.7:
        return "comfortable"
    if model_ram_gb <= usable:
        return "tight"
    return "heavy"


_WARN = {
    "comfortable": None,
    "tight": "Tight for this machine's RAM — close other apps and avoid heavy "
             "audio analysis while chatting, or it may stutter.",
    "heavy":  "Exceeds this machine's comfortable RAM — it will swap and run "
              "slowly. Only pick this if you know your setup can take it.",
}


def catalog(installed: set = None) -> list:
    """The model list annotated with suitability, warning, and installed state.

    Args:
        installed: set of model tags already pulled (from the Ollama client).
    Returns:
        list of dicts ready for the UI / API.
    """
    installed = installed or set()
    sys_ram = round(system_ram_gb(), 1)
    out = []
    for m in MODELS:
        v = verdict(m["ram_gb"], sys_ram)
        out.append({**m, "verdict": v, "warning": _WARN[v],
                    "installed": m["tag"] in installed,
                    "recommended": m["tag"] == default_chat_model(sys_ram)})
    return out


def default_chat_model(system_ram: float = None) -> str:
    """The recommended CHAT model for this machine.

    We prefer an INSTANT (no-thinking) build: the thinking models reason on every
    turn and are too slow for the live paths (re-check, KB gate), while the
    instruct build answers and tool-calls just as well for this router-style
    assistant. Among instant models we take the LIGHTEST that fits — capability
    was never the bottleneck (the assistant routes to deterministic engines), so
    we keep RAM free for the Essentia/TF audio stack that shares this machine.
    Only when no instant model exists do we fall back to the most capable
    'comfortable' thinking model, then to the smallest of all (tiny machine)."""
    sys_ram = system_ram if system_ram is not None else system_ram_gb()
    chat = [m for m in MODELS if m["role"] == "chat"]
    comfy = [m for m in chat if verdict(m["ram_gb"], sys_ram) == "comfortable"]
    pool = comfy or chat                                 # tiny machine → still pick something
    instant = [m for m in pool if m.get("instant")]
    if instant:
        return min(instant, key=lambda m: m["ram_gb"])["tag"]
    pick = max(comfy, key=lambda m: m["ram_gb"]) if comfy else \
        min(chat, key=lambda m: m["ram_gb"])
    return pick["tag"]


def fastest_chat_model(installed: set = None) -> str | None:
    """The lightest INSTALLED instant (no-thinking) chat model, or None if the
    user has not pulled one. Latency-critical callers (the live re-check) use this
    to run the fast build regardless of which chat model is active."""
    installed = installed or set()
    instant = [m for m in MODELS if m["role"] == "chat" and m.get("instant")
               and (m["tag"] in installed or f"{m['tag']}:latest" in installed)]
    return min(instant, key=lambda m: m["ram_gb"])["tag"] if instant else None


EMBED_MODEL = "nomic-embed-text"
