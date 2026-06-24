"""
test_assistant.py — The Crate, Phase 1: AI assistant (no real LLM needed)
========================================================================
Covers the assistant's deterministic parts: the model registry + RAM verdicts,
the domain guardrail, the four tools (DB-backed), and the agent's tool-calling
wiring via PydanticAI's TestModel (a MOCK model — no Ollama, no real LLM, fast).

Run:  uv run python ab_tests/test_assistant.py
Skips the DB-backed tool checks cleanly when the DB is down.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import database
from assistant import models, scope, tools, agent


def test_model_registry():
    # 16 GB verdicts: small models comfortable, 32B heavy.
    assert models.verdict(3.0, 16) == "comfortable"
    assert models.verdict(9.5, 16) == "tight"
    assert models.verdict(20.0, 16) == "heavy"
    # Default prefers the instant (no-thinking) instruct build at every RAM tier:
    # fast wins, and the lightest instant model keeps RAM free for the audio stack.
    assert models.default_chat_model(16) == "qwen3:4b-instruct"
    assert models.default_chat_model(8) == "qwen3:4b-instruct"   # tiny machine → still instant
    # fastest_chat_model: lightest INSTALLED instant model, else None.
    assert models.fastest_chat_model({"qwen3:4b-instruct"}) == "qwen3:4b-instruct"
    assert models.fastest_chat_model({"qwen3:8b"}) is None       # no instant build pulled
    cat = models.catalog(installed={"qwen3:4b-instruct"})
    qi = next(m for m in cat if m["tag"] == "qwen3:4b-instruct")
    assert qi["installed"] and qi["verdict"] == "comfortable" and qi["recommended"]
    assert any(m["warning"] for m in cat if m["verdict"] != "comfortable")
    print(f"  ✓ registry: 16GB→default qwen3:4b-instruct · verdicts comfortable/tight/heavy · warnings present")


def test_scope_guardrail():
    assert "specialized in electronic music" in scope.REFUSAL
    assert "audio_similarity" in scope.SYSTEM_PROMPT
    assert "DO NOT call any tool" in scope.SYSTEM_PROMPT or "DO NOT" in scope.SYSTEM_PROMPT
    print("  ✓ scope: refusal string + tool-aware system prompt")


def test_tools():
    if not database.DB_AVAILABLE:
        print("  ⚠ DB down — skipping tool checks"); return
    arts = database.list_artists()
    assert arts, "no artists — run Phase 0 backfill first"
    name = arts[0]["name"]
    # similar_artists: known → results; unknown → clean error (no fabrication)
    r = tools.similar_artists(name, 3)
    assert "resolved" in r and isinstance(r["results"], list)
    bad = tools.similar_artists("__definitely_not_an_artist__")
    assert "error" in bad and not bad["results"]
    # audio_similarity by artist returns track rows with similarity
    a = tools.audio_similarity(query=name, n=3)
    assert a["results"] and "similarity" in a["results"][0] and "artists" in a["results"][0]
    # metadata_search by BPM band
    m = tools.metadata_search(bpm_min=100, bpm_max=200, n=5)
    assert m["count"] >= 1 and "bpm" in m["results"][0]
    print(f"  ✓ tools: similar_artists/audio_similarity/metadata_search shapes · unknown→error")


def test_agent_tool_wiring():
    # MOCK model: PydanticAI's TestModel drives the agent through its tools
    # without any real LLM — proves the tools are registered and callable.
    from pydantic_ai.models.test import TestModel
    a = agent._build_agent("qwen3:4b")           # tag irrelevant; model is overridden
    with a.override(model=TestModel()):
        result = a.run_sync("recommend artists like techno")
    assert result.output is not None
    # TestModel calls each registered tool once — confirm there were tool calls.
    names = [p.tool_name for m in result.all_messages() for p in getattr(m, "parts", [])
             if getattr(p, "tool_name", None)]
    assert any(n in names for n in
               ("audio_similarity", "similar_artists", "metadata_search")), \
        f"agent did not call its tools: {names}"
    print(f"  ✓ agent wiring (TestModel): tools registered & called {sorted(set(names))}")


if __name__ == "__main__":
    print("\nThe Crate — assistant tests\n" + "─" * 40)
    test_model_registry()
    test_scope_guardrail()
    test_tools()
    test_agent_tool_wiring()
    print("─" * 40 + "\nAll assistant tests passed.\n")
