"""
Supervisor — the routing brain of the multi-agent system.

Routing strategy (dual-mode)
-----------------------------
1. LLM mode    — when OPENAI_API_KEY is set, a structured LLM call produces
                 a RouterDecision.  Understands nuanced, context-rich prompts
                 and Hindi text.
2. Keyword mode — zero-dependency fallback; works offline with no API key.
                 Scores every registered agent's keyword list against the
                 user message.

How to add a new agent
-----------------------
Just register it in chatbot/agents/ — this file needs no changes.
The keyword map and the LLM system prompt are built from the registry
at runtime, so new agents are picked up automatically.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from chatbot.state import AgentState

load_dotenv()

# Registry is imported after agents so all @registry.register decorators
# have already fired by the time supervisor_node is first called.
# (workflow_graph.py imports chatbot.agents before chatbot.agent_router)
from chatbot.registry import registry  # noqa: E402


# ---------------------------------------------------------------------------
# Pydantic model for structured LLM output
# ---------------------------------------------------------------------------

class RouterDecision(BaseModel):
    """
    Structured decision returned by the LLM router.

    ``next_agent`` must be one of the routing keys listed in the system
    prompt, or "general" as the fallback.
    """
    next_agent: str = Field(
        description=(
            "Routing key of the specialist agent that should handle this "
            "message.  Must be one of the keys listed in the system prompt, "
            "or 'general' if none apply."
        )
    )
    intent: str = Field(
        description="One-sentence description of what the user wants."
    )
    detected_language: str = Field(
        description="Language code: 'en', 'hi', or 'unknown'."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# High-confidence patterns checked before keyword scoring.
# Format: (regex_pattern, routing_key)
_PRIORITY_PATTERNS: list[tuple[str, str]] = [
    (r"inv-\d+",                       "billing"),     # invoice ID
    (r"\border\s*(id|#|no)?\s*ord-\d+", "order"),      # order ID
    (r"\btranslat",                    "translator"),
    (r"\bhindi\b|\benglish\b",         "translator"),
    (r"\bcancel\b",                    "cancellation"),
    (r"\breturn\b|\bexchange\b",       "returns"),
    (r"\breward|points\b",             "loyalty"),
]


def _get_user_text(state: AgentState) -> str:
    last = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    return last.content if last else ""


# ---------------------------------------------------------------------------
# Keyword-based fallback router (no LLM required)
# ---------------------------------------------------------------------------

def _keyword_route(message: str) -> RouterDecision:
    """Score every registered agent's keywords against the message."""
    lower = message.lower()

    if _DEVANAGARI_RE.search(message):
        return RouterDecision(
            next_agent="translator",
            intent="Message written in Hindi — routing to translator.",
            detected_language="hi",
        )

    for pattern, agent in _PRIORITY_PATTERNS:
        if re.search(pattern, lower):
            return RouterDecision(
                next_agent=agent,
                intent=f"Priority pattern match → {agent} agent.",
                detected_language="en",
            )

    # Score all registered agents
    scores: dict[str, int] = {key: 0 for key in registry.keyword_map()}
    for agent_key, keywords in registry.keyword_map().items():
        for kw in keywords:
            if kw in lower:
                scores[agent_key] += 1

    best = max(scores, key=lambda k: scores[k]) if scores else "general"

    if not scores or scores[best] == 0:
        return RouterDecision(
            next_agent="general",
            intent="No keyword match — falling back to general agent.",
            detected_language="en",
        )

    return RouterDecision(
        next_agent=best,
        intent=f"Keyword match → {best} agent (score={scores[best]}).",
        detected_language="en",
    )


# ---------------------------------------------------------------------------
# LLM-based router
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Build the system prompt dynamically from the registry."""
    agent_list = registry.describe()
    return (
        "You are the routing supervisor for a Smart Retail e-commerce assistant.\n\n"
        "Given the user's latest message, decide which specialist agent should handle it.\n\n"
        "Available agents:\n"
        f"{agent_list}\n"
        "- general         : Greetings, small talk, help requests, "
        "and anything outside the domains above.\n\n"
        "Rules:\n"
        "• Choose 'general' only when no specialist fits.\n"
        "• Prefer the most specific agent (e.g. 'cancellation' over 'order').\n"
        "• Respond ONLY with the structured JSON — no extra text."
    )


def _llm_route(state: AgentState) -> RouterDecision:
    """LLM-based routing using structured output."""
    from langchain_openai import ChatOpenAI  # lazy import

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(RouterDecision)

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_get_user_text(state)),
    ]
    return structured_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Public supervisor node
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> AgentState:
    """
    Classifies the user's intent and sets routing fields on the state.

    Uses the LLM when OPENAI_API_KEY is configured, otherwise falls back
    to deterministic keyword scoring.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if api_key:
        try:
            decision = _llm_route(state)
        except Exception as exc:
            print(f"[supervisor] LLM routing failed ({exc!r}), falling back to keywords.")
            decision = _keyword_route(_get_user_text(state))
    else:
        decision = _keyword_route(_get_user_text(state))

    # Validate — ensure the LLM didn't hallucinate an unknown key
    valid_keys = set(registry.routing_keys()) | {"general"}
    if decision.next_agent not in valid_keys:
        print(
            f"[supervisor] Unknown agent {decision.next_agent!r} from LLM — "
            "falling back to keyword route."
        )
        decision = _keyword_route(_get_user_text(state))

    print(
        f"[supervisor] → agent={decision.next_agent!r}  "
        f"lang={decision.detected_language!r}  "
        f"intent={decision.intent!r}"
    )

    return {
        "next_agent": decision.next_agent,
        "intent": decision.intent,
        "detected_language": decision.detected_language,
        "rag_context": None,
        "metadata": state.get("metadata", {}),
    }


# ---------------------------------------------------------------------------
# Conditional edge function used by LangGraph
# ---------------------------------------------------------------------------

def route_decision(state: AgentState) -> str:
    """
    Returns the node name to route to.

    Falls back to 'general' for any unrecognised key so the graph never
    reaches a dead end when a new agent key hasn't been wired yet.
    """
    agent = state.get("next_agent", "general")
    valid_nodes = set(registry.routing_keys()) | {"general"}
    return agent if agent in valid_nodes else "general"
