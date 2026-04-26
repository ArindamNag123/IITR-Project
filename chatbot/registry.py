"""
Agent Registry — single source of truth for all specialist agents.

How to add a new agent
-----------------------
1. Create  chatbot/agents/my_agent.py
2. Decorate the node function with @registry.register(...)::

       from chatbot.registry import registry

       @registry.register(
           routing_key="my_agent",
           keywords=["keyword1", "keyword2"],
           description="What this agent handles.",
       )
       def my_agent_node(state: AgentState) -> AgentState:
           ...

3. Import the module in chatbot/agents/__init__.py so the decorator fires.

That's it — supervisor.py and graph.py pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot.state import AgentState


# ---------------------------------------------------------------------------
# Data model for a registered agent
# ---------------------------------------------------------------------------

@dataclass
class AgentDefinition:
    """All metadata and the callable for one specialist agent."""

    routing_key: str
    node_fn: Callable
    keywords: list[str]
    description: str
    enabled: bool = True


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class AgentRegistry:
    """
    Collects agent definitions at import time via the @register decorator.

    The supervisor reads ``keyword_map()`` to route without an LLM.
    The graph reads ``node_map()`` to wire nodes — no manual list to keep.
    The LLM system prompt is built from ``describe()`` so it always reflects
    every registered agent.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        routing_key: str,
        keywords: list[str],
        description: str = "",
        enabled: bool = True,
    ) -> Callable:
        """
        Decorator that registers an agent node function.

        Example::

            @registry.register("billing", keywords=["bill", "invoice"],
                               description="Handles invoices and payments.")
            def billing_agent_node(state): ...
        """
        def decorator(fn: Callable) -> Callable:
            self._agents[routing_key] = AgentDefinition(
                routing_key=routing_key,
                node_fn=fn,
                keywords=keywords,
                description=description,
                enabled=enabled,
            )
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Queries — used by supervisor.py and graph.py
    # ------------------------------------------------------------------

    @property
    def active(self) -> dict[str, AgentDefinition]:
        """All enabled agent definitions, keyed by routing_key."""
        return {k: v for k, v in self._agents.items() if v.enabled}

    def routing_keys(self) -> list[str]:
        """Sorted list of all active routing keys."""
        return sorted(self.active)

    def keyword_map(self) -> dict[str, list[str]]:
        """routing_key → keyword list (used by keyword-based fallback router)."""
        return {k: v.keywords for k, v in self.active.items()}

    def node_map(self) -> dict[str, Callable]:
        """routing_key → node callable (used by graph.py to wire nodes)."""
        return {k: v.node_fn for k, v in self.active.items()}

    def describe(self) -> str:
        """
        Human-readable agent table for the LLM system prompt.

        Example output::
            - billing       : Handles invoices, payments, GST, refunds.
            - cancellation  : Handles order cancellation requests.
        """
        return "\n".join(
            f"- {key:<16}: {defn.description}"
            for key, defn in self.active.items()
        )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------

registry = AgentRegistry()
