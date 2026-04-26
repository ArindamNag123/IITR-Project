"""
Shared utilities for all agent node functions.

make_stub_agent()
    Factory that creates a placeholder agent node.  Use it for any module
    that is not yet implemented — it returns a friendly "coming soon" reply
    so routing works end-to-end without breaking the graph.

    Example::

        from chatbot.agents.base import make_stub_agent
        from chatbot.registry import registry

        cancellation_agent_node = registry.register(
            routing_key="cancellation",
            keywords=["cancel", "cancellation"],
            description="Handles order cancellation requests.",
        )(make_stub_agent("Cancellation"))
"""

from langchain_core.messages import AIMessage

from chatbot.state import AgentState


def _last_human_text(state: AgentState) -> str:
    """Extract the text of the most recent human message."""
    msg = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    return msg.content if msg else ""


def make_stub_agent(module_name: str):
    """
    Return an agent node that emits a placeholder response.

    Parameters
    ----------
    module_name:
        Human-readable name shown in the reply (e.g. "Cancellation").
    """
    def _node(state: AgentState) -> AgentState:
        reply = (
            f"**[{module_name} Module]** — This specialist is under active "
            f"development and will be fully available soon.\n\n"
            f"For now, please contact support or try again later."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": {**state.get("metadata", {}), "stub": module_name},
        }

    _node.__name__ = f"{module_name.lower()}_agent_node"
    _node.__qualname__ = _node.__name__
    return _node
