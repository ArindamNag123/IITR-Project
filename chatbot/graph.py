"""
LangGraph StateGraph — wires the supervisor and all registered agents.

Graph topology
--------------

    START
      │
      ▼
  [supervisor]          ← classifies intent, sets next_agent
      │
      │  conditional edge  (route_decision)
      │
      ├──► [product]       ──► END
      ├──► [billing]       ──► END
      ├──► [order]         ──► END
      ├──► [translator]    ──► END
      ├──► [cancellation]  ──► END   (stub)
      ├──► [returns]       ──► END   (stub)
      ├──► [loyalty]       ──► END   (stub)
      ├──► [support]       ──► END   (stub)
      └──► [general]       ──► END   ← fallback for out-of-scope queries

Extending the graph
--------------------
1. Create chatbot/agents/my_agent.py and decorate its node with
   @registry.register(routing_key="my_agent", keywords=[...]).
2. Add one import line in chatbot/agents/__init__.py.
3. Done — this file needs no changes.
"""

import chatbot.agents  # triggers all @registry.register decorators  # noqa: F401

from langgraph.graph import END, START, StateGraph

from chatbot.agents.general_agent import general_agent_node
from chatbot.registry import registry
from chatbot.state import AgentState
from chatbot.supervisor import route_decision, supervisor_node


def build_graph():
    """
    Compile and return the LangGraph StateGraph.

    Reads the active agent map from the registry at build time so new agents
    are picked up without touching this function.
    """
    graph = StateGraph(AgentState)

    # --- Supervisor node (always present) ---
    graph.add_node("supervisor", supervisor_node)

    # --- Specialist agent nodes (auto-discovered from registry) ---
    agent_nodes = registry.node_map()
    for routing_key, node_fn in agent_nodes.items():
        graph.add_node(routing_key, node_fn)

    # --- Fallback / general node (not in registry, always wired) ---
    graph.add_node("general", general_agent_node)

    # --- Entry point ---
    graph.add_edge(START, "supervisor")

    # --- Conditional routing: supervisor → specialist or general ---
    all_node_names = {**{k: k for k in agent_nodes}, "general": "general"}
    graph.add_conditional_edges("supervisor", route_decision, all_node_names)

    # --- All agents → END (single-turn, stateless per request) ---
    for name in agent_nodes:
        graph.add_edge(name, END)
    graph.add_edge("general", END)

    return graph.compile()


# Module-level singleton — imported by app.py and any other entry point.
retail_graph = build_graph()
