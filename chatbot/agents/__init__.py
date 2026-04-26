"""
chatbot/agents — auto-discovery of all agent modules.

Importing this package is enough to register every agent with the
AgentRegistry singleton.  graph.py and supervisor.py then read the
registry dynamically — no manual list to maintain.

To add a new agent:
    1. Create chatbot/agents/my_agent.py and decorate its node function
       with @registry.register(...).
    2. Add a single import line below.
    Done — routing and graph wiring update automatically.
"""

# ---------- implemented agents ----------
from chatbot.agents import product_agent      # noqa: F401
from chatbot.agents import billing_agent      # noqa: F401
from chatbot.agents import order_agent        # noqa: F401
from chatbot.agents import translator_agent   # noqa: F401

# ---------- stub agents (skeleton, ready for future implementation) ----------
from chatbot.agents import cancellation_agent  # noqa: F401
from chatbot.agents import returns_agent       # noqa: F401
from chatbot.agents import loyalty_agent       # noqa: F401
from chatbot.agents import support_agent       # noqa: F401

# ---------- fallback (not in registry — wired directly by graph.py) ----------
from chatbot.agents.general_agent import general_agent_node  # noqa: F401
