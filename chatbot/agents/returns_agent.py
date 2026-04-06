"""
Returns Agent — handles product return and exchange requests.

Status : STUB (skeleton only)
-------
  This module is registered and routed correctly.  Replace the stub node
  below with real returns logic when the feature is ready:

  1. Validate order ID and return eligibility window.
  2. Determine return reason (defective, wrong item, change of mind).
  3. Schedule a reverse-pickup via the logistics API.
  4. Issue store credit or initiate refund.
"""

from chatbot.agents.base import make_stub_agent
from chatbot.registry import registry

returns_agent_node = registry.register(
    routing_key="returns",
    keywords=[
        "return", "returns", "refund", "send back", "replace",
        "damaged", "defective", "wrong item", "exchange",
    ],
    description="Handles product returns, exchanges, and reverse-logistics.",
)(make_stub_agent("Returns & Exchanges"))
