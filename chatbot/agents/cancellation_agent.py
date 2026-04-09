"""
Cancellation Agent — handles order cancellation requests.

Status : STUB (skeleton only)
-------
  This module is registered and routed correctly.  Replace the stub node
  below with real cancellation logic when the feature is ready:

  1. Validate the order ID.
  2. Check cancellation eligibility (time window, dispatch status).
  3. Call the Order Management Service (OMS) cancel endpoint.
  4. Confirm cancellation and trigger any refund flow.
"""

from chatbot.agents.base import make_stub_agent
from chatbot.registry import registry

cancellation_agent_node = registry.register(
    routing_key="cancellation",
    keywords=[
        "cancel", "cancellation", "cancel order", "stop order",
        "don't want", "dont want", "withdraw",
    ],
    description="Handles order cancellation requests and eligibility checks.",
)(make_stub_agent("Cancellation"))
