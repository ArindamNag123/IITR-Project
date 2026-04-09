"""
Support Agent — escalation path for complaints and human-agent handoff.

Status : STUB (skeleton only)
-------
  This module is registered and routed correctly.  Replace the stub node
  below with real support logic when the feature is ready:

  1. Categorise the complaint (product quality, delivery, payment, other).
  2. Create a support ticket in the CRM (Freshdesk / Zendesk / custom).
  3. Provide ticket ID and estimated response time.
  4. Optionally trigger live-chat handoff to a human agent.
"""

from chatbot.agents.base import make_stub_agent
from chatbot.registry import registry

support_agent_node = registry.register(
    routing_key="support",
    keywords=[
        "support", "help me", "complaint", "issue", "problem",
        "grievance", "escalate", "agent", "human", "talk to someone",
        "not working", "broken", "damaged", "wrong",
    ],
    description="Handles complaints, escalations, and human-agent handoff.",
)(make_stub_agent("Customer Support"))
