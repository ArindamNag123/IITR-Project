"""
Loyalty Agent — manages reward points and membership tiers.

Status : STUB (skeleton only)
-------
  This module is registered and routed correctly.  Replace the stub node
  below with real loyalty logic when the feature is ready:

  1. Look up the customer's points balance from the Loyalty Service.
  2. Calculate tier status (Silver / Gold / Platinum).
  3. Show available redemption options.
  4. Apply points during checkout (integrate with billing agent).
"""

from chatbot.agents.base import make_stub_agent
from chatbot.registry import registry

loyalty_agent_node = registry.register(
    routing_key="loyalty",
    keywords=[
        "loyalty", "points", "reward", "rewards", "redeem",
        "membership", "tier", "silver", "gold", "platinum",
        "cashback", "coupon", "voucher", "discount",
    ],
    description="Manages reward points, membership tiers, and redemptions.",
)(make_stub_agent("Loyalty & Rewards"))
