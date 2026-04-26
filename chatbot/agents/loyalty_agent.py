"""
Loyalty Agent — reward points and membership (generalized; no per-user tracking yet).
"""

from langchain_core.messages import AIMessage

from chatbot.registry import registry
from chatbot.state import AgentState

_GENERAL_LOYALTY_REPLY = (
    "We don’t yet link this chat to individual shopper accounts, so I can’t look up "
    "your personal **points balance** or **order history** here.\n\n"
    "**How rewards usually work** (when we plug in accounts later):\n"
    "• You earn **points on eligible purchases** (often a small percentage of what you spend).\n"
    "• Points can be **redeemed** toward discounts or offers, subject to program rules.\n"
    "• Some programs use **tiers** (e.g. standard / plus / premium) with extra perks.\n\n"
    "For now, treat this as **general information** only. "
    "If you have a **billing or order** question, say **invoice**, **order status**, or **place order** and I’ll route you to the right help."
)


@registry.register(
    routing_key="loyalty",
    keywords=[
        "loyalty", "points", "reward", "rewards", "redeem",
        "membership", "tier", "silver", "gold", "platinum",
        "cashback", "coupon", "voucher", "discount",
    ],
    description="Manages reward points, membership tiers, and redemptions.",
)
def loyalty_agent_node(state: AgentState) -> AgentState:
    return {
        "messages": [AIMessage(content=_GENERAL_LOYALTY_REPLY)],
        "metadata": {
            **state.get("metadata", {}),
            "loyalty": {"mode": "generalized", "personalized_balance": False},
        },
    }
