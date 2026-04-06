"""
Order Agent — tracks orders, cancellations, returns.

Production wiring:
  Replace _MOCK_ORDERS with a call to your Order Management Service (OMS).
  The node signature stays the same — only the data source changes.
"""

from langchain_core.messages import AIMessage

from chatbot.registry import registry
from chatbot.state import AgentState

# ---------------------------------------------------------------------------
# Mock order store — replace with OMS API / DB in production
# ---------------------------------------------------------------------------

_MOCK_ORDERS = {
    "ORD-1001": {
        "status": "Delivered",
        "items": ["Neutrogena Face Wash", "SUKIN Natural Cleanser"],
        "delivery_date": "2 Apr 2026",
        "courier": "Delhivery",
        "tracking_id": "DL9823741",
    },
    "ORD-1002": {
        "status": "Out for Delivery",
        "items": ["BioGaia Probiotic Drops"],
        "delivery_date": "Expected: 6 Apr 2026",
        "courier": "BlueDart",
        "tracking_id": "BD4512098",
    },
    "ORD-1003": {
        "status": "Processing",
        "items": ["Vitagummies Vitamins", "Gripe Water for Infants"],
        "delivery_date": "Expected: 8–10 Apr 2026",
        "courier": "Ekart",
        "tracking_id": "EK1190034",
    },
}


def _format_order(order_id: str, order: dict) -> str:
    items_str = ", ".join(order["items"])
    return (
        f"**Order {order_id}**\n"
        f"Status   : {order['status']}\n"
        f"Items    : {items_str}\n"
        f"Delivery : {order['delivery_date']}\n"
        f"Courier  : {order['courier']}  (Tracking: `{order['tracking_id']}`)"
    )


@registry.register(
    routing_key="order",
    keywords=[
        "order", "track", "status", "delivered", "shipment", "dispatch",
        "return", "exchange", "delivery", "courier", "package",
    ],
    description="Tracks orders, delivery status, shipments, and exchanges.",
)
def order_agent_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    query = (last_human.content if last_human else "").lower()

    # Order ID lookup
    for ord_id, data in _MOCK_ORDERS.items():
        if ord_id.lower() in query:
            reply = _format_order(ord_id, data)
            return {
                "messages": [AIMessage(content=reply)],
                "metadata": {**state.get("metadata", {}), "order": data},
            }

    # General order intents
    if any(w in query for w in ("cancel",)):
        reply = (
            "To cancel an order, please provide your order ID (e.g. ORD-1001).\n"
            "Cancellations are accepted within **24 hours** of placing the order "
            "if it hasn't been dispatched yet."
        )
    elif any(w in query for w in ("return", "exchange")):
        reply = (
            "Returns and exchanges are accepted within **7 days** of delivery.\n"
            "Please share your order ID to initiate the process."
        )
    elif any(w in query for w in ("all order", "my order", "history", "past")):
        lines = ["**Your recent orders:**\n"]
        for oid, data in _MOCK_ORDERS.items():
            lines.append(f"• {oid} — {data['status']} ({', '.join(data['items'])})")
        reply = "\n".join(lines)
    else:
        reply = (
            "I can help you track or manage orders.\n"
            "Please share your order ID (e.g. **ORD-1001**) or ask about "
            "cancellations, returns, or delivery status."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": state.get("metadata", {}),
    }
