"""
Billing Agent — handles invoices, payments, and billing queries.

Production wiring:
  Replace the mock data below with calls to your real billing service / DB.
  The interface contract is the same: return a reply string and optionally
  populate state["metadata"]["billing"] with a structured payload.
"""

from langchain_core.messages import AIMessage

from chatbot.registry import registry
from chatbot.state import AgentState

# ---------------------------------------------------------------------------
# Mock billing data — replace with DB / API calls in production
# ---------------------------------------------------------------------------

_MOCK_INVOICES = {
    "INV-001": {"items": [("Sebamed Baby Lotion", 550), ("Vitagummies Vitamins", 680)], "gst_rate": 0.05},
    "INV-002": {"items": [("Neutrogena Face Wash", 650)], "gst_rate": 0.05},
}

_GST_FAQ = (
    "GST on healthcare / personal-care products is typically **5%**.\n"
    "Your final amount = Subtotal + 5% GST."
)


def _format_invoice(inv_id: str, data: dict) -> str:
    subtotal = sum(p for _, p in data["items"])
    gst = int(subtotal * data["gst_rate"])
    lines = [f"**Invoice {inv_id}**\n"]
    for name, price in data["items"]:
        lines.append(f"  • {name} — ₹{price}")
    lines += [
        f"\nSubtotal : ₹{subtotal}",
        f"GST (5%) : ₹{gst}",
        f"**Total  : ₹{subtotal + gst}**",
    ]
    return "\n".join(lines)


@registry.register(
    routing_key="billing",
    keywords=[
        "bill", "invoice", "payment", "pay", "charge", "receipt", "total",
        "gst", "tax", "amount", "due", "refund", "money",
    ],
    description="Handles invoices, payments, GST, receipts, and refunds.",
)
def billing_agent_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    query = (last_human.content if last_human else "").lower()

    # Invoice lookup
    for inv_id, data in _MOCK_INVOICES.items():
        if inv_id.lower() in query:
            reply = _format_invoice(inv_id, data)
            return {
                "messages": [AIMessage(content=reply)],
                "metadata": {**state.get("metadata", {}), "billing": data},
            }

    # GST FAQ
    if any(w in query for w in ("gst", "tax", "rate")):
        reply = _GST_FAQ
    elif any(w in query for w in ("refund", "return payment")):
        reply = (
            "Refunds are processed within **5–7 business days** after the "
            "return is picked up.  The amount is credited to your original "
            "payment method."
        )
    elif any(w in query for w in ("pay", "payment", "method", "upi", "card")):
        reply = (
            "We accept the following payment methods:\n"
            "• UPI (GPay, PhonePe, Paytm)\n"
            "• Credit / Debit cards\n"
            "• Net banking\n"
            "• Cash on delivery (select pincodes)"
        )
    else:
        reply = (
            "I can help with invoices, GST, refunds, and payment methods.\n"
            "Please share your invoice number (e.g. INV-001) or ask a "
            "specific billing question."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": state.get("metadata", {}),
    }
