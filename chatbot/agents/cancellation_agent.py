"""
Cancellation Agent — handles order cancellation requests per chatbot/policies.py.
"""

from langchain_core.messages import AIMessage

from chatbot.database import (
    ORDER_STATUS_CANCELLED,
    falkor_db,
    get_invoice_record,
    parse_invoice_number_from_text,
    set_invoice_status,
)
from chatbot.policies import CANCELLATION_WINDOW_DAYS, is_cancellation_eligible
from chatbot.registry import registry
from chatbot.state import AgentState


@registry.register(
    routing_key="cancellation",
    keywords=[
        "cancel", "cancellation", "cancel order", "stop order",
        "don't want", "dont want", "withdraw",
    ],
    description="Handles order cancellation requests and eligibility checks.",
)
def cancellation_agent_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    raw = last_human.content if last_human else ""

    if falkor_db is None:
        return {
            "messages": [AIMessage(content="Database unavailable. Try again later.")],
            "metadata": state.get("metadata", {}),
        }

    invoice_no = parse_invoice_number_from_text(raw)

    if not invoice_no:
        reply = (
            "To cancel your order, please provide your invoice number (e.g., **INV-123456**). "
            "You can find this on your order confirmation or invoice.\n\n"
            f"**Policy:** we only cancel orders that are still **Order Placed** (not shipped) and were placed within "
            f"the last **{CANCELLATION_WINDOW_DAYS} days**. After shipment, use **returns** or **support**."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": state.get("metadata", {}),
        }

    rec = get_invoice_record(invoice_no)
    if rec is None:
        reply = (
            f"I'm sorry, I couldn't find an order with invoice number **{invoice_no}** in our system. "
            "Please double-check the number and try again."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": state.get("metadata", {}),
        }

    status = (rec.get("status") or "").strip()
    if status == ORDER_STATUS_CANCELLED:
        reply = (
            f"Invoice **{invoice_no}** is already **{ORDER_STATUS_CANCELLED.replace('_', ' ')}**. "
            "If you need help with a refund, say **refund** or **billing**."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": state.get("metadata", {}),
        }

    ok, reason = is_cancellation_eligible(rec.get("date"), status)
    if not ok:
        reply = f"I'm not able to cancel invoice **{invoice_no}** under our current rules.\n\n{reason}"
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": state.get("metadata", {}),
        }

    if set_invoice_status(invoice_no, ORDER_STATUS_CANCELLED):
        reply = (
            f"I have cancelled order **{invoice_no}**. "
            "The refund will be processed to the original payment method within **5–7 business days**."
        )
    else:
        reply = (
            "I could not update your order status in the system. "
            "Please try again in a moment or contact support."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": state.get("metadata", {}),
    }
