"""
Cancellation Agent — handles order cancellation requests.

1. Asks for an invoice number if not provided.
2. Validates the invoice against FalkorDB.
3. Confirms cancellation status to the user.
4. Explains the refund process.
"""

from chatbot.database import (
    ORDER_STATUS_CANCELLED,
    get_invoice_status,
    parse_invoice_number_from_text,
    set_invoice_status,
)
from chatbot.registry import registry
from chatbot.state import AgentState
from langchain_core.messages import AIMessage

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

    invoice_no = parse_invoice_number_from_text(raw)

    if invoice_no:
        status = get_invoice_status(invoice_no)
        if status is None:
            reply = (
                f"I'm sorry, I couldn't find an order with invoice number **{invoice_no}** in our system. "
                "Please double-check the number and try again."
            )
        elif status == ORDER_STATUS_CANCELLED:
            reply = (
                f"Invoice **{invoice_no}** is already **{ORDER_STATUS_CANCELLED.replace('_', ' ')}**. "
                "If you need help with a refund, say **refund** or **billing**."
            )
        else:
            if set_invoice_status(invoice_no, ORDER_STATUS_CANCELLED):
                reply = (
                    f"I have found your order with invoice number **{invoice_no}**. "
                    "The order has been successfully cancelled. "
                    "The refund will be processed to the original payment method within 5-7 business days."
                )
            else:
                reply = (
                    "I could not update your order status in the system. "
                    "Please try again in a moment or contact support."
                )
    else:
        reply = (
            "To cancel your order, please provide your invoice number (e.g., **INV-123456**). "
            "You can find this on your order confirmation or invoice."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": state.get("metadata", {}),
    }
