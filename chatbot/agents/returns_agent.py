"""
Returns Agent — registers a return in FalkorDB per chatbot/policies.py.
"""

from langchain_core.messages import AIMessage

from chatbot.database import (
    ORDER_STATUS_CANCELLED,
    ORDER_STATUS_RETURN_SUCCESSFUL,
    falkor_db,
    get_invoice_record,
    parse_invoice_number_from_text,
    set_invoice_status,
)
from chatbot.policies import RETURN_WINDOW_DAYS, is_return_eligible
from chatbot.registry import registry
from chatbot.state import AgentState


@registry.register(
    routing_key="returns",
    keywords=[
        "return", "returns", "refund", "send back", "replace",
        "damaged", "defective", "wrong item", "exchange",
    ],
    description="Handles product returns, exchanges, and reverse-logistics.",
)
def returns_agent_node(state: AgentState) -> AgentState:
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
            "To process a return, please provide your invoice number "
            "(e.g. **INV-123456**). You can find it on your order confirmation or invoice.\n\n"
            f"**Policy:** returns are accepted within **{RETURN_WINDOW_DAYS} days** of the order date for eligible "
            "statuses (**Order Placed** or **Order In Transit**), and not after cancellation or a completed return."
        )
    else:
        rec = get_invoice_record(invoice_no)
        if rec is None:
            reply = (
                f"I could not find an order with invoice **{invoice_no}** in our system. "
                "Please check the number and try again."
            )
        else:
            status = (rec.get("status") or "").strip()
            if status == ORDER_STATUS_RETURN_SUCCESSFUL:
                st = ORDER_STATUS_RETURN_SUCCESSFUL.replace("_", " ")
                reply = (
                    f"Invoice **{invoice_no}** is already marked as **{st}**. "
                    "Your refund should reach your bank account within 3-5 business days if it has not already."
                )
            elif status == ORDER_STATUS_CANCELLED:
                reply = (
                    f"Invoice **{invoice_no}** is **{ORDER_STATUS_CANCELLED.replace('_', ' ')}**. "
                    "Returns are not needed for cancelled orders; any refund follows the cancellation timeline."
                )
            else:
                ok, reason = is_return_eligible(rec.get("date"), status)
                if not ok:
                    reply = (
                        f"I'm not able to register a return for **{invoice_no}** under our current rules.\n\n{reason}"
                    )
                elif set_invoice_status(invoice_no, ORDER_STATUS_RETURN_SUCCESSFUL):
                    st = ORDER_STATUS_RETURN_SUCCESSFUL.replace("_", " ")
                    reply = (
                        f"Your return for invoice **{invoice_no}** has been registered successfully. "
                        f"**Current status:** {st}. "
                        "Your refund will be processed to your bank account within **3-5 business days**."
                    )
                else:
                    reply = (
                        "I could not update your return status in the system. "
                        "Please try again shortly or contact support."
                    )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": state.get("metadata", {}),
    }
