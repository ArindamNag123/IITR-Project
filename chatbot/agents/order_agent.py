"""
Order Agent — fetch an :Invoice from FalkorDB by invoice number and reply with stored fields.
"""

from langchain_core.messages import AIMessage

from chatbot.database import falkor_db, get_invoice_record, parse_invoice_number_from_text
from chatbot.registry import registry
from chatbot.state import AgentState


def _format_invoice(rec: dict) -> str:
    """Plain text: whatever FalkorDB stored on this invoice."""

    def dash(v) -> str:
        if v is None or v == "":
            return "—"
        return str(v)

    def rupee(key: str) -> str:
        v = rec.get(key)
        return f"₹{v}" if v is not None else "—"

    st = (rec.get("status") or "").strip().replace("_", " ") or "—"
    return "\n".join(
        [
            f"Invoice: **{dash(rec.get('invoice_number'))}**",
            f"Status: **{st}**",
            f"Order ID: `{dash(rec.get('order_id'))}`",
            f"Date: {dash(rec.get('date'))}",
            f"Customer: {dash(rec.get('customer_name'))}",
            f"Items: {dash(rec.get('itemized_list'))}",
            f"Subtotal: {rupee('subtotal')}",
            f"GST: {rupee('gst')}",
            f"Total: {rupee('final_total')}",
        ]
    )


@registry.register(
    routing_key="order",
    keywords=[
        "order", "track", "status", "invoice", "delivery", "shipment",
        "my order", "order id", "invoice number",
    ],
    description="Fetches order/invoice details from the database by invoice number.",
)
def order_agent_node(state: AgentState) -> AgentState:
    last = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    text = last.content if last else ""

    if falkor_db is None:
        return {
            "messages": [AIMessage(content="Database unavailable. Try again later.")],
            "metadata": state.get("metadata", {}),
        }

    inv = parse_invoice_number_from_text(text)
    if not inv:
        return {
            "messages": [
                AIMessage(
                    content="Share your **invoice number** (e.g. **INV-1234567890** or **NV-1234567890**)."
                )
            ],
            "metadata": state.get("metadata", {}),
        }

    rec = get_invoice_record(inv)
    if rec is None:
        return {
            "messages": [
                AIMessage(
                    content=f"No invoice **{inv}** in our system. Check the number and try again."
                )
            ],
            "metadata": state.get("metadata", {}),
        }

    return {
        "messages": [AIMessage(content=_format_invoice(rec))],
        "metadata": {**state.get("metadata", {}), "invoice": rec},
    }
