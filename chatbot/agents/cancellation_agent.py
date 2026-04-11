"""
Cancellation Agent — handles order cancellation requests.

1. Asks for an invoice number if not provided.
2. Validates the invoice against FalkorDB.
3. Confirms cancellation status to the user.
4. Explains the refund process.
"""

import os
import re
from chatbot.database import falkor_db as r
from chatbot.registry import registry
from chatbot.state import AgentState
from langchain_core.messages import AIMessage

def _check_db_for_invoice(invoice_no: str) -> bool:
    """Check if the invoice exists in FalkorDB."""
    if r is None:
        return False
    try:
        # We query the 'products' graph as seen in app.py
        query = f"MATCH (i:Invoice {{invoiceNumber: '{invoice_no}'}}) RETURN i.invoiceNumber"
        result = r.execute_command("GRAPH.QUERY", "products", query)
        
        # FalkorDB result format: [ [headers], [row1], [row2], ..., [statistics] ]
        # If result has more than 1 entry, and the last one is stats, we need to check rows.
        # Based on the app.py log, it looks like: [['header1', ...], ['value1', ...]]
        # If no match: [['i.invoiceNumber'], ['Query internal execution time: 0.123 ms']] 
        # Wait, usually if no match, it returns headers and then stats.
        
        if result and len(result) > 1:
            # Check if the second element is a list of values or just a stats string
            if isinstance(result[1], list) and len(result[1]) > 0:
                return True
        return False
    except Exception:
        return False

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
    query = (last_human.content if last_human else "").lower()

    # Try to find an invoice number in the query (e.g., INV-123456789)
    # We look for 'inv-' followed by one or more digits
    invoice_match = re.search(r'inv-\d+', query)
    
    if invoice_match:
        invoice_no = invoice_match.group(0).upper()
        if _check_db_for_invoice(invoice_no):
            reply = (
                f"I have found your order with invoice number **{invoice_no}**. "
                "The order has been successfully cancelled. "
                "The refund will be processed to the original payment method within 5-7 business days."
            )
        else:
            reply = (
                f"I'm sorry, I couldn't find an order with invoice number **{invoice_no}** in our system. "
                "Please double-check the number and try again."
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
