"""
Billing Agent — invoices, payments, GST, and chat-based checkout (catalog + FalkorDB).
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import AIMessage

from chatbot.catalog_lookup import resolve_product_from_user_text
from chatbot.database import save_new_invoice_order
from chatbot.registry import registry
from chatbot.state import AgentState

# ---------------------------------------------------------------------------
# Mock billing data — demo invoice lookup by id
# ---------------------------------------------------------------------------

_MOCK_INVOICES = {
    "INV-001": {"items": [("Sebamed Baby Lotion", 550), ("Vitagummies Vitamins", 680)], "gst_rate": 0.05},
    "INV-002": {"items": [("Neutrogena Face Wash", 650)], "gst_rate": 0.05},
}

_GST_FAQ = (
    "GST on healthcare / personal-care products is typically **5%**.\n"
    "Your final amount = Subtotal + 5% GST."
)

_PLACE_TRIGGERS = (
    "place order",
    "checkout",
    "purchase order",
    "order placed",
    "complete purchase",
    "buy now",
    "pay for",
    "bill for",
    "invoice for",
    "get invoice for",
)

# Greetings only — LLM sometimes routes these to billing; skip mock invoice / catalog checkout.
_GREETING_ONLY_RE = re.compile(
    r"^\s*("
    r"hi\b|hello\b|hey\b|hiya\b|namaste\b|namaskar\b|hola\b|"
    r"good\s+(morning|afternoon|evening|day)\b|"
    r"what'?s\s+up\b|howdy\b|sup\b"
    r")[\s!?.]*$",
    re.I,
)

_MOCK_INV_IN_MESSAGE_RE = re.compile(r"(?:\binv|\bnv)-\d+", re.I)


def _is_greeting_only(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if _GREETING_ONLY_RE.match(raw):
        return True
    if len(raw) <= 16 and raw.lower() in {
        "hi", "hello", "hey", "hola", "namaste", "namaskar", "yo", "sup",
    }:
        return True
    return False


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


def _should_place_order(text: str, resolved_name: str) -> bool:
    tl = text.lower().strip()
    if any(t in tl for t in _PLACE_TRIGGERS):
        return True
    from chatbot.catalog_lookup import normalize_catalog_text

    if normalize_catalog_text(resolved_name) == normalize_catalog_text(text):
        return True
    return False


def _purchase_intent_but_no_product(text: str, resolved: Optional[dict]) -> bool:
    if resolved is not None:
        return False
    tl = text.lower()
    return any(t in tl for t in _PLACE_TRIGGERS) and len(text.strip()) > 12


@registry.register(
    routing_key="billing",
    keywords=[
        "bill", "invoice", "payment", "pay", "charge", "receipt", "total",
        "gst", "tax", "amount", "due", "refund", "money",
        "place order", "checkout", "purchase", "purchase order",
    ],
    description="Handles invoices, payments, GST, receipts, refunds, and catalog checkout.",
)
def billing_agent_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    text = (last_human.content if last_human else "") or ""
    query_lower = text.lower()

    if _is_greeting_only(text):
        reply = (
            "Hello! For **GST, payments, refunds**, or **checkout by product name**, tell me what you need "
            "(for example: **place order for …** or a full product title).\n\n"
            "For **general help** or **product search**, just ask in your own words—the assistant will route you."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": state.get("metadata", {}),
        }

    # --- Demo invoice lookup by id (require an INV-/NV- token; avoid loose substring false positives) ---
    if _MOCK_INV_IN_MESSAGE_RE.search(text):
        for inv_id, data in _MOCK_INVOICES.items():
            if inv_id.lower() in query_lower:
                reply = _format_invoice(inv_id, data)
                return {
                    "messages": [AIMessage(content=reply)],
                    "metadata": {**state.get("metadata", {}), "billing": data},
                }

    resolved = resolve_product_from_user_text(text)

    # --- Checkout: product in catalog → FalkorDB :Invoice ---
    if resolved is not None and _should_place_order(text, resolved["name"]):
        order = save_new_invoice_order(
            products_list=[resolved["name"]],
            prices_list=[int(resolved["price"])],
            user_name="Guest",
        )
        if order:
            reply = (
                "Your order has been placed successfully. "
                f"**Invoice:** {order['invoice_no']} · **Order ID:** `{order['order_id']}` · "
                f"**Total (incl. GST):** ₹{order['final_total']}"
            )
            return {
                "messages": [AIMessage(content=reply)],
                "metadata": {
                    **state.get("metadata", {}),
                    "billing": {"placed_order": order, "product": resolved},
                },
            }
        reply = (
            "The product is in our catalog, but the order could not be saved right now. "
            "Please try again in a moment or contact support."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": state.get("metadata", {}),
        }

    if resolved is not None and not _should_place_order(text, resolved["name"]):
        reply = (
            f"I found **{resolved['name']}** in our catalog at **₹{resolved['price']}**. "
            "Say **place order** (or include **checkout** / **invoice for** with the product) "
            "to complete billing."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": {**state.get("metadata", {}), "billing": {"product": resolved}},
        }

    if _purchase_intent_but_no_product(text, resolved):
        reply = "Sorry, this product doesn't exist in our inventory."
        return {
            "messages": [AIMessage(content=reply)],
            "metadata": state.get("metadata", {}),
        }

    # --- GST FAQ ---
    if any(w in query_lower for w in ("gst", "tax", "rate")):
        reply = _GST_FAQ
    elif any(w in query_lower for w in ("refund", "return payment")):
        reply = (
            "Refunds are processed within **5–7 business days** after the "
            "return is picked up.  The amount is credited to your original "
            "payment method."
        )
    elif any(w in query_lower for w in ("pay", "payment", "method", "upi", "card")):
        reply = (
            "We accept the following payment methods:\n"
            "• UPI (GPay, PhonePe, Paytm)\n"
            "• Credit / Debit cards\n"
            "• Net banking\n"
            "• Cash on delivery (select pincodes)"
        )
    else:
        reply = (
            "I can help with **GST, refunds, payment methods**, and **placing an order** "
            "for a product by name.\n"
            "Share a product name (e.g. *Flying Machine Men Brown Casual Shoes*) or say "
            "**place order for …** to checkout."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": state.get("metadata", {}),
    }
