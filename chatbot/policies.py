"""
Agent policies — rules specialists should follow (enforced where data exists in FalkorDB).

Human-readable summary: docs/AGENT_POLICIES.md
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Tuple

from chatbot.database import (
    ORDER_STATUS_CANCELLED,
    ORDER_STATUS_IN_TRANSIT,
    ORDER_STATUS_PLACED,
    ORDER_STATUS_RETURN_SUCCESSFUL,
)

# --- Windows (calendar days from invoice :Invoice.date) ---
CANCELLATION_WINDOW_DAYS = 5
RETURN_WINDOW_DAYS = 5


def _coalesce_status(status: Optional[str]) -> str:
    s = (status or "").strip()
    return s if s else ORDER_STATUS_PLACED


def invoice_age_days(iso_date: Optional[str]) -> Optional[int]:
    """Days since invoice date (YYYY-MM-DD); None if unparseable or missing."""
    if not iso_date:
        return None
    try:
        d = datetime.strptime(str(iso_date).strip()[:10], "%Y-%m-%d").date()
        return (date.today() - d).days
    except (TypeError, ValueError):
        return None


def is_cancellation_eligible(
    invoice_date: Optional[str],
    status: Optional[str],
) -> Tuple[bool, str]:
    """
    Cancellation is allowed only if:
      • Status is Order_Placed (not yet shipped),
      • Order is not already cancelled or fully returned,
      • Invoice date is within CANCELLATION_WINDOW_DAYS (inclusive of day 0).

    Order_In_Transit is treated as shipped — cancellation is not allowed (contact support).
    """
    st = _coalesce_status(status)

    if st == ORDER_STATUS_CANCELLED:
        return False, "This order is already cancelled."
    if st == ORDER_STATUS_RETURN_SUCCESSFUL:
        return False, "This order has already been returned; cancellation does not apply."
    if st == ORDER_STATUS_IN_TRANSIT:
        return (
            False,
            "This order has already shipped. Our policy does not allow cancellation after "
            "shipment — please contact **support** or use **returns** if the item is eligible.",
        )
    if st != ORDER_STATUS_PLACED:
        return False, f"Orders with status **{st.replace('_', ' ')}** cannot be cancelled through this assistant."

    age = invoice_age_days(invoice_date)
    if age is None:
        return (
            False,
            "We could not verify the order date. Please contact **support** with your invoice number.",
        )
    if age > CANCELLATION_WINDOW_DAYS:
        return (
            False,
            f"Our policy allows cancellation only within **{CANCELLATION_WINDOW_DAYS} days** of purchase. "
            "This order is outside that window — please contact **support**.",
        )
    if age < 0:
        return False, "This invoice date appears invalid. Please contact **support**."
    return True, ""


def is_return_eligible(
    invoice_date: Optional[str],
    status: Optional[str],
) -> Tuple[bool, str]:
    """
    Returns are allowed only if:
      • Status is Order_Placed or Order_In_Transit (not cancelled / not already returned),
      • Invoice date is within RETURN_WINDOW_DAYS.

    (Demo policy: same window as cancellation; adjust RETURN_WINDOW_DAYS for longer retail windows.)
    """
    st = _coalesce_status(status)

    if st == ORDER_STATUS_CANCELLED:
        return False, "This order was cancelled. Returns do not apply; any refund follows cancellation rules."
    if st == ORDER_STATUS_RETURN_SUCCESSFUL:
        return False, "A return has already been recorded for this invoice."

    if st not in (ORDER_STATUS_PLACED, ORDER_STATUS_IN_TRANSIT):
        return False, f"Returns are not available for status **{st.replace('_', ' ')}**. Contact **support**."

    age = invoice_age_days(invoice_date)
    if age is None:
        return (
            False,
            "We could not verify the order date. Please contact **support** with your invoice number.",
        )
    if age > RETURN_WINDOW_DAYS:
        return (
            False,
            f"Our policy accepts returns only within **{RETURN_WINDOW_DAYS} days** of the order date. "
            "Please contact **support** for options.",
        )
    if age < 0:
        return False, "This invoice date appears invalid. Please contact **support**."
    return True, ""
