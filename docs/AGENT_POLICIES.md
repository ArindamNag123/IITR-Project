# Agent policies — rules and regulations

This document describes what each specialist agent **should** do and the **business rules** they follow.  
Enforcement: **cancellation** and **returns** use `chatbot/policies.py` plus FalkorDB invoice `date` and `status`. Other sections are **behavioral / informational** for the LLM and maintainers.

---

## Cancellation agent

| Rule | Detail |
|------|--------|
| **When cancellation is allowed** | Invoice status is **Order_Placed** (not shipped), and the order date is within **5 calendar days** of today. |
| **When cancellation is refused** | Status is **Order_In_Transit** (treated as shipped), **Order_Cancelled**, **Order_Return_Successful**, or the order is **older than 5 days**. |
| **After successful cancel** | Status set to **Order_Cancelled**; user informed refund timeline (e.g. 5–7 business days to original payment method). |
| **Invoice required** | User must provide **INV-…** (or **NV-…** parsed as INV). |

---

## Returns agent

| Rule | Detail |
|------|--------|
| **When a return is allowed** | Status is **Order_Placed** or **Order_In_Transit**, not already returned or cancelled, and order date is within **5 calendar days** (demo window; increase `RETURN_WINDOW_DAYS` in code for longer policies). |
| **When a return is refused** | Cancelled orders, already returned, outside the return window, or unverifiable date. |
| **After successful return** | Status set to **Order_Return_Successful**; user informed bank refund **3–5 business days** (messaging in agent). |
| **Invoice required** | User must provide **INV-…**. |

---

## Order agent

| Rule | Detail |
|------|--------|
| **Scope** | Read-only lookup: invoice / order fields from FalkorDB by **INV-…** or **order ID** (hex), per routing. |
| **No mutations** | Does not cancel, return, or change status — directs users to **cancellation** / **returns** / **support** as needed. |
| **Accuracy** | Reflects whatever is stored on **:Invoice**; if missing, asks user to verify the number. |

---

## Billing agent

| Rule | Detail |
|------|--------|
| **Catalog checkout** | Product must exist in inventory (CSV / search). Places **:Invoice** with **Order_Placed** when save succeeds. |
| **GST** | **5%** on subtotal for generated amounts (aligned with app checkout). |
| **Demo invoices** | **INV-001** / **INV-002** mock lines only when message contains an **INV-/NV-** token. |
| **Greetings** | Pure greetings get a short info reply — no mock invoice or checkout without intent. |
| **No user accounts** | Checkout is under a generic customer (e.g. **Guest**) unless extended later. |

---

## Product agent

| Rule | Detail |
|------|--------|
| **Scope** | Search and describe catalog items; no order placement (that is **billing**). |
| **Data source** | Results come from the local catalog / similarity engine — not live stock in physical stores. |
| **Pricing** | Shown as in catalog data; final totals at checkout may include taxes per **billing**. |

---

## Loyalty agent

| Rule | Detail |
|------|--------|
| **Applicability** | **No per-user tracking** — no personal points balance or order count in this build. |
| **Messaging** | General explanation of how loyalty programs *typically* work; not a binding offer. |
| **Escalation** | Directs billing / order questions to the appropriate intents. |

---

## Translator agent

| Rule | Detail |
|------|--------|
| **Role** | Language normalization for the supervisor and other agents; does not change order or billing policy. |

---

## Support agent

| Rule | Detail |
|------|--------|
| **Role** | Stub / generic help; edge cases outside automated policy should be escalated to human support in a full deployment. |

---

## Adjusting windows and rules

Edit **`chatbot/policies.py`**:

- `CANCELLATION_WINDOW_DAYS`
- `RETURN_WINDOW_DAYS`
- Logic in `is_cancellation_eligible` / `is_return_eligible` (e.g. whether **Order_In_Transit** may cancel)

Then keep this document in sync for stakeholders.
