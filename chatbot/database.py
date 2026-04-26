import os
import re
import uuid
import logging
from datetime import datetime
from typing import Any, Optional

import redis

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Stored on FalkorDB :Invoice nodes — use these values in MATCH/SET queries
ORDER_STATUS_PLACED = "Order_Placed"
ORDER_STATUS_CANCELLED = "Order_Cancelled"
ORDER_STATUS_IN_TRANSIT = "Order_In_Transit"
ORDER_STATUS_RETURN_SUCCESSFUL = "Order_Return_Successful"

GRAPH_NAME = "products"

_INVOICE_IN_TEXT = re.compile(r"(?:inv|nv)-(\d+)", re.IGNORECASE)


def parse_invoice_number_from_text(text: str) -> Optional[str]:
    """
    Extract invoice key as INV-<digits>. Accepts common typo NV-… (read as INV-…).
    """
    if not text:
        return None
    m = _INVOICE_IN_TEXT.search(text)
    if not m:
        return None
    return f"INV-{m.group(1)}"


def _cypher_escape(s: str) -> str:
    """Escape single backslashes and quotes for Cypher string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _normalize_falkor_row(row: Any) -> list[Any]:
    """
    Falkor/redis-py often returns one outer row whose single element is the
    full list of column values — flatten so _row_to_invoice_dict sees 9 cells.
    """
    if row is None:
        return []
    r = list(row) if isinstance(row, (list, tuple)) else [row]
    if len(r) == 1 and isinstance(r[0], (list, tuple)) and len(r[0]) > 1:
        return list(r[0])
    return r


def _parse_properties_cell(raw: Any) -> dict[str, Any]:
    """Turn properties(i) / map cell from GRAPH.QUERY into a Python dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            key = _norm_str_prop(k)
            out[key] = v
        return out
    if isinstance(raw, str):
        import json

        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                return {str(k): v for k, v in d.items()}
        except Exception:
            pass
        return {}
    if isinstance(raw, (list, tuple)):
        out: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                k = _norm_str_prop(item[0])
                out[k] = item[1]
        return out
    return {}


def _invoice_dict_from_property_map(pm: dict[str, Any]) -> dict[str, Any]:
    """Map Falkor node property names (camelCase from app.py) to our dict."""
    return {
        "invoice_number": _norm_str_prop(
            pm.get("invoiceNumber") or pm.get("invoice_number")
        ),
        "order_id": _norm_str_prop(pm.get("orderID") or pm.get("order_id")),
        "date": _norm_str_prop(pm.get("date")),
        "customer_name": _norm_str_prop(
            pm.get("customerName") or pm.get("customer_name")
        ),
        "status": _norm_str_prop(pm.get("status")),
        "final_total": _norm_num_prop(pm.get("finalTotal") or pm.get("final_total")),
        "itemized_list": _norm_str_prop(
            pm.get("itemizedList") or pm.get("itemized_list")
        ),
        "subtotal": _norm_num_prop(pm.get("subtotal")),
        "gst": _norm_num_prop(pm.get("gst")),
    }


def _merge_invoice_fields(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Fill empty / None slots in base from extra."""
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        cur = base.get(k)
        if cur is None or (isinstance(cur, str) and not str(cur).strip()):
            base[k] = v
    return base


def _invoice_row_incomplete(rec: dict[str, Any]) -> bool:
    """True if we only got an invoice id but no other meaningful fields."""
    if not rec.get("invoice_number"):
        return True
    has_other = any(
        [
            (rec.get("order_id") or "").strip(),
            (rec.get("date") or "").strip(),
            (rec.get("customer_name") or "").strip(),
            (rec.get("status") or "").strip(),
            (rec.get("itemized_list") or "").strip(),
            rec.get("final_total") is not None,
            rec.get("subtotal") is not None,
            rec.get("gst") is not None,
        ]
    )
    return not has_other


def _unwrap_graph_scalar(v: Any) -> Any:
    """FalkorDB sometimes returns a property as a nested one-element list."""
    while isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return v


def _norm_str_prop(v: Any) -> str:
    v = _unwrap_graph_scalar(v)
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (list, tuple)):
        return _norm_str_prop(v[0]) if v else ""
    return str(v)


def _norm_num_prop(v: Any) -> Any:
    v = _unwrap_graph_scalar(v)
    if isinstance(v, (list, tuple)) and v:
        return _norm_num_prop(v[0])
    if isinstance(v, (int, float)):
        return v
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


# --- FalkorDB Connection ---
try:
    falkor_db = redis.Redis(
        host=os.getenv("FALKORDB_HOST"),
        port=int(os.getenv("FALKORDB_PORT", 6379)),
        username=os.getenv("FALKORDB_USER"),
        password=os.getenv("FALKORDB_PASS"),
        decode_responses=True
    )
    # Simple ping to verify connection
    falkor_db.ping()
    logger.info("✅ Connected to FalkorDB")
except Exception as e:
    logger.error(f"❌ FalkorDB Connection Error: {e}")
    falkor_db = None


def _graph_data_rows(result) -> list[list[Any]]:
    """Data rows from GRAPH.QUERY (skips header and execution-time rows)."""
    rows: list[list[Any]] = []
    if not result or len(result) < 2:
        return rows
    for i in range(1, len(result)):
        row = result[i]
        if isinstance(row, str):
            continue
        if not isinstance(row, (list, tuple)) or not row:
            continue
        first = row[0]
        if isinstance(first, str) and (
            "execution time" in first.lower() or first.lower().startswith("cached")
        ):
            continue
        rows.append(list(row))
    return rows


def _graph_result_first_value(result) -> Optional[str]:
    """First cell of the first data row from GRAPH.QUERY, or None."""
    if not result or len(result) < 2:
        return None
    for i in range(1, len(result)):
        row = result[i]
        if isinstance(row, str):
            continue
        if not isinstance(row, (list, tuple)) or not row:
            continue
        cell = _unwrap_graph_scalar(row[0])
        if not isinstance(cell, str):
            continue
        if "execution time" in cell.lower():
            continue
        return cell
    return None


def get_invoice_status(invoice_no: str) -> Optional[str]:
    """Return Invoice.status for an existing invoice; None if not found or DB down."""
    if falkor_db is None:
        return None
    try:
        esc = _cypher_escape(invoice_no)
        q = (
            f"MATCH (i:Invoice {{invoiceNumber: '{esc}'}}) "
            f"RETURN coalesce(i.status, '{ORDER_STATUS_PLACED}') AS status"
        )
        result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
        v = _graph_result_first_value(result)
        if v is None:
            return None
        s = _norm_str_prop(v)
        return s if s else None
    except Exception as e:
        logger.warning("get_invoice_status failed: %s", e)
        return None


def _invoice_return_columns() -> str:
    """All :Invoice fields used for order lookup (raw i.status — no default)."""
    return (
        "i.invoiceNumber, i.orderID, i.date, i.customerName, i.status, i.finalTotal, "
        "i.itemizedList, i.subtotal, i.gst"
    )


def get_invoice_record(invoice_no: str) -> Optional[dict[str, Any]]:
    """
    Full :Invoice row as stored in FalkorDB (no filtering by status).
    None if the invoice does not exist.
    """
    if falkor_db is None:
        return None
    inv = (invoice_no or "").strip()
    if not inv:
        return None
    esc = _cypher_escape(inv)
    cols = _invoice_return_columns()
    try:
        # 1) Exact property match (as written by app.py)
        q1 = f"MATCH (i:Invoice {{invoiceNumber: '{esc}'}}) RETURN {cols}"
        result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q1)
        rows = _graph_data_rows(result)
        if not rows:
            # 2) Case / whitespace differences in stored invoiceNumber (best-effort)
            for q2 in (
                f"MATCH (i:Invoice) WHERE toLower(i.invoiceNumber) = toLower('{esc}') RETURN {cols} LIMIT 1",
                f"MATCH (i:Invoice) WHERE toLower(trim(toString(i.invoiceNumber))) = toLower(trim('{esc}')) RETURN {cols} LIMIT 1",
            ):
                try:
                    result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q2)
                    rows = _graph_data_rows(result)
                    if rows:
                        break
                except Exception as ex:
                    logger.debug("get_invoice_record fallback query skipped: %s", ex)
        if not rows:
            logger.info("get_invoice_record: no rows for invoiceNumber=%r", inv)
            return None
        rec = _row_to_invoice_dict(rows[0])
        if rec is None:
            logger.warning(
                "get_invoice_record: unparsable row raw_len=%s",
                len(_normalize_falkor_row(rows[0])),
            )
            return None
        if _invoice_row_incomplete(rec):
            props = _try_invoice_properties_by_match(esc)
            if props:
                rec = _merge_invoice_fields(rec, props)
        return rec
    except Exception as e:
        logger.warning("get_invoice_record failed: %s", e)
        return None


def _row_to_invoice_dict(row: list[Any]) -> Optional[dict[str, Any]]:
    """Map RETURN row to dict; pad short rows (older Falkor clients may omit null tail)."""
    r = _normalize_falkor_row(row)
    while len(r) < 9:
        r.append(None)
    if len(r) < 6:
        return None
    return {
        "invoice_number": _norm_str_prop(r[0]),
        "order_id": _norm_str_prop(r[1]),
        "date": _norm_str_prop(r[2]),
        "customer_name": _norm_str_prop(r[3]),
        "status": _norm_str_prop(r[4]),
        "final_total": _norm_num_prop(r[5]),
        "itemized_list": _norm_str_prop(r[6]),
        "subtotal": _norm_num_prop(r[7]),
        "gst": _norm_num_prop(r[8]),
    }


def _try_invoice_properties_by_match(esc_invoice: str) -> Optional[dict[str, Any]]:
    """Load all node properties when scalar RETURN columns are empty or mis-parsed."""
    if falkor_db is None:
        return None
    queries = [
        f"MATCH (i:Invoice {{invoiceNumber: '{esc_invoice}'}}) RETURN properties(i) AS p",
        f"MATCH (i:Invoice) WHERE toLower(i.invoiceNumber) = toLower('{esc_invoice}') RETURN properties(i) AS p LIMIT 1",
    ]
    for q in queries:
        try:
            result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
            rows = _graph_data_rows(result)
            if not rows or not rows[0]:
                continue
            cell = rows[0][0]
            pm = _parse_properties_cell(cell)
            if pm:
                return _invoice_dict_from_property_map(pm)
        except Exception as ex:
            logger.debug("properties(i) invoice lookup skipped: %s", ex)
    return None


def _try_invoice_properties_by_order_id(esc_oid: str) -> Optional[dict[str, Any]]:
    if falkor_db is None:
        return None
    q = f"MATCH (i:Invoice {{orderID: '{esc_oid}'}}) RETURN properties(i) AS p"
    try:
        result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
        rows = _graph_data_rows(result)
        if not rows or not rows[0]:
            return None
        pm = _parse_properties_cell(rows[0][0])
        if pm:
            return _invoice_dict_from_property_map(pm)
    except Exception as ex:
        logger.debug("properties(i) orderID lookup skipped: %s", ex)
    return None


def get_invoice_record_by_order_id(order_id: str) -> Optional[dict[str, Any]]:
    """
    Same shape as get_invoice_record, keyed by :Invoice.orderID (8-char UUID prefix).
    """
    if falkor_db is None:
        return None
    oid = order_id.strip().lower()
    if not oid:
        return None
    try:
        esc = _cypher_escape(oid)
        q = (
            f"MATCH (i:Invoice {{orderID: '{esc}'}}) "
            f"RETURN {_invoice_return_columns()}"
        )
        result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
        rows = _graph_data_rows(result)
        if not rows:
            return None
        rec = _row_to_invoice_dict(rows[0])
        if rec is None:
            return None
        if _invoice_row_incomplete(rec):
            props = _try_invoice_properties_by_order_id(esc)
            if props:
                rec = _merge_invoice_fields(rec, props)
        return rec
    except Exception as e:
        logger.warning("get_invoice_record_by_order_id failed: %s", e)
        return None


def list_invoices(limit: int = 40) -> list[dict[str, Any]]:
    """Recent invoices from the graph (best-effort; capped)."""
    if falkor_db is None:
        return []
    lim = max(1, min(int(limit), 200))
    try:
        q = (
            "MATCH (i:Invoice) "
            f"RETURN {_invoice_return_columns()} "
            f"LIMIT {lim}"
        )
        result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
        rows = _graph_data_rows(result)
        out: list[dict[str, Any]] = []
        for row in rows:
            rec = _row_to_invoice_dict(row)
            if rec is None:
                continue
            if _invoice_row_incomplete(rec):
                esc_inv = _cypher_escape(rec.get("invoice_number", ""))
                if esc_inv:
                    props = _try_invoice_properties_by_match(esc_inv)
                    if props:
                        rec = _merge_invoice_fields(rec, props)
            out.append(rec)
        return out
    except Exception as e:
        logger.warning("list_invoices failed: %s", e)
        return []


def set_invoice_status(invoice_no: str, status: str) -> bool:
    """Set Invoice.status. Returns True if a row was updated."""
    if falkor_db is None:
        return False
    try:
        esc_inv = _cypher_escape(invoice_no)
        esc_st = _cypher_escape(status)
        q = (
            f"MATCH (i:Invoice {{invoiceNumber: '{esc_inv}'}}) "
            f"SET i.status = '{esc_st}' RETURN i.invoiceNumber"
        )
        result = falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
        v = _graph_result_first_value(result)
        return v is not None
    except Exception as e:
        logger.warning("set_invoice_status failed: %s", e)
        return False


def save_new_invoice_order(
    products_list: list[str],
    prices_list: list[int],
    user_name: str = "Guest",
) -> Optional[dict[str, Any]]:
    """
    Create a new :Invoice in FalkorDB (same shape as Streamlit checkout in app.py).
    Returns order fields on success, or None if DB is unavailable or the write fails.
    """
    if falkor_db is None:
        logger.warning("save_new_invoice_order: FalkorDB not connected")
        return None
    if not products_list or len(products_list) != len(prices_list):
        return None

    order_id = str(uuid.uuid4())[:8]
    invoice_no = f"INV-{int(datetime.now().timestamp())}"
    ts = datetime.now()
    total_price = int(sum(prices_list))
    gst = int(total_price * 0.05)
    final_total = int(total_price * 1.05)

    items_literal = str(products_list).replace("'", '"')
    esc_inv = _cypher_escape(invoice_no)
    esc_oid = _cypher_escape(order_id)
    esc_name = _cypher_escape(user_name)
    esc_items = _cypher_escape(items_literal)

    try:
        q = (
            "CREATE (:Invoice {"
            f"invoiceNumber: '{esc_inv}', "
            f"orderID: '{esc_oid}', "
            f"date: '{ts.strftime('%Y-%m-%d')}', "
            f"customerName: '{esc_name}', "
            f"itemizedList: '{esc_items}', "
            f"subtotal: {total_price}, "
            f"gst: {gst}, "
            f"finalTotal: {final_total}, "
            f"status: '{ORDER_STATUS_PLACED}'"
            "})"
        )
        falkor_db.execute_command("GRAPH.QUERY", GRAPH_NAME, q)
        return {
            "invoice_no": invoice_no,
            "order_id": order_id,
            "date": ts,
            "products_list": list(products_list),
            "prices_list": list(prices_list),
            "total_price": total_price,
            "gst": gst,
            "final_total": final_total,
        }
    except Exception as e:
        logger.warning("save_new_invoice_order failed: %s", e)
        return None


def get_db():
    return falkor_db
