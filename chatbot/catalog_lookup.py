"""
Shared catalog resolution for billing checkout and routing (CSV + optional semantic search).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

import pandas as pd

from config import DATA_PATH

_SEMANTIC_MIN_SCORE = 0.18


@lru_cache(maxsize=1)
def _catalog_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def normalize_catalog_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def whole_message_matches_catalog_product(text: str) -> bool:
    """True if the message (trimmed) is exactly one productDisplayName in the CSV."""
    if not text or len(text.strip()) < 6:
        return False
    n = normalize_catalog_text(text)
    for _, row in _catalog_df().iterrows():
        name = str(row["productDisplayName"]).strip()
        if normalize_catalog_text(name) == n:
            return True
    return False


def resolve_product_from_user_text(text: str) -> Optional[dict[str, Any]]:
    """
    Match the user message to a single catalog row: exact name, substring, or semantic search.
    Returns dict with id, name, price or None.
    """
    if not text or not text.strip():
        return None

    raw = text.strip()
    norm_msg = normalize_catalog_text(raw)
    df = _catalog_df()

    for _, row in df.iterrows():
        name = str(row["productDisplayName"]).strip()
        if normalize_catalog_text(name) == norm_msg:
            return {"id": int(row["id"]), "name": name, "price": int(row["price"])}

    matches: list[tuple[int, Any]] = []
    for _, row in df.iterrows():
        name = str(row["productDisplayName"]).strip()
        nl = name.lower()
        if nl and nl in raw.lower():
            matches.append((len(name), row))
    if matches:
        row = max(matches, key=lambda x: x[0])[1]
        name = str(row["productDisplayName"]).strip()
        return {"id": int(row["id"]), "name": name, "price": int(row["price"])}

    if len(norm_msg) >= 8:
        subs: list[Any] = []
        for _, row in df.iterrows():
            name = str(row["productDisplayName"]).strip()
            if norm_msg in name.lower():
                subs.append(row)
        if len(subs) == 1:
            row = subs[0]
            return {
                "id": int(row["id"]),
                "name": str(row["productDisplayName"]).strip(),
                "price": int(row["price"]),
            }

    try:
        from similarity_engine import search_by_text
    except Exception:
        return None

    hits = search_by_text(raw, top_k=1, gender_filter=None)
    if hits and float(hits[0]["score"]) >= _SEMANTIC_MIN_SCORE:
        h = hits[0]
        return {"id": int(h["id"]), "name": str(h["name"]), "price": int(h["price"])}

    return None
