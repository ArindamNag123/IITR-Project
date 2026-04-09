"""
Product Agent — searches the product catalog.

Production wiring:
  • Uses `search_by_text` from the existing similarity_engine directly.
  • If a rag_context is already populated (by the RAG retriever) it is
    prepended to the search query for better grounding.
  • Returns top-k results formatted as a human-readable reply plus a
    structured list in state["metadata"]["products"] for the UI to render.
"""

from langchain_core.messages import AIMessage

from chatbot.registry import registry
from chatbot.state import AgentState


@registry.register(
    routing_key="product",
    keywords=[
        "product", "search", "find", "show", "recommend", "buy", "item",
        "price", "category", "shirt", "shoe", "jeans", "kurta", "dress",
        "men", "women", "colour", "color", "brand", "skincare", "vitamin",
        "supplement", "soap", "lotion", "wash", "cream",
    ],
    description="Searches / filters / recommends products, prices, and categories.",
)
def product_agent_node(state: AgentState) -> AgentState:
    from similarity_engine import search_by_text  # lazy import — avoids heavy init at import time

    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    query = last_human.content if last_human else ""

    # --- RAG grounding (future) ---
    rag_ctx = state.get("rag_context")
    if rag_ctx:
        query = f"{query} {rag_ctx}"

    # Gender inference from the intent label
    intent = state.get("intent", "").lower()
    gender_filter = None
    if "women" in intent or "female" in intent:
        gender_filter = "Women"
    elif "men" in intent or "male" in intent:
        gender_filter = "Men"

    results = search_by_text(query, top_k=3, gender_filter=gender_filter)

    if results:
        lines = ["Here are the products I found:\n"]
        for r in results:
            match_pct = int(max(0.0, min(1.0, r["score"])) * 100)
            lines.append(
                f"• **{r['name']}** — ₹{r['price']}  ({match_pct}% match)"
            )
        reply = "\n".join(lines)
    else:
        reply = (
            "I couldn't find any products matching your query. "
            "Try different keywords — for example a category (shoes, shirt) "
            "or a colour."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": {**state.get("metadata", {}), "products": results},
    }
