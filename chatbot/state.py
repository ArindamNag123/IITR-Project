"""
Shared state schema for the multi-agent LangGraph graph.

Every node reads from and writes to AgentState.  The `messages` field uses
LangGraph's add_messages reducer so each node appends rather than replaces.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation history (append-only via LangGraph reducer)
    messages: Annotated[list, add_messages]

    # Routing decision made by the supervisor
    next_agent: str  # "product" | "billing" | "order" | "translator" | "general"

    # Human-readable intent label (for logging / debugging)
    intent: str

    # Detected language so the translator agent knows direction
    detected_language: str  # "en" | "hi" | "unknown"

    # --- RAG slot (future-ready) -----------------------------------
    # Any agent can populate this before generating its reply.
    # The RAGRetriever in chatbot/rag/retriever.py is the standard way
    # to fill this field.  Leave as None when RAG is not needed.
    rag_context: Optional[str]

    # Generic metadata bag — agents can attach structured payloads
    # (e.g. product list, order dict) for downstream use without
    # polluting the message stream.
    metadata: dict
