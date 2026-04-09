"""
General Agent — fallback for greetings, help requests, and out-of-scope
queries that don't match any specialist agent.

This node is NOT registered in AgentRegistry because it is always the
default fallback — the graph wires it directly as the "general" node.
"""

from langchain_core.messages import AIMessage

from chatbot.state import AgentState

_GREETINGS = {"hi", "hello", "hey", "namaste", "hola", "good morning", "good evening"}

_HELP_TEXT = """I'm your **Smart Retail Assistant**. Here's what I can help with:

- **Products**      — *"Show me face wash for women"*
- **Orders**        — *"Where is my order ORD-1001?"*
- **Billing**       — *"Show invoice INV-001"* or *"What is the GST rate?"*
- **Cancellations** — *"Cancel my order ORD-1002"*
- **Returns**       — *"I want to return my purchase"*
- **Loyalty**       — *"How many reward points do I have?"*
- **Translation**   — *"Translate: साबुन"* (Hindi ↔ English)

Just type your question and I'll route it to the right specialist!"""

_OUT_OF_SCOPE_REPLY = (
    "**Out of scope** — that topic isn't something I can help with right now.\n\n"
    + _HELP_TEXT
)


def general_agent_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    text = (last_human.content if last_human else "").strip().lower()

    if any(g in text for g in _GREETINGS):
        reply = "Hello! Welcome to the Smart Retail Assistant.\n\n" + _HELP_TEXT
    elif any(w in text for w in ("help", "what can you do", "capabilities")):
        reply = _HELP_TEXT
    elif any(w in text for w in ("thank", "thanks", "bye", "goodbye")):
        reply = "Thank you for shopping with us! Have a great day. 🛍️"
    else:
        reply = _OUT_OF_SCOPE_REPLY

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": state.get("metadata", {}),
    }
