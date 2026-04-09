# Chatbot Module — `feature/development/jaydeep`

> All changes documented here were introduced on the `feature/development/jaydeep` branch.
> The base app (product search, image search, cart) already existed on `main`.

---

## What Was Added

| Area | Files |
|---|---|
| Multi-agent graph | `chatbot/graph.py`, `chatbot/state.py` |
| Routing brain | `chatbot/supervisor.py` |
| Agent registry | `chatbot/registry.py` |
| Agent base helper | `chatbot/agents/base.py` |
| Implemented agents | `agents/product_agent.py`, `billing_agent.py`, `order_agent.py`, `translator_agent.py` |
| Stub agents (skeleton) | `agents/cancellation_agent.py`, `returns_agent.py`, `loyalty_agent.py`, `support_agent.py` |
| Fallback agent | `agents/general_agent.py` |
| RAG slot (future) | `chatbot/rag/retriever.py` |
| UI integration | `app.py` — floating chat bubble + chat panel wired to the graph |
| Config | `.env.example` — `OPENAI_API_KEY`, `OPENAI_MODEL` |

---

## High-Level Flow

```
User types a message in the chat panel
              │
              ▼
      ┌───────────────┐
      │   Supervisor  │  ← classifies intent
      │               │    (LLM if present,
      │               │     keyword fallback otherwise)
      └──────┬────────┘
             │  routes to →
     ┌───────┴────────────────────────────────────────────┐
     │                                                    │
  [product]  [billing]  [order]  [translator]  ...stubs  [general]
     │            │         │          │                      │
  Search      Invoice    Track      Translate            Out-of-scope
  catalog     / GST      order      text                 / greetings
     │            │         │          │                      │
     └────────────┴─────────┴──────────┴──────────────────────┘
                                    │
                              Reply displayed
                              in chat panel
```

---

## Detailed Component Diagram

```
chatbot/
├── __init__.py              exports retail_graph (public API)
│
├── state.py                 AgentState (TypedDict)
│                             • messages       — full conversation history
│                             • next_agent     — routing decision
│                             • intent         — human-readable label
│                             • detected_language
│                             • rag_context    — future RAG grounding slot
│                             • metadata       — structured payload bag
│
├── registry.py              AgentRegistry singleton
│                             • @registry.register(routing_key, keywords, description)
│                             • registry.keyword_map()  → used by supervisor
│                             • registry.node_map()     → used by graph
│                             • registry.describe()     → injected into LLM prompt
│
├── supervisor.py            Routes each message to the right agent
│                             • LLM mode  — gpt-4o-mini with structured output
│                             • Keyword mode — deterministic fallback (no API key needed)
│                             • Priority patterns — regex shortcuts (e.g. INV-xxx → billing)
│
├── graph.py                 LangGraph StateGraph
│                             • Reads all nodes from registry.node_map() automatically
│                             • Adding a new agent requires zero changes here
│
└── agents/
    ├── base.py              make_stub_agent(name) — placeholder factory
    │
    ├── product_agent.py     IMPLEMENTED — calls similarity_engine.search_by_text()
    ├── billing_agent.py     IMPLEMENTED — invoice lookup, GST FAQ, payment methods
    ├── order_agent.py       IMPLEMENTED — order tracking, returns, cancellations
    ├── translator_agent.py  IMPLEMENTED — Hindi ↔ English (LLM or glossary fallback)
    │
    ├── cancellation_agent.py  STUB — ready for OMS cancel API
    ├── returns_agent.py       STUB — ready for reverse-logistics integration
    ├── loyalty_agent.py       STUB — ready for rewards/points service
    ├── support_agent.py       STUB — ready for CRM / live-chat handoff
    │
    └── general_agent.py     FALLBACK — greetings, help menu, out-of-scope replies
```

---

## Routing Logic (Supervisor)

### Step 1 — Hindi detection
If the message contains Devanagari script → route directly to `translator`.

### Step 2 — Priority regex patterns (checked before scoring)

| Pattern | Routes to |
|---|---|
| `INV-\d+` | billing |
| `ORD-\d+` | order |
| `\bcancel\b` | cancellation |
| `\breturn\b` or `\bexchange\b` | returns |
| `\breward` or `points\b` | loyalty |
| `\btranslat` | translator |

### Step 3 — Keyword scoring
Every registered agent has a keyword list. The supervisor scores the message against all lists and picks the highest-scoring agent.

### Step 4 — Fallback
If score is 0 across all agents → `general` → returns **"Out of scope"** reply with help menu.

---

## Dual-Mode Routing

```
OPENAI_API_KEY set?
        │
       YES ──► LLM router (gpt-4o-mini)
        │       • understands nuanced phrasing
        │       • handles Hindi natively
        │       • falls back to keywords on API error
        │
        NO ──► Keyword router
                • zero dependencies
                • works offline / demo / CI
```

---

## Agent Status

| Agent | Routing Key | Status | Triggered by |
|---|---|---|---|
| Product | `product` | Implemented | "show me", "find", "buy", product names, categories |
| Billing | `billing` | Implemented | "bill", "invoice", "GST", "refund", INV-xxx |
| Order | `order` | Implemented | "track", "order", "delivery", ORD-xxx |
| Translator | `translator` | Implemented | Hindi text, "translate", "hindi" |
| Cancellation | `cancellation` | Stub | "cancel", "withdraw", "don't want" |
| Returns | `returns` | Stub | "return", "refund", "damaged", "exchange" |
| Loyalty | `loyalty` | Stub | "points", "reward", "coupon", "tier" |
| Support | `support` | Stub | "complaint", "issue", "talk to someone" |
| General | `general` | Implemented | anything unmatched |

---

## How to Add a New Agent (3 Steps)

**Step 1** — Create `chatbot/agents/my_agent.py`:

```python
from chatbot.registry import registry
from chatbot.agents.base import make_stub_agent

my_agent_node = registry.register(
    routing_key="my_agent",
    keywords=["keyword1", "keyword2"],
    description="What this agent handles.",
)(make_stub_agent("My Agent"))
```

Replace `make_stub_agent(...)` with real logic when ready.

**Step 2** — Add one import to `chatbot/agents/__init__.py`:

```python
from chatbot.agents import my_agent  # noqa: F401
```

**Step 3** — Done. The supervisor and graph pick it up automatically.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | No | — | Enables LLM-based routing and translation |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Which OpenAI model to use |

Copy `.env.example` → `.env` and fill in values. The app runs without any key using keyword fallback.

---

## UI Integration (`app.py`)

- Floating **💬 chat button** (bottom-right) — opens/closes the panel via `?chat=open` query param
- When open: page splits into **products (60%)** + **chat panel (40%)**
- Chat panel title: **🛍️ Smart Retail Assistant**
- Each reply shows `→ <agent> agent` caption so users can see which specialist responded
- Chat history persists within the Streamlit session

---

## Files Changed on This Branch

| File | Change Type | Description |
|---|---|---|
| `app.py` | Modified | Added chat panel, floating button, graph invocation; refactored into named functions |
| `config.py` | Modified | Any config additions for chatbot paths |
| `chatbot/` | New directory | Entire chatbot module |
| `.env.example` | New | Template for API keys |
