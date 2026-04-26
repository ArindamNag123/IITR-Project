# IITR-Project

# Agentic Multimodal Commerce Platform

**Conversational product discovery, multilingual retrieval, and transparent billing—driven by an agentic, explainable AI stack.**

---

## 🚀 What is this?

The **Agentic Multimodal Commerce Platform** is a research and engineering initiative that lets shoppers find products and complete checkout-style flows using **natural language** (including multiple languages) and **product images**, not just keyword search.

Today's commerce assistants often treat text and images in isolation and automate billing without enough transparency. This project builds a **single conversational system** that routes queries intelligently across specialized agents, retrieves products using text and image similarity, and **explains** what it matched and why—so it can be taken seriously in real retail and e-commerce settings.

---

## 🎯 Why it exists

| Gap | How this project addresses it |
|-----|--------------------------------|
| Discovery is text-only or image-only, rarely both in one flow | Unified chat + image upload with semantic search across modalities |
| Weak multilingual support | **Translation** to a canonical language for search and response localization |
| Billing feels like a black box | **Billing agent** with order tracking and transparent invoice generation |
| Brittle, linear AI pipelines | **LangGraph**: explicit supervisor routing, pluggable agents, and policy enforcement |
| No audit trail for commerce decisions | **FalkorDB** order storage and **agent routing logs** for full traceability |

The goal is an architecture that is **agentic** (intelligent routing), **explainable**, **multilingual**, and **safe-by-design**—aligned with how production teams expect to deploy AI in commerce.

---

## 🧠 Key ideas / concepts

- **Agentic orchestration** — A supervisor LLM/keyword router classifies user intent and routes to specialized agents; workflows adapt dynamically without hardcoded chains.
- **LangGraph supervisor routing** — A single supervisor node routes to product agents, billing, order lookup, translation, returns, cancellation, or a fallback general agent.
- **Multimodal retrieval** — Text search via TF-IDF + FAISS and image search via CLIP embeddings both query a product catalog; results are ranked and returned in user's language.
- **Order & invoice storage** — **FalkorDB** persists orders, invoices, and customer data for audit, returns, and cancellation workflows.
- **Guardrails & policies** — Enforced rules for cancellation windows, return eligibility, and billing thresholds (see `chatbot/policies.py`).
- **Open-source stack** — Python, Streamlit, LangGraph, open LLMs and embedding models, enabling reproducibility and cost control.

**What makes the platform "agentic"**  
Dynamic supervisor-based routing, intent classification, specialized agent responsibility, policy-driven automation, and explainable decisions tied to FalkorDB records.


---

## 🛠️ What’s inside

>*Repository layout reflects the current implementation.*

| Area | Role |
|------|------|
| **Streamlit app** (`app.py`) | Conversational UI: chat, image upload, streaming responses, cart, billing, order status lookup |
| **LangGraph runtime** (`chatbot/chatbot_controller.py`) | Graph definition: supervisor → agents → END; state management |
| **Agent layer** (`chatbot/agents/`) | Specialized agents: product, billing, order, translator, returns, cancellation, loyalty, support, general |
| **Supervisor & routing** (`chatbot/agent_router.py`) | LLM-based or keyword-based intent detection and agent selection |
| **Order & invoice storage** (`chatbot/database.py`) | FalkorDB connection for persisting and retrieving orders, invoices, and order status |
| **Policy enforcement** (`chatbot/policies.py`) | Business rules for returns, cancellations, and billing |
| **Retrieval** (`chatbot/rag/retriever.py`) | TF-IDF + FAISS-based text retrieval; ready for embedding model swap |
| **Image search** (`similarity_engine.py`) | CLIP-based image embeddings and similarity search over product catalog |
| **Config / prompts** | Model endpoints, thresholds, and system prompts for agents |

**Actual folder structure**

```text
iitr-project/
├── README.md                 # This file
├── requirements.txt
├── app.py                    # Streamlit entry point
├── config.py                 # Configuration and paths
├── similarity_engine.py       # CLIP-based image search utility
├── chatbot/
│   ├── __init__.py
│   ├── agent_router.py       # Supervisor: LLM-based intent routing
│   ├── chatbot_controller.py  # LangGraph StateGraph definition
│   ├── database.py           # FalkorDB connection and invoice management
│   ├── policies.py           # Policy rules (return windows, cancellation, etc.)
│   ├── registry.py           # Agent registration and discovery
│   ├── state.py              # AgentState schema
│   ├── translator_module.py  # Multilingual translation utilities
│   ├── agents/
│   │   ├── __init__.py       # Auto-loads all agents via @registry.register decorators
│   │   ├── base.py           # Base agent class
│   │   ├── product_agent.py  # Product search and retrieval
│   │   ├── billing_agent.py  # Checkout, pricing, tax, invoice generation
│   │   ├── order_agent.py    # Order lookup and status
│   │   ├── translator_agent.py # Language translation
│   │   ├── returns_agent.py  # Return processing
│   │   ├── cancellation_agent.py # Cancellation processing
│   │   ├── loyalty_agent.py  # Loyalty program (stub)
│   │   ├── support_agent.py  # Customer support (stub)
│   │   └── general_agent.py  # Fallback for out-of-scope queries
│   └── rag/
│       ├── __init__.py
│       └── retriever.py      # TF-IDF + FAISS retriever
├── data/
│   └── styles.csv            # Product catalog
├── dataset/
│   └── images/               # Product images for similarity search
└── docs/
    ├── AGENT_POLICIES.md     # Detailed policy rules
    └── CHATBOT_MODULE.md     # Agent system documentation

**Core technology choices (from the proposal)**

| Layer | Technologies |
|-------|----------------|
| Frontend | **Streamlit** — chat UI, image upload, streaming, cart, billing |
| Backend & orchestration | **Python**, **LangGraph**, **pydantic** |
| Routing & intent | **OpenAI API** (LLM-based supervisor) or keyword fallback (zero-dependency) |
| LLMs (open) | OpenAI GPT, LLaMA, Mistral, or Mixtral (configurable) |
| Text Search | TF-IDF + FAISS (in-memory, production-ready with vector DB swap) |
| Image Search | CLIP embeddings + cosine similarity |
| Vector & RAG | **FalkorDB** — text and image vectors (Redis-based graph database) |

### Agent responsibilities (at a glance)

| Agent |	Responsibility |
|-------|------------------|
| **Supervisor** | Classify user intent (text vs order lookup vs returns, etc.); route to specialist agent |
| **Product** | Retrieve products via text/image similarity; return ranked candidates with descriptions and prices |
| **Translator** | Normalize user queries and responses to/from user's language |
| **Billing & validation** | Price calculation, tax, discounts, invoice generation; checkout workflow |
| **Order** | Retrieve order status and invoice details from FalkorDB by order ID or invoice number |
| **Returns** | Process returns per policy rules (e.g., 30-day window, condition); update FalkorDB status |
| **Cancellation** | Process order cancellation per policy rules; update FalkorDB status |
| **Loyalty** | Loyalty points and rewards (stub — ready for implementation) |
| **Support** | Escalation and customer support workflows (stub — ready for implementation) |
| **General** | Fallback for out-of-scope or ambiguous queries |

---

## 📊 Examples / outputs

**LangGraph Flow A — Text query**

```text
START
  │
  ▼
[supervisor]  ← Classifies intent, routes to next_agent
  │
  │ conditional edge (route_decision)
  │
  ├──► [product]       ──► END
  ├──► [billing]       ──► END
  ├──► [order]         ──► END
  ├──► [translator]    ──► END
  ├──► [returns]       ──► END
  ├──► [cancellation]  ──► END
  ├──► [loyalty]       ──► END
  ├──► [support]       ──► END
  └──► [general]       ──► END (fallback)
```

User uploads an image and asks in Hindi to generate a bill for that product. The system may: use vision first; fall back to ANOCR when vision confidence is low; normalize entities; confirm the product via FalkorDB, produces billing and then returns a short explanation in the user’s language.

Scenario A — Product search by text:
User: "I need a face wash for dry skin"
  → Supervisor: Intent = "product_search"
  → Product agent: Text embedding → FAISS similarity search → returns 3–5 products
  → Translator: Localizes product names and descriptions to user's language
  → Response: Ranked products with prices and details

Scenario B — Product search by image:
User: [uploads product image]
  → Supervisor: Intent = "image_search"
  → Product agent: CLIP image embedding → similarity search → returns 3–5 products
  → Response: Ranked products with prices and details

Scenario C — Order lookup:
User: "Where's my order? INV-12345"
  → Supervisor: Intent = "order_lookup"
  → Order agent: Parse invoice number from FalkorDB → retrieve status, total, items
  → Response: Order status, expected delivery, and billing details

Scenario D — Multilingual query:
User: "मुझे बिल दिखाओ" (Hindi: "Show me the bill")
  → Supervisor: Detects Hindi intent = "billing_help"
  → Translator: Translates to English for retrieval
  → Billing agent: Retrieves or generates invoice
  → Translator: Translates response back to Hindi
  → Response: Invoice in Hindi

Scenario E — Return request:
User: "I want to return order ORD-ABC123"
  → Supervisor: Intent = "return_request"
  → Returns agent: Fetch order from FalkorDB → check 30-day window → issue RMA → update status
  → Response: Return approved with RMA number and instructions


**Illustrative architecture**

```
┌────────────────────────────────────┐
│ Streamlit UI                       │
│ (Chat + Image Upload + Cart)       │
└──────────────┬─────────────────────┘
               │ user input
               ▼
┌────────────────────────────────────┐
│ LangGraph Supervisor Node          │
│ Intent → Route Decision            │
└──────────────┬─────────────────────┘
               │ route to specialist
               ▼
┌────────────────────────────────────┐
│ Specialized Agent Nodes            │
│ Product · Billing · Order          │
│ Translator · Returns · Cancel      │
│ Loyalty · Support · General        │
└──────────────┬─────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    ┌──────────┐  ┌──────────────┐
    │ FAISS    │  │ FalkorDB     │
    │ (Text &  │  │ (Orders,     │
    │ Images)  │  │ Invoices)    │
    └──────────┘  └──────────────┘
```

*Add screenshots, sample invoices, and retrieval metrics here as the implementation matures.*

---

## 🧩 Use cases

- **Retail & e-commerce teams** prototyping conversational and multimodal search without vendor lock-in.
- **Global or multilingual catalogs** where queries and product copy mix languages.
- **Operations** that need **auditable** steps: why a SKU was chosen, what confidence was, and what the user confirmed.
- **Research and coursework (IITR)** demonstrating modern RAG, agents, and graph-based orchestration on realistic commerce scenarios.

**Business value (from the proposal)**  
Faster product discovery (text + image)
Reduced manual intervention via automated routing and policies
Multilingual global support without separate systems
Full audit trail for orders, returns, and billing
Enterprise-grade trust and safety via policy enforcement

---

## 🗺️ Roadmap

| Status | Phase | Focus |
|--------|-------|--------|
| Complete | **Foundation** | Streamlit shell, FalkorDB connectivity, baseline embeddings and ingestion |
| Complete | **Core agents** | Product search, billing, order lookup, translation, returns, cancellation |
| Complete | **Supervisor routing** | LLM-based and keyword-based intent classification and routing
| In Progress | **Policy refinement** | Return window tuning, cancellation rules, tax/discount logic
| Planned | **Loyalty integration** | Points system, rewards, personalized recommendations
| Planned | **Support escalation** | Human handoff workflows and ticket integration
| Planned | **Eval & metrics** | Retrieval quality checks, routing accuracy, agent performance benchmarks
| Planned | **Production hardening** | Load testing, rate limiting, analytics, monitoring

**Retry, retake & fallback (by design)**

- **Retry** — User rephrases the query, uploads a new image, or the graph loops back to retrieval nodes after confirmation.
- **Fallback** — Ask for text, ask for image, or offer barcode / manual selection.

---

## For running instructions, see [Run_Setup.md](Run_Setup.md).

## 🤝 Contributing

1. **Branching** — Use short-lived feature branches; keep `main` stable.
2. **New agents** — Create chatbot/agents/my_agent.py, decorate with @registry.register(routing_key="my_agent", keywords=[...]), and add one import line to __init__.py. The supervisor and graph are auto-updated.
3. **Changes** — Small, reviewable PRs; describe behavior and any new config or environment variables.
4. **Quality** — Run tests before opening a PR; document policy thresholds (e.g., return windows, cancellation cutoffs).
5. **Safety** — Do not weaken guardrails or billing checks without explicit review and tests.


For alignment with project scope, use this README together with the documentation in docs and maintainer guidance.

---

## 📄 License

Specify a license before public release (for example **MIT** or **Apache-2.0**) and add a `LICENSE` file at the repository root. Until then, treat the code as **all rights reserved** unless your institution states otherwise.

---

## 📑 Exporting this document to PDF

Headings, tables, and code blocks in this file are chosen so the document converts cleanly to PDF (for example with [Pandoc](https://pandoc.org/)):

```bash
pandoc README.md -o IITR-Project-README.pdf --pdf-engine=pdflatex
```

Install Pandoc and a LaTeX engine if you use PDF output. For a quick HTML or PDF from Markdown, many IDEs and Git hosts also offer export or print-to-PDF.

---

*Derived from the project proposal: Agentic Multimodal Commerce Platform — Conversational Product Discovery, Multilingual Retrieval & Automatic Billing.*
