# IITR-Project

# Agentic Multimodal Commerce Platform

**Conversational product discovery, multilingual retrieval, and transparent billing—driven by an agentic, explainable AI stack.**

---

## 🚀 What is this?

The **Agentic Multimodal Commerce Platform** is a research and engineering initiative that lets shoppers find products and complete checkout-style flows using **natural language** (including multiple languages) and **product images**, not just keyword search.

Today’s commerce assistants often treat text and images in isolation, treat OCR as plain text without product semantics, and automate billing without enough transparency. This project builds a **single conversational system** that reasons across modalities, adapts its workflow dynamically, and **explains** what it matched and why—so it can be taken seriously in real retail and e-commerce settings.

---

## 🎯 Why it exists

| Gap | How this project addresses it |
|-----|--------------------------------|
| Discovery is text-only or image-only, rarely both in one flow | Unified chat + image upload with orchestrated multimodal retrieval |
| OCR yields text, not product semantics (brand, size, variant) | **ANOCR** (advanced named OCR) for structured entities and confidence |
| Weak multilingual support | **Translation** to a canonical language plus multilingual embeddings |
| Billing feels like a black box | **Billing & explanation** agents with RAG-backed rationale |
| Brittle, linear AI pipelines | **LangGraph**: explicit nodes, edges, conditions, retries, and fallbacks |

The goal is an architecture that is **agentic** (dynamic routing), **explainable**, **multilingual**, and **safe-by-design**—aligned with how production teams expect to deploy AI in commerce.

---

## 🧠 Key ideas / concepts

- **Agentic orchestration** — A conversation manager coordinates specialized agents; workflows are not a single fixed chain.
- **LangGraph control flow** — Nodes (agents/steps), edges (transitions), and **conditions** (which path to take) are first-class; supports loops for retry and user confirmation.
- **Multimodal RAG** — **FalkorDB** stores text, image, OCR, and multilingual vectors for retrieval-augmented grounding.
- **Confidence-aware automation** — High / medium / low confidence gates drive billing, user confirmation, or fallback—not blind automation.
- **Guardrails** — Input validation, policy constraints, and thresholds before irreversible actions (e.g. billing).
- **Open-source stack** — Python, Streamlit, LangGraph, open LLMs and embedding models, enabling reproducibility and cost control.

**What makes the platform “agentic” (from the proposal)**  
Dynamic workflow selection, conditional execution and loops, confidence-aware automation, built-in safety guardrails, and explainable decisions.

---

## 🛠️ What’s inside

> *Repository layout reflects the implementation plan; adjust names as the codebase grows.*

| Area | Role |
|------|------|
| **Streamlit app** | Conversational UI: chat, image upload, streaming responses, confirmation controls |
| **LangGraph runtime** | Python application wiring graph definition, state, and execution |
| **Agent layer** | Pluggable agents: conversation manager, guardrails, intent, retrieval, vision, ANOCR, translation, billing, explanation |
| **FalkorDB** | Vector store for multimodal RAG (text, image, OCR, multilingual embeddings) |
| **Config / prompts** | Model endpoints, thresholds, safety rules, and prompt templates |
| **Tests & eval** | Unit/integration tests and optional retrieval-quality checks |

**Suggested folder sketch**

```text
iitr-project/
├── README.md                 # This file
├── requirements.txt          # or pyproject.toml
├── app/                      # Streamlit entrypoints and UI components
├── graph/                    # LangGraph graph definition(s), state schema
├── agents/                   # Agent implementations (thin wrappers over tools)
├── retrieval/                # FalkorDB client, embedding pipelines, RAG helpers
├── models/                   # Model loading / routing (LLM, embedders, CLIP/SigLIP)
├── config/                   # YAML/TOML for thresholds, policies, locales
└── tests/                    # Automated tests
```

**Core technology choices (from the proposal)**

| Layer | Technologies |
|-------|----------------|
| Frontend | **Streamlit** — chat UI, image upload, streaming, confirmation buttons |
| Backend & orchestration | **Python**, **LangGraph** |
| LLMs (open) | LLaMA, Mistral, Mixtral, Phi (as appropriate) |
| Text embeddings | Sentence-Transformers, BGE, Instructor |
| Image embeddings | CLIP / SigLIP |
| Vector & RAG | **FalkorDB** — text, image, OCR, multilingual vectors |

### Agent responsibilities (at a glance)

| Agent | Responsibility |
|-------|------------------|
| **Conversation Manager** | Session state, intent (text vs image path), LangGraph route selection |
| **Guardrails** | Validate inputs; enforce product/policy rules; confidence thresholds for automation |
| **Intent classifier** | Distinguish text query vs image scan and downstream branches |
| **Text retrieval** | Multilingual query embedding; semantic search in FalkorDB; ranked candidates |
| **Vision search** | Image embedding; similarity search in FalkorDB |
| **ANOCR** | Extract brand, product name, size, attributes; structured entities + confidence |
| **Translation** | Normalize text to a canonical language for search and responses |
| **Billing & validation** | Price, tax, discount; invoice generation |
| **Explanation** | Use RAG context to justify decisions to the user |

**LangGraph core nodes (from the proposal)**  
`ChatInput`, `ConversationManager`, `Guardrails`, `IntentClassifier`, `TextRetrieval`, `VisionSearch`, `ANOCR`, `Translation`, `ConfidenceEvaluation`, `UserConfirmation`, `Billing`, `ExplainResponse`, `Fallback`.

**Conceptual path**

```text
ChatInput → ConversationManager → Guardrails → IntentClassifier
  → (TextRetrieval | VisionSearch / ANOCR / Translation)
  → ConfidenceEvaluation → (Billing | UserConfirmation | Fallback) → ExplainResponse
```

---

## 📊 Examples / outputs

**LangGraph Flow A — Text query**

```text
ChatInput
  → ConversationManager
  → Guardrails
  → IntentClassifier (TEXT_QUERY)
  → Translation
  → TextRetrieval (FalkorDB)
  → ConfidenceEvaluation
      ├─ High   → Billing
      ├─ Medium → UserConfirmation → Retry
      └─ Low    → Fallback
```

**LangGraph Flow B — Image upload**

```text
ChatInput
  → ConversationManager
  → Guardrails
  → IntentClassifier (IMAGE_SCAN)
  → VisionSearch (FalkorDB)
  → ConfidenceEvaluation
      ├─ High → Billing
      └─ Low  → ANOCR
                  → Translation
                  → TextRetrieval (FalkorDB)
                  → ConfidenceEvaluation (Merged)
                      ├─ High   → Billing
                      ├─ Medium → UserConfirmation → Retry
                      └─ Low    → Fallback
```

**Narrative summaries**

- **Flow A** — User asks in any supported language → translation → text retrieval in FalkorDB → confidence branch → billing, user confirmation, or fallback → explained answer.
- **Flow B** — Image → vision search → if confidence is low, **ANOCR** → translation → text retrieval → merged confidence → billing / confirmation / fallback → explanation.

**End-to-end story (from the proposal)**  
User uploads an image and asks in Hindi to generate a bill for that product. The system may: use vision first; fall back to ANOCR when vision confidence is low; normalize entities; confirm the product via FalkorDB; produce billing; return a short explanation in the user’s language.

**Illustrative architecture**

```text
┌────────────────────┐
│ Streamlit UI       │
│ (Chat + Image)     │
└─────────┬──────────┘
          │ user input
          ▼
┌────────────────────┐
│ LangGraph Runtime  │
│ (Python)           │
└─────────┬──────────┘
          │ state + control
          ▼
┌──────────────────────────────────────────────┐
│ Agent orchestration layer                    │
│ Conversation Manager · Guardrails · Intent   │
│ Text / Vision · ANOCR · Translation          │
│ Billing · Explain                            │
└─────────┬────────────────────────────────────┘
          │ vector queries
          ▼
┌──────────────────────────────────────────────┐
│ FalkorDB                                     │
│ Text · Image · OCR & multilingual vectors    │
└──────────────────────────────────────────────┘
```

*Add screenshots, sample invoices, and retrieval metrics here as the implementation matures.*

---

## 🧩 Use cases

- **Retail & e-commerce teams** prototyping conversational and multimodal search without vendor lock-in.
- **Global or multilingual catalogs** where queries and product copy mix languages.
- **Operations** that need **auditable** steps: why a SKU was chosen, what confidence was, and what the user confirmed.
- **Research and coursework (IITR)** demonstrating modern RAG, agents, and graph-based orchestration on realistic commerce scenarios.

**Business value (from the proposal)**  
Faster checkout, reduced manual intervention, multilingual global support, and enterprise-grade trust and safety.

---

## 🗺️ Roadmap

| Status | Phase | Focus |
|--------|-------|--------|
| Planned | **Foundation** | Streamlit shell, FalkorDB connectivity, baseline embeddings and ingestion |
| Planned | **Graph & agents** | LangGraph graph, guardrails, intent, text + vision retrieval, ANOCR + translation paths |
| Planned | **Product experience** | Confidence gates, user confirmation, retry/fallback flows, streaming UX |
| Planned | **Trust & quality** | Explanation quality, eval sets, guardrail tuning, documentation and demos |

**Retry, retake & fallback (by design)**

- **Retry** — User rephrases the query, uploads a new image, or the graph loops back to retrieval nodes after confirmation.
- **Fallback** — Ask for text, ask for image, or offer barcode / manual selection.

---

## 🤝 Contributing

1. **Branching** — Use short-lived feature branches; keep `main` stable.
2. **Changes** — Small, reviewable PRs; describe behavior and any new config or environment variables.
3. **Quality** — Run tests and linters before opening a PR; document non-obvious thresholds (for example confidence cutoffs).
4. **Safety** — Do not weaken guardrails or billing checks without explicit review and tests.

For alignment with project scope, use this README together with the original proposal and maintainer guidance.

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
