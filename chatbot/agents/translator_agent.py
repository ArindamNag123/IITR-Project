"""
Translator Agent — Hindi ↔ English translation.

Production wiring (two options):
  Option A — LLM translation (best quality):
      Set OPENAI_API_KEY and the agent uses ChatOpenAI to translate fluently.
  Option B — External API:
      Swap _translate_with_llm for a call to Google Cloud Translation API
      or Azure Cognitive Services Translator.

Fallback behaviour (no API key):
  Returns the detected language and provides a polite message explaining
  that a translation service needs to be configured.

Retail glossary seed (works without any API):
  A small built-in dictionary covers common retail terms so basic
  Hindi queries can be partially understood.
"""

import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from chatbot.registry import registry
from chatbot.state import AgentState

load_dotenv()

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Retail-domain mini-dictionary (Hindi → English)
_RETAIL_GLOSSARY: dict[str, str] = {
    "साबुन": "soap",
    "शैंपू": "shampoo",
    "क्रीम": "cream",
    "लोशन": "lotion",
    "दवा": "medicine",
    "विटामिन": "vitamins",
    "कीमत": "price",
    "ऑर्डर": "order",
    "बिल": "bill",
    "खोजें": "search",
    "खरीदें": "buy",
    "महिला": "women",
    "पुरुष": "men",
    "बच्चे": "baby",
}


def _glossary_translate(text: str) -> str:
    """Shallow glossary replacement — useful for demos without an LLM."""
    for hindi, english in _RETAIL_GLOSSARY.items():
        text = text.replace(hindi, english)
    return text


def _translate_with_llm(text: str, source_lang: str, target_lang: str) -> str:
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.2)
    direction = f"{source_lang} to {target_lang}"
    messages = [
        SystemMessage(
            content=(
                f"You are a precise translator for an e-commerce retail assistant. "
                f"Translate the following text from {direction}. "
                f"Output only the translated text, nothing else."
            )
        ),
        HumanMessage(content=text),
    ]
    result = llm.invoke(messages)
    return result.content.strip()


@registry.register(
    routing_key="translator",
    keywords=[
        "translate", "translation", "hindi", "english", "anuvad",
        "\u0905\u0928\u0941\u0935\u093e\u0926",  # anuvad (Devanagari)
        "\u0939\u093f\u0902\u0926\u0940",          # hindi (Devanagari)
    ],
    description="Translates between English and Hindi; handles Hindi-script messages.",
)
def translator_agent_node(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    text = last_human.content if last_human else ""
    detected_lang = state.get("detected_language", "unknown")

    source_lang = "Hindi" if detected_lang == "hi" or _DEVANAGARI_RE.search(text) else "English"
    target_lang = "English" if source_lang == "Hindi" else "Hindi"

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if api_key:
        try:
            translated = _translate_with_llm(text, source_lang, target_lang)
            reply = (
                f"**Translation ({source_lang} → {target_lang}):**\n\n{translated}\n\n"
                f"---\n*Original:* {text}"
            )
        except Exception as exc:
            # Graceful degradation to glossary
            translated = _glossary_translate(text)
            reply = (
                f"Translation service error ({exc}).\n"
                f"Partial glossary translation: {translated}"
            )
    else:
        # No LLM — use glossary + helpful message
        translated = _glossary_translate(text)
        reply = (
            f"*Translation service (LLM) is not configured.*\n\n"
            f"Detected language: **{source_lang}**\n"
            f"Partial glossary translation: `{translated}`\n\n"
            f"Set `OPENAI_API_KEY` in your `.env` file to enable full "
            f"{source_lang} → {target_lang} translation."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "metadata": {
            **state.get("metadata", {}),
            "translation": {"source": source_lang, "target": target_lang, "text": text},
        },
    }
