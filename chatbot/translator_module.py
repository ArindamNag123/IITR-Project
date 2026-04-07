"""
Language Translator Module - Agentic LLM-based Translation

Goal
----
Allow users to chat in any language. The app:
1) Detects the user's language (fast heuristic + optional LLM)
2) Translates user text → English before sending to the supervisor/agents
3) Translates the final English response → user's language for display

Implementation notes
--------------------
- Uses LangChain's ChatOpenAI when OPENAI_API_KEY is configured (same stack as router/agents).
- If no API key is configured, translation becomes a no-op (passes text through).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")  # Hindi
_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")       # Thai
_ENGLISH_LIKE_RE = re.compile(r"^[A-Za-z0-9\s.,!?\'\"-]+$")


@dataclass(frozen=True)
class TranslationResult:
    original: str
    detected_language: str
    english: str


class ChatbotTranslator:
    """
    Translator wrapper intended for the Streamlit app layer.

    - If OpenAI is configured, uses LLM translation for quality + broad language support.
    - Otherwise, falls back to returning inputs unchanged.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._llm = None

        if self._api_key:
            # Lazy import so the app can still run without langchain_openai installed
            # (though this repo already uses it in other modules).
            try:
                from langchain_openai import ChatOpenAI
            except Exception:
                # If the dependency isn't installed, fall back to no-op translation
                # instead of crashing the Streamlit app.
                self._llm = None
            else:
                self._llm = ChatOpenAI(model=self.model, temperature=0.2)

    def _heuristic_detect(self, text: str) -> str:
        if _DEVANAGARI_RE.search(text):
            return "hi"
        if _THAI_RE.search(text):
            return "th"
        # Romanized Hindi / Hinglish quick heuristics (helps routing even without an LLM)
        lower = (text or "").lower()
        if any(
            w in lower.split()
            for w in (
                "mujhe",
                "mujh",
                "chahiye",
                "chaiye",
                "kripya",
                "kya",
                "kaise",
                "mera",
                "meri",
                "hamara",
                "aap",
                "krdo",
                "kardo",
            )
        ):
            return "hi"
        if _ENGLISH_LIKE_RE.match(text or ""):
            return "en"
        # Could be any Latin-script language; let LLM handle it if available.
        return "unknown"

    def translate_to_english(self, text: str) -> TranslationResult:
        detected = self._heuristic_detect(text)

        # No LLM configured → no-op translation
        if not self._llm:
            return TranslationResult(original=text, detected_language=detected, english=text)

        # If already English-like, avoid extra calls
        if detected == "en":
            return TranslationResult(original=text, detected_language="en", english=text)

        # LLM translation (also implicitly handles unknown languages)
        english = self._llm.invoke(
            [
                (
                    "system",
                    "You are a professional translator. Detect the input language and translate it to English. "
                    "Output only the translated text, nothing else.",
                ),
                ("user", text),
            ]
        ).content.strip()

        # If we don't know, still store unknown; supervisor can use its own logic.
        return TranslationResult(original=text, detected_language=detected, english=english or text)

    def translate_from_english(self, english_text: str, target_language: str, reference_text: str) -> str:
        # If no translation needed / no LLM configured → no-op
        if not self._llm or target_language in ("en", "", None):
            return english_text

        translated = self._llm.invoke(
            [
                (
                    "system",
                    "You are a professional translator. Translate the English text back to the SAME language as the reference text. "
                    "Output only the translated text, nothing else.",
                ),
                (
                    "user",
                    f"Reference (language to match): {reference_text}\n\nEnglish text to translate:\n{english_text}",
                ),
            ]
        ).content.strip()

        return translated or english_text

