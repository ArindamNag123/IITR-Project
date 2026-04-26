"""
RAGRetriever — future-ready Retrieval-Augmented Generation layer.

Current implementation:
    TF-IDF vectorisation + FAISS IndexFlatIP for similarity search.
    This is intentionally lightweight — it uses packages already installed
    (scikit-learn, faiss-cpu, numpy) and requires zero API keys.

Migration path to production RAG:
    1. Swap `_default_embed` for an embedding model:
           from langchain_openai import OpenAIEmbeddings
           embed_fn = OpenAIEmbeddings().embed_documents
       or a local model via sentence-transformers.
    2. Replace FAISS with a persistent store (Pinecone, Chroma, Weaviate)
       for cross-session memory.
    3. Call `retriever.retrieve(query)` inside any agent node and store
       the result in `state["rag_context"]` before generating the reply.

Usage example (inside an agent node):
    rag = RAGRetriever()
    rag.add_documents(["product A is a face wash", "product B is a vitamin"])
    ctx = rag.retrieve("face wash for dry skin")
    state["rag_context"] = ctx
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class RAGRetriever:
    """
    In-memory FAISS-backed retriever.

    Parameters
    ----------
    embed_fn : callable, optional
        Function that accepts list[str] and returns a 2-D numpy float32 array.
        Defaults to TF-IDF.  Swap for any embedding model in production.
    top_k : int
        Number of chunks to return per query.
    """

    def __init__(self, embed_fn=None, top_k: int = 3):
        self._embed_fn = embed_fn  # None → use TF-IDF fallback
        self._top_k = top_k
        self._documents: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._index = None  # faiss index — built lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[str]) -> None:
        """Index new documents.  Can be called multiple times (rebuilds index)."""
        if not docs:
            return
        self._documents.extend(docs)
        self._build_index()

    def retrieve(self, query: str) -> str:
        """
        Return top-k relevant chunks joined as a single context string.
        Returns empty string when the index is empty.
        """
        if not self._documents or self._index is None:
            return ""

        q_vec = self._embed_query(query)
        distances, indices = self._index.search(q_vec, min(self._top_k, len(self._documents)))

        chunks = [self._documents[i] for i in indices[0] if i >= 0]
        return "\n".join(chunks)

    def is_ready(self) -> bool:
        return bool(self._documents)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        import faiss  # local import — faiss-cpu already installed

        embeddings = self._embed_documents(self._documents)
        dim = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def _embed_documents(self, docs: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return np.array(self._embed_fn(docs), dtype="float32")

        # TF-IDF fallback
        self._vectorizer = TfidfVectorizer(stop_words="english")
        mat = self._vectorizer.fit_transform(docs).toarray().astype("float32")
        return self._normalise(mat)

    def _embed_query(self, query: str) -> np.ndarray:
        if self._embed_fn is not None:
            vec = np.array(self._embed_fn([query]), dtype="float32")
            return self._normalise(vec)

        if self._vectorizer is None:
            raise RuntimeError("Call add_documents() before retrieve().")
        vec = self._vectorizer.transform([query]).toarray().astype("float32")
        return self._normalise(vec)

    @staticmethod
    def _normalise(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
