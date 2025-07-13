"""Cadena QA + heurística de precios."""
from __future__ import annotations

import logging
from typing import Optional

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI

from .constants import CURRENCY_RE, PRICE_QUERY_RE
from .processors import pick_best_price
from .retrievers import build_hybrid_retriever
from .prompts import CUSTOM_PROMPT
from .config import LLM_MODEL

logger = logging.getLogger(__name__)

__all__ = ["ask"]


def ask(query: str) -> str:
    """Resuelve *query* y devuelve la respuesta en texto plano."""
    price_query = bool(PRICE_QUERY_RE.search(query))

    retriever = build_hybrid_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=LLM_MODEL, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )

    res = qa_chain.invoke({"query": query, "question": query})
    answer: str = (res.get("result") or "").strip()

    # Fallback específico para preguntas de precio
    if price_query and not CURRENCY_RE.search(answer):
        best = pick_best_price(query, res.get("source_documents", []))
        if best:
            logger.debug("Fallback de precio → %s", best)
            answer = best

    return answer or "No lo encuentro."
