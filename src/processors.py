"""Funciones de pre‑procesado: split + heurísticas de precios."""
from __future__ import annotations

import logging
from typing import List, Optional

from rapidfuzz import fuzz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .constants import CURRENCY_RE, MONEY_RE, PRICE_QUERY_RE

logger = logging.getLogger(__name__)

__all__ = [
    "chunk_documents",
    "pick_best_price",
]


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Separa documentos largos manteniendo ítems de precio sin tocar."""
    price_docs = [d for d in docs if CURRENCY_RE.search(d.page_content)]
    other_docs = [d for d in docs if d not in price_docs]

    avg_len = sum(len(d.page_content)
                  for d in other_docs) / max(len(other_docs), 1)
    chunk_size = 800 if avg_len > 1800 else 400

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.15),
        separators=["\n\n", "\n", " "],
        add_start_index=True,
    )

    return price_docs + splitter.split_documents(other_docs)


def pick_best_price(query: str, docs: List[Document], threshold: int = 60) -> Optional[str]:
    """Heurística simple basada en *fuzz* para elegir un precio concreto."""
    best_score = 0
    best_price: Optional[str] = None
    query_norm = query.lower()

    for d in docs:
        m = MONEY_RE.search(d.page_content)
        if not m:
            continue
        price_val = m.group(0)
        label = (d.metadata.get("service")
                 or d.metadata.get("section") or "").lower()
        score = fuzz.partial_ratio(query_norm, label)
        if score > best_score:
            best_score = score
            best_price = price_val

    if best_score >= threshold:
        logger.debug("Mejor precio elegido (score %s): %s",
                     best_score, best_price)
    return best_price if best_score >= threshold else None
