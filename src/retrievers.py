"""Composición híbrida BM25 + embeddings."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from .config import EMBEDDING_MODEL, PERSIST_DIR

__all__ = ["build_hybrid_retriever"]


def _load_vectordb() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


def build_hybrid_retriever(k: int = 15) -> EnsembleRetriever:
    """Combina similitud vectorial y BM25 con pesos 0.7 / 0.3."""
    vectordb = _load_vectordb()

    vec_ret = vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": k})

    raw = vectordb.get(include=["documents", "metadatas"])
    bm_docs = [Document(page_content=t, metadata=m)
               for t, m in zip(raw["documents"], raw["metadatas"])]
    bm25 = BM25Retriever.from_documents(bm_docs)
    bm25.k = k

    return EnsembleRetriever(retrievers=[bm25, vec_ret], weights=[0.3, 0.7])
