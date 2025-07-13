"""Creación y mantenimiento del Chroma DB."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .config import EMBEDDING_MODEL, PERSIST_DIR

__all__ = ["build_vector_store"]


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Indexa *chunks* en una colección persistente y la devuelve."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma(persist_directory=str(PERSIST_DIR)).delete_collection()
    vectordb = Chroma(persist_directory=str(PERSIST_DIR),
                      embedding_function=embeddings)
    vectordb.add_documents(chunks)
    return vectordb
