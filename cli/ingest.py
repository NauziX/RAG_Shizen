"""Indexa todo el contenido."""
from __future__ import annotations

import logging

from src.loaders import load_all
from src.processors import chunk_documents
from src.vector_store import build_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")


def main() -> None:
    docs = load_all()
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    logging.info("Ingestados %s chunks", len(chunks))


if __name__ == "__main__":
    main()
