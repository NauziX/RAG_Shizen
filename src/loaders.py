"""Carga y limpieza de datos externos (web, PDFs, JSON)."""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import List, Set

from bs4 import MarkupResemblesLocatorWarning
from langchain_community.document_loaders import RecursiveUrlLoader, UnstructuredPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document

from .config import BASE_URL, DOCS_DIR, PRICE_JSON
from .constants import FAQ_HEADER_RE

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

__all__ = [
    "load_website",
    "load_pdf",
    "load_all_pdfs",
    "load_json",
    "load_all",
]


def load_website(url: str = BASE_URL, max_depth: int = 2) -> List[Document]:
    """Rastrea *url* (profundidad limitada) y devuelve *Documents* limpios."""
    selectors_to_remove = [
        "header",
        "footer",
        "[role='navigation']",
        "[role='banner']",
        "[aria-label='cookies']",
        ".cookie",
        ".cookies",
        ".banner",
    ]

    loader = RecursiveUrlLoader(url=url, max_depth=max_depth, timeout=10)
    try:
        raw_docs = loader.load()
    except Exception as exc:  # pragma: no cover
        logger.warning("Error al rastrear %s: %s", url, exc)
        return []

    try:
        cleaner = BeautifulSoupTransformer(
            remove_selectors=selectors_to_remove)
    except TypeError:  # langchain < 0.2.x compat
        cleaner = BeautifulSoupTransformer()
    cleaned_docs = cleaner.transform_documents(raw_docs)

    docs: List[Document] = []
    seen_urls: Set[str] = set()
    for d in cleaned_docs:
        src = d.metadata.get("url", url)
        if src in seen_urls:
            continue
        seen_urls.add(src)

        text = d.page_content.strip()
        if "<" in text or len(text) < 200:
            continue

        d.metadata["source"] = src
        docs.append(d)
    return docs


def load_pdf(path: Path) -> List[Document]:
    """Parsea un PDF Shizen (FAQ) agrupando por secciones."""
    elements = UnstructuredPDFLoader(str(path), mode="elements").load()
    sections: List[List[str]] = []
    buff: List[str] = []
    for el in elements:
        line = el.page_content.strip()
        if FAQ_HEADER_RE.match(line):
            if buff:
                sections.append(buff)
                buff = []
        buff.append(line)
    if buff:
        sections.append(buff)

    return [
        Document(
            page_content="\n".join(block),
            metadata={
                "source": str(path.relative_to(DOCS_DIR.parent)),
                "section": block[0].split(":")[0].title(),
            },
        )
        for block in sections
    ]


def load_all_pdfs(folder: Path = DOCS_DIR) -> List[Document]:
    docs: List[Document] = []
    for pdf in folder.glob("*.pdf"):
        docs.extend(load_pdf(pdf))
    return docs


def load_json(path: Path = PRICE_JSON) -> List[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=f"{row['service']} {row['price']}",
            metadata={
                "source": str(path.relative_to(DOCS_DIR.parent)),
                "category": row["category"],
                "service": row["service"],
            },
        )
        for row in data
    ]


def load_all() -> List[Document]:
    """Carga TODO lo disponible (web, PDFs y lista de precios)."""
    return load_website() + load_all_pdfs() + load_json()
