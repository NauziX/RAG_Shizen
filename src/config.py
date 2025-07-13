"""Gestión de rutas y variables de entorno."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
PRICE_JSON = DOCS_DIR / "precios_shizen.json"
BASE_URL = os.getenv("SHIZEN_BASE_URL", "https://www.organicshizen.com/")

PERSIST_DIR = ROOT_DIR / "data" / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL: str = os.getenv(
    "RAG_EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL: str = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
