"""
src/ingest.py – Pipeline RAG (v4)
=================================
Ahora usa **precios_shizen.json** (docs/) en lugar del PDF para los precios.

• Carga web + JSON estructurado
• JSONLoader crea un Document por fila "service price"
• Web chunk 1 000 / 200
• Chroma lotes 100
• Retriever híbrido (k=25) + fallback regex
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    RecursiveUrlLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader

# ── Paths & env ───────────────────────────────────────────────────────────────
load_dotenv()
ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
PRICE_JSON = DOCS_DIR / "precios_shizen.json"  # <── nuevo
BASE_URL = "https://www.organicshizen.com/"
PERSIST_DIR = ROOT_DIR / "data" / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ── Config -------------------------------------------------------------------
CHUNK_SIZE, CHUNK_OVERLAP = 1_000, 200
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

# ── Loaders ------------------------------------------------------------------


def load_website(url: str, max_depth: int = 2) -> List[Document]:
    docs = RecursiveUrlLoader(url=url, max_depth=max_depth).load()
    for d in docs:
        d.metadata["source"] = d.metadata.get("url", url)
    return docs


def load_json(path: Path) -> List[Document]:
    """Cada fila JSON → Document con "service price" como contenido."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Document] = []
    for row in data:
        docs.append(
            Document(
                page_content=f"{row['service']} {row['price']}",
                metadata={"source": str(path.relative_to(
                    ROOT_DIR)), "category": row["category"]},
            )
        )
    return docs

# ── Chunking -----------------------------------------------------------------


def load_pdf(path: Path) -> List[Document]:
    """Agrupa cabecera + líneas de precio; o devuelve cada página para FAQs."""
    els = UnstructuredPDFLoader(str(path), mode="elements").load()
    return [
        Document(
            page_content=el.page_content.strip(),
            metadata={"source": str(path.relative_to(ROOT_DIR))}
        )
        for el in els
    ]


def load_all_pdfs(folder: Path) -> List[Document]:
    docs = []
    for pdf in folder.glob("*.pdf"):
        docs += load_pdf(pdf)
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    price_docs = [d for d in docs if "€" in d.page_content]
    web_docs = [d for d in docs if d not in price_docs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
        add_start_index=True,
    )
    web_chunks = splitter.split_documents(web_docs)

    return price_docs + web_chunks

# ── Vector-store -------------------------------------------------------------


def build_vector_store(chunks: List[Document]):
    vectordb = Chroma(persist_directory=str(PERSIST_DIR),
                      embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL))
    for i in range(0, len(chunks), 100):
        vectordb.add_documents(chunks[i: i + 100])
    return vectordb

# ── Ingest -------------------------------------------------------------------


def ingest():
    docs = (
        load_website(BASE_URL) +  # web
        load_all_pdfs(DOCS_DIR) +
        load_json(PRICE_JSON)          # precios estructurados
    )
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print(f"Ingestados {len(chunks)} chunks → {PERSIST_DIR}")


# ── Retriever híbrido --------------------------------------------------------


def build_hybrid_retriever():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=str(PERSIST_DIR),
                      embedding_function=embeddings)

    vec_ret = vectordb.as_retriever(search_kwargs={"k": 25})
    raw = vectordb.get(include=["documents", "metadatas"])
    bm_docs = [Document(page_content=t, metadata=m)
               for t, m in zip(raw["documents"], raw["metadatas"])]
    bm25 = BM25Retriever.from_documents(bm_docs)
    bm25.k = 25

    return EnsembleRetriever(retrievers=[bm25, vec_ret], weights=[0.4, 0.6])


# ── Prompt -------------------------------------------------------------------
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres un asistente conciso. Si el contexto incluye un precio, "
        "responde solo con la cifra exacta y la divisa; de lo contrario indica que no lo encuentras."
        "\n{context}\nPregunta: {question}\nRespuesta:"
    ),
)

# ── QA ----------------------------------------------------------------------


def ask(query: str):
    retriever = build_hybrid_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=LLM_MODEL, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )
    res = qa_chain.invoke({"query": query})
    answer = res["result"].strip()

    # Fallback regex si no hay €
    if "€" not in answer:
        key = unicodedata.normalize("NFKD", query).encode(
            "ascii", "ignore").decode().lower()
        key = re.sub(r"cuanto|cuánto|cuesta|\?", "", key).strip()
        for doc in res["source_documents"]:
            txt = unicodedata.normalize("NFKD", doc.page_content).encode(
                "ascii", "ignore").decode().lower()
            if key in txt and (m := re.search(r"(\d+[\.,]?\d*)\s*€", txt)):
                answer = f"{m.group(1)} €"
                break

    print("\n💬  Respuesta:\n", answer or "No lo encuentro.", "\n")
    seen = set()  # ← añadir estas dos líneas
    unique_sources = [d for d in res["source_documents"]
                      if not (d.metadata.get("source") in seen or seen.add(d.metadata.get("source")))]

    print("📚  Fuentes:")
    for d in unique_sources:  # ← usar unique_sources
        print(" •", d.metadata.get("source"))


# ── CLI ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", help="Pregunta (si se omite se re-indexa)")
    arg = p.parse_args()
    if arg.query:
        ask(arg.query)
    else:
        ingest()
