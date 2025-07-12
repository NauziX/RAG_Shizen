from typing import Set
import argparse
import json
import os
import re
import unicodedata
import warnings
from pathlib import Path
from typing import List, Optional, Set

from bs4 import MarkupResemblesLocatorWarning
from dotenv import load_dotenv, find_dotenv
from rapidfuzz import fuzz

from langchain_community.document_loaders import RecursiveUrlLoader, UnstructuredPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# ─────────────────────────── Environment & Paths ────────────────────────────
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
load_dotenv(find_dotenv())

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
PRICE_JSON = DOCS_DIR / "precios_shizen.json"
BASE_URL = os.getenv("SHIZEN_BASE_URL", "https://www.organicshizen.com/")

PERSIST_DIR = ROOT_DIR / "data" / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────── Config ────────────────────────────────────
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")

# ─────────────────────────────── Regex ─────────────────────────────────────
FAQ_HEADER_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ0-9 &'/-]{4,}:?$")
CURRENCY_RE = re.compile(r"(€|eur(?:os?)?)", re.I)
MONEY_RE = re.compile(
    r"(€\s*\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*(?:€|eur(?:os?)?))", re.I)
PRICE_QUERY_RE = re.compile(
    r"(cu[aá]nt[oa]?|precio|vale|cuesta|coste|importe|tarifa)", re.I)

# ─────────────────────────────── Loaders ────────────────────────────────────


def load_website(url: str, max_depth: int = 2) -> List[Document]:
    """Recursively crawl *url* (depth-limited) and return cleaned Documents."""
    selectors_to_remove = [
        "header", "footer", "[role='navigation']", "[role='banner']",
        "[aria-label='cookies']", ".cookie", ".cookies", ".banner",
    ]

    loader = RecursiveUrlLoader(url=url, max_depth=max_depth, timeout=10)
    try:
        raw_docs = loader.load()
    except Exception as exc:
        print(f"⚠️  Error al rastrear {url}: {exc}")
        return []

    try:
        cleaner = BeautifulSoupTransformer(
            remove_selectors=selectors_to_remove)
    except TypeError:
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
                "source": str(path.relative_to(ROOT_DIR)),
                "section": block[0].split(":")[0].title(),
            },
        )
        for block in sections
    ]


def load_all_pdfs(folder: Path) -> List[Document]:
    docs: List[Document] = []
    for pdf in folder.glob("*.pdf"):
        docs.extend(load_pdf(pdf))
    return docs


def load_json(path: Path) -> List[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=f"{row['service']} {row['price']}",
            metadata={
                "source": str(path.relative_to(ROOT_DIR)),
                "category": row["category"],
                "service": row["service"],
            },
        )
        for row in data
    ]

# ─────────────────────────────── Chunking ───────────────────────────────────


def chunk_documents(docs: List[Document]) -> List[Document]:
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

# ───────────────────────────── Vector Store ─────────────────────────────────


def build_vector_store(chunks: List[Document]):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma(persist_directory=str(PERSIST_DIR)).delete_collection()
    vectordb = Chroma(persist_directory=str(PERSIST_DIR),
                      embedding_function=embeddings)
    vectordb.add_documents(chunks)
    return vectordb

# ───────────────────────────── Retrievers ───────────────────────────────────


def build_hybrid_retriever():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=str(PERSIST_DIR),
                      embedding_function=embeddings)

    vec_ret = vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": 15})

    raw = vectordb.get(include=["documents", "metadatas"])
    bm_docs = [Document(page_content=t, metadata=m)
               for t, m in zip(raw["documents"], raw["metadatas"])]
    bm25 = BM25Retriever.from_documents(bm_docs)
    bm25.k = 15

    return EnsembleRetriever(retrievers=[bm25, vec_ret], weights=[0.3, 0.7])

# ─────────────────────────── Helper: Price Pick ─────────────────────────────


def pick_best_price(query: str, docs: List[Document], threshold: int = 60) -> Optional[str]:
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

    return best_price if best_score >= threshold else None


# ─────────────────────────────── Prompt ─────────────────────────────────────
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres un asistente experto en servicios de peluquería y estética de Shizen Organic.\n"
        "Responde en español de forma concisa usando solo la información del contexto.\n"
        "Si la pregunta es sobre un precio y el contexto incluye importe, responde exclusivamente con la cifra y la divisa.\n"
        "Si no encuentras la respuesta, devuelve 'No lo encuentro.'.\n"
        "\n"
        "{context}\n"
        "Pregunta: {question}\n"
        "Respuesta:"
    ),
)


# ─────────────────────────────── QA ─────────────────────────────────────────


def ask(query: str) -> None:
    """Run a query and print answer plus distinct sources."""
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
    answer = (res.get("result") or "").strip()

    # Fallback específico para preguntas de precio
    if price_query and not CURRENCY_RE.search(answer):
        best = pick_best_price(query, res.get("source_documents", []))
        if best:
            answer = best
    if not answer:
        answer = "No lo encuentro."

    # ── Salida formateada ──────────────────────────────────────────────
    print(f"\n💬  Respuesta:\n{answer}\n")

    print("📚  Fuentes:")
    seen: Set[str] = set()
    for d in res.get("source_documents", []):
        src = d.metadata.get("source")
        if src and src not in seen:
            print(" •", src)
            seen.add(src)


# ─────────────────────────────── CLI ────────────────────────────────────────


def ingest() -> None:
    docs = load_website(BASE_URL) + \
        load_all_pdfs(DOCS_DIR) + load_json(PRICE_JSON)
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print(f"Ingestados {len(chunks)} chunks → {PERSIST_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Pregunta (si se omite se re-indexa)")
    args = parser.parse_args()
    if args.query:
        ask(args.query)
    else:
        ingest()
