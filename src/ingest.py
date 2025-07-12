import argparse
import json
import re
import unicodedata
import warnings
from pathlib import Path
from typing import List

from bs4 import MarkupResemblesLocatorWarning
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    RecursiveUrlLoader,
    JSONLoader,
    UnstructuredPDFLoader,
)
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# ── Suppress BS4 noisy warnings ─────────────────────────────────────────────
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ── Paths & env ───────────────────────────────────────────────────────────────
load_dotenv()
ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
PRICE_JSON = DOCS_DIR / "precios_shizen.json"
BASE_URL = "https://www.organicshizen.com/"

PERSIST_DIR = ROOT_DIR / "data" / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ── Config -------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

FAQ_HEADER_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ ]{4,}:?$")
CURRENCY_RE = re.compile(r"(€|euros?)", re.I)
MONEY_RE = re.compile(r"(\d+[.,]?\d*)\s*(€|euros?)", re.I)
PRICE_QUERY_RE = re.compile(r"(cuanto|cuánto|precio|vale|cuesta)", re.I)

# ── Loaders ------------------------------------------------------------------


def load_website(url: str, max_depth: int = 2) -> List[Document]:
    """Recursively load a website, clean HTML and add source metadata."""
    docs = RecursiveUrlLoader(url=url, max_depth=max_depth).load()
    cleaner = BeautifulSoupTransformer()
    docs = cleaner.transform_documents(docs)
    # remove non‑HTML docs accidentally captured
    docs = [d for d in docs if "<" in d.page_content]
    for d in docs:
        d.metadata["source"] = d.metadata.get("url", url)
    return docs


def load_pdf(path: Path) -> List[Document]:
    """Group each FAQ section into a single Document."""
    elements = UnstructuredPDFLoader(str(path), mode="elements").load()
    sections, buff = [], []
    for el in elements:
        line = el.page_content.strip()
        if FAQ_HEADER_RE.match(line):
            if buff:
                sections.append(buff)
                buff = []
        buff.append(line)
    if buff:
        sections.append(buff)

    docs = [
        Document(
            page_content="\n".join(block),
            metadata={
                "source": str(path.relative_to(ROOT_DIR)),
                "section": block[0].split(":")[0].title(),
            },
        )
        for block in sections
    ]
    return docs


def load_all_pdfs(folder: Path) -> List[Document]:
    docs: List[Document] = []
    for pdf in folder.glob("*.pdf"):
        docs.extend(load_pdf(pdf))
    return docs


def load_json(path: Path) -> List[Document]:
    """Each JSON row -> Document with service and price."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Document] = []
    for row in data:
        docs.append(
            Document(
                page_content=f"{row['service']} {row['price']}",
                metadata={
                    "source": str(path.relative_to(ROOT_DIR)),
                    "category": row["category"],
                },
            )
        )
    return docs


# ── Chunking -----------------------------------------------------------------

def chunk_documents(docs: List[Document]) -> List[Document]:
    price_docs = [d for d in docs if CURRENCY_RE.search(d.page_content)]
    web_docs = [d for d in docs if d not in price_docs]

    avg_len = (
        sum(len(d.page_content) for d in web_docs) / max(len(web_docs), 1)
    )
    chunk_size = 800 if avg_len > 1_800 else 400
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.15),
        separators=["\n\n", "\n", " "],
        add_start_index=True,
    )
    web_chunks = splitter.split_documents(web_docs)

    return price_docs + web_chunks


# ── Vector-store -------------------------------------------------------------

def build_vector_store(chunks: List[Document]):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    # Clean previous collection to avoid duplicates
    Chroma(persist_directory=str(PERSIST_DIR)).delete_collection()
    vectordb = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )
    vectordb.add_documents(chunks)
    return vectordb


# ── Retriever híbrido --------------------------------------------------------

def build_hybrid_retriever():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    vec_ret = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.22, "k": 15},
    )

    raw = vectordb.get(include=["documents", "metadatas"])
    bm_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(raw["documents"], raw["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(bm_docs)
    bm25.k = 15

    return EnsembleRetriever(retrievers=[bm25, vec_ret], weights=[0.3, 0.7])


# ── Prompt -------------------------------------------------------------------
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres un asistente experto en servicios de peluquería y estética de Shizen Organic. "
        "Responde en español de forma concisa usando solo la información proporcionada en el contexto. "
        "Si la pregunta es sobre un precio y el contexto incluye un importe, responde únicamente con la cifra exacta y la divisa (ej. '35 €'). "
        "Si no encuentras información suficiente en el contexto, responde 'No lo encuentro.'."
        "\n{context}\nPregunta: {question}\nRespuesta:"
    ),
)


# ── QA -----------------------------------------------------------------------

def ask(query: str):
    price_query = bool(PRICE_QUERY_RE.search(query))

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

    # Fallback regex only for price queries
    if price_query and not CURRENCY_RE.search(answer):
        key = (
            unicodedata.normalize("NFKD", query)
            .encode("ascii", "ignore")
            .decode()
            .lower()
        )
        key = re.sub(r"cuanto|cuánto|cuesta|precio|vale|\?", "", key).strip()
        for doc in res["source_documents"]:
            txt = (
                unicodedata.normalize("NFKD", doc.page_content)
                .encode("ascii", "ignore")
                .decode()
                .lower()
            )
            if key and key in txt and (m := MONEY_RE.search(txt)):
                answer = f"{m.group(1).replace(',', '.')} €"
                break

    if not answer:
        answer = "No lo encuentro."

    print("\n💬  Respuesta:\n", answer, "\n")
    seen = set()
    unique_sources = [
        d
        for d in res["source_documents"]
        if not (d.metadata.get("source") in seen or seen.add(d.metadata.get("source")))
    ]
    print("📚  Fuentes:")
    for d in unique_sources:
        print(" •", d.metadata.get("source"))


# ── CLI ----------------------------------------------------------------------

def ingest():
    docs = (
        load_website(BASE_URL)
        + load_all_pdfs(DOCS_DIR)
        + load_json(PRICE_JSON)
    )
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print(f"Ingestados {len(chunks)} chunks → {PERSIST_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Pregunta (si se omite se re‑indexa)")
    args = parser.parse_args()
    if args.query:
        ask(args.query)
    else:
        ingest()
