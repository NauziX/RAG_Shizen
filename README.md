RAG CLI para Shizen Organic

## Resumen

`ingest.py` actúa como **punto de entrada** para el sistema RAG local de Shizen Organic.

- **Sin argumentos**: recorre las fuentes (web, PDFs, JSON), limpia y fragmenta el texto y construye la base vectorial Chroma persistente.
- **Con **``: carga la base vectorial, recupera contexto con un *retriever híbrido* (BM25 + embeddings) y genera una respuesta usando un LLM y un *prompt* especializado.



## Diagrama de flujo

```mermaid
flowchart TD
    A[Arranque ingest.py] -->|--query vacío| B(Ingestar)
    B --> C[load_website]
    B --> D[load_all_pdfs]
    B --> E[load_json]
    C & D & E --> F[chunk_documents]
    F --> G[build_vector_store]
    A -->|--query "texto"| H(ask)
    H --> I[build_hybrid_retriever]
    I --> J[Retrieval]
    J --> K[LLM con CUSTOM_PROMPT]
    K --> L[Respuesta]
```

---

## Cómo usarlo

```bash
# 1) Reindexar todas las fuentes (se crea data/chroma/)
python ingest.py

# 2) Hacer una pregunta
python ingest.py --query "¿Cuánto cuesta la Pedicura Orgánica?"
```

---

## Dependencias clave

- `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `langchain-chroma`, `langchain-core`
- `chromadb`
- `openai`, `python-dotenv`, `beautifulsoup4`, `rapidfuzz`, `unstructured`

> **Tip**: coloca tu `OPENAI_API_KEY` y demás variables en `.env` antes de ejecutar.

