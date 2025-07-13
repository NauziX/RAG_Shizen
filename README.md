# ingest.py – RAG CLI para Shizen Organic

## Resumen

`ingest.py` actúa como **punto de entrada** para el sistema RAG local de Shizen Organic.

- **Sin argumentos**: recorre las fuentes (web, PDFs, JSON), limpia y fragmenta el texto y construye la base vectorial Chroma persistente.
- **Con **``: carga la base vectorial, recupera contexto con un *retriever híbrido* (BM25 + embeddings) y genera una respuesta usando un LLM y un *prompt* especializado.

---

## Estructura del archivo

| Sección                                                       | Descripción                                                                                           | Líneas   |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------- |
| **1. Imports**                                                | Librerías estándar (argparse, pathlib, regex, …), 3rd‑party (LangChain, Chroma, OpenAI, rapidfuzz)    |  1‑28    |
| **2. Environment & Paths**                                    | Carga `.env`, define rutas (`ROOT_DIR`, `DOCS_DIR`, `PERSIST_DIR`, `PRICE_JSON`, `BASE_URL`)          |  30‑46   |
| **3. Config**                                                 | Modelos por defecto: `text‑embedding‑3‑large` y `gpt‑4o‑mini`                                         |  48‑52   |
| **4. Regex**                                                  | Patrones para detectar encabezados FAQ, importes en € y consultas sobre precio                        |  54‑66   |
| **5. Loaders**                                                | ‑ `load_website()` → crawl + limpieza HTML                                                            |          |
| ‑ `load_pdf()` → segmenta PDFs por secciones                  |                                                                                                       |          |
| ‑ `load_all_pdfs()`                                           |                                                                                                       |          |
| ‑ `load_json()` → convierte cada servicio+precio en documento |  69‑127                                                                                               |          |
| **6. Chunking**                                               | `chunk_documents()` divide texto largo en chunks (400‑800 car.) y separa docs con precios             |  130‑153 |
| **7. Vector Store**                                           | `build_vector_store()` crea/borra colección Chroma y añade embeddings                                 |  156‑164 |
| **8. Retrievers**                                             | `build_hybrid_retriever()` combina buscador de similitud (k=15) y BM25 (k=15) con pesos 0.7/0.3       |  167‑187 |
| **9. Helper**                                                 | `pick_best_price()` usa rapidfuzz para elegir el mejor precio cuando el LLM no devuelve importe       |  190‑212 |
| **10. Prompt**                                                | `CUSTOM_PROMPT` instruye al LLM a responder en español, conciso, y devolver sólo la info del contexto |  215‑229 |
| **11. QA Chain**                                              | `ask(query)` construye cadena Retrieval‑QA, aplica fallback de precio y muestra fuentes               |  232‑270 |
| **12. CLI**                                                   | `ingest()` indexa; `__main__` → si `--query` llama a `ask`, si no llama a `ingest`                    |  273‑296 |

---

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

