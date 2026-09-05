# TrifectaRAG

Local multi-modal RAG: **HNSW** (vectors) + **BM25** (keywords) + a **knowledge graph**, fused with RRF and reranked with MMR. Use it from the **browser tutor** or from the **Python SDK**.

The browser agent decides when to search your indexed library, pull a figure, or look on the web. Answers that used the library include source, page, and chunk id.

---

## What you need

- Python 3.11+
- Node 18+ (UI only)
- [Ollama](https://ollama.com) with a tool-capable model (default `qwen2.5:7b`)

```powershell
pip install -r requirements.txt
ollama pull qwen2.5:7b
cd frontend
npm install
```

Build the C++ engine once if `import trifecta` fails (from the repo root, with your usual pybind11 / CMake setup).

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` / `TRIFECTA_OLLAMA_MODEL` | `qwen2.5:7b` | Chat model |
| `OLLAMA_CHAT_TIMEOUT_SECONDS` | `150` | Per agent turn |
| `TRIFECTA_AGENT_ROUNDS` | `6` | Max tool-call rounds |

---

## Run the UI

Two processes, then open [http://localhost:5172/](http://localhost:5172/).

```powershell
# terminal 1 — API (port 8001)
python api.py

# terminal 2 — Vite (port 5172, proxies /chat, /library, /image, …)
cd frontend
npm run dev
```

On first start the API loads the textbook snapshot if one exists under `examples/data/` (`*_page_*.trifecta`). That can take a minute while CLIP loads. The header should show something like **Library · Numerical_Analysis_page_p11-444 · 448 chunks · HNSW+BM25+KG+MMR**.

---

## Using the tutor (browser)

There is **one chat**. There is no Study / Math / Research mode picker. The agent picks tools from the question.

### Ask a question

1. Type in the box (Shift+Enter for a new line).
2. Wait — the first answer after a restart can take about a minute.
3. Math should render as formulas (KaTeX). The renderer also accepts `\(...\)` and `\[...\]` if the model emits those.
4. If the library was used, open **Sources** under the reply: source name, page, `gid`, RRF score, and which retrievers ran.
5. **Used search_corpus · web_search** (or similar) tells you which tools ran.

Typical questions:

- Exam / textbook: *Construct a Lagrange interpolation polynomial \(p_1\) on \([-1,1]\) with \(x_0=-1\), \(x_1=1\).*
- Figure: *Show the bisection method figure and explain it.*
- Web: *What is retrieval-augmented generation? Give sources with URLs.*

### Attach vs index

| Control | What it does |
|---|---|
| Image / paperclip in the input | **This turn only.** Image, PDF, DOCX, TXT, MD, or CSV is parsed and given to the agent. Images are also matched against the library with CLIP. The file is **not** added as a lasting RAG source. |
| File-up in the input, or **Library → Index a file** | **Indexes** the file. Builds HNSW and BM25 together (plus KG links for PDFs). The agent can search it on later questions. |

Supported index types: PDF, DOCX, TXT, MD, CSV.

### Library walkthrough (index a laptop file and choose what to search)

1. Sidebar → **Library**.
2. **Index a file** → pick a PDF/DOCX/notes from your computer. Wait until the row says it is indexed (pages + chunk count).
3. Each source has **On / Off**. Off means that book is skipped at query time.
4. Under **Retrieval modes**:
   - **Vector (HNSW)** — semantic / embedding search.
   - **Keyword (BM25)** — lexical search.
   - Both are created on ingest. You can run one or both. At least one must stay on.
   - **Knowledge graph** stays automatic (1-hop expansion from retrieved seeds). It is not a user toggle.
5. **Back** to chat and ask. The header shows the active retriever label (`HNSW+BM25+KG+MMR`, `BM25+KG+MMR`, …).

If a PDF is already sitting in `uploaded_pdfs/` or `examples/data/` but is not indexed, Library lists it with **Create RAG DB**.

---

## Using the SDK (Python)

```python
from trifecta import TrifectaClient, PDFIngestor
from trifecta import trifecta_py as tr
```

### Ingest text, images, and a PDF

```python
client = TrifectaClient(device="cpu")

gid = client.add_document(
    "Lagrange interpolation uses basis polynomials L_i(x).",
    metadata={"source": "notes", "page": 1},
)
client.add_image("figure.png", caption="Bisection interval", metadata={"source": "notes", "page": 2})

# PDF: page mode = one chunk per page + cropped figures. HNSW + BM25 are filled together.
ingestor = PDFIngestor(client, mode="page")
stats = ingestor.ingest_pdf("textbook.pdf", output_dir="extracted_images")
print(stats)  # pages, text_chunks, images, kg_edges

client.add_edge(gid, other_gid, tr.EdgeType.RELATES_TO)  # or EXPLAINS / DEPICTS
client.save_snapshot("examples/data/my_corpus")  # writes .trifecta + .meta.gz
```

`mode="classical"` uses overlapping word chunks instead of one page per node.

Reload later:

```python
client = TrifectaClient.from_snapshot("examples/data/my_corpus.trifecta", device="cpu")
```

### Query (hybrid, or one retriever)

```python
# Both HNSW and BM25 (default). KG 1-hop still applies in the C++ engine.
hits = client.query(text="Lagrange interpolation", top_k=8)
for row in client.get_results(hits):
    meta = row["metadata"]
    print(row["global_id"], row["score"], meta.get("source"), meta.get("page"))

# Keyword only (skip the vector index)
hits = client.query(text="bisection", top_k=5, use_hnsw=False, use_bm25=True)

# Vector only
hits = client.query(text="bisection", top_k=5, use_hnsw=True, use_bm25=False)

# Image + text late fusion (CLIP)
hits = client.query(text="root finding figure", image="scan.png", top_k=5)
```

Page map:

```python
print(client.list_sources())
print(client.list_pages("Numerical_Analysis"))
print(client.get_page_chunks("Numerical_Analysis", 36))
print(client.get_chunk_page(209))  # -> ("Numerical_Analysis", 213)
```

### Same search the tutor uses (MMR + provenance)

```python
from trifecta.retrieve import hybrid_search, format_passages, serialize_sources

hits = hybrid_search(client, "Lagrange interpolation", top_k=4, use_hnsw=True, use_bm25=True)
print(serialize_sources(hits))
# [{global_id, score, source, page, text_preview, retrieval: "HNSW+BM25+KG+MMR"}, ...]
```

### Tutor agent from Python

```python
from trifecta.agent import run_agent

result = run_agent(
    "agent",
    "Construct p1 in P1 for f on [-1,1] with nodes x0=-1, x1=1.",
)
print(result["answer_markdown"])
print(result["tools_used"])   # e.g. ["search_corpus"]
print(result["sources"])      # page + gid provenance
```

`run_agent` is what `POST /chat` calls. It needs `python api.py`’s process (or an import of `api`) so the engine and retrieval toggles are shared.

### Example scripts

From the repo root:

```powershell
python examples/00_basic_usage.py
python examples/01_ingest_textbook.py
python examples/02_query_textbook.py
python examples/05_page_index_demo.py
```

Put a PDF in `examples/data/` (or set `TRIFECTA_TEXTBOOK_PDF`). `01` writes a snapshot; `02` queries it.

---

## HTTP API (what the UI calls)

Base URL when using Vite: same origin (`/chat` is proxied to `127.0.0.1:8001`). Direct: `http://127.0.0.1:8001`.

| Method | Path | Use |
|---|---|---|
| GET | `/health` | Chunks, corpus, retriever label |
| POST | `/chat` | Agent: `{ mode, messages, question?, attachments? }` |
| POST | `/ask` | Same agent, one question |
| POST | `/upload` | Index PDF/DOCX/TXT/MD/CSV → `{ task_id }` |
| GET | `/ingest-status/{task_id}` | Poll until `done` / `error` |
| POST | `/attach` | Parse a file for this chat turn |
| GET | `/corpora` | Indexed + on-disk sources |
| POST | `/corpora/toggle` | `{ name, enabled }` |
| POST | `/corpora/ingest` | Index a known on-disk PDF |
| GET/POST | `/settings/retrieval` | `{ use_hnsw, use_bm25 }` |
| GET | `/image?path=` | Sandboxed figure from extract dirs |

---

## How retrieval works (short)

On ingest, **HNSW and BM25 are always built together**. At query time:

1. Optional HNSW (if Vector is on).
2. Optional BM25 (if Keyword is on).
3. KG 1-hop from those seeds (always).
4. Reciprocal rank fusion, then MMR so you do not get four near-duplicate pages.

Provenance on each hit: `source`, `page`, `global_id`, RRF `score`. That is what **Sources** in the UI shows.
