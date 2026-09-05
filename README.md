# TrifectaRAG

Local multi-modal RAG: **HNSW** (vectors) + **BM25** (keywords) + a **knowledge graph**, fused with Reciprocal Rank Fusion and optionally reranked with MMR.

Install the Python SDK and CLI from PyPI, or run the optional browser tutor from this repo.

```bash
pip install trifectarag[pdf]
```

```python
from trifecta import TrifectaClient, PDFIngestor

client = TrifectaClient(device="cpu")
PDFIngestor(client, mode="page").ingest_pdf("textbook.pdf", output_dir="extracted")
client.save_snapshot("my.index")

hits = client.get_results(client.query("Newton's method", top_k=5))
for row in hits:
    print(row["global_id"], row["score"], row["metadata"].get("page"))
```

```bash
trifecta ingest textbook.pdf --index ./my.index
trifecta query "Newton's method" --index ./my.index --top-k 5
trifecta info --index ./my.index
```

---

## Install

Requires Python 3.10+ and a C++17 toolchain only when building from source. PyPI wheels ship the compiled `trifecta_py` extension.

| Extra | What it adds |
|---|---|
| *(none)* | Core SDK + CLI (hash embeddings, no PyTorch) |
| `[pdf]` | PyMuPDF ingestion + vector-figure extraction |
| `[ml]` | CLIP embeddings via `torch` / `transformers` |
| `[mcp]` | Model Context Protocol server |
| `[agent]` | Ollama tutor agent helpers |
| `[all]` | Everything above |

```bash
pip install trifectarag          # library + CLI
pip install trifectarag[pdf]     # + PDF ingestion
pip install trifectarag[all]     # + CLIP, MCP, agent
```

From a clone of this repository:

```bash
pip install -e ".[pdf,ml]"
```

Default index path is `./trifecta.index` (or `$TRIFECTA_INDEX`). Snapshots are two files: `<stem>.trifecta` (C++ engine) and `<stem>.meta.gz` (Python metadata).

---

## Python SDK

```python
from trifecta import TrifectaClient, PDFIngestor
from trifecta import trifecta_py as tr

client = TrifectaClient(device="cpu")

gid = client.add_document(
    "Lagrange interpolation uses basis polynomials L_i(x).",
    metadata={"source": "notes", "page": 1},
)
client.add_image("figure.png", caption="Bisection interval", metadata={"source": "notes", "page": 2})
client.add_edge(gid, other_gid, tr.EdgeType.RELATES_TO)  # or EXPLAINS / DEPICTS

# PDF: page mode = one chunk per page + cropped figures. HNSW + BM25 are filled together.
stats = PDFIngestor(client, mode="page").ingest_pdf("textbook.pdf", output_dir="extracted")
print(stats)  # pages, text_chunks, images, kg_edges

client.save_snapshot("my.index")
client = TrifectaClient.from_snapshot("my.index", device="cpu")
```

`mode="classical"` uses overlapping word chunks instead of one page per node.

### Query

```python
hits = client.query(text="Lagrange interpolation", top_k=8)
for row in client.get_results(hits):
    meta = row["metadata"]
    print(row["global_id"], row["score"], meta.get("source"), meta.get("page"))

# Keyword only / vector only
client.query(text="bisection", top_k=5, use_hnsw=False, use_bm25=True)
client.query(text="bisection", top_k=5, use_hnsw=True, use_bm25=False)

# Image + text late fusion (CLIP, or hash embeddings if [ml] is not installed)
client.query(text="root finding figure", image="scan.png", top_k=5)
```

### Page index

```python
client.list_sources()
client.list_pages("Numerical_Analysis")
client.get_page_chunks("Numerical_Analysis", 36)
client.get_chunk_page(209)  # -> ("Numerical_Analysis", 213)
```

### Hybrid search the tutor uses (MMR + provenance)

```python
from trifecta import hybrid_search, serialize_sources

hits = hybrid_search(client, "Lagrange interpolation", top_k=4)
print(serialize_sources(hits))
```

---

## CLI

```text
trifecta ingest FILE.pdf [FILE.pdf ...] [--mode page|classical] [--pages START:END]
trifecta query TEXT [--image FILE] [--top-k N] [--no-hnsw] [--no-bm25]
trifecta add-text TEXT [--source NAME] [--page N]
trifecta add-image FILE [--caption TEXT]
trifecta info | sources | pages [--source NAME] [--page N]
trifecta get GID
trifecta mcp [--ingest FILE.pdf]
```

Shared flags: `--index PATH`, `--device cpu|cuda`, `--json`.

```bash
trifecta ingest examples/data/Numerical_Analysis.pdf --index ./na.index --pages 11:444
trifecta query "trapezoidal rule" --index ./na.index --json
python -m trifecta info --index ./na.index
```

---

## How retrieval works

On ingest, **HNSW and BM25 are always built together**. At query time:

1. Optional HNSW (vector / embedding search)
2. Optional BM25 (lexical search)
3. Knowledge-graph 1-hop expansion from those seeds
4. Reciprocal rank fusion, then optional MMR so near-duplicate pages are dropped

Each hit carries `source`, `page`, `global_id`, and RRF `score`.

---

## Browser tutor (this repo)

The optional UI is **not** on PyPI. From a clone:

- Python 3.11+
- Node 18+
- [Ollama](https://ollama.com) with a tool-capable model (default `qwen2.5:7b`)

```powershell
pip install -e ".[all]"
pip install -r requirements.txt
ollama pull qwen2.5:7b
cd frontend
npm install
```

```powershell
# terminal 1 — API (port 8001)
python api.py

# terminal 2 — Vite (port 5172)
cd frontend
npm run dev
```

Open [http://localhost:5172/](http://localhost:5172/). On first start the API loads a textbook snapshot under `examples/data/` if one exists (`*_page_*.trifecta`).

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` / `TRIFECTA_OLLAMA_MODEL` | `qwen2.5:7b` | Chat model |
| `OLLAMA_CHAT_TIMEOUT_SECONDS` | `150` | Per agent turn |
| `TRIFECTA_AGENT_ROUNDS` | `6` | Max tool-call rounds |

Index a file from **Library**, then ask exam / figure / web questions in the single chat. Retrieval toggles (HNSW / BM25) live in the sidebar.

### HTTP API (what the UI calls)

Base URL with Vite: same origin. Direct: `http://127.0.0.1:8001`.

| Method | Path | Use |
|---|---|---|
| GET | `/health` | Chunks, corpus, retriever label |
| POST | `/chat` | Agent: `{ mode, messages, question?, attachments? }` |
| POST | `/upload` | Index PDF/DOCX/TXT/MD/CSV → `{ task_id }` |
| GET | `/ingest-status/{task_id}` | Poll until `done` / `error` |
| GET | `/corpora` | Indexed + on-disk sources |
| GET/POST | `/settings/retrieval` | `{ use_hnsw, use_bm25 }` |

---

## Example scripts

From the repo root, after `pip install -e ".[pdf,ml]"`:

```powershell
python examples/00_basic_usage.py
python examples/01_ingest_textbook.py
python examples/02_query_textbook.py
python examples/05_page_index_demo.py
```

Put a PDF in `examples/data/` or set `TRIFECTA_TEXTBOOK_PDF`.

---

## Publishing

```bash
pip install build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

GitHub Releases trigger `.github/workflows/publish.yml` (cibuildwheel + trusted PyPI publishing).
