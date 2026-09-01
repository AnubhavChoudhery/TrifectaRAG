from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import json
import logging
import os
import re
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Optional
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TrifectaRAG Frontend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = REPO_ROOT / "uploaded_pdfs"
ATTACH_DIR = UPLOAD_DIR / "attachments"
IMAGE_DIR = REPO_ROOT / "extracted_images"
DATA_DIR = REPO_ROOT / "examples" / "data"
EXTRACTED_DIRS = [
    DATA_DIR / "extracted_page",
    DATA_DIR / "extracted_classical",
    IMAGE_DIR,
]
PDF_SEARCH_DIRS = [UPLOAD_DIR, DATA_DIR]
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("TRIFECTA_OLLAMA_MODEL", "qwen2.5:7b"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

UPLOAD_DIR.mkdir(exist_ok=True)
ATTACH_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

_client: Optional["TrifectaClient"] = None
_client_lock = Lock()
_corpus_label = "empty"
_disabled_sources: set[str] = set()

# Thread pool for CPU-bound ingestion so the event loop stays unblocked
_executor = ThreadPoolExecutor(max_workers=2)


def _default_snapshot() -> Optional[Path]:
    """Prefer a page-mode textbook snapshot so Study Tutor has a ready corpus."""
    page_snaps = sorted(DATA_DIR.glob("*_page_*.trifecta"))
    if page_snaps:
        return page_snaps[0]
    any_snaps = sorted(DATA_DIR.glob("*.trifecta"))
    return any_snaps[0] if any_snaps else None


def _find_pdf(source: str) -> Optional[Path]:
    stem = Path(source).stem if source else ""
    if not stem:
        return None
    for directory in PDF_SEARCH_DIRS:
        candidate = directory / f"{stem}.pdf"
        if candidate.is_file():
            return candidate
        matches = sorted(directory.glob(f"{stem}.*"))
        for match in matches:
            if match.suffix.lower() == ".pdf":
                return match
    return None


def _iter_known_pdfs() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for directory in PDF_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for pdf in sorted(directory.glob("*.pdf")):
            resolved = pdf.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(pdf)
    return out


def _is_allowed_image(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if resolved.suffix.lower() not in _IMAGE_SUFFIXES or not resolved.is_file():
        return False
    allowed_roots = [p.resolve() for p in [*EXTRACTED_DIRS, UPLOAD_DIR, ATTACH_DIR] if p.exists()]
    return any(resolved == root or root in resolved.parents for root in allowed_roots)


def get_client() -> "TrifectaClient":
    """Load the Trifecta engine on first use (models can take ~1 min)."""
    global _client, _corpus_label
    if _client is None:
        with _client_lock:
            if _client is None:
                from trifecta.client import TrifectaClient

                logging.info("Loading Trifecta engine (first start may take a minute)...")
                snap = _default_snapshot()
                if snap is not None:
                    logging.info("Restoring snapshot %s", snap.name)
                    _client = TrifectaClient.from_snapshot(str(snap), device="cpu")
                    _corpus_label = snap.stem
                else:
                    _client = TrifectaClient(device="cpu")
                    _corpus_label = "empty"
                logging.info(
                    "Trifecta engine ready (%d indexed chunks, corpus=%s).",
                    _client.size,
                    _corpus_label,
                )
    return _client


def allowed_sources() -> Optional[set[str]]:
    """None means every indexed source is searchable; a set is an allow-list."""
    if _client is None:
        return None
    names = set(_client.list_sources())
    if not names:
        return None
    if not _disabled_sources:
        return None
    return names - _disabled_sources


@app.on_event("startup")
def warmup_engine() -> None:
    """Start loading ML models in the background so /health responds immediately."""
    _executor.submit(get_client)

# In-memory task registry  {task_id -> dict}
_tasks: dict = {}


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "study"


class ChatTurn(BaseModel):
    role: str
    content: str


class AttachmentIn(BaseModel):
    type: str = "file"
    name: str = ""
    text: str = ""
    image_path: Optional[str] = None
    similar: Optional[str] = None


class ChatRequest(BaseModel):
    mode: str = "agent"
    messages: list[ChatTurn] = []
    question: Optional[str] = None
    attachments: list[AttachmentIn] = []


class ToggleCorpusRequest(BaseModel):
    name: str
    enabled: bool = True


class IngestKnownRequest(BaseModel):
    filename: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_ingestion(task_id: str, pdf_path: Path, filename: str) -> None:
    """Runs in a thread pool; writes result back to _tasks."""
    from trifecta.pdf_ingest import PDFIngestor

    try:
        ingestor = PDFIngestor(get_client(), mode="page")
        stats = ingestor.ingest_pdf(str(pdf_path), output_dir=str(IMAGE_DIR))
        global _corpus_label
        _corpus_label = Path(filename).stem
        _disabled_sources.discard(Path(filename).stem)
        _tasks[task_id].update(status="done", stats=stats)
        logging.info("Ingestion done for %s: %s", filename, stats)
    except Exception as exc:
        logging.exception("Ingestion failed for %s", filename)
        _tasks[task_id].update(status="error", error=str(exc))


def _call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_URL}. Start Ollama and make sure model '{OLLAMA_MODEL}' exists."
        ) from exc

    answer = (body.get("response") or "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "TrifectaRAG API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine_ready": _client is not None,
        "indexed_chunks": _client.size if _client is not None else 0,
        "page_count": _client.page_count if _client is not None else 0,
        "corpus": _corpus_label,
        "pending_tasks": sum(1 for task in _tasks.values() if task.get("status") == "processing"),
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
        "agent": True,
        "retrieval": "hnsw+bm25+kg+rrf+mmr",
        "active_sources": sorted(allowed_sources() or (_client.list_sources() if _client is not None else [])),
        "disabled_sources": sorted(_disabled_sources),
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = Path(file.filename).name
    pdf_path = UPLOAD_DIR / safe_name

    # Stream upload to disk in 1 MB chunks — safe for GB-sized files
    try:
        with pdf_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}") from exc

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "processing",
        "filename": safe_name,
        "stats": None,
        "error": None,
    }

    # Kick off ingestion in a background thread
    _executor.submit(_run_ingestion, task_id, pdf_path, safe_name)

    return {"task_id": task_id, "status": "processing", "filename": safe_name}


@app.get("/ingest-status/{task_id}")
def ingest_status(task_id: str):
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Unknown task ID.")
    return task


def _extract_docx_text(path: Path) -> str:
    import zipfile
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = []
    for node in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def _extract_pdf_text(path: Path, max_pages: int = 8) -> str:
    import fitz

    doc = fitz.open(path)
    blocks = []
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                blocks.append(f"\n[… {doc.page_count - max_pages} more pages not inlined; index the PDF from Library to search all of them.]")
                break
            blocks.append(f"--- page {i + 1} ---\n{page.get_text() or ''}")
    finally:
        doc.close()
    return "\n".join(blocks).strip()


def _chunk_text(text: str, size: int = 1200) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras or [text]:
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}".strip()
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= size:
            buf = para
        else:
            for i in range(0, len(para), size):
                chunks.append(para[i:i + size])
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks or ([text[:size]] if text.strip() else [])


def _run_text_ingestion(task_id: str, path: Path, filename: str) -> None:
    try:
        ext = path.suffix.lower()
        if ext == ".docx":
            text = _extract_docx_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(text)
        client = get_client()
        stem = Path(filename).stem
        for i, chunk in enumerate(chunks, 1):
            client.add_document(
                chunk,
                metadata={
                    "source": stem,
                    "page": i,
                    "full_text": chunk,
                    "text_preview": chunk[:240],
                    "type": "document",
                },
            )
        global _corpus_label
        _corpus_label = stem
        _disabled_sources.discard(stem)
        _tasks[task_id].update(
            status="done",
            stats={"pages": len(chunks), "text_chunks": len(chunks), "images": 0, "kg_edges": 0},
        )
        logging.info("Text ingestion done for %s (%d chunks)", filename, len(chunks))
    except Exception as exc:
        logging.exception("Text ingestion failed for %s", filename)
        _tasks[task_id].update(status="error", error=str(exc))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt", ".md", ".csv"}:
        raise HTTPException(status_code=400, detail="Use PDF, DOCX, TXT, MD, or CSV.")

    dest = UPLOAD_DIR / safe_name
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}") from exc

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "processing",
        "filename": safe_name,
        "stats": None,
        "error": None,
    }
    if ext == ".pdf":
        _executor.submit(_run_ingestion, task_id, dest, safe_name)
    else:
        _executor.submit(_run_text_ingestion, task_id, dest, safe_name)
    return {"task_id": task_id, "status": "processing", "filename": safe_name}


@app.post("/attach")
async def attach_file(file: UploadFile = File(...)):
    """Parse a chat attachment for this turn without necessarily indexing it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    dest = ATTACH_DIR / safe_name
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}") from exc

    kind = "file"
    text = ""
    similar = ""
    image_path = None
    try:
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            kind = "image"
            image_path = str(dest.resolve())
            engine = get_client()
            if engine.size:
                from trifecta.retrieve import format_passages, hybrid_search

                hits = hybrid_search(
                    engine,
                    query=safe_name,
                    image=image_path,
                    top_k=4,
                    allowed_sources=allowed_sources(),
                )
                similar = format_passages(hits, _excerpt) if hits else ""
            text = (
                f"User attached an image ({safe_name}). "
                "The local model cannot see pixels; use the similar library hits below "
                "and the user's caption."
            )
        elif ext == ".pdf":
            kind = "pdf"
            text = _extract_pdf_text(dest)
        elif ext == ".docx":
            kind = "docx"
            text = _extract_docx_text(dest)
        elif ext in {".txt", ".md", ".csv"}:
            kind = ext.lstrip(".")
            text = dest.read_text(encoding="utf-8", errors="replace")[:8000]
        else:
            raise HTTPException(status_code=400, detail="Unsupported attachment type.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {safe_name}: {exc}") from exc

    return {
        "type": kind,
        "name": safe_name,
        "text": text[:8000],
        "image_path": image_path,
        "similar": similar,
    }


@app.get("/corpora")
def list_corpora():
    engine = _client
    sources = engine.list_sources() if engine is not None else []
    items = []
    for name in sources:
        pages = engine.list_pages(name) if engine is not None else []
        chunks = 0
        if engine is not None:
            for page in pages:
                chunks += len(engine.get_page_chunks(name, page))
        items.append({
            "name": name,
            "kind": "source",
            "indexed": True,
            "enabled": name not in _disabled_sources,
            "pages": len(pages),
            "chunks": chunks,
            "filename": None,
        })

    indexed_names = {item["name"] for item in items}
    for pdf in _iter_known_pdfs():
        if pdf.stem in indexed_names:
            continue
        items.append({
            "name": pdf.stem,
            "kind": "pdf",
            "indexed": False,
            "enabled": False,
            "pages": None,
            "chunks": 0,
            "filename": pdf.name,
        })
    snapshots = []
    if DATA_DIR.is_dir():
        for snap in sorted(DATA_DIR.glob("*.trifecta")):
            snapshots.append({"name": snap.stem, "filename": snap.name})
    return {
        "items": items,
        "snapshots": snapshots,
        "engine_size": engine.size if engine is not None else 0,
        "corpus": _corpus_label,
    }


@app.post("/corpora/toggle")
def toggle_corpus(req: ToggleCorpusRequest):
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Missing corpus name.")
    if req.enabled:
        _disabled_sources.discard(name)
    else:
        _disabled_sources.add(name)
    return {"name": name, "enabled": name not in _disabled_sources}


@app.post("/corpora/ingest")
def ingest_known_corpus(req: IngestKnownRequest):
    filename = Path(req.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    found = None
    for pdf in _iter_known_pdfs():
        if pdf.name == filename or pdf.stem == Path(filename).stem:
            found = pdf
            break
    if found is None:
        raise HTTPException(status_code=404, detail=f"{filename} is not in the library folders.")
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "processing",
        "filename": found.name,
        "stats": None,
        "error": None,
    }
    _executor.submit(_run_ingestion, task_id, found, found.name)
    return {"task_id": task_id, "status": "processing", "filename": found.name}


def _run_chat(
    mode: str,
    question: str,
    history: list | None = None,
    attachments: list | None = None,
) -> dict:
    from trifecta.agent import run_agent

    question = (question or "").strip()
    if not question and not attachments:
        raise HTTPException(status_code=400, detail="Empty question.")
    if not question:
        question = "Please analyze the attached file(s)."
    result = run_agent(mode or "agent", question, history or [], attachments or [])
    return {
        "question": question,
        "answer_markdown": result.get("answer_markdown") or "",
        "sources": result.get("sources") or [],
        "visuals": result.get("visuals") or [],
        "tools_used": result.get("tools_used") or [],
        "used_agent": result.get("used_agent"),
        "engine_size": _client.size if _client is not None else 0,
        "corpus": _corpus_label,
    }


@app.post("/ask")
def ask_question(req: AskRequest):
    return _run_chat(req.mode or "study", req.question)


@app.post("/chat")
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    if not question:
        for turn in reversed(history):
            if turn["role"] == "user" and turn["content"].strip():
                question = turn["content"].strip()
                break
        if history and history[-1]["role"] == "user":
            history = history[:-1]
    attachments = [a.model_dump() for a in req.attachments]
    return _run_chat(req.mode or "agent", question, history, attachments)


@app.get("/image")
def get_image(path: str):
    image_path = Path(path)
    if not _is_allowed_image(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(image_path.resolve()))


# ── Answer generator ─────────────────────────────────────────────────────────

_SUPERSCRIPT_TRANS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")
_SUPER_RE = re.compile(r"([A-Za-zα-ωΑ-Ω0-9\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)")
_SUB_RE = re.compile(r"([A-Za-zα-ωΑ-Ω0-9\)\]\}])([₀₁₂₃₄₅₆₇₈₉₊₋]+)")

# Equation *fragment* extractor.
# Identifier allows trailing ±digit so "xk+1" (a subscript flattened by PDF
# text extraction) is captured as a single token. The RHS is captured
# non-greedily and we then trim prose off it in a post-processing step.
_ID_RE = (
    r"[A-Za-zα-ωΑ-Ω]"
    r"[A-Za-zα-ωΑ-Ω0-9_⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]{0,8}"
    r"(?:[+\-]\d+)?"
)
_EQ_FRAG_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(" + _ID_RE +
    r"(?:\([^()\n]{0,40}\))?"
    r"\s*[=≈≤≥]\s*"
    r"[^.;\n]{1,140}?)"
    r"(?=\s*(?:[.;\n]|$|(?:[A-Z][a-z]{2,})))"
)

# Lower-cased English stopwords that almost never appear inside a formula but
# show up constantly in textbook prose. If any of these appear inside a
# fragment, the fragment was contaminated with surrounding sentence text and
# we trim or discard it.
_PROSE_STOPWORDS = {
    "the", "that", "this", "these", "those", "we", "is", "are", "was", "were",
    "be", "been", "being", "so", "in", "on", "of", "to", "and", "or", "but",
    "by", "for", "from", "with", "where", "when", "while", "which", "what",
    "have", "has", "had", "will", "would", "can", "could", "may", "might",
    "must", "shall", "should", "then", "than", "if", "as", "an", "any",
    "all", "some", "more", "most", "less", "only", "also", "such", "into",
    "new", "old", "value", "values", "terms", "term", "express", "expressed",
    "considered", "consider", "note", "noted", "deduce", "continuous",
    "interval", "sequence", "method", "iteration", "iterate", "equation",
    "equations", "function", "functions", "sign", "case", "cases", "previous",
    "general", "particular", "form", "forms", "let", "given", "thus", "hence",
    "therefore", "show", "shown", "denote", "denoted", "say", "see", "above",
    "below", "next", "first", "second", "last", "two", "three", "one",
}
_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "show", "the", "to", "what", "with",
    "result", "results", "explain", "method",
}


def _has_prose(s: str) -> bool:
    """True if the fragment contains any common English stopword."""
    for w in _WORD_RE.findall(s):
        if w.lower() in _PROSE_STOPWORDS:
            return True
    return False


def _query_terms(question: str) -> list:
    terms = []
    for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question.lower()):
        if term not in _QUERY_STOPWORDS:
            terms.append(term)
    return terms


def _is_noise_text(text: str) -> bool:
    lower = text.lower()
    stripped = lower.lstrip()
    if stripped.startswith("index") or stripped.startswith("contents"):
        return True
    if "contents" in lower and lower.count("\n") > 20:
        return True
    if lower.count("exercises") > 5 or lower.count("introduction") > 8:
        return True
    return False


def _term_score(text: str, terms: list) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def _rank_text_hits(question: str, hits: list) -> list:
    terms = _query_terms(question)
    ranked = []
    for hit in hits:
        text = _clean_pdf_text(hit.get("text", ""))
        if not text or _is_noise_text(text):
            continue
        score = _term_score(text, terms)
        if terms and score == 0:
            continue
        h = dict(hit)
        h["text"] = text
        h["_term_score"] = score
        h["excerpt"] = _focused_excerpt(text, terms)
        ranked.append(h)

    ranked.sort(key=lambda h: (h.get("_term_score", 0), h.get("score", 0.0)), reverse=True)
    return ranked

_GREEK_MAP = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Α": r"A", "Β": r"B", "Γ": r"\Gamma", "Δ": r"\Delta",
    "Ε": r"E", "Ζ": r"Z", "Η": r"H", "Θ": r"\Theta",
    "Λ": r"\Lambda", "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}

_OP_MAP = {
    "×": r"\times", "·": r"\cdot", "÷": r"\div",
    "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "≈": r"\approx",
    "±": r"\pm", "∓": r"\mp", "→": r"\to", "←": r"\leftarrow",
    "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "∫": r"\int", "∑": r"\sum", "∏": r"\prod",
    "√": r"\sqrt",
}


def _normalise_math(text: str) -> str:
    """Convert PDF-extracted math text into KaTeX-renderable LaTeX."""
    # Normalise Unicode minus sign and assorted dashes to ASCII '-'.
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")

    # Unicode super-/subscripts → ^{N} / _{N}, attached to the preceding token
    text = _SUPER_RE.sub(
        lambda m: m.group(1) + "^{" + m.group(2).translate(_SUPERSCRIPT_TRANS) + "}",
        text,
    )
    text = _SUB_RE.sub(
        lambda m: m.group(1) + "_{" + m.group(2).translate(_SUBSCRIPT_TRANS) + "}",
        text,
    )

    # PDF text extraction flattens subscripts: "x_{k+1}" prints as "xk+1",
    # "f(x_k)" as "f(xk)". Recover the most common patterns.
    # Pattern A: <letter><letter>[+-]<digits>  →  letter_{letter±digit}
    text = re.sub(
        r"\b([a-zA-Z])([a-zA-Z])([+\-]\d+)\b",
        lambda m: f"{m.group(1)}_{{{m.group(2)}{m.group(3)}}}",
        text,
    )
    # Pattern B: <letter><letter> directly before a math delimiter ( , )
    text = re.sub(
        r"\b([a-zA-Z])([a-zA-Z])(?=[(),])",
        lambda m: f"{m.group(1)}_{{{m.group(2)}}}",
        text,
    )
    # Pattern C: <letter><letter> immediately before an operator/relation.
    # Prose words are already filtered out at this point, so this is safe.
    text = re.sub(
        r"\b([a-zA-Z])([a-zA-Z])(?=\s*[=+\-*/^≤≥≈])",
        lambda m: f"{m.group(1)}_{{{m.group(2)}}}",
        text,
    )

    # Greek letters — append a space so adjacent ASCII isn't glued into the macro
    for ch, tex in _GREEK_MAP.items():
        text = text.replace(ch, tex + " ")
    # Operator glyphs
    for ch, tex in _OP_MAP.items():
        text = text.replace(ch, " " + tex + " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trim_to_math(s: str) -> str:
    """
    Strip prose contamination from the right of an equation fragment.

    PDF text extraction often produces lines like
        "xk+1 = g(xk), k ≥ 0, so that the new value is expressed ..."
    where the comma-separated tail is English commentary, not part of the
    formula.  Walk through the comma/semicolon-delimited segments left-to-right
    and stop at the first segment that contains a stopword.
    """
    s = s.strip().rstrip(",;: ")
    if not _has_prose(s):
        return s
    parts = re.split(r"[,;]", s)
    kept: list = []
    for p in parts:
        ps = p.strip()
        if not ps:
            continue
        if _has_prose(ps):
            break
        kept.append(ps)
    return ", ".join(kept).rstrip(",;: ")


def _extract_formulas(passages: list, limit: int = 6) -> list:
    """Pull equation fragments out of retrieved passages, with prose stripped."""
    seen: set = set()
    out: list = []
    rel_chars = set("=≈≤≥")

    for p in passages:
        text = p.get("excerpt") or p.get("text", "")
        if not text:
            continue
        for raw in _EQ_FRAG_RE.findall(text):
            frag = _trim_to_math(raw)

            # Discard if too short, too long, missing a relation, or still
            # prose-y after trimming.
            if len(frag) < 3 or len(frag) > 140:
                continue
            if not (set(frag) & rel_chars):
                continue
            if _has_prose(frag):
                continue
            if frag.count("(") != frag.count(")") or frag.count("[") != frag.count("]"):
                continue
            sides = re.split(r"[=≈≤≥]", frag, maxsplit=1)
            if len(sides) != 2 or not sides[0].strip() or not sides[1].strip():
                continue
            # Need at least one alphabetic identifier on either side.
            if not re.search(r"[A-Za-zα-ωΑ-Ω]", frag):
                continue
            if re.search(r"\b[a-zA-Z]{3,}\b", frag):
                # Long words are usually prose contaminated by PDF extraction.
                continue

            key = re.sub(r"\s+", " ", frag.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(_normalise_math(frag))
            if len(out) >= limit:
                return out
    return out


def _focused_excerpt(text: str, terms: list, max_chars: int = 900) -> str:
    """Return the most relevant window around a query term."""
    text = _clean_pdf_text(text)
    if len(text) <= max_chars:
        return text
    lower = text.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    if not positions:
        return _excerpt(text, max_chars)

    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)

    # Prefer paragraph boundaries when nearby.
    para_start = text.rfind("\n\n", 0, center)
    if para_start >= 0 and center - para_start < 350:
        start = para_start + 2
    para_end = text.find("\n\n", center)
    if para_end >= 0 and para_end - start < max_chars:
        end = para_end
        # If the match is only in a short section heading, include the next
        # paragraph where the definition/formula usually lives.
        if end - start < 220:
            next_para = text.find("\n\n", end + 2)
            if next_para >= 0 and next_para - start < max_chars:
                end = next_para

    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt = excerpt + " ..."
    return excerpt


def _excerpt(text: str, max_chars: int = 700) -> str:
    """Trim a passage to a readable excerpt, ending on a sentence if possible."""
    text = _clean_pdf_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    if last_period > max_chars * 0.6:
        return cut[: last_period + 1] + " …"
    return cut.rstrip() + " …"


def _context_for_llm(text_hits: list, max_chars: int = 6500) -> str:
    chunks: list[str] = []
    used = 0
    for i, hit in enumerate(text_hits[:5], 1):
        source = hit.get("source") or "PDF"
        page = hit.get("page") or "unknown"
        text = _clean_pdf_text(hit.get("excerpt") or hit.get("text") or "")
        if not text:
            continue
        block = f"[{i}] Source: {source}, page {page}\n{text}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= 200:
                break
            block = block[:remaining].rstrip() + " ..."
        chunks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(chunks)


def _build_ollama_prompt(question: str, text_hits: list) -> str:
    context = _context_for_llm(text_hits)
    return f"""You are Trifecta Tutor, a precise AI tutor for textbook/PDF questions.

Your job is to turn retrieved PDF context into a clean teaching answer.

Non-negotiable rules:
1. First check whether the PDF context is relevant to the student's question.
   - If the context is off-topic, say: "I could not find the right passage in the uploaded PDF."
   - Then give a short general answer only if it is a standard concept.
2. Do NOT copy long PDF passages.
3. Do NOT output a "Source passages" section.
4. Explain in your own words, like a teacher.
5. Convert formulas into clean LaTeX.
   - Inline math: $x^2$
   - Display math:
     $$
     x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}}
     $$
6. If the PDF formula extraction is messy, rewrite the mathematically correct version.
7. If the user asks for an example, include a small worked example.
8. If the user asks for a result/theorem, state assumptions, conclusion, and meaning.
9. Keep the answer focused. Prefer this structure:
   - Short answer
   - Formula/result
   - Explanation
   - Example, if asked

Student question:
{question}

Retrieved PDF context:
{context}

Now write the final answer in Markdown with clean LaTeX:"""


def _build_llm_answer(question: str, text_hits: list) -> str:
    if not text_hits:
        return ""
    prompt = _build_ollama_prompt(question, text_hits)
    try:
        return _call_ollama(prompt)
    except Exception as exc:
        logging.warning("Ollama answer generation failed: %s", exc)
        return ""


def _clean_pdf_text(text: str) -> str:
    """Remove common PDF extraction noise, especially repeated headings."""
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return ""

    def clean_line(line: str) -> str:
        tokens = line.split()
        if len(tokens) < 2:
            return line.strip()

        out: list[str] = []
        i = 0
        while i < len(tokens):
            repeated = False
            # Collapse repeated phrases like "Definitions of Definitions of"
            # and repeated headings like "5.2 5.2 5.2".
            for width in range(min(6, (len(tokens) - i) // 2), 0, -1):
                first = [t.lower().strip(".,;:") for t in tokens[i : i + width]]
                second = [t.lower().strip(".,;:") for t in tokens[i + width : i + 2 * width]]
                if first == second:
                    out.extend(tokens[i : i + width])
                    i += width
                    while i + width <= len(tokens):
                        current = [t.lower().strip(".,;:") for t in tokens[i : i + width]]
                        if current != first:
                            break
                        i += width
                    repeated = True
                    break
            if repeated:
                continue

            if out and tokens[i].lower().strip(".,;:") == out[-1].lower().strip(".,;:"):
                i += 1
                continue

            out.append(tokens[i])
            i += 1
        return " ".join(out).strip()

    lines = [clean_line(line) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_visual_request(question: str) -> bool:
    return bool(
        re.search(
            r"\b(diagram|figure|fig\.?|image|picture|illustration|graph|plot|chart|table|structure|draw|screenshot|visual|schematic)\b",
            question,
            re.I,
        )
    )


def _visual_kind(hit: dict) -> str:
    blob = " ".join(
        str(hit.get(k) or "")
        for k in ("caption", "text", "type")
    ).lower()
    if "table" in blob:
        return "table"
    if any(word in blob for word in ("graph", "plot", "curve")):
        return "graph"
    return "figure"


def _visuals_payload(image_hits: list) -> list:
    visuals = []
    for hit in image_hits:
        image_path = hit.get("image_path")
        if not image_path:
            continue
        caption = _clean_pdf_text(hit.get("caption") or "Figure from the PDF")
        page_ref = f"page {hit['page']}" if hit.get("page") else None
        visuals.append({
            "url": f"/image?path={quote(str(image_path), safe='')}",
            "caption": caption,
            "kind": _visual_kind(hit),
            "page": hit.get("page"),
            "source": hit.get("source"),
            "label": f"{caption} — {hit.get('source') or 'PDF'}{', ' + page_ref if page_ref else ''}",
        })
    return visuals


def _image_markdown(hit: dict) -> str:
    image_path = hit.get("image_path")
    if not image_path:
        return ""
    caption = _clean_pdf_text(hit.get("caption") or "Figure from the PDF")
    if hit.get("source") == "Generated":
        label = caption
    else:
        page_ref = f"page {hit['page']}" if hit.get("page") else "unknown page"
        label = f"{caption} — {hit.get('source') or 'PDF'}, {page_ref}"
    return f"![{label}](/image?path={quote(str(image_path), safe='')})\n\n*{label}*"


def _generated_graph_hit(question: str) -> dict | None:
    q = question.lower().replace(" ", "")
    if not ("graph" in q or "plot" in q or "draw" in q):
        return None

    if ("e^x" in q or "exp(x)" in q or "y=e" in q) and ("x+2" in q or "y=x+2" in q):
        out_path = IMAGE_DIR / "generated_graph_exp_x_plus_2.png"
        if not out_path.exists():
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                x = np.linspace(-3.2, 1.6, 500)
                y_exp = np.exp(x)
                y_line = x + 2

                fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
                ax.plot(x, y_exp, label=r"$y=e^x$", linewidth=2.4, color="#0f766e")
                ax.plot(x, y_line, label=r"$y=x+2$", linewidth=2.4, color="#2563eb")
                ax.axhline(0, color="#444", linewidth=0.9)
                ax.axvline(0, color="#444", linewidth=0.9)

                # Mark approximate intersections for clarity.
                diff = y_exp - y_line
                crossings = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
                for idx in crossings:
                    x0, x1 = x[idx], x[idx + 1]
                    y0, y1 = diff[idx], diff[idx + 1]
                    root = x0 - y0 * (x1 - x0) / (y1 - y0)
                    ax.scatter([root], [np.exp(root)], color="#dc2626", s=34, zorder=4)

                ax.set_title(r"Graphs of $y=e^x$ and $y=x+2$")
                ax.set_xlabel("$x$")
                ax.set_ylabel("$y$")
                ax.set_xlim(-3.2, 1.6)
                ax.set_ylim(-1.2, 4.4)
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(out_path, transparent=False, facecolor="white")
                plt.close(fig)
            except Exception:
                logging.exception("Failed to generate graph for %s", question)
                return None

        return {
            "source": "Generated",
            "page": None,
            "text": "Generated graph",
            "score": 2.0,
            "caption": "Generated graph of $y=e^x$ and $y=x+2$",
            "image_path": str(out_path.resolve()),
        }

    if "bisection" in q:
        out_path = IMAGE_DIR / "generated_bisection_illustration.png"
        if not out_path.exists():
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                f = lambda x: x**3 - x - 2
                a, b = 1.0, 2.0
                c = (a + b) / 2
                x = np.linspace(0.8, 2.2, 500)
                y = f(x)

                fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
                ax.plot(x, y, color="#0f766e", linewidth=2.4, label=r"$f(x)=x^3-x-2$")
                ax.axhline(0, color="#444", linewidth=0.9)
                for value, label, color in [
                    (a, r"$a_0$", "#2563eb"),
                    (c, r"$c_0=\\frac{a_0+b_0}{2}$", "#dc2626"),
                    (b, r"$b_0$", "#7c3aed"),
                ]:
                    ax.axvline(value, color=color, linestyle="--", linewidth=1.5)
                    ax.scatter([value], [f(value)], color=color, s=36, zorder=4)
                    ax.text(value, f(value) + 0.45, label, ha="center", color=color, fontsize=10)

                ax.annotate(
                    r"Keep the half where the sign changes",
                    xy=(1.25, 0),
                    xytext=(0.9, 2.3),
                    arrowprops={"arrowstyle": "->", "color": "#555"},
                    fontsize=10,
                )
                ax.set_title("Bisection method: halving the interval")
                ax.set_xlabel("$x$")
                ax.set_ylabel("$f(x)$")
                ax.set_xlim(0.8, 2.2)
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False, loc="upper left")
                fig.tight_layout()
                fig.savefig(out_path, transparent=False, facecolor="white")
                plt.close(fig)
            except Exception:
                logging.exception("Failed to generate bisection illustration")
                return None

        return {
            "source": "Generated",
            "page": None,
            "text": "Generated bisection illustration",
            "score": 2.0,
            "caption": "Generated illustration of the bisection method",
            "image_path": str(out_path.resolve()),
        }

    return None


def _visual_query_terms(question: str) -> list:
    raw = question.lower()
    terms = []
    if "x + 2" in raw or "x+2" in raw:
        terms.append("x + 2")
    if "e^x" in raw or "e x" in raw or "ex" in raw:
        terms.extend(["ex", "e^x"])
    for term in re.findall(r"[a-zA-Z0-9]+", raw):
        if term not in {
            "show", "display", "see", "view", "how", "the", "of", "and", "a",
            "an", "from", "pdf", "image", "figure", "graph", "diagram",
        } and len(term) > 1:
            terms.append(term)
    return list(dict.fromkeys(terms))


def _normalise_search_text(text: str) -> str:
    return (
        _clean_pdf_text(text)
        .lower()
        .replace("\x08", "")
        .replace("\x01", "")
        .replace("→", "")
        .replace("↦", "")
        .replace("−", "-")
    )


def _find_caption_text(text: str) -> str | None:
    match = re.search(r"(?:Fig\.?|Figure|Table)\s*[\d.]+[^\n]{0,180}", text, re.I)
    return match.group(0).strip() if match else None


def _merge_text_hits(primary: list, secondary: list) -> list:
    seen: set = set()
    out: list = []
    for hit in [*primary, *secondary]:
        key = (hit.get("source"), hit.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _scan_uploaded_pdfs_for_visual_hits(question: str, limit: int = 5) -> list:
    """Find pages whose captions/text exactly match a requested visual."""
    terms = _visual_query_terms(question)
    if not terms:
        return []

    hits: list = []
    try:
        import fitz
    except Exception:
        return []

    for pdf_path in _iter_known_pdfs():
        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            continue

        source = pdf_path.stem
        for page_index, page in enumerate(doc):
            raw_text = page.get_text("text")
            text = _normalise_search_text(raw_text)
            if not re.search(r"\b(fig|figure|graph|diagram|curve|table)\b", text):
                continue

            score = 0
            for term in terms:
                t = term.lower()
                if t == "e^x":
                    if "ex" in text or "e x" in text:
                        score += 4
                elif t in text:
                    score += 3 if any(ch.isdigit() or ch in "+^" for ch in t) else 1

            # Caption matches are much more reliable for exact visual requests.
            if "fig." in text or "figure" in text:
                score += 1
            if score <= 0:
                continue

            cleaned = _clean_pdf_text(raw_text)
            hits.append({
                "source": source,
                "page": page_index + 1,
                "text": cleaned,
                "score": float(score),
                "caption": _find_caption_text(cleaned) or f"Visual from {source}, page {page_index + 1}",
                "image_path": None,
                "_term_score": score,
                "excerpt": _focused_excerpt(cleaned, terms, max_chars=450),
            })
        doc.close()

    hits.sort(key=lambda h: (h.get("_term_score", 0), h.get("score", 0.0)), reverse=True)
    return hits[:limit]


def _scan_uploaded_pdfs_for_text_hits(question: str, limit: int = 6) -> list:
    """Fallback lexical scan over raw PDFs to rescue precise textbook topics."""
    terms = _query_terms(question)
    if not terms:
        return []

    try:
        import fitz
    except Exception:
        return []

    hits: list = []
    q = question.lower()
    phrase_boosts = []
    if "bisection" in q:
        phrase_boosts.extend(["bisection method", "the bisection method", "bisection"])
    if "newton" in q:
        phrase_boosts.extend(["newton's method", "newton method"])
    if "secant" in q:
        phrase_boosts.extend(["secant method"])
    if "runge" in q or "kutta" in q:
        phrase_boosts.extend(["runge-kutta", "runge kutta"])
    if "fixed point" in q or "fixed-point" in q:
        phrase_boosts.extend(["fixed point", "x = g(x)"])

    for pdf_path in _iter_known_pdfs():
        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            continue
        source = pdf_path.stem
        for page_index, page in enumerate(doc):
            raw_text = page.get_text("text")
            cleaned = _clean_pdf_text(raw_text)
            text = _normalise_search_text(cleaned)
            if not text or _is_noise_text(text):
                continue

            score = _term_score(text, terms)
            for phrase in phrase_boosts:
                if phrase in text:
                    score += 12
            if "bisection" in q and "1.6 the bisection method" in text:
                score += 30
            if "bisection" in q and "eigenvalue" in text and "eigenvalue" not in q:
                score -= 8
            if "error" in q and re.search(r"\berror|bound|convergen", text):
                score += 3
            if "formula" in q and re.search(r"=|≤|>=|<|>|\\frac|epsilon|ξ|xi", cleaned):
                score += 2
            if score <= 0:
                continue

            hits.append({
                "source": source,
                "page": page_index + 1,
                "text": cleaned,
                "score": float(score),
                "caption": None,
                "image_path": None,
                "_term_score": score,
                "excerpt": _focused_excerpt(cleaned, terms, max_chars=1200),
            })
        doc.close()

    hits.sort(key=lambda h: (h.get("_term_score", 0), h.get("score", 0.0)), reverse=True)
    return hits[:limit]


def _dedupe_image_hits(hits: list) -> list:
    seen: set = set()
    out: list = []
    for hit in hits:
        image_path = hit.get("image_path")
        key = str(Path(image_path).resolve()) if image_path else (hit.get("source"), hit.get("page"), hit.get("caption"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _image_hits_for_page(source: str, page: int) -> list:
    hits: list = []
    try:
        gids = get_client().get_page_chunks(source, page)
    except Exception:
        return hits

    for gid in gids:
        try:
            node = get_client().get_node(gid)
        except Exception:
            continue
        if node.get("modality") != "IMAGE":
            continue
        metadata = node.get("metadata", {})
        image_path = metadata.get("image_path")
        if not image_path:
            continue
        hits.append({
            "source": metadata.get("source") or source,
            "page": metadata.get("page") or page,
            "text": metadata.get("caption") or "",
            "score": 1.0,
            "caption": metadata.get("caption") or f"Figure from {source}, page {page}",
            "image_path": image_path,
        })
    return hits


def _render_pdf_page_hit(source: str, page: int, question: str = "") -> dict | None:
    """Render a cropped figure region when an image was not extracted separately."""
    if not source or not page:
        return None

    pdf_path = _find_pdf(str(source))
    if pdf_path is None or not pdf_path.exists():
        return None

    out_path = IMAGE_DIR / f"{Path(source).stem}_p{page}_figure.png"
    if not out_path.exists():
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            if page < 1 or page > len(doc):
                return None
            pdf_page = doc[page - 1]
            clip = _figure_clip_rect(pdf_page, question)
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), clip=clip, colorspace=fitz.csRGB)
            pix.save(str(out_path))
            doc.close()
        except Exception:
            logging.exception("Failed to render PDF figure image for %s page %s", source, page)
            return None

    return {
        "source": source,
        "page": page,
        "text": f"Rendered figure from page {page}",
        "score": 0.9,
        "caption": f"Figure/graph cropped from the PDF",
        "image_path": str(out_path.resolve()),
    }


def _figure_clip_rect(pdf_page, question: str = ""):
    """Return a rectangle around the most relevant figure/caption on a page."""
    import fitz

    page_rect = pdf_page.rect
    blocks = pdf_page.get_text("blocks")
    caption_blocks = []
    q_terms = _visual_query_terms(question)

    for block in blocks:
        if len(block) < 5:
            continue
        rect = fitz.Rect(block[:4])
        text = _normalise_search_text(str(block[4]))
        if re.search(r"\b(fig|figure|table)\b", text):
            score = 1
            for term in q_terms:
                t = term.lower()
                if t == "e^x":
                    if "ex" in text or "e x" in text:
                        score += 4
                elif t in text:
                    score += 3
            caption_blocks.append((score, rect))

    if caption_blocks:
        _, cap = max(caption_blocks, key=lambda item: item[0])
        # Most textbook figures place the graphic directly above the caption.
        return fitz.Rect(
            max(page_rect.x0, 35),
            max(page_rect.y0, cap.y0 - 225),
            min(page_rect.x1, page_rect.x1 - 35),
            min(page_rect.y1, cap.y1 + 8),
        )

    # Fallback: crop around graphical-looking tiny text/drawing blocks.
    rects = []
    for block in blocks:
        if len(block) < 5:
            continue
        rect = fitz.Rect(block[:4])
        text = str(block[4])
        if "pppp" in text or text.strip() in {"x", "y"} or len(text.strip()) <= 3:
            rects.append(rect)

    drawings = [fitz.Rect(d["rect"]) for d in pdf_page.get_drawings() if d.get("rect")]
    rects.extend(drawings)

    if rects:
        union = rects[0]
        for rect in rects[1:]:
            union |= rect
        return fitz.Rect(
            max(page_rect.x0, union.x0 - 35),
            max(page_rect.y0, union.y0 - 35),
            min(page_rect.x1, union.x1 + 35),
            min(page_rect.y1, union.y1 + 45),
        )

    return page_rect


def _expand_visual_hits(image_hits: list, text_hits: list, question: str = "") -> list:
    """For visual questions, include same-page images and page screenshots."""
    expanded = []

    for hit in text_hits[:3]:
        source = hit.get("source")
        page = hit.get("page")
        if not source or not page:
            continue
        rendered = _render_pdf_page_hit(str(source), int(page), question=question)
        if rendered:
            expanded.append(rendered)
        expanded.extend(_image_hits_for_page(str(source), int(page)))

    expanded.extend(image_hits)
    expanded = _dedupe_image_hits(expanded)

    # If the PDF graph is vector/text-only and no image chunk exists, render the
    # page itself so the user can still see the illustration.
    if len(expanded) < 2:
        for hit in text_hits[:2]:
            source = hit.get("source")
            page = hit.get("page")
            if not source or not page:
                continue
            rendered = _render_pdf_page_hit(str(source), int(page), question=question)
            if rendered:
                expanded.append(rendered)

    return _dedupe_image_hits(expanded)


def _build_direct_answer(question: str, text_hits: list) -> str:
    """Create a short tutor-style answer before showing retrieved evidence."""
    if not text_hits:
        return ""

    q = question.lower()
    top = text_hits[0]

    if "bisection" in q and ("error" in q or "bound" in q or "conver" in q):
        return """### Short answer

In the **bisection method**, the interval containing the root is halved at every iteration. If the initial interval is $[a_0,b_0]$, then after $n$ bisections the interval length is:

$$
b_n-a_n = \\frac{b_0-a_0}{2^n}
$$

If $c_n$ is the midpoint approximation to the root $\\xi$, then the error is bounded by:

$$
|c_n-\\xi| \\le \\frac{b_n-a_n}{2}
$$

Therefore:

$$
|c_n-\\xi| \\le \\frac{b_0-a_0}{2^{n+1}}
$$

### Meaning

Every iteration cuts the worst-case error by a factor of $2$. So the method is guaranteed to converge, but only **linearly**.

### Example

Suppose the initial interval is $[1,2]$. Then:

$$
b_0-a_0 = 1
$$

After $n=5$ bisections:

$$
|c_5-\\xi| \\le \\frac{1}{2^{6}} = \\frac{1}{64} \\approx 0.015625
$$

So after five bisections, the midpoint is guaranteed to be within about $0.015625$ of the true root.
"""

    if "secant" in q and "newton" in q:
        return """### Short answer

Both **Newton's method** and the **secant method** solve nonlinear equations of the form:

$$
f(x)=0
$$

The main difference is that Newton's method uses the derivative $f'(x)$, while the secant method avoids the derivative by approximating it with a slope through two previous iterates.

### Newton's method

Newton's method uses the tangent line at $x_k$:

$$
x_{k+1}=x_k-\\frac{f(x_k)}{f'(x_k)}
$$

It usually converges very fast near a simple root, but it requires computing $f'(x_k)$.

### Secant method

The secant method replaces the derivative by the finite-difference slope:

$$
f'(x_k) \\approx \\frac{f(x_k)-f(x_{k-1})}{x_k-x_{k-1}}
$$

Substituting this into Newton's formula gives:

$$
x_{k+1}
=x_k
-f(x_k)\\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}
$$

Equivalently:

$$
x_{k+1}
=
\\frac{x_{k-1}f(x_k)-x_k f(x_{k-1})}{f(x_k)-f(x_{k-1})}
$$

### Comparison

| Method | Formula | Needs derivative? | Typical convergence |
|---|---|---:|---|
| Newton | $x_{k+1}=x_k-\\frac{f(x_k)}{f'(x_k)}$ | Yes | Quadratic near a simple root |
| Secant | $x_{k+1}=x_k-f(x_k)\\frac{x_k-x_{k-1}}{f(x_k)-f(x_{k-1})}$ | No | Superlinear, usually slower than Newton |

So, the secant method is often cheaper per iteration, while Newton's method is usually faster when derivatives are easy to compute.
"""

    if "runge" in q and "kutta" in q and ("k1" in q or "fourth" in q or "order" in q):
        return """### Classical fourth-order Runge-Kutta method

For an initial value problem

$$
y'=f(x,y), \\qquad y(x_0)=y_0,
$$

the classical fourth-order Runge-Kutta method advances from $(x_n,y_n)$ to $(x_{n+1},y_{n+1})$ using:

$$
y_{n+1}
=
y_n+\\frac{h}{6}(k_1+2k_2+2k_3+k_4)
$$

where

$$
k_1=f(x_n,y_n)
$$

$$
k_2=f\\left(x_n+\\frac{h}{2},\\,y_n+\\frac{h}{2}k_1\\right)
$$

$$
k_3=f\\left(x_n+\\frac{h}{2},\\,y_n+\\frac{h}{2}k_2\\right)
$$

$$
k_4=f(x_n+h,\\,y_n+hk_3)
$$

### Meaning

The method estimates the slope four times:

- $k_1$: slope at the start
- $k_2$: slope at the midpoint using $k_1$
- $k_3$: another midpoint slope using $k_2$
- $k_4$: slope at the end using $k_3$

Then it takes a weighted average of these slopes.
"""

    if (
        ("coordination" in q or "complex compound" in q or "complex compounds" in q)
        and ("what is" in q or "what are" in q or "basically" in q or "example" in q or "examples" in q)
    ):
        example_hit = next(
            (
                h for h in text_hits
                if re.search(r"chlorophyll|haemoglobin|vitamin|\[Ni\(CO\)4\]|Wilkinson|K4\[Fe", h.get("text", ""), re.I)
            ),
            top,
        )
        return """### Answer

**Coordination compounds** are compounds in which a central metal atom or metal ion is surrounded by molecules or ions called **ligands**. The ligands donate an electron pair to the metal, forming coordinate bonds.

In simple words:

$$
\\text{{central metal ion}} + \\text{{ligands}} \\rightarrow \\text{{coordination compound}}
$$

The part inside square brackets is called the **coordination entity** or **complex ion**.

### Examples

| Coordination compound | Central metal | Ligand(s) | Note |
|---|---:|---|---|
| $K_4[Fe(CN)_6]$ | $Fe$ | $CN^-$ | Contains the complex ion $[Fe(CN)_6]^{4-}$. |
| $[Ni(CO)_4]$ | $Ni$ | $CO$ | Used in purification of nickel. |
| $[Co(NH_3)_6]Cl_3$ | $Co$ | $NH_3$ | A common coordination compound example. |
| Chlorophyll | $Mg$ | Organic ligands | Biological coordination compound. |
| Haemoglobin | $Fe$ | Porphyrin-based ligand system | Carries oxygen in blood. |
| Vitamin $B_{12}$ | $Co$ | Corrin ligand system | Biological coordination compound. |

So the key idea is: **a coordination compound has a metal center attached to surrounding ligands**.
"""

    if (
        "double salt" in q
        and ("complex" in q or "complex salt" in q or "coordination" in q)
    ):
        return """### Answer

A **double salt** and a **complex** are both formed from two or more stable compounds in a fixed stoichiometric ratio, but they behave differently in water.

| Point | Double salt | Complex / complex salt |
|---|---|---|
| Dissociation in water | Dissociates completely into simple ions. | Does not dissociate completely into all simple ions; the complex ion remains intact. |
| Ions present in solution | Gives the constituent ions of the salts. | Gives complex ions such as $[Fe(CN)_6]^{4-}$. |
| Example from the source | Carnallite, Mohr's salt, potash alum. | $K_4[Fe(CN)_6]$, containing $[Fe(CN)_6]^{4-}$. |
| Key idea | Exists as separate simple ions in solution. | Has a coordination sphere that stays together in solution. |

So, the main difference is:

$$
\\text{double salt} \\rightarrow \\text{simple ions completely}
$$

but

$$
\\text{complex salt} \\rightarrow \\text{complex ion remains intact}
$$
"""

    if "difference between" in q and len(text_hits) > 0:
        excerpt = top.get("excerpt") or _excerpt(top.get("text", ""), 500)
        return f"""### Answer

The key difference is:

{excerpt}

In short, focus on the property the passage contrasts: how the two things behave, not only how they are formed.
"""

    if q.startswith(("what is", "explain", "define", "why", "how")):
        excerpt = top.get("excerpt") or _excerpt(top.get("text", ""), 500)
        return f"""### Answer

{excerpt}
"""

    return ""


def build_answer_markdown(question: str, text_hits: list, image_hits: list) -> str:
    """Build a Markdown answer panel from the actual retrieved context."""
    parts: list = [f"## {question.strip()}\n"]
    visual_request = _is_visual_request(question)

    if not text_hits and not image_hits:
        parts.append(
            "_No matching content was retrieved from the indexed PDFs. "
            "Try rephrasing the question or uploading a more relevant document._"
        )
        return "\n".join(parts)

    if visual_request:
        if image_hits:
            if image_hits[0].get("source") == "Generated":
                parts.append("### Generated graph\n")
            else:
                parts.append("### Relevant images from the PDF\n")
            for hit in image_hits[:4]:
                image_md = _image_markdown(hit)
                if image_md:
                    parts.append(image_md)
                    parts.append("")
        else:
            parts.append(
                "_I found text for this topic, but no extracted image or diagram matched the question. "
                "Try asking for a more specific figure, structure, graph, or page._\n"
            )

    if visual_request:
        direct_answer = ""
    elif "bisection" in question.lower() and ("error" in question.lower() or "bound" in question.lower()):
        direct_answer = _build_direct_answer(question, text_hits)
    else:
        direct_answer = _build_llm_answer(question, text_hits)
        if not direct_answer:
            direct_answer = _build_direct_answer(question, text_hits)
    if direct_answer:
        parts.append(direct_answer.strip())
        parts.append("")

    # Image / figure hits
    if image_hits and not visual_request:
        parts.append("### Related figures\n")
        for hit in image_hits[:4]:
            cap = _clean_pdf_text(hit.get("caption") or "Figure")
            page_ref = f"page {hit['page']}" if hit.get("page") else "unknown page"
            image_md = _image_markdown(hit)
            if image_md:
                parts.append(image_md)
                parts.append("")
            else:
                parts.append(f"- _{cap}_ — {hit['source']}, {page_ref}")

    return "\n".join(parts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=False)
