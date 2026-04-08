"""
03_agent_image_response.py — Real RAG loop: TrifectaRAG retrieval + Ollama LLM.

Flow per question:
  1. Query TrifectaRAG (HNSW + BM25 + KG) -> top-5 chunks
  2. Print every retrieved chunk (full text / figure path)
  3. Build a context string and send to Ollama -> print full answer

Model: TRIFECTA_OLLAMA_MODEL env var, default qwen2.5:7b
       Override: $env:TRIFECTA_OLLAMA_MODEL = "qwen3:14b"

Usage:
    python examples/03_agent_image_response.py

Env vars:
    TRIFECTA_RAG_MODE       page | classical  (default: page)
    TRIFECTA_OLLAMA_MODEL   Ollama model name (default: qwen2.5:7b)
    TRIFECTA_TEXTBOOK_PDF   path to PDF (default: data/Numerical_Analysis.pdf)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
sys.path.insert(0, str(_EX.parent))
sys.path.insert(0, str(_EX))

import fitz
import ollama
import os

from textbook_config import (
    DATA_DIR,
    ensure_utf8_stdout,
    ollama_model,
    page_range_for_document,
    print_retrieval_result,
    resolve_pdf_path,
    snapshot_exists,
    snapshot_path,
)
from trifecta import TrifectaClient, PDFIngestor

SYSTEM_PROMPT = (
    "You are a concise numerical analysis tutor. "
    "Answer the student's question using only the context provided. "
    "If figures are referenced, describe what the path suggests they show "
    "and relate it to the mathematical content. "
    "Be precise and cite page numbers where relevant."
)

QUESTIONS = [
    "What is Newton's method for solving nonlinear equations? Show the iteration formula.",
    "Explain the bisection method and its convergence rate.",
    "What is LU factorization and when is it used for solving linear systems?",
    "How does the trapezoidal rule work for numerical integration?",
]


def build_engine(pdf: Path, n: int, mode: str) -> TrifectaClient:
    pr = page_range_for_document(n, pdf)
    snap = snapshot_path(pdf, mode, pr)

    if snapshot_exists(snap):
        print(f"  Loading {mode!r} snapshot: {snap.name}  (run 01 to re-ingest)")
        client = TrifectaClient.from_snapshot(str(snap), device="cpu")
        print(f"  Engine: {client.size} nodes, {client.page_count} pages\n")
        return client

    print(f"  No snapshot found for mode={mode!r}. Ingesting pages {pr.start + 1}..{pr.stop} ({len(pr)} pages)...")
    print("  Tip: run 01_ingest_textbook.py first for fast reloads.\n")
    client = TrifectaClient(device="cpu")
    ingestor = PDFIngestor(client, mode=mode, min_img_px=60)
    stats = ingestor.ingest_pdf(
        str(pdf),
        output_dir=str(DATA_DIR / f"extracted_{mode}"),
        page_range=pr,
    )
    print(
        f"  Engine: {stats['text_chunks']} text chunks, "
        f"{stats['images']} images, {stats['kg_edges']} KG edges\n"
    )
    return client


def build_context(results: list) -> str:
    """Turn retrieved chunks into an LLM-ready context string."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        page = meta.get("page", "?")
        score = r["score"]

        if r["modality"] == "IMAGE":
            ip = meta.get("image_path", "")
            cap = meta.get("caption", "")
            parts.append(
                f"[Context {i} | Figure | page {page} | score {score:.4f}]\n"
                f"  Path: {ip}\n  Caption: {cap}"
            )
        else:
            text = (meta.get("full_text") or meta.get("text_preview") or "").strip()
            parts.append(
                f"[Context {i} | Text | page {page} | score {score:.4f}]\n{text}"
            )
    return "\n\n".join(parts)


def ask_ollama(question: str, context: str, model: str) -> str:
    user_msg = (
        "Use the following context retrieved from the textbook:\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.message.content.strip()
    except ollama.ResponseError as e:
        return f"[Ollama error: {e}]"
    except Exception as e:
        return f"[Unexpected error calling Ollama: {e}]"


def main() -> None:
    ensure_utf8_stdout()
    model = ollama_model()

    mode = os.environ.get("TRIFECTA_RAG_MODE", "page").strip()
    if mode not in ("page", "classical"):
        print(f"  Warning: unknown TRIFECTA_RAG_MODE={mode!r}, defaulting to 'page'")
        mode = "page"

    print(f"TrifectaRAG + Ollama ({model}) — Numerical Analysis RAG demo  (mode={mode!r})\n")
    print("=" * 72)

    pdf = resolve_pdf_path()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
    finally:
        doc.close()

    client = build_engine(pdf, n, mode)

    for q in QUESTIONS:
        print(f"\n{'=' * 72}")
        print(f"USER: {q}")
        print("=" * 72)

        raw = client.query(text=q, top_k=5)
        results = client.get_results(raw)

        print(f"\n--- Retrieved context ({len(results)} chunks) ---")
        for rank, r in enumerate(results, 1):
            print(f"\n  ### Rank {rank}")
            print_retrieval_result(r)

        context = build_context(results)
        print(f"\n--- Ollama ({model}) response ---\n")
        answer = ask_ollama(q, context, model)
        print(answer)
        print()

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
