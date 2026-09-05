"""
Hybrid retrieval helpers: source filter, MMR diversity, provenance labels.

The C++ engine already fuses HNSW + BM25 + KG via RRF. This module reranks
the fused list so exam answers cite diverse pages instead of near-duplicate
chunks, and attaches reproducible (source, page, gid, score) provenance.
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def mmr_rerank(hits: list[dict], k: int = 4, lambda_mult: float = 0.72) -> list[dict]:
    """Maximal Marginal Relevance over already-fused RRF hits."""
    if not hits:
        return []
    k = max(1, min(k, len(hits)))
    scores = [float(h.get("score") or 0.0) for h in hits]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0

    remaining = list(hits)
    selected: list[dict] = []
    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best_i = 0
        best = -1e9
        for i, hit in enumerate(remaining):
            rel = (float(hit.get("score") or 0.0) - lo) / span
            overlap = max(
                _jaccard(hit.get("text") or "", prev.get("text") or "")
                for prev in selected
            )
            if overlap > 0.86:
                continue
            mmr = lambda_mult * rel - (1.0 - lambda_mult) * overlap
            if mmr > best:
                best = mmr
                best_i = i
        selected.append(remaining.pop(best_i))
    return selected


def _hit_from_result(result: dict) -> dict:
    meta = result.get("metadata") or {}
    text = (
        meta.get("full_text")
        or meta.get("text_preview")
        or meta.get("caption")
        or ""
    )
    return {
        "global_id": result.get("global_id"),
        "score": float(result.get("score") or 0.0),
        "modality": result.get("modality"),
        "source": meta.get("source") or "PDF",
        "page": meta.get("page"),
        "text": text,
        "caption": meta.get("caption"),
        "image_path": meta.get("image_path"),
        "type": meta.get("type"),
        "raw": result,
    }


def _active_flags() -> tuple[bool, bool, str]:
    try:
        import api as api_mod

        settings = api_mod.retrieval_settings()
        use_hnsw = bool(settings.get("use_hnsw", True))
        use_bm25 = bool(settings.get("use_bm25", True))
        return use_hnsw, use_bm25, settings.get("label") or _label(use_hnsw, use_bm25)
    except Exception:
        return True, True, "HNSW+BM25+KG+MMR"


def _label(use_hnsw: bool, use_bm25: bool) -> str:
    parts = []
    if use_hnsw:
        parts.append("HNSW")
    if use_bm25:
        parts.append("BM25")
    parts.extend(["KG", "MMR"])
    return "+".join(parts)


def hybrid_search(
    engine: Any,
    query: str,
    top_k: int = 4,
    allowed_sources: set[str] | None = None,
    image: str | None = None,
    fetch: int = 16,
    use_hnsw: bool | None = None,
    use_bm25: bool | None = None,
) -> list[dict]:
    """Query the fused engine, drop disabled sources, then MMR-trim."""
    if engine is None or getattr(engine, "size", 0) == 0:
        return []
    flag_hnsw, flag_bm25, label = _active_flags()
    if use_hnsw is None:
        use_hnsw = flag_hnsw
    if use_bm25 is None:
        use_bm25 = flag_bm25
    k = max(1, min(int(top_k or 4), 8))
    raw = engine.query(
        text=query or None,
        image=image,
        top_k=max(fetch, k * 4),
        use_hnsw=use_hnsw,
        use_bm25=use_bm25,
    )
    hits = [_hit_from_result(r) for r in engine.get_results(raw)]
    for hit in hits:
        hit["retrieval"] = label
        if hit.get("raw") is not None:
            hit["raw"]["retrieval"] = label
    if allowed_sources is not None:
        hits = [
            h for h in hits
            if (h.get("source") or "") in allowed_sources
        ]
    return mmr_rerank(hits, k=k)


def format_passages(hits: list[dict], excerpt_fn) -> str:
    if not hits:
        return "No relevant passages in the active library."
    blocks = []
    for i, hit in enumerate(hits, 1):
        excerpt = excerpt_fn(hit.get("text") or "", 700)
        page = hit.get("page")
        page_s = f"page {page}" if page is not None else "page ?"
        gid = hit.get("global_id")
        score = hit.get("score") or 0.0
        blocks.append(
            f"[{i}] {hit.get('source')}, {page_s} · gid={gid} · rrf={score:.4f} · hybrid={hit.get('retrieval') or 'HNSW+BM25+KG+MMR'}\n"
            f"{excerpt}"
        )
    return "\n\n".join(blocks)


def serialize_sources(hits: list[dict]) -> list[dict]:
    out = []
    for hit in hits:
        out.append({
            "global_id": hit.get("global_id"),
            "score": hit.get("score"),
            "modality": hit.get("modality"),
            "source": hit.get("source"),
            "page": hit.get("page"),
            "text_preview": (hit.get("text") or "")[:240],
            "image_path": hit.get("image_path"),
            "retrieval": hit.get("retrieval") or "HNSW+BM25+KG+MMR",
        })
    return out
