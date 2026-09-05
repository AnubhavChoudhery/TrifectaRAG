"""
Agential chat loop for Trifecta Tutor.

The model decides when to search the textbook index, pull figures/tables,
or search the web.  Each tool returns a short payload so Ollama is not
fed a 6k-character PDF dump in a single generate() call (that is what
was timing out).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

logger = logging.getLogger(__name__)

MAX_ROUNDS = int(os.getenv("TRIFECTA_AGENT_ROUNDS", "6"))
CHAT_TIMEOUT = float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "150"))

_TOOL_TAG_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_QWEN_FN_RE = re.compile(
    r"<function=([A-Za-z_][\w]*)>(.*?)</function>",
    re.DOTALL,
)
_QWEN_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z_][\w]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_QWEN_NAMED_CALL_RE = re.compile(
    r"<tool_call>\s*([A-Za-z_][\w]*)\s*(?:```(?:json)?\s*)?(\{.*?\})?(?:\s*```)?\s*</tool_call>",
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_VISUAL_HINT = re.compile(
    r"\b(figure|fig\.|graph|plot|diagram|table|image|picture|illustration|heatmap|chart)\b",
    re.I,
)
_TEXTBOOK_HINT = re.compile(
    r"\b(lagrange|newton|bisection|interpolat|quadrature|chebyshev|hilbert|"
    r"runge|ode|finite difference|spline|gauss|cholesky|richardson|"
    r"textbook|notes|according to|theorem|lemma|construct|polynomial|"
    r"numerical analysis|midterm|exam)\b",
    re.I,
)


# ── Tool schemas (Ollama / OpenAI style) ─────────────────────────────────────

_SEARCH_CORPUS = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the indexed textbook/PDF (vector + BM25 + knowledge graph). "
            "Use this for definitions, theorems, formulas, worked examples, and "
            "exam-style questions that come from the uploaded notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query, e.g. 'Lagrange interpolation polynomial degree 1'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to return (1-6). Default 4.",
                },
            },
            "required": ["query"],
        },
    },
}

_SEARCH_VISUALS = {
    "type": "function",
    "function": {
        "name": "search_visuals",
        "description": (
            "Find figures, graphs, plots, diagrams, or tables in the indexed PDF. "
            "Call this when a picture, graph, or table would help the student "
            "(or when they explicitly ask to see one)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, e.g. 'bisection method figure' or 'Hilbert matrix table'",
                },
            },
            "required": ["query"],
        },
    },
}

_GET_PAGE = {
    "type": "function",
    "function": {
        "name": "get_page",
        "description": "Read one specific textbook page after you already know the page number.",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "1-based page number"},
                "source": {
                    "type": "string",
                    "description": "PDF stem, default Numerical_Analysis",
                },
            },
            "required": ["page"],
        },
    },
}

_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web (DuckDuckGo + Wikipedia). Use for current events, "
            "papers, citations, or topics not in the uploaded PDF."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query"},
            },
            "required": ["query"],
        },
    },
}

_FETCH_URL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch a web page and return a plain-text excerpt. Use after web_search.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http(s) URL to fetch"},
            },
            "required": ["url"],
        },
    },
}

ALL_TOOLS = [_SEARCH_CORPUS, _SEARCH_VISUALS, _GET_PAGE, _WEB_SEARCH, _FETCH_URL]

UNIFIED_PROMPT = """You are Trifecta Tutor — one agent for study, math, code, and research.

Decide which tools to use. Do not wait for a mode:
- search_corpus: indexed library (hybrid HNSW + BM25 + knowledge graph + MMR). Homework, theorems, formulas.
- search_visuals: figures, graphs, tables from those PDFs. Use when a picture would help.
- get_page: read one textbook page after you know the number.
- web_search / fetch_url: papers, current events, anything not in the library.

Exam / homework workflow:
1. search_corpus with the method name (e.g. "Lagrange interpolation").
2. search_visuals only if a figure or table would clarify the construction.
3. SOLVE the student's problem. Cite source + page for retrieved claims.
4. Markdown only. LaTeX MUST use $inline$ and $$display$$. Never write \\( \\), \\[ \\], or [equation] brackets.

If the library has nothing relevant, say so and solve from first principles or the web."""

MODE_TOOLS: dict[str, list] = {
    "agent": ALL_TOOLS,
    "study": ALL_TOOLS,
    "research": ALL_TOOLS,
    "math": ALL_TOOLS,
    "code": ALL_TOOLS,
    "general": ALL_TOOLS,
}

MODE_PROMPTS: dict[str, str] = {
    "agent": UNIFIED_PROMPT,
    "study": UNIFIED_PROMPT,
    "research": UNIFIED_PROMPT,
    "math": UNIFIED_PROMPT,
    "code": UNIFIED_PROMPT,
    "general": UNIFIED_PROMPT,
}


def _ollama() -> tuple[str, str]:
    import api as api_mod

    return api_mod.OLLAMA_URL.rstrip("/"), api_mod.OLLAMA_MODEL


def _chat(messages: list[dict], tools: list | None) -> dict:
    url, model = _ollama()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 6144,
            "num_predict": 900,
        },
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama chat failed: {exc}") from exc


def _parse_tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    out = []
    for call in calls:
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name")
        raw_args = fn.get("arguments", call.get("arguments") or {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                raw_args = {}
        if name:
            out.append({"name": name, "arguments": raw_args or {}})

    content = message.get("content") or ""
    for match in _TOOL_TAG_RE.finditer(content):
        try:
            blob = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = blob.get("name") or blob.get("tool")
        args = blob.get("arguments") or blob.get("parameters") or {}
        if name:
            out.append({"name": name, "arguments": args})
    for match in _QWEN_FN_RE.finditer(content):
        name, body = match.group(1), match.group(2)
        args = {p.group(1): p.group(2).strip() for p in _QWEN_PARAM_RE.finditer(body)}
        if name:
            out.append({"name": name, "arguments": args})
    for match in _QWEN_NAMED_CALL_RE.finditer(content):
        name, raw = match.group(1), match.group(2)
        args = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        if name and not any(c["name"] == name and c["arguments"] == args for c in out):
            out.append({"name": name, "arguments": args})
    return out


def _focus_query(question: str) -> str:
    q = " ".join((question or "").split())
    named = re.search(
        r"(Lagrange interpolation|Newton(?:-Cotes|-Raphson)?|bisection|"
        r"Chebyshev|Hermite interpolation|divided difference|"
        r"Gaussian quadrature|Runge[- ]Kutta|finite difference|"
        r"piecewise linear|interpolating polynomial)[^\n.]{0,48}",
        q,
        re.I,
    )
    if named:
        return named.group(0).strip()
    return q[:240]


_WEB_HINT = re.compile(
    r"\b(search the web|online|paper|arxiv|doi|cite|citation|survey|recent|"
    r"according to wikipedia|look up|latest|news)\b",
    re.I,
)


def _forced_tool_calls(mode: str, question: str, tools_used: list[str]) -> list[dict]:
    """If the model skips tools, still retrieve so exam/research questions are grounded."""
    forced: list[dict] = []
    names = set(tools_used)
    want_corpus = _TEXTBOOK_HINT.search(question) or len(question) > 160
    if want_corpus and "search_corpus" not in names:
        forced.append({
            "name": "search_corpus",
            "arguments": {"query": _focus_query(question), "top_k": 4},
        })
    if _WEB_HINT.search(question) and "web_search" not in names:
        forced.append({
            "name": "web_search",
            "arguments": {"query": _focus_query(question)},
        })
    if _VISUAL_HINT.search(question) and "search_visuals" not in names:
        forced.append({
            "name": "search_visuals",
            "arguments": {"query": _focus_query(question)},
        })
    return forced


def _allowed_sources():
    import api as api_mod

    fn = getattr(api_mod, "allowed_sources", None)
    return fn() if callable(fn) else None


def _tool_search_corpus(query: str, top_k: int = 4) -> tuple[str, list]:
    import api as api_mod
    from trifecta.retrieve import format_passages, hybrid_search

    engine = api_mod.get_client()
    if engine.size == 0:
        return "Library is empty. Index a PDF or DOCX from the Library dashboard.", []

    hits = hybrid_search(
        engine,
        query,
        top_k=max(1, min(int(top_k or 4), 6)),
        allowed_sources=_allowed_sources(),
    )
    text_hits = [h for h in hits if h.get("modality") != "IMAGE" and h.get("text")]
    if not text_hits:
        return f"No relevant passages in the active library for {query!r}.", [h.get("raw") for h in hits if h.get("raw")]

    ranked = api_mod._rank_text_hits(
        query,
        [{"source": h.get("source"), "page": h.get("page"), "text": h.get("text"), "score": h.get("score")} for h in text_hits],
    ) or text_hits
    # Keep MMR order but prefer lexical-ranked excerpts when available.
    by_key = {(h.get("source"), h.get("page"), (h.get("text") or "")[:80]): h for h in text_hits}
    ordered = []
    for item in ranked:
        key = (item.get("source"), item.get("page"), (item.get("text") or "")[:80])
        ordered.append(by_key.get(key) or item)
    if not ordered:
        ordered = text_hits
    return format_passages(ordered[:4], api_mod._excerpt), [h.get("raw") or h for h in hits]


def _tool_search_visuals(query: str) -> tuple[str, list, list]:
    import api as api_mod

    engine = api_mod.get_client()
    text_hits: list = []
    image_hits: list = []
    if engine.size:
        raw = engine.query(text=query, top_k=12)
        results = engine.get_results(raw)
        allowed = _allowed_sources()
        for r in results:
            meta = r.get("metadata") or {}
            src = meta.get("source") or "PDF"
            if allowed is not None and src not in allowed:
                continue
            hit = {
                "source": meta.get("source") or "PDF",
                "page": meta.get("page"),
                "text": meta.get("full_text") or meta.get("text_preview") or meta.get("caption") or "",
                "score": r.get("score", 0.0),
                "caption": meta.get("caption"),
                "image_path": meta.get("image_path"),
                "type": meta.get("type"),
            }
            if r.get("modality") == "IMAGE":
                image_hits.append(hit)
            elif hit["text"]:
                text_hits.append(hit)
        text_hits = api_mod._rank_text_hits(query, text_hits)[:4]

    text_hits = api_mod._merge_text_hits(
        api_mod._scan_uploaded_pdfs_for_visual_hits(query),
        text_hits,
    )[:5]
    image_hits = api_mod._expand_visual_hits(image_hits, text_hits, query)
    generated = api_mod._generated_graph_hit(query)
    if generated:
        image_hits = api_mod._dedupe_image_hits([*image_hits, generated])
    visuals = api_mod._visuals_payload(image_hits[:6])
    if not visuals:
        return "No matching figure, graph, or table was found in the PDF.", [], []

    lines = []
    for v in visuals:
        lines.append(
            f"- {v.get('kind', 'figure')} | {v.get('source')} page {v.get('page')}: {v.get('caption')}\n"
            f"  markdown: {api_mod._image_markdown({'image_path': _path_from_visual(v), 'caption': v.get('caption'), 'page': v.get('page'), 'source': v.get('source')})}"
        )
    return "Found visuals. Include the markdown image tags in your answer if they help.\n" + "\n".join(lines), visuals, image_hits


def _attachment_context(attachments: list[dict] | None) -> str:
    if not attachments:
        return ""
    parts = ["Attached files for this turn:"]
    for item in attachments:
        name = item.get("name") or item.get("type") or "file"
        kind = item.get("type") or "file"
        text = (item.get("text") or "").strip()
        path = item.get("image_path") or ""
        similar = item.get("similar") or ""
        parts.append(f"### {name} ({kind})")
        if path:
            parts.append(f"Saved image path: {path}")
        if similar:
            parts.append(similar)
        if text:
            parts.append(text[:3500])
    return "\n".join(parts)


def _path_from_visual(visual: dict) -> str:
    url = visual.get("url") or ""
    if "path=" in url:
        return urllib.parse.unquote(url.split("path=", 1)[1])
    return ""


def _finalize_answer(answer: str, visuals: list) -> str:
    """Keep retrieved figures, but repair URLs the model often mangles."""
    import api as api_mod

    answer = answer or ""
    answer = re.sub(r"\]\(https?://image\?path=", "](/image?path=", answer)

    def _encode_path(match: re.Match) -> str:
        raw = match.group(1)
        if raw.startswith(("C:", "c:")) or "\\" in raw:
            return f"](/image?path={urllib.parse.quote(raw, safe='')})"
        return match.group(0)

    answer = re.sub(r"\]\(/image\?path=([^)]+)\)", _encode_path, answer)

    if visuals and not re.search(r"!\[[^\]]*\]\(/image\?path=", answer):
        answer = re.sub(r"!\[([^\]]*)\]\([^)]+\)\s*", "", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        parts = ["\n\n### Retrieved figures / tables\n"]
        for hit in visuals[:4]:
            md = api_mod._image_markdown({
                "image_path": _path_from_visual(hit),
                "caption": hit.get("caption") or hit.get("label"),
                "page": hit.get("page"),
                "source": hit.get("source"),
            })
            if md:
                parts.append(md)
        answer += "".join(parts)
    return answer.strip()


def _tool_get_page(page: int, source: str | None = None) -> str:
    import api as api_mod

    engine = api_mod.get_client()
    sources = engine.list_sources() if hasattr(engine, "list_sources") else []
    src = source or (sources[0] if sources else "Numerical_Analysis")
    gids = engine.get_page_chunks(src, int(page))
    if not gids:
        return f"No chunks indexed for {src} page {page}."
    parts = []
    for gid in gids:
        node = engine.get_node(gid)
        meta = node.get("metadata") or {}
        if node.get("modality") == "IMAGE":
            cap = meta.get("caption") or "figure"
            path = meta.get("image_path")
            parts.append(f"[IMAGE] {cap} path={path}")
            continue
        text = api_mod._clean_pdf_text(meta.get("full_text") or meta.get("text_preview") or "")
        if text:
            parts.append(text[:1800])
    return "\n\n".join(parts) or f"Page {page} had no readable text."


def _web_search(query: str) -> str:
    bits: list[str] = []
    wiki = _wikipedia_search(query)
    if wiki:
        bits.append(wiki)
    ddg = _duckduckgo_search(query)
    if ddg:
        bits.append(ddg)
    return "\n\n".join(bits) or f"No web results for {query!r}."


def _wikipedia_search(query: str) -> str:
    api = (
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        })
    )
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "TrifectaTutor/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            titles, descs, urls = payload[1], payload[2], payload[3]
    except Exception as exc:
        logger.info("Wikipedia search failed: %s", exc)
        return ""
    if not titles:
        return ""
    lines = ["Wikipedia:"]
    for title, desc, url in zip(titles, descs, urls):
        lines.append(f"- {title}: {desc} ({url})")
    return "\n".join(lines)


def _duckduckgo_search(query: str) -> str:
    instant = _duckduckgo_instant(query)
    html_hits = _duckduckgo_html(query)
    bits = [part for part in (instant, html_hits) if part]
    return "\n\n".join(bits)


def _duckduckgo_instant(query: str) -> str:
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    })
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrifectaTutor/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.info("DuckDuckGo instant search failed: %s", exc)
        return ""

    lines: list[str] = []
    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = data.get("AbstractURL") or ""
    heading = data.get("Heading") or query
    if abstract:
        lines.append(f"- {heading}: {abstract[:400]} ({abstract_url})")
    for topic in (data.get("RelatedTopics") or [])[:4]:
        if not isinstance(topic, dict):
            continue
        text = (topic.get("Text") or "").strip()
        first = topic.get("FirstURL") or ""
        if text:
            lines.append(f"- {text[:220]} ({first})")
    return ("Web:\n" + "\n".join(lines)) if lines else ""


def _duckduckgo_html(query: str) -> str:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TrifectaTutor/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.info("DuckDuckGo HTML search failed: %s", exc)
        return ""

    results = []
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    ):
        href, title = m.group(1), m.group(2)
        title = unescape(_TAG_RE.sub("", title)).strip()
        href = unescape(href)
        if "uddg=" in href:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = parsed.get("uddg", [href])[0]
        if title and href:
            results.append(f"- {title} ({href})")
        if len(results) >= 5:
            break
    return ("Web:\n" + "\n".join(results)) if results else ""


def _fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrifectaTutor/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read(80_000).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Failed to fetch {url}: {exc}"
    text = unescape(_TAG_RE.sub(" ", raw))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2500] or "Page had no readable text."


def _run_tool(name: str, args: dict) -> tuple[str, list, list]:
    """Return (text, visuals, sources)."""
    try:
        if name == "search_corpus":
            text, sources = _tool_search_corpus(
                str(args.get("query") or ""),
                int(args.get("top_k") or 4),
            )
            return text, [], sources
        if name == "search_visuals":
            text, visuals, _hits = _tool_search_visuals(str(args.get("query") or ""))
            return text, visuals, []
        if name == "get_page":
            return _tool_get_page(int(args.get("page") or 0), args.get("source")), [], []
        if name == "web_search":
            return _web_search(str(args.get("query") or "")), [], []
        if name == "fetch_url":
            return _fetch_url(str(args.get("url") or "")), [], []
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return f"Tool {name} failed: {exc}", [], []
    return f"Unknown tool {name}", [], []


def run_agent(
    mode: str,
    question: str,
    history: list[dict] | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    """
    Run the tool-calling loop.

    Returns {answer_markdown, visuals, sources, tools_used, used_agent}.
    """
    mode = (mode or "agent").lower()
    if mode not in MODE_PROMPTS:
        mode = "agent"

    tools = MODE_TOOLS[mode]
    messages: list[dict] = [{"role": "system", "content": MODE_PROMPTS[mode]}]
    extra = _attachment_context(attachments)
    if extra:
        question = f"{question}\n\n{extra}" if question else extra
    for item in (history or [])[-8:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:4000]})
    if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != question:
        messages.append({"role": "user", "content": question})

    visuals: list = []
    sources: list = []
    tools_used: list[str] = []

    try:
        for _round in range(MAX_ROUNDS):
            body = _chat(messages, tools)
            message = body.get("message") or {}
            calls = _parse_tool_calls(message)
            content = (message.get("content") or "").strip()

            if not calls and _round == 0:
                calls = _forced_tool_calls(mode, question, tools_used)
                if calls:
                    logger.info("Forcing tools %s for mode=%s", [c["name"] for c in calls], mode)
                    content = ""

            if not calls:
                answer = _TOOL_TAG_RE.sub("", content).strip()
                answer = _QWEN_FN_RE.sub("", answer).strip()
                return {
                    "answer_markdown": _finalize_answer(answer, visuals) or "_The model returned an empty answer._",
                    "visuals": visuals,
                    "sources": _serialize_sources(sources),
                    "tools_used": tools_used,
                    "used_agent": True,
                }

            native_calls = message.get("tool_calls") or [
                {"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                for c in calls
            ]
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": native_calls,
            })
            for call in calls:
                name = call["name"]
                args = call.get("arguments") or {}
                tools_used.append(name)
                logger.info("Agent tool %s(%s)", name, args)
                text, extra_visuals, extra_sources = _run_tool(name, args)
                visuals.extend(extra_visuals)
                sources.extend(extra_sources)
                messages.append({
                    "role": "tool",
                    "name": name,
                    "tool_name": name,
                    "content": text[:4000],
                })
    except Exception as exc:
        logger.warning("Agent loop failed (%s); falling back to compact retrieve+answer", exc)

    return _fallback_answer(mode, question, visuals, sources, tools_used)


def _serialize_sources(raw: list) -> list:
    out = []
    seen = set()
    for r in raw:
        if not isinstance(r, dict):
            continue
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        gid = r.get("global_id")
        source = r.get("source") or meta.get("source")
        page = r.get("page") if r.get("page") is not None else meta.get("page")
        key = (gid, source, page)
        if key in seen:
            continue
        seen.add(key)
        preview = r.get("text_preview") or r.get("text") or meta.get("text_preview") or ""
        out.append({
            "global_id": gid,
            "score": r.get("score"),
            "modality": r.get("modality"),
            "source": source,
            "page": page,
            "text_preview": preview[:240],
            "image_path": r.get("image_path") or meta.get("image_path"),
            "retrieval": r.get("retrieval") or "HNSW+BM25+KG+MMR",
        })
    return out


def _fallback_answer(mode: str, question: str, visuals: list, sources: list, tools_used: list) -> dict:
    """One compact retrieve + one short chat turn if the agent loop dies/times out."""
    context = ""
    extra_sources = []
    try:
        context, extra_sources = _tool_search_corpus(question, 3)
        sources = sources + extra_sources
        tools_used = [*tools_used, "search_corpus"]
    except Exception:
        context = ""

    prompt = (
        f"{MODE_PROMPTS.get(mode, MODE_PROMPTS['general'])}\n\n"
        f"Student question:\n{question}\n\n"
        f"Retrieved notes (may be incomplete):\n{context[:2200]}\n\n"
        "Write the final Markdown answer now. Solve the problem; do not stall."
    )
    try:
        body = _chat(
            [
                {"role": "system", "content": "Answer concisely with LaTeX. Solve the asked problem."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        answer = ((body.get("message") or {}).get("content") or "").strip()
    except Exception as exc:
        logger.warning("Fallback chat also failed: %s", exc)
        answer = (
            "I could not reach the local model in time. "
            "Retrieved notes are below — use them to continue, or retry.\n\n"
            + (context[:1800] if context else "_No passages retrieved._")
        )

    answer = _finalize_answer(answer, visuals)

    return {
        "answer_markdown": answer,
        "visuals": visuals,
        "sources": _serialize_sources(sources),
        "tools_used": tools_used,
        "used_agent": False,
    }
