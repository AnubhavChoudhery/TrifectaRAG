"""
trifecta.cli — Command-line interface for the TrifectaRAG SDK.

Install:
    pip install trifectarag[pdf]

Usage:
    trifecta ingest notes.pdf --index ./my.index
    trifecta query "Newton's method" --index ./my.index
    trifecta info --index ./my.index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from trifecta._version import __version__

_LOG = logging.getLogger("trifecta.cli")

DEFAULT_INDEX = os.environ.get("TRIFECTA_INDEX", "trifecta.index")


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _die(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _index_stem(path: str | Path) -> str:
    raw = str(path)
    for suffix in (".snap.gz", ".meta.gz", ".gz", ".meta", ".trifecta"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    return raw


def index_exists(path: str | Path) -> bool:
    stem = _index_stem(path)
    return Path(stem + ".trifecta").exists() or Path(stem + ".snap.gz").exists()


def _parse_pages(spec: str) -> range:
    """Parse ``START:END`` (0-based, end exclusive) or a single integer."""
    spec = spec.strip()
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s)
        if end <= start:
            raise ValueError(f"empty page range: {spec!r}")
        return range(start, end)
    page = int(spec)
    return range(page, page + 1)


def _dump(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
        return
    if isinstance(obj, str):
        print(obj)
        return
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _load_or_create(index: str, device: str):
    from trifecta.client import TrifectaClient

    if index_exists(index):
        _LOG.info("Loading index %s", index)
        return TrifectaClient.from_snapshot(index, device=device)
    return TrifectaClient(device=device)


def _save(client, index: str) -> None:
    Path(index).parent.mkdir(parents=True, exist_ok=True)
    client.save_snapshot(index)
    stem = _index_stem(index)
    print(f"saved  {stem}.trifecta + {stem}.meta.gz  ({client.size} chunks)")


def _print_hit(rank: int, row: dict[str, Any]) -> None:
    meta = row.get("metadata") or {}
    page = meta.get("page", "?")
    src = meta.get("source", "?")
    kind = row.get("modality", "?")
    print(
        f"  #{rank}  gid={row.get('global_id')}  [{kind}]  "
        f"score={float(row.get('score') or 0):.6f}  "
        f"source={src}  page={page}"
    )
    if kind == "IMAGE":
        if meta.get("image_path"):
            print(f"       image: {meta['image_path']}")
        if meta.get("caption"):
            print(f"       caption: {meta['caption']}")
        return
    body = (meta.get("full_text") or meta.get("text_preview") or "").strip()
    if body:
        preview = body.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:217] + "..."
        print(f"       {preview}")


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_version(_args: argparse.Namespace) -> int:
    extra = ""
    try:
        from trifecta import trifecta_py as tr

        extra = f"  engine={tr.TrifectaEngine.__name__}"
    except Exception:
        extra = "  engine=not-built"
    print(f"trifectarag {__version__}{extra}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        from trifecta.pdf_ingest import PDFIngestor
    except Exception as exc:
        _die(f"PDF support is missing ({exc}). Install with: pip install trifectarag[pdf]")

    pdfs = [Path(p) for p in args.pdfs]
    missing = [p for p in pdfs if not p.is_file()]
    if missing:
        _die("file not found: " + ", ".join(str(p) for p in missing))

    page_range = None
    if args.pages:
        try:
            page_range = _parse_pages(args.pages)
        except ValueError as exc:
            _die(str(exc))

    extract_dir = args.extract_dir or str(Path(_index_stem(args.index)).parent / "extracted")
    client = _load_or_create(args.index, args.device)
    ingestor = PDFIngestor(
        client,
        mode=args.mode,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_img_px=args.min_img_px,
    )

    if len(pdfs) == 1:
        stats = ingestor.ingest_pdf(
            str(pdfs[0]),
            output_dir=extract_dir,
            page_range=page_range,
        )
    else:
        ranges = None
        if page_range is not None:
            ranges = {p.stem: page_range for p in pdfs}
        stats = ingestor.ingest_pdfs(
            [str(p) for p in pdfs],
            output_dir=extract_dir,
            page_ranges=ranges,
        )

    payload = {
        "index": args.index,
        "pages": stats.get("pages", 0),
        "text_chunks": stats.get("text_chunks", 0),
        "images": stats.get("images", 0),
        "kg_edges": stats.get("kg_edges", 0),
        "engine_size": client.size,
        "page_count": client.page_count,
        "sources": client.list_sources(),
    }
    if args.json:
        _dump(payload, True)
    else:
        print("ingest complete")
        print(f"  pages        {payload['pages']}")
        print(f"  text chunks  {payload['text_chunks']}")
        print(f"  images       {payload['images']}")
        print(f"  kg edges     {payload['kg_edges']}")
        print(f"  engine size  {payload['engine_size']}")
        print(f"  pages indexed {payload['page_count']}")
    _save(client, args.index)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    if not index_exists(args.index):
        _die(f"index not found: {args.index}  (run: trifecta ingest <pdf> --index {args.index})")

    from trifecta.client import TrifectaClient

    client = TrifectaClient.from_snapshot(args.index, device=args.device)
    raw = client.query(
        text=args.text,
        image=args.image,
        top_k=args.top_k,
        use_hnsw=not args.no_hnsw,
        use_bm25=not args.no_bm25,
    )
    rows = client.get_results(raw)
    if args.json:
        _dump(rows, True)
        return 0
    if not rows:
        print("no results")
        return 0
    print(f"{len(rows)} result(s)  index={args.index}  size={client.size}")
    for i, row in enumerate(rows, start=1):
        _print_hit(i, row)
    return 0


def cmd_add_text(args: argparse.Namespace) -> int:
    client = _load_or_create(args.index, args.device)
    metadata: dict[str, Any] = {}
    if args.source:
        metadata["source"] = args.source
    if args.page is not None:
        metadata["page"] = args.page
    gid = client.add_document(args.text, metadata=metadata or None)
    if args.json:
        _dump({"global_id": gid, "size": client.size}, True)
    else:
        print(f"added text  gid={gid}")
    _save(client, args.index)
    return 0


def cmd_add_image(args: argparse.Namespace) -> int:
    path = Path(args.image)
    if not path.is_file():
        _die(f"image not found: {path}")
    client = _load_or_create(args.index, args.device)
    metadata: dict[str, Any] = {}
    if args.source:
        metadata["source"] = args.source
    if args.page is not None:
        metadata["page"] = args.page
    gid = client.add_image(path, caption=args.caption or "", metadata=metadata or None)
    if args.json:
        _dump({"global_id": gid, "size": client.size}, True)
    else:
        print(f"added image  gid={gid}")
    _save(client, args.index)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    if not index_exists(args.index):
        _die(f"index not found: {args.index}")
    from trifecta.client import TrifectaClient

    client = TrifectaClient.from_snapshot(args.index, device=args.device)
    payload = {
        "index": args.index,
        "version": __version__,
        "size": client.size,
        "dim": client.dim,
        "device": client.device,
        "page_count": client.page_count,
        "sources": client.list_sources(),
    }
    if args.json:
        _dump(payload, True)
        return 0
    print(f"trifectarag {__version__}")
    print(f"  index       {args.index}")
    print(f"  chunks      {payload['size']}")
    print(f"  dim         {payload['dim']}")
    print(f"  device      {payload['device']}")
    print(f"  pages       {payload['page_count']}")
    print(f"  sources     {', '.join(payload['sources']) or '(none)'}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    if not index_exists(args.index):
        _die(f"index not found: {args.index}")
    from trifecta.client import TrifectaClient

    client = TrifectaClient.from_snapshot(args.index, device=args.device)
    sources = client.list_sources()
    if args.json:
        _dump(sources, True)
        return 0
    if not sources:
        print("(no sources)")
        return 0
    for name in sources:
        pages = client.list_pages(name)
        print(f"  {name}  ({len(pages)} pages)")
    return 0


def cmd_pages(args: argparse.Namespace) -> int:
    if not index_exists(args.index):
        _die(f"index not found: {args.index}")
    from trifecta.client import TrifectaClient

    client = TrifectaClient.from_snapshot(args.index, device=args.device)
    if args.source:
        pages = client.list_pages(args.source)
        payload: Any = {"source": args.source, "pages": pages}
        if args.page is not None:
            gids = client.get_page_chunks(args.source, args.page)
            payload = {"source": args.source, "page": args.page, "global_ids": gids}
    else:
        payload = {
            src: client.list_pages(src) for src in client.list_sources()
        }
    if args.json:
        _dump(payload, True)
        return 0
    if isinstance(payload, dict) and "global_ids" in payload:
        print(
            f"{payload['source']} page {payload['page']}: "
            f"{payload['global_ids'] or '(empty)'}"
        )
        return 0
    if isinstance(payload, dict) and "pages" in payload:
        print(f"{payload['source']}: {payload['pages']}")
        return 0
    for src, pages in payload.items():
        print(f"  {src}: {pages}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    if not index_exists(args.index):
        _die(f"index not found: {args.index}")
    from trifecta.client import TrifectaClient

    client = TrifectaClient.from_snapshot(args.index, device=args.device)
    try:
        info = client.get_node(args.gid)
    except Exception as exc:
        _die(str(exc))
    loc = client.get_chunk_page(args.gid)
    info["page_index"] = list(loc) if loc else None
    if args.json:
        _dump(info, True)
        return 0
    meta = info.get("metadata") or {}
    print(f"gid={info['global_id']}  modality={info['modality']}")
    if loc:
        print(f"  page-index  source={loc[0]}  page={loc[1]}")
    body = (meta.get("full_text") or meta.get("text_preview") or meta.get("caption") or "")
    if body:
        print(body)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from trifecta.mcp_server import run_server
    except Exception as exc:
        _die(f"MCP extra is missing ({exc}). Install with: pip install trifectarag[mcp]")

    client = _load_or_create(args.index, args.device)
    if args.ingest:
        try:
            from trifecta.pdf_ingest import PDFIngestor
        except Exception as exc:
            _die(f"PDF support is missing ({exc}). Install with: pip install trifectarag[pdf]")
        PDFIngestor(client, mode=args.mode).ingest_pdf(
            args.ingest,
            output_dir=args.extract_dir or "extracted",
        )
        _save(client, args.index)
    run_server(client)
    return 0


# ── Parser ────────────────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Snapshot base path (default: {DEFAULT_INDEX} or $TRIFECTA_INDEX)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trifecta",
        description="TrifectaRAG — multi-modal RAG SDK and CLI (HNSW + BM25 + KG).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  trifecta ingest textbook.pdf --index ./na.index --mode page\n"
            "  trifecta query \"Newton method\" --index ./na.index --top-k 5\n"
            "  trifecta info --index ./na.index\n"
            "  trifecta pages --index ./na.index --source Numerical_Analysis --page 36\n"
        ),
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"trifectarag {__version__}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="Print package version")
    p.set_defaults(func=cmd_version)

    p = sub.add_parser("ingest", help="Ingest one or more PDFs into an index")
    _add_common(p)
    p.add_argument("pdfs", nargs="+", help="PDF file(s)")
    p.add_argument("--mode", choices=("page", "classical"), default="page")
    p.add_argument(
        "--pages",
        default=None,
        help="0-based page window START:END (end exclusive), e.g. 11:444",
    )
    p.add_argument("--extract-dir", default=None, help="Directory for extracted figures")
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--min-img-px", type=int, default=60)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="Search an index")
    _add_common(p)
    p.add_argument("text", help="Query text")
    p.add_argument("--image", default=None, help="Optional query image path (late fusion)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-hnsw", action="store_true", help="Disable vector search")
    p.add_argument("--no-bm25", action="store_true", help="Disable keyword search")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("add-text", help="Add a text document to an index")
    _add_common(p)
    p.add_argument("text", help="Document text")
    p.add_argument("--source", default=None)
    p.add_argument("--page", type=int, default=None)
    p.set_defaults(func=cmd_add_text)

    p = sub.add_parser("add-image", help="Add an image to an index")
    _add_common(p)
    p.add_argument("image", help="Image file path")
    p.add_argument("--caption", default="")
    p.add_argument("--source", default=None)
    p.add_argument("--page", type=int, default=None)
    p.set_defaults(func=cmd_add_image)

    p = sub.add_parser("info", help="Show index statistics")
    _add_common(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("sources", help="List indexed sources")
    _add_common(p)
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("pages", help="List pages or chunks on a page")
    _add_common(p)
    p.add_argument("--source", default=None)
    p.add_argument("--page", type=int, default=None)
    p.set_defaults(func=cmd_pages)

    p = sub.add_parser("get", help="Fetch a chunk by global_id")
    _add_common(p)
    p.add_argument("gid", type=int)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("mcp", help="Start the Model Context Protocol server")
    _add_common(p)
    p.add_argument("--ingest", default=None, help="Optional PDF to ingest on startup")
    p.add_argument("--mode", choices=("page", "classical"), default="page")
    p.add_argument("--extract-dir", default=None)
    p.set_defaults(func=cmd_mcp)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
