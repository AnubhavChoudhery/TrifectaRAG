"""Package metadata and public-export smoke tests."""

from __future__ import annotations

import trifecta
from trifecta import PDFIngestor, TrifectaClient
from trifecta._version import __version__ as file_version
from trifecta.cli import build_parser, index_exists, _index_stem, _parse_pages


def test_version_exported():
    assert trifecta.__version__ == file_version
    assert isinstance(trifecta.__version__, str)
    assert trifecta.__version__.count(".") >= 2


def test_public_exports():
    assert callable(TrifectaClient)
    assert callable(PDFIngestor)
    assert hasattr(trifecta, "trifecta_py")


def test_lazy_retrieve_exports():
    from trifecta import hybrid_search, mmr_rerank, serialize_sources

    assert callable(hybrid_search)
    assert callable(mmr_rerank)
    assert callable(serialize_sources)


def test_cli_parser_has_core_commands():
    parser = build_parser()
    ns = parser.parse_args(["version"])
    assert ns.command == "version"
    ns = parser.parse_args(["query", "newton method", "--index", "x.index", "--top-k", "3"])
    assert ns.text == "newton method"
    assert ns.top_k == 3
    ns = parser.parse_args(["ingest", "a.pdf", "b.pdf", "--mode", "classical"])
    assert ns.pdfs == ["a.pdf", "b.pdf"]
    assert ns.mode == "classical"


def test_index_stem_strips_known_suffixes():
    assert _index_stem("foo.trifecta") == "foo"
    assert _index_stem("foo.meta.gz") == "foo"
    assert _index_stem("foo.snap.gz") == "foo"
    assert _index_stem("dir/na.index") == "dir/na.index"


def test_parse_pages():
    r = _parse_pages("11:444")
    assert r.start == 11 and r.stop == 444
    single = _parse_pages("7")
    assert list(single) == [7]


def test_index_exists_missing(tmp_path):
    assert index_exists(tmp_path / "nope.index") is False


if __name__ == "__main__":
    tests = [
        test_version_exported,
        test_public_exports,
        test_lazy_retrieve_exports,
        test_cli_parser_has_core_commands,
        test_index_stem_strips_known_suffixes,
        test_parse_pages,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__} -- {exc}")
    raise SystemExit(1 if failed else 0)
