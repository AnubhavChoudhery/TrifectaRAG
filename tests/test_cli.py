"""CLI unit tests — parser + command handlers with a mocked client."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from trifecta.cli import (
    build_parser,
    cmd_add_text,
    cmd_info,
    cmd_query,
    cmd_version,
    main,
)


def test_main_version_exits_zero():
    assert main(["version"]) == 0


def test_cmd_version_prints(capsys_like=None):
    assert cmd_version(SimpleNamespace()) == 0


def test_query_missing_index(tmp_path):
    args = build_parser().parse_args(
        ["query", "newton", "--index", str(tmp_path / "missing.index")]
    )
    try:
        cmd_query(args)
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 1


def test_query_with_mock_client(tmp_path, monkeypatch=None):
    stem = tmp_path / "demo.index"
    (tmp_path / "demo.index.trifecta").write_bytes(b"x")
    (tmp_path / "demo.index.meta.gz").write_bytes(b"x")

    mock_client = MagicMock()
    mock_client.size = 2
    mock_client.query.return_value = [(0, 0.9)]
    mock_client.get_results.return_value = [
        {
            "global_id": 0,
            "score": 0.9,
            "modality": "TEXT",
            "metadata": {"source": "notes", "page": 1, "full_text": "Newton"},
        }
    ]

    args = build_parser().parse_args(
        ["query", "Newton", "--index", str(stem), "--json"]
    )
    with patch("trifecta.client.TrifectaClient.from_snapshot", return_value=mock_client):
        assert cmd_query(args) == 0
    mock_client.query.assert_called_once()


def test_add_text_creates_and_saves(tmp_path):
    mock_client = MagicMock()
    mock_client.size = 1
    mock_client.add_document.return_value = 0

    args = build_parser().parse_args(
        ["add-text", "hello world", "--index", str(tmp_path / "new.index"),
         "--source", "notes", "--page", "3"]
    )
    with patch("trifecta.client.TrifectaClient", return_value=mock_client):
        assert cmd_add_text(args) == 0
    mock_client.add_document.assert_called_once()
    mock_client.save_snapshot.assert_called_once()


def test_info_with_mock(tmp_path):
    (tmp_path / "demo.index.trifecta").write_bytes(b"x")
    mock_client = MagicMock()
    mock_client.size = 4
    mock_client.dim = 512
    mock_client.device = "cpu"
    mock_client.page_count = 2
    mock_client.list_sources.return_value = ["notes"]

    args = build_parser().parse_args(
        ["info", "--index", str(tmp_path / "demo.index"), "--json"]
    )
    with patch("trifecta.client.TrifectaClient.from_snapshot", return_value=mock_client):
        assert cmd_info(args) == 0


if __name__ == "__main__":
    from pathlib import Path
    import tempfile

    passed = 0
    failed = 0

    def run(name, fn):
        global passed, failed
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name} -- {exc}")
            failed += 1

    run("main_version", test_main_version_exits_zero)
    run("cmd_version", test_cmd_version_prints)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run("query_missing", lambda: test_query_missing_index(tmp))
        run("query_mock", lambda: test_query_with_mock_client(tmp))
        run("add_text", lambda: test_add_text_creates_and_saves(tmp))
        run("info", lambda: test_info_with_mock(tmp))

    print(f"\nCLI tests: {passed}/{passed + failed} passed")
    raise SystemExit(1 if failed else 0)
