"""Tests for obsassist.search."""
from __future__ import annotations

from pathlib import Path

import pytest

from obsassist.config import Config
from obsassist.indexer import build_index
from obsassist.search import search

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


def _write_note(vault: Path, rel: str, content: str) -> Path:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _cfg(vault: Path) -> Config:
    cfg = Config()
    cfg.vault_root = str(vault)
    return cfg


def _build(tmp_path: Path) -> tuple[Path, Path]:
    vault = _make_vault(tmp_path)
    _write_note(
        vault,
        "Areas/creativity.md",
        "# Creativity\n\nThoughts on #creativity and self-expression. [[Deep Work]].",
    )
    _write_note(
        vault,
        "Daily/2026-01-01.md",
        "# Daily Note\n\nToday I focused on writing and exercise.",
    )
    _write_note(
        vault,
        "Projects/alpha.md",
        "# Alpha Project\n\nProject goals: deliver MVP by Q1.",
    )
    idx = tmp_path / "index.sqlite"
    build_index(vault, idx, _cfg(vault))
    return vault, idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_empty_list_for_missing_index(self, tmp_path: Path):
        idx = tmp_path / "nonexistent.sqlite"
        results = search(idx, "anything")
        assert results == []

    def test_finds_matching_note(self, tmp_path: Path):
        _, idx = _build(tmp_path)
        results = search(idx, "creativity")
        paths = [r.path for r in results]
        assert any("creativity" in p for p in paths)

    def test_returns_multiple_results(self, tmp_path: Path):
        _, idx = _build(tmp_path)
        # "writing" appears in the Daily note; check we get at least one result
        results = search(idx, "writing")
        assert len(results) >= 1

    def test_no_match_returns_empty(self, tmp_path: Path):
        _, idx = _build(tmp_path)
        results = search(idx, "zzznomatchxxx")
        assert results == []

    def test_result_has_expected_fields(self, tmp_path: Path):
        _, idx = _build(tmp_path)
        results = search(idx, "creativity")
        assert results, "Expected at least one result"
        r = results[0]
        assert r.path
        assert r.title
        # rank is a float (negative in FTS5 — lower is better)
        assert isinstance(r.rank, float)

    def test_snippet_contains_query_term(self, tmp_path: Path):
        _, idx = _build(tmp_path)
        results = search(idx, "creativity")
        assert results
        # The snippet should contain the matched term (possibly highlighted)
        assert "reativity" in results[0].snippet  # catches both "creativity" and "[creativity]"

    def test_limit_respected(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        for i in range(10):
            _write_note(vault, f"note_{i}.md", f"# Note {i}\n\nContent about searchterm.")
        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))
        results = search(idx, "searchterm", limit=3)
        assert len(results) <= 3

    def test_specific_note_found_by_title(self, tmp_path: Path):
        _, idx = _build(tmp_path)
        results = search(idx, "Alpha Project")
        paths = [r.path for r in results]
        assert any("alpha" in p.lower() for p in paths)
