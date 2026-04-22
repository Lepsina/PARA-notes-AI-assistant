"""Tests for obsassist.embeddings (build/update pipeline)."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from obsassist.config import Config, EmbeddingsConfig
from obsassist.embeddings import build_embeddings, update_embeddings


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
    cfg.embeddings = EmbeddingsConfig(
        model="test-model",
        chunk_size=200,
        chunk_overlap=20,
        batch_size=4,
    )
    return cfg


def _dummy_embed(texts: list[str]) -> list[list[float]]:
    """Return a deterministic fake 4-dim vector for each text."""
    return [[float(len(t) % 10), 0.5, 0.1, 0.9] for t in texts]


def _row_count(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(str(db_path))
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# build_embeddings
# ---------------------------------------------------------------------------


class TestBuildEmbeddings:
    def test_creates_chunks_table(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Hello\n\nWorld.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert _row_count(idx, "chunks") >= 1

    def test_creates_embeddings_table(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Hello\n\nWorld.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert _row_count(idx, "embeddings") >= 1

    def test_chunks_and_embeddings_count_match(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "a.md", "# A\n\nText A.")
        _write_note(vault, "b.md", "# B\n\nText B.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert _row_count(idx, "chunks") == _row_count(idx, "embeddings")

    def test_full_rebuild_clears_old_data(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Title\n\nContent.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        first_count = _row_count(idx, "chunks")

        # Rebuild — count should be the same (not doubled)
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert _row_count(idx, "chunks") == first_count

    def test_vector_stored_as_blob(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Title\n\nContent.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        conn = sqlite3.connect(str(idx))
        blob = conn.execute("SELECT vector FROM embeddings LIMIT 1").fetchone()[0]
        conn.close()
        assert isinstance(blob, bytes)
        # Should decode as 4 floats (our dummy returns 4-dim vectors)
        vals = struct.unpack("4f", blob)
        assert len(vals) == 4

    def test_excluded_paths_not_indexed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Note\n\nContent.")
        _write_note(vault, "Templates/tmpl.md", "# Template")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        conn = sqlite3.connect(str(idx))
        paths = {r[0] for r in conn.execute("SELECT path FROM chunks")}
        conn.close()
        assert not any(p.startswith("Templates/") for p in paths)

    def test_returns_embedded_count(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "a.md", "# A\n\nText.")
        _write_note(vault, "b.md", "# B\n\nText.")
        idx = tmp_path / "index.sqlite"
        embedded, skipped = build_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert embedded >= 2
        assert skipped == 0


# ---------------------------------------------------------------------------
# update_embeddings
# ---------------------------------------------------------------------------


class TestUpdateEmbeddings:
    def test_new_file_indexed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "a.md", "# A\n\nText A.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)

        _write_note(vault, "b.md", "# B\n\nText B.")
        updated, deleted, skipped = update_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert updated == 1
        assert deleted == 0

        conn = sqlite3.connect(str(idx))
        paths = {r[0] for r in conn.execute("SELECT DISTINCT path FROM chunks")}
        conn.close()
        assert "b.md" in paths

    def test_deleted_file_removed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note_a = _write_note(vault, "a.md", "# A\n\nText A.")
        _write_note(vault, "b.md", "# B\n\nText B.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)

        note_a.unlink()
        updated, deleted, skipped = update_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert deleted == 1

        conn = sqlite3.connect(str(idx))
        paths = {r[0] for r in conn.execute("SELECT DISTINCT path FROM chunks")}
        conn.close()
        assert "a.md" not in paths
        assert "b.md" in paths

    def test_unchanged_file_skipped(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Title\n\nContent.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)

        embed_calls: list[int] = []

        def counting_embed(texts: list[str]) -> list[list[float]]:
            embed_calls.append(len(texts))
            return _dummy_embed(texts)

        updated, deleted, skipped = update_embeddings(
            vault, idx, _cfg(vault), counting_embed
        )
        assert updated == 0
        assert embed_calls == []  # No embedding calls for unchanged file

    def test_changed_file_reembedded(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", "# Old\n\nOld content.")
        idx = tmp_path / "index.sqlite"
        build_embeddings(vault, idx, _cfg(vault), _dummy_embed)

        import os
        import time

        time.sleep(0.01)
        note.write_text("# New\n\nNew content.", encoding="utf-8")
        os.utime(note, None)

        updated, deleted, skipped = update_embeddings(vault, idx, _cfg(vault), _dummy_embed)
        assert updated == 1

        conn = sqlite3.connect(str(idx))
        heading = conn.execute(
            "SELECT heading FROM chunks WHERE path='note.md' ORDER BY chunk_index LIMIT 1"
        ).fetchone()[0]
        conn.close()
        assert heading == "New"
