"""Unit and integration tests for obsassist.indexer."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from obsassist.config import Config
from obsassist.indexer import (
    build_index,
    extract_metadata,
    update_index,
)

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


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    def test_title_from_h1(self):
        content = "# My Great Title\n\nSome body text."
        meta = extract_metadata(Path("note.md"), content)
        assert meta["title"] == "My Great Title"
        assert "My Great Title" in meta["headings"]

    def test_title_fallback_to_filename(self):
        content = "No heading here."
        meta = extract_metadata(Path("my-note.md"), content)
        assert meta["title"] == "my-note"

    def test_headings_extracted(self):
        content = "# H1\n\n## H2\n\n### H3\n"
        meta = extract_metadata(Path("n.md"), content)
        assert "H1" in meta["headings"]
        assert "H2" in meta["headings"]
        assert "H3" in meta["headings"]

    def test_inline_tags(self):
        content = "Some text #creativity and #writing here."
        meta = extract_metadata(Path("n.md"), content)
        assert "creativity" in meta["tags"]
        assert "writing" in meta["tags"]

    def test_yaml_frontmatter_tags(self):
        content = "---\ntags:\n  - productivity\n  - focus\n---\n\nBody."
        meta = extract_metadata(Path("n.md"), content)
        assert "productivity" in meta["tags"]
        assert "focus" in meta["tags"]

    def test_wikilinks(self):
        content = "See [[Deep Work]] and [[Cal Newport|author]]."
        meta = extract_metadata(Path("n.md"), content)
        assert "Deep Work" in meta["links"]
        assert "Cal Newport" in meta["links"]

    def test_frontmatter_not_in_body(self):
        content = "---\ntitle: secret\n---\n\nActual body."
        meta = extract_metadata(Path("n.md"), content)
        assert "secret" not in meta["body"]
        assert "Actual body" in meta["body"]

    def test_wikilink_with_anchor(self):
        content = "See [[MyNote#Section|alias]]."
        meta = extract_metadata(Path("n.md"), content)
        assert "MyNote" in meta["links"]


# ---------------------------------------------------------------------------
# Path exclusion during indexing
# ---------------------------------------------------------------------------


class TestPathExclusionDuringIndex:
    def test_excluded_dirs_not_indexed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "Areas/health.md", "# Health\n\nGood stuff.")
        _write_note(vault, "Resources/article.md", "# Article\n\nReference.")
        _write_note(vault, "Templates/daily.md", "# Template")
        _write_note(vault, "Files/image.md", "# image")
        _write_note(vault, "Excalidraw/diagram.md", "# diagram")
        _write_note(vault, ".obsidian/config.md", "obsidian config")

        idx = tmp_path / "index.sqlite"
        cfg = _cfg(vault)
        build_index(vault, idx, cfg)

        conn = sqlite3.connect(str(idx))
        paths = {row[0] for row in conn.execute("SELECT path FROM documents")}
        conn.close()

        assert "Areas/health.md" in paths
        assert not any(p.startswith("Resources/") for p in paths)
        assert not any(p.startswith("Templates/") for p in paths)
        assert not any(p.startswith("Files/") for p in paths)
        assert not any(p.startswith("Excalidraw/") for p in paths)
        assert not any(p.startswith(".obsidian/") for p in paths)

    def test_only_md_indexed_by_default(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# MD note")
        (vault / "note.txt").write_text("plain text")
        (vault / "note.pdf").write_bytes(b"%PDF")

        idx = tmp_path / "index.sqlite"
        cfg = _cfg(vault)
        build_index(vault, idx, cfg)

        conn = sqlite3.connect(str(idx))
        paths = {row[0] for row in conn.execute("SELECT path FROM documents")}
        conn.close()

        assert "note.md" in paths
        assert "note.txt" not in paths
        assert "note.pdf" not in paths


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_creates_db_file(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Hello\n\nWorld")
        idx = tmp_path / "subdir" / "index.sqlite"
        build_index(vault, idx, _cfg(vault))
        assert idx.exists()

    def test_indexes_all_md_files(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "Daily/2026-01-01.md", "# Daily\n\nToday.")
        _write_note(vault, "Areas/health.md", "# Health\n\nExercise.")
        _write_note(vault, "Projects/alpha/plan.md", "# Plan\n\nSteps.")

        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))

        conn = sqlite3.connect(str(idx))
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert count == 3

    def test_full_rebuild_clears_old_data(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note_a.md", "# A\n\nText A.")
        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))

        # Add a new note and rebuild
        _write_note(vault, "note_b.md", "# B\n\nText B.")
        build_index(vault, idx, _cfg(vault))

        conn = sqlite3.connect(str(idx))
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert count == 2  # not 3 (would be 3 if old rows were kept)

    def test_metadata_stored_correctly(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        content = (
            "---\ntags:\n  - health\n---\n\n"
            "# Fitness Goals\n\n"
            "Work on #strength and [[Running Program]].\n"
        )
        _write_note(vault, "note.md", content)
        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))

        conn = sqlite3.connect(str(idx))
        row = conn.execute(
            "SELECT title, tags, links FROM documents WHERE path='note.md'"
        ).fetchone()
        conn.close()

        assert row[0] == "Fitness Goals"
        tags = json.loads(row[1])
        assert "health" in tags
        assert "strength" in tags
        links = json.loads(row[2])
        assert "Running Program" in links


# ---------------------------------------------------------------------------
# update_index
# ---------------------------------------------------------------------------


class TestUpdateIndex:
    def test_added_file_indexed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note_a.md", "# A\n\nText A.")
        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))

        _write_note(vault, "note_b.md", "# B\n\nText B.")
        added, deleted, skipped = update_index(vault, idx, _cfg(vault))
        assert added == 1
        assert deleted == 0

        conn = sqlite3.connect(str(idx))
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert count == 2

    def test_deleted_file_removed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note_a = _write_note(vault, "note_a.md", "# A\n\nText A.")
        _write_note(vault, "note_b.md", "# B\n\nText B.")
        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))

        note_a.unlink()
        added, deleted, skipped = update_index(vault, idx, _cfg(vault))
        assert deleted == 1

        conn = sqlite3.connect(str(idx))
        paths = {row[0] for row in conn.execute("SELECT path FROM documents")}
        conn.close()
        assert "note_a.md" not in paths
        assert "note_b.md" in paths

    def test_changed_file_reindexed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", "# Old Title\n\nOld content.")
        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))

        # Overwrite with new content (ensure mtime changes)
        import time

        time.sleep(0.01)
        note.write_text("# New Title\n\nNew content.", encoding="utf-8")
        # Force mtime change in case filesystem resolution is low
        import os

        os.utime(note, None)

        added, deleted, skipped = update_index(vault, idx, _cfg(vault))
        assert added == 1

        conn = sqlite3.connect(str(idx))
        title = conn.execute(
            "SELECT title FROM documents WHERE path='note.md'"
        ).fetchone()[0]
        conn.close()
        assert title == "New Title"

    def test_unchanged_file_skipped(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", "# Title\n\nContent.")
        idx = tmp_path / "index.sqlite"
        build_index(vault, idx, _cfg(vault))

        added, deleted, skipped = update_index(vault, idx, _cfg(vault))
        assert added == 0
        assert deleted == 0
