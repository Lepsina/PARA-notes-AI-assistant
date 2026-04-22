"""Tests for obsassist.retrieval (FTS, vector, hybrid)."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from obsassist.config import Config, EmbeddingsConfig
from obsassist.embeddings import build_embeddings
from obsassist.indexer import build_index
from obsassist.retrieval import (
    ChunkResult,
    cosine_similarity,
    retrieve_fts,
    retrieve_hybrid,
    retrieve_vector,
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
    cfg.embeddings = EmbeddingsConfig(
        model="test-model",
        chunk_size=500,
        chunk_overlap=50,
        batch_size=4,
    )
    return cfg


def _dummy_embed_factory(dim: int = 4):
    """Return an embed_fn that produces deterministic vectors based on text hash."""
    def embed_fn(texts: list[str]) -> list[list[float]]:
        vecs = []
        for text in texts:
            h = hash(text) % 1000
            # Simple deterministic vector
            vecs.append([float(h % 10) / 10, float((h // 10) % 10) / 10,
                          float((h // 100) % 10) / 10, float(h % 7) / 10])
        return vecs
    return embed_fn


def _seed_db(tmp_path: Path, notes: dict[str, str]) -> tuple[Path, Path, Config]:
    """Create vault with notes, build FTS + embeddings index."""
    vault = _make_vault(tmp_path)
    for rel, content in notes.items():
        _write_note(vault, rel, content)
    cfg = _cfg(vault)
    idx = tmp_path / "index.sqlite"
    build_index(vault, idx, cfg)
    build_embeddings(vault, idx, cfg, _dummy_embed_factory())
    return vault, idx, cfg


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.5, 0.2]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# retrieve_fts
# ---------------------------------------------------------------------------


class TestRetrieveFts:
    def test_returns_results_for_matching_query(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            "a.md": "# Alpha\n\nThis note is about machine learning.",
            "b.md": "# Beta\n\nThis note is about cooking recipes.",
        })
        results = retrieve_fts(idx, "machine learning")
        assert len(results) >= 1
        paths = [r.path for r in results]
        assert "a.md" in paths

    def test_no_results_for_unmatched_query(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            "a.md": "# Alpha\n\nPlain text note."
        })
        results = retrieve_fts(idx, "xyzzyquuxfrobnicator")
        assert results == []

    def test_returns_at_most_k(self, tmp_path: Path):
        notes = {f"note_{i}.md": f"# Note {i}\n\nCommon word here." for i in range(10)}
        _, idx, _ = _seed_db(tmp_path, notes)
        results = retrieve_fts(idx, "Common word", k=3)
        assert len(results) <= 3

    def test_nonexistent_index(self, tmp_path: Path):
        idx = tmp_path / "nonexistent.sqlite"
        results = retrieve_fts(idx, "anything")
        assert results == []

    def test_result_has_correct_fields(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            "note.md": "# My Heading\n\nSome content about testing."
        })
        results = retrieve_fts(idx, "testing")
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, ChunkResult)
        assert isinstance(r.path, str)
        assert isinstance(r.content, str)
        assert isinstance(r.score, float)


# ---------------------------------------------------------------------------
# retrieve_vector
# ---------------------------------------------------------------------------


class TestRetrieveVector:
    def test_returns_results(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            "a.md": "# Alpha\n\nText about apples.",
            "b.md": "# Beta\n\nText about bananas.",
        })
        query_vec = [0.5, 0.5, 0.5, 0.5]
        results = retrieve_vector(idx, query_vec, k=5)
        assert len(results) >= 1

    def test_results_sorted_by_score_descending(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            f"note_{i}.md": f"# Note {i}\n\nContent {i}." for i in range(5)
        })
        query_vec = [0.9, 0.1, 0.0, 0.0]
        results = retrieve_vector(idx, query_vec, k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_at_most_k(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            f"note_{i}.md": f"# Note {i}\n\nContent." for i in range(10)
        })
        query_vec = [0.5, 0.5, 0.5, 0.5]
        results = retrieve_vector(idx, query_vec, k=3)
        assert len(results) <= 3

    def test_nonexistent_index(self, tmp_path: Path):
        idx = tmp_path / "nonexistent.sqlite"
        results = retrieve_vector(idx, [0.1, 0.2, 0.3], k=5)
        assert results == []


# ---------------------------------------------------------------------------
# retrieve_hybrid
# ---------------------------------------------------------------------------


class TestRetrieveHybrid:
    def test_returns_results_for_matching_query(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            "a.md": "# Alpha\n\nThis note is about deep learning.",
            "b.md": "# Beta\n\nThis note is about gardening.",
        })
        query_vec = [0.5, 0.5, 0.5, 0.5]
        results = retrieve_hybrid(idx, "deep learning", query_vec, k=5, candidates=10)
        assert len(results) >= 1
        paths = [r.path for r in results]
        assert "a.md" in paths

    def test_results_sorted_by_score_descending(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            f"note_{i}.md": f"# Note {i}\n\nContent." for i in range(5)
        })
        query_vec = [0.9, 0.1, 0.0, 0.0]
        results = retrieve_hybrid(idx, "Content", query_vec, k=10, candidates=20)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_at_most_k(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            f"note_{i}.md": f"# Note {i}\n\nCommon content." for i in range(10)
        })
        query_vec = [0.5, 0.5, 0.5, 0.5]
        results = retrieve_hybrid(idx, "content", query_vec, k=3, candidates=20)
        assert len(results) <= 3

    def test_no_fts_match_returns_empty(self, tmp_path: Path):
        _, idx, _ = _seed_db(tmp_path, {
            "note.md": "# Title\n\nSome plain text."
        })
        query_vec = [0.5, 0.5, 0.5, 0.5]
        results = retrieve_hybrid(idx, "xyzzyquux", query_vec, k=5)
        assert results == []

    def test_nonexistent_index(self, tmp_path: Path):
        idx = tmp_path / "nonexistent.sqlite"
        results = retrieve_hybrid(idx, "query", [0.1, 0.2], k=5)
        assert results == []
