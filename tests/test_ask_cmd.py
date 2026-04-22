"""Tests for the obsassist ask CLI command."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from obsassist.cli import main
from obsassist.config import Config, EmbeddingsConfig
from obsassist.embeddings import build_embeddings
from obsassist.indexer import build_index


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


def _dummy_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _encode_vec(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _seed_full_index(tmp_path: Path) -> tuple[Path, Path, str]:
    """Create vault + FTS + embeddings index. Returns (vault, idx_path, config_path)."""
    vault = _make_vault(tmp_path)
    _write_note(vault, "note.md", "# Obsidian Tips\n\nUse daily notes for journaling.")
    _write_note(vault, "arch.md", "# Architecture\n\nMicroservices pattern is popular.")

    cfg = Config()
    cfg.vault_root = str(vault)
    cfg.embeddings = EmbeddingsConfig(
        model="nomic-embed-text",
        chunk_size=500,
        chunk_overlap=50,
        batch_size=4,
    )

    idx = tmp_path / "index.sqlite"
    build_index(vault, idx, cfg)
    build_embeddings(vault, idx, cfg, _dummy_embed)

    # Write config file
    config_text = f"""
vault_root: "{vault}"
ollama:
  base_url: "http://127.0.0.1:11434"
  model: "llama3:8b"
embeddings:
  base_url: "http://127.0.0.1:11434"
  model: "nomic-embed-text"
  chunk_size: 500
  chunk_overlap: 50
index_path: "{idx}"
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(config_text, encoding="utf-8")

    return vault, idx, str(config_path)


# ---------------------------------------------------------------------------
# ask command – FTS mode (no embedding client needed)
# ---------------------------------------------------------------------------


class TestAskFtsMode:
    def test_ask_fts_returns_answer_and_sources(self, tmp_path: Path):
        vault, idx, config_path = _seed_full_index(tmp_path)
        runner = CliRunner()

        with patch("obsassist.cli._build_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.health_check.return_value = True
            mock_client.generate.return_value = "Obsidian is great for note-taking."
            mock_client_factory.return_value = mock_client

            result = runner.invoke(
                main,
                ["ask", "What is Obsidian?", "--mode", "fts",
                 "--config", config_path],
            )

        assert result.exit_code == 0, result.output
        assert "Sources" in result.output

    def test_ask_fts_output_contains_file_path(self, tmp_path: Path):
        vault, idx, config_path = _seed_full_index(tmp_path)
        runner = CliRunner()

        with patch("obsassist.cli._build_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.health_check.return_value = True
            mock_client.generate.return_value = "Daily notes help with journaling."
            mock_client_factory.return_value = mock_client

            result = runner.invoke(
                main,
                ["ask", "daily notes journaling", "--mode", "fts",
                 "--config", config_path],
            )

        assert result.exit_code == 0, result.output
        # Sources should include the matched file
        assert "note.md" in result.output or "arch.md" in result.output


# ---------------------------------------------------------------------------
# ask command – vector mode
# ---------------------------------------------------------------------------


class TestAskVectorMode:
    def test_ask_vector_returns_answer_and_sources(self, tmp_path: Path):
        vault, idx, config_path = _seed_full_index(tmp_path)
        runner = CliRunner()

        with (
            patch("obsassist.cli._build_client") as mock_llm_factory,
            patch("obsassist.cli.OllamaClient") as MockOllamaClient,
        ):
            mock_llm = MagicMock()
            mock_llm.health_check.return_value = True
            mock_llm.generate.return_value = "Vector search uses embeddings."
            mock_llm_factory.return_value = mock_llm

            mock_emb = MagicMock()
            mock_emb.health_check.return_value = True
            mock_emb.embed.return_value = [0.1, 0.2, 0.3, 0.4]
            MockOllamaClient.return_value = mock_emb

            result = runner.invoke(
                main,
                ["ask", "semantic search", "--mode", "vector",
                 "--config", config_path],
            )

        assert result.exit_code == 0, result.output
        assert "Sources" in result.output


# ---------------------------------------------------------------------------
# ask command – hybrid mode
# ---------------------------------------------------------------------------


class TestAskHybridMode:
    def test_ask_hybrid_default_mode(self, tmp_path: Path):
        vault, idx, config_path = _seed_full_index(tmp_path)
        runner = CliRunner()

        with (
            patch("obsassist.cli._build_client") as mock_llm_factory,
            patch("obsassist.cli.OllamaClient") as MockOllamaClient,
        ):
            mock_llm = MagicMock()
            mock_llm.health_check.return_value = True
            mock_llm.generate.return_value = "Hybrid retrieval combines FTS and vectors."
            mock_llm_factory.return_value = mock_llm

            mock_emb = MagicMock()
            mock_emb.health_check.return_value = True
            mock_emb.embed.return_value = [0.1, 0.2, 0.3, 0.4]
            MockOllamaClient.return_value = mock_emb

            result = runner.invoke(
                main,
                ["ask", "microservices architecture", "--config", config_path],
            )

        assert result.exit_code == 0, result.output
        assert "Sources" in result.output


# ---------------------------------------------------------------------------
# ask command – error cases
# ---------------------------------------------------------------------------


class TestAskErrors:
    def test_missing_index_exits_with_error(self, tmp_path: Path):
        """ask should fail gracefully when index doesn't exist."""
        runner = CliRunner()
        config_path = tmp_path / "config.yml"
        config_path.write_text(
            f'index_path: "{tmp_path / "nonexistent.sqlite"}"\n', encoding="utf-8"
        )
        result = runner.invoke(
            main, ["ask", "test question", "--config", str(config_path)]
        )
        assert result.exit_code != 0
        assert "Index not found" in result.output

    def test_ollama_unavailable_exits_with_error(self, tmp_path: Path):
        """ask should fail gracefully when Ollama is not running."""
        vault, idx, config_path = _seed_full_index(tmp_path)
        runner = CliRunner()

        with (
            patch("obsassist.cli._build_client") as mock_llm_factory,
            patch("obsassist.cli.OllamaClient") as MockOllamaClient,
        ):
            # Embedding client unavailable
            mock_emb = MagicMock()
            mock_emb.health_check.return_value = False
            MockOllamaClient.return_value = mock_emb

            mock_llm = MagicMock()
            mock_llm.health_check.return_value = False
            mock_llm_factory.return_value = mock_llm

            result = runner.invoke(
                main,
                ["ask", "test question", "--mode", "vector",
                 "--config", config_path],
            )

        assert result.exit_code != 0
        assert "Cannot reach Ollama" in result.output or "Error" in result.output

    def test_save_to_creates_file(self, tmp_path: Path):
        vault, idx, config_path = _seed_full_index(tmp_path)
        save_path = tmp_path / "answer.md"
        runner = CliRunner()

        with patch("obsassist.cli._build_client") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.health_check.return_value = True
            mock_llm.generate.return_value = "Here is the answer."
            mock_llm_factory.return_value = mock_llm

            result = runner.invoke(
                main,
                ["ask", "journaling tips", "--mode", "fts",
                 "--config", config_path, "--save-to", str(save_path)],
            )

        assert result.exit_code == 0, result.output
        assert save_path.exists()
        text = save_path.read_text(encoding="utf-8")
        assert "Sources" in text
        assert "journaling tips" in text
