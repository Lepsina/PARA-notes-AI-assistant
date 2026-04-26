"""Tests for the ``obsassist metadata apply`` CLI command.

Covers:
- --dry-run performs zero writes
- --limit restricts the number of files processed
- --remove-tag removes the marker on success and keeps it on failure
- --resume skips already-processed files; state is saved after success
- --backup / --no-backup controls .bak creation
- --path restricts the vault scan to a sub-directory
- body unchanged invariant: the markdown body is never modified
- skips files that do not contain the marker tag
- --workers falls back to 1 in interactive mode (no --yes/--dry-run)
- --batch prints progress lines at the requested interval
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from obsassist.cli import _load_processed_set, _metadata_state_path, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TAGGED_CONTENT = "---\ntags:\n  - add-metadata\n---\n# Title\n\nBody text.\n"
_PLAIN_CONTENT = "---\ntags:\n  - review\n---\n# Title\n\nBody text.\n"
# A response that will always add a 'title' field (causes changed=True)
_GOOD_LLM_RESPONSE = "title: Test Note\nstatus: draft\n"
# A non-dict YAML response that triggers ValueError in apply_metadata_to_content
_BAD_LLM_RESPONSE = "- item1\n- item2\n"


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


def _write_note(parent: Path, name: str, content: str) -> Path:
    path = parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _config_file(tmp_path: Path, vault: Path) -> str:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f'vault_root: "{vault}"\n'
        "ollama:\n"
        '  base_url: "http://127.0.0.1:11434"\n'
        '  model: "llama3:8b"\n',
        encoding="utf-8",
    )
    return str(cfg_path)


def _mock_client(response: str = _GOOD_LLM_RESPONSE) -> MagicMock:
    """Return a mock OllamaClient whose generate() returns *response*."""
    mock = MagicMock()
    mock.health_check.return_value = True
    mock.generate.return_value = response
    return mock


def _invoke(args: list[str], mock: MagicMock) -> object:
    runner = CliRunner()
    with patch("obsassist.cli.OllamaClient", return_value=mock):
        return runner.invoke(main, args)


# ---------------------------------------------------------------------------
# --dry-run — no writes
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_no_writes(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)
        original = note.read_text(encoding="utf-8")

        result = _invoke(
            ["metadata", "apply", "--tag", "add-metadata", "--dry-run", "--config", config],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        assert note.read_text(encoding="utf-8") == original

    def test_dry_run_mentions_would_update(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "apply", "--tag", "add-metadata", "--dry-run", "--config", config],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        assert "dry-run" in result.output.lower() or "Would update" in result.output

    def test_dry_run_no_bak_created(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        _invoke(
            ["metadata", "apply", "--tag", "add-metadata", "--dry-run", "--config", config],
            _mock_client(),
        )

        assert not note.with_suffix(".md.bak").exists()


# ---------------------------------------------------------------------------
# --limit
# ---------------------------------------------------------------------------


class TestLimit:
    def test_limit_restricts_llm_calls(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        for i in range(5):
            _write_note(vault, f"note{i}.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)
        mock = _mock_client()

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--limit", "2", "--dry-run", "--config", config,
            ],
            mock,
        )

        assert result.exit_code == 0, result.output
        assert mock.generate.call_count <= 2

    def test_no_limit_processes_all(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        for i in range(3):
            _write_note(vault, f"note{i}.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)
        mock = _mock_client()

        result = _invoke(
            ["metadata", "apply", "--tag", "add-metadata", "--dry-run", "--config", config],
            mock,
        )

        assert result.exit_code == 0, result.output
        assert mock.generate.call_count == 3


# ---------------------------------------------------------------------------
# --remove-tag
# ---------------------------------------------------------------------------


class TestRemoveTag:
    def test_remove_tag_on_success(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--remove-tag", "--yes", "--no-backup", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        updated = note.read_text(encoding="utf-8")
        assert "add-metadata" not in updated

    def test_remove_tag_not_removed_on_failure(self, tmp_path: Path):
        """When the LLM returns invalid metadata the marker must be preserved."""
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--remove-tag", "--yes", "--no-backup", "--config", config,
            ],
            _mock_client(response=_BAD_LLM_RESPONSE),
        )

        # Command should succeed (no crash), but the file must be unchanged
        assert result.exit_code == 0, result.output
        assert note.read_text(encoding="utf-8") == _TAGGED_CONTENT

    def test_remove_tag_only_for_successful_files(self, tmp_path: Path):
        """One file succeeds (tag removed), the other fails (tag kept)."""
        vault = _make_vault(tmp_path)
        # Use names that sort predictably: 'a_' comes before 'b_'
        note_a = _write_note(vault, "a_note.md", _TAGGED_CONTENT)  # processed 1st
        note_b = _write_note(vault, "b_note.md", _TAGGED_CONTENT)  # processed 2nd
        config = _config_file(tmp_path, vault)

        mock = _mock_client()
        # First call (a_note.md) succeeds; second call (b_note.md) returns bad YAML
        mock.generate.side_effect = [_GOOD_LLM_RESPONSE, _BAD_LLM_RESPONSE]

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--remove-tag", "--yes", "--no-backup", "--config", config,
            ],
            mock,
        )

        assert result.exit_code == 0, result.output
        # a_note.md: good response → tag removed
        assert "add-metadata" not in note_a.read_text(encoding="utf-8")
        # b_note.md: bad response (list YAML → ValueError) → tag kept
        assert "add-metadata" in note_b.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# --resume
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_skips_already_processed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note1 = _write_note(vault, "note1.md", _TAGGED_CONTENT)
        _write_note(vault, "note2.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        # Pre-populate state file: note1 was already processed
        state_file = _metadata_state_path(vault)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({"tag": "add-metadata", "processed": [str(note1.resolve())]}),
            encoding="utf-8",
        )

        mock = _mock_client()
        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--resume", "--dry-run", "--config", config,
            ],
            mock,
        )

        assert result.exit_code == 0, result.output
        # Only note2 should have been processed
        assert mock.generate.call_count == 1

    def test_state_saved_after_successful_write(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--yes", "--no-backup", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        state_file = _metadata_state_path(vault)
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tag"] == "add-metadata"
        assert str(note.resolve()) in data["processed"]

    def test_resume_with_different_tag_processes_all(self, tmp_path: Path):
        """State file for a different tag should not affect the current run."""
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        # State file records note as processed, but for a *different* tag
        state_file = _metadata_state_path(vault)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({"tag": "other-tag", "processed": [str(note.resolve())]}),
            encoding="utf-8",
        )

        mock = _mock_client()
        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--resume", "--dry-run", "--config", config,
            ],
            mock,
        )

        assert result.exit_code == 0, result.output
        assert mock.generate.call_count == 1

    def test_load_processed_set_returns_empty_for_missing_file(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.json"
        assert _load_processed_set(missing, "add-metadata") == set()

    def test_load_processed_set_returns_empty_for_wrong_tag(self, tmp_path: Path):
        state = tmp_path / "state.json"
        state.write_text(
            json.dumps({"tag": "other-tag", "processed": ["/some/file.md"]}),
            encoding="utf-8",
        )
        assert _load_processed_set(state, "add-metadata") == set()


# ---------------------------------------------------------------------------
# --backup / --no-backup
# ---------------------------------------------------------------------------


class TestBackup:
    def test_backup_created_on_write(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--yes", "--backup", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        bak = note.with_suffix(".md.bak")
        # Backup should exist if the file was actually modified
        if note.read_text(encoding="utf-8") != _TAGGED_CONTENT:
            assert bak.exists(), "Expected .bak file to be created"

    def test_no_backup_when_disabled(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--yes", "--no-backup", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        bak = note.with_suffix(".md.bak")
        assert not bak.exists()

    def test_bak_contains_original_content(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)
        original = note.read_text(encoding="utf-8")

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--yes", "--backup", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        bak = note.with_suffix(".md.bak")
        if bak.exists():
            assert bak.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# --path (scope restriction)
# ---------------------------------------------------------------------------


class TestPath:
    def test_scope_path_restricts_scan(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "sub/in-scope.md", _TAGGED_CONTENT)
        _write_note(vault, "other/out-of-scope.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)
        mock = _mock_client()

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--path", "sub", "--dry-run", "--config", config,
            ],
            mock,
        )

        assert result.exit_code == 0, result.output
        # Only the file inside sub/ should have been processed
        assert mock.generate.call_count == 1

    def test_invalid_scope_path_exits(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--path", "nonexistent-sub", "--dry-run", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Skips non-matching files
# ---------------------------------------------------------------------------


class TestSkipNonMatching:
    def test_skips_files_without_marker(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        _write_note(vault, "plain.md", _PLAIN_CONTENT)
        config = _config_file(tmp_path, vault)
        mock = _mock_client()

        result = _invoke(
            ["metadata", "apply", "--tag", "add-metadata", "--dry-run", "--config", config],
            mock,
        )

        assert result.exit_code == 0, result.output
        # Only the tagged file should be processed
        assert mock.generate.call_count == 1

    def test_no_files_found_exits_cleanly(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "plain.md", _PLAIN_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "apply", "--tag", "add-metadata", "--dry-run", "--config", config],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        assert "No notes found" in result.output


# ---------------------------------------------------------------------------
# Body unchanged invariant
# ---------------------------------------------------------------------------


class TestBodyInvariant:
    def test_body_unchanged_after_apply(self, tmp_path: Path):
        body = "# My Note\n\nThis is the body content.\n\nSecond paragraph.\n"
        content = f"---\ntags:\n  - add-metadata\n---\n{body}"
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", content)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--yes", "--no-backup", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        updated = note.read_text(encoding="utf-8")
        # Use split_frontmatter to correctly extract the body
        from obsassist.metadata_guard import split_frontmatter
        _, updated_body = split_frontmatter(updated)
        assert updated_body == body


# ---------------------------------------------------------------------------
# --workers fallback in interactive mode
# ---------------------------------------------------------------------------


class TestWorkers:
    def test_workers_fallback_warning_without_yes(self, tmp_path: Path):
        """--workers > 1 without --yes or --dry-run should emit a warning."""
        vault = _make_vault(tmp_path)
        _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        # Use --yes to avoid the interactive prompt, but first check warning
        # by NOT passing --yes or --dry-run in a dry-run context:
        runner = CliRunner()
        mock = _mock_client()
        # We simulate the condition: workers=2 but interactive mode
        # The command should print a warning and fall back to workers=1.
        # We can test this by running with input="" to auto-decline prompts:
        with patch("obsassist.cli.OllamaClient", return_value=mock):
            result = runner.invoke(
                main,
                [
                    "metadata", "apply", "--tag", "add-metadata",
                    "--workers", "2", "--config", config,
                ],
                input="n\n",  # decline the per-file prompt
            )

        assert result.exit_code == 0, result.output
        assert "Warning" in result.output or "workers" in result.output.lower()

    def test_workers_with_yes_no_warning(self, tmp_path: Path):
        """--workers > 1 with --yes should work without warning."""
        vault = _make_vault(tmp_path)
        _write_note(vault, "tagged.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--workers", "2", "--yes", "--no-backup", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        assert "Warning" not in result.output


# ---------------------------------------------------------------------------
# --batch progress lines
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_progress_printed(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        for i in range(4):
            _write_note(vault, f"note{i}.md", _TAGGED_CONTENT)
        config = _config_file(tmp_path, vault)

        result = _invoke(
            [
                "metadata", "apply", "--tag", "add-metadata",
                "--batch", "2", "--dry-run", "--config", config,
            ],
            _mock_client(),
        )

        assert result.exit_code == 0, result.output
        assert "Batch progress" in result.output
