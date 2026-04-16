"""Unit tests for obsassist.filters."""
from __future__ import annotations

from pathlib import Path

import pytest

from obsassist.filters import DEFAULT_EXCLUDE_PATHS, is_excluded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsExcluded:
    # --- Excluded paths ---

    def test_resources_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert is_excluded(vault / "Resources" / "note.md", vault, DEFAULT_EXCLUDE_PATHS)

    def test_resources_subdir_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert is_excluded(
            vault / "Resources" / "SubDir" / "deep.md",
            vault,
            DEFAULT_EXCLUDE_PATHS,
        )

    def test_templates_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert is_excluded(vault / "Templates" / "template.md", vault, DEFAULT_EXCLUDE_PATHS)

    def test_files_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert is_excluded(vault / "Files" / "image.png", vault, DEFAULT_EXCLUDE_PATHS)

    def test_excalidraw_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert is_excluded(
            vault / "Excalidraw" / "diagram.excalidraw",
            vault,
            DEFAULT_EXCLUDE_PATHS,
        )

    def test_obsidian_config_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert is_excluded(vault / ".obsidian" / "config.json", vault, DEFAULT_EXCLUDE_PATHS)

    # --- Allowed paths ---

    def test_daily_not_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert not is_excluded(vault / "Daily" / "2026-01-01.md", vault, DEFAULT_EXCLUDE_PATHS)

    def test_projects_not_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert not is_excluded(
            vault / "Projects" / "project_a" / "plan.md",
            vault,
            DEFAULT_EXCLUDE_PATHS,
        )

    def test_areas_not_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert not is_excluded(vault / "Areas" / "health.md", vault, DEFAULT_EXCLUDE_PATHS)

    def test_archive_not_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert not is_excluded(
            vault / "Archive" / "old_project.md", vault, DEFAULT_EXCLUDE_PATHS
        )

    def test_buffer_not_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert not is_excluded(vault / "Buffer" / "inbox.md", vault, DEFAULT_EXCLUDE_PATHS)

    # --- Outside vault ---

    def test_outside_vault_excluded(self, tmp_path: Path):
        vault = _vault(tmp_path)
        outside = tmp_path / "outside" / "note.md"
        assert is_excluded(outside, vault, DEFAULT_EXCLUDE_PATHS)

    # --- vault_root=None ---

    def test_no_vault_root_excluded_prefix(self, tmp_path: Path):
        """When vault_root is None, path is treated as relative and prefix-matched."""
        path = Path("Resources") / "note.md"
        assert is_excluded(path, None, ["Resources/"])

    def test_no_vault_root_not_excluded(self):
        path = Path("Daily") / "2026-01-01.md"
        assert not is_excluded(path, None, DEFAULT_EXCLUDE_PATHS)

    # --- Custom exclude list ---

    def test_custom_exclude_list(self, tmp_path: Path):
        vault = _vault(tmp_path)
        assert is_excluded(vault / "MyFolder" / "note.md", vault, ["MyFolder/"])
        assert not is_excluded(vault / "Other" / "note.md", vault, ["MyFolder/"])
