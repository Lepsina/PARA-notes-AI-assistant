"""Tests for the ``obsassist metadata audit`` CLI command.

Covers:
- counts total markdown files correctly
- detects missing frontmatter
- detects invalid YAML frontmatter
- counts marker tag occurrences (frontmatter and body)
- reports type/status/lang distributions
- detects invalid type / status / lang values
- archive type support (4. Archive folder → type: archive)
- --path restricts scan to sub-directory
- --format json produces valid JSON
- --format md produces Markdown and writes default report file
- --output writes report to a custom path
- read-only: no note files are modified during audit
- vault_root missing → non-zero exit
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from obsassist.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        f'vault_root: "{vault}"\n',
        encoding="utf-8",
    )
    return str(cfg_path)


def _invoke(args: list[str]) -> object:
    runner = CliRunner()
    return runner.invoke(main, args)


# ---------------------------------------------------------------------------
# Fixtures / shared notes
# ---------------------------------------------------------------------------

_NOTE_COMPLETE = """\
---
type: note
status: draft
lang: ru
tags:
  - python
---
# Complete note

Body text.
"""

_NOTE_NO_FM = "# No frontmatter\n\nJust a body.\n"

_NOTE_INVALID_YAML = "---\n: this is invalid yaml: [\n---\nBody.\n"

_NOTE_MARKER_FM = """\
---
type: note
status: draft
lang: ru
tags:
  - add-metadata
---
# Marker in frontmatter
"""

_NOTE_MARKER_BODY = """\
---
type: note
status: active
lang: en
---
# Marker in body

See #add-metadata here.
"""

_NOTE_INVALID_TYPE = """\
---
type: weirdo
status: draft
lang: ru
---
# Invalid type
"""

_NOTE_INVALID_STATUS = """\
---
type: note
status: published
lang: ru
---
# Invalid status
"""

_NOTE_INVALID_LANG = """\
---
type: note
status: active
lang: klingon
---
# Invalid lang
"""

_NOTE_ARCHIVE = """\
---
type: archive
status: archived
lang: ru
---
# Archived note
"""


# ---------------------------------------------------------------------------
# Test: counts files correctly
# ---------------------------------------------------------------------------


class TestAuditFileCounts:
    def test_counts_total(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "a.md", _NOTE_COMPLETE)
        _write_note(vault, "b.md", _NOTE_COMPLETE)
        _write_note(vault, "c.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(["metadata", "audit", "--config", cfg])

        assert result.exit_code == 0, result.output
        assert "3" in result.output

    def test_only_md_files_counted(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _NOTE_COMPLETE)
        (vault / "image.png").write_bytes(b"")
        (vault / "data.json").write_text("{}", encoding="utf-8")
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["total"] == 1


# ---------------------------------------------------------------------------
# Test: missing frontmatter
# ---------------------------------------------------------------------------


class TestMissingFrontmatter:
    def test_detects_missing_frontmatter(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "no_fm.md", _NOTE_NO_FM)
        _write_note(vault, "ok.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["missing_frontmatter"]) == 1
        assert "no_fm.md" in data["missing_frontmatter"][0]

    def test_no_missing_frontmatter(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "ok.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["missing_frontmatter"] == []


# ---------------------------------------------------------------------------
# Test: invalid YAML frontmatter
# ---------------------------------------------------------------------------


class TestInvalidYaml:
    def test_detects_invalid_yaml(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "bad.md", _NOTE_INVALID_YAML)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["invalid_yaml"]) == 1
        assert "bad.md" in data["invalid_yaml"][0]

    def test_no_invalid_yaml(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "ok.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["invalid_yaml"] == []


# ---------------------------------------------------------------------------
# Test: marker tag count (frontmatter and body)
# ---------------------------------------------------------------------------


class TestMarkerTagCount:
    def test_detects_marker_in_frontmatter(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "fm_marker.md", _NOTE_MARKER_FM)
        _write_note(vault, "clean.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--tag", "add-metadata", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["marker_files"]) == 1
        assert "fm_marker.md" in data["marker_files"][0]

    def test_detects_marker_in_body(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "body_marker.md", _NOTE_MARKER_BODY)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--tag", "add-metadata", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["marker_files"]) == 1

    def test_counts_both_locations(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "fm.md", _NOTE_MARKER_FM)
        _write_note(vault, "body.md", _NOTE_MARKER_BODY)
        _write_note(vault, "clean.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--tag", "add-metadata", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["marker_files"]) == 2

    def test_marker_in_file_without_frontmatter(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "no_fm_marker.md", "# Title\n\n#add-metadata here.\n")
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--tag", "add-metadata", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["marker_files"]) == 1
        assert len(data["missing_frontmatter"]) == 1


# ---------------------------------------------------------------------------
# Test: type/status/lang distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_type_distribution(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "n1.md", _NOTE_COMPLETE)  # type: note
        _write_note(vault, "n2.md", _NOTE_COMPLETE)  # type: note
        _write_note(vault, "a1.md", _NOTE_ARCHIVE)   # type: archive
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["type_distribution"]["note"] == 2
        assert data["type_distribution"]["archive"] == 1

    def test_status_distribution(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "draft.md", _NOTE_COMPLETE)  # status: draft
        _write_note(vault, "arch.md", _NOTE_ARCHIVE)    # status: archived
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["status_distribution"]["draft"] == 1
        assert data["status_distribution"]["archived"] == 1

    def test_lang_distribution(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "ru.md", _NOTE_COMPLETE)     # lang: ru
        _write_note(vault, "body.md", _NOTE_MARKER_BODY)  # lang: en
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["lang_distribution"]["ru"] == 1
        assert data["lang_distribution"]["en"] == 1

    def test_invalid_type_detected(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "bad_type.md", _NOTE_INVALID_TYPE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["invalid_type"]) == 1
        assert "bad_type.md" in data["invalid_type"][0]

    def test_invalid_status_detected(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "bad_status.md", _NOTE_INVALID_STATUS)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["invalid_status"]) == 1

    def test_invalid_lang_detected(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "bad_lang.md", _NOTE_INVALID_LANG)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["invalid_lang"]) == 1


# ---------------------------------------------------------------------------
# Test: Archive type support
# ---------------------------------------------------------------------------


class TestArchiveTypeSupport:
    def test_archive_folder_infers_archive_type(self, tmp_path: Path):
        """Notes under 4. Archive/ have type: archive in the allowed set."""
        vault = _make_vault(tmp_path)
        archive_dir = vault / "4. Archive"
        _write_note(archive_dir, "old_note.md", _NOTE_ARCHIVE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["invalid_type"] == []
        assert data["type_distribution"].get("archive", 0) == 1

    def test_archive_type_not_counted_as_invalid(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "arch.md", _NOTE_ARCHIVE)  # type: archive
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["invalid_type"] == []

    def test_archived_status_not_counted_as_invalid(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "arch.md", _NOTE_ARCHIVE)  # status: archived
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        assert data["invalid_status"] == []


# ---------------------------------------------------------------------------
# Test: --path restricts scan
# ---------------------------------------------------------------------------


class TestScopePath:
    def test_path_restricts_scan(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        sub = vault / "sub"
        _write_note(sub, "in_scope.md", _NOTE_COMPLETE)
        _write_note(vault, "out_of_scope.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--path", "sub", "--format", "json", "--config", cfg]
        )

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["total"] == 1

    def test_invalid_path_exits(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--path", "nonexistent", "--config", cfg]
        )

        assert result.exit_code != 0 or "does not exist" in result.output


# ---------------------------------------------------------------------------
# Test: --format json
# ---------------------------------------------------------------------------


class TestFormatJson:
    def test_json_output_is_valid(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "total" in data
        assert "missing_frontmatter" in data
        assert "type_distribution" in data
        assert "status_distribution" in data
        assert "lang_distribution" in data
        assert "top_tags" in data

    def test_json_top_tags_is_list_of_dicts(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--config", cfg]
        )

        data = json.loads(result.output)
        for item in data["top_tags"]:
            assert "tag" in item
            assert "count" in item


# ---------------------------------------------------------------------------
# Test: --format md writes report file
# ---------------------------------------------------------------------------


class TestFormatMd:
    def test_md_writes_default_report(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "md", "--config", cfg]
        )

        assert result.exit_code == 0, result.output
        from datetime import date
        today = date.today().isoformat()
        report_path = vault / ".obsassist" / "reports" / f"metadata-audit-{today}.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# Metadata Audit" in content

    def test_md_output_contains_sections(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "md", "--config", cfg]
        )

        assert "## Summary" in result.output
        assert "Total markdown files" in result.output


# ---------------------------------------------------------------------------
# Test: --output writes to custom path
# ---------------------------------------------------------------------------


class TestOutputPath:
    def test_custom_output_path(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)
        out = tmp_path / "report.txt"

        result = _invoke(
            ["metadata", "audit", "--output", str(out), "--config", cfg]
        )

        assert result.exit_code == 0, result.output
        assert out.exists()
        assert "3" in out.read_text(encoding="utf-8") or "1" in out.read_text(encoding="utf-8")

    def test_custom_output_md(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        _write_note(vault, "note.md", _NOTE_COMPLETE)
        cfg = _config_file(tmp_path, vault)
        out = tmp_path / "sub" / "report.md"

        result = _invoke(
            [
                "metadata", "audit",
                "--format", "md",
                "--output", str(out),
                "--config", cfg,
            ]
        )

        assert result.exit_code == 0, result.output
        assert out.exists()


# ---------------------------------------------------------------------------
# Test: read-only — notes are never modified
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_notes_unchanged(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", _NOTE_COMPLETE)
        original = note.read_text(encoding="utf-8")
        cfg = _config_file(tmp_path, vault)

        _invoke(["metadata", "audit", "--config", cfg])

        assert note.read_text(encoding="utf-8") == original

    def test_notes_unchanged_format_md(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        note = _write_note(vault, "note.md", _NOTE_MARKER_FM)
        original = note.read_text(encoding="utf-8")
        cfg = _config_file(tmp_path, vault)

        _invoke(["metadata", "audit", "--format", "md", "--config", cfg])

        assert note.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Test: missing vault_root
# ---------------------------------------------------------------------------


class TestVaultRootMissing:
    def test_no_vault_root_exits(self, tmp_path: Path):
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("", encoding="utf-8")

        result = _invoke(["metadata", "audit", "--config", str(cfg_path)])

        assert result.exit_code != 0 or "vault_root" in result.output


# ---------------------------------------------------------------------------
# Test: top-N tags
# ---------------------------------------------------------------------------


class TestTopTags:
    def test_top_tags_in_json(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        multi_tag = """\
---
type: note
status: draft
lang: ru
tags:
  - python
  - python
  - blockchain
---
# Multi tag note
"""
        _write_note(vault, "n1.md", multi_tag)
        _write_note(vault, "n2.md", _NOTE_COMPLETE)  # tags: python
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--top-tags", "5", "--config", cfg]
        )

        data = json.loads(result.output)
        tag_names = [item["tag"] for item in data["top_tags"]]
        assert "python" in tag_names

    def test_top_n_limits_output(self, tmp_path: Path):
        vault = _make_vault(tmp_path)
        many_tags = "---\ntype: note\nstatus: draft\nlang: ru\ntags:\n" + \
                    "\n".join(f"  - tag{i}" for i in range(20)) + "\n---\n# Many tags\n"
        _write_note(vault, "many.md", many_tags)
        cfg = _config_file(tmp_path, vault)

        result = _invoke(
            ["metadata", "audit", "--format", "json", "--top-tags", "3", "--config", cfg]
        )

        data = json.loads(result.output)
        assert len(data["top_tags"]) <= 3
