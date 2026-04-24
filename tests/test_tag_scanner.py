"""Tests for obsassist.tag_scanner.

Covers:
- has_marker_tag: frontmatter tags list detection
- has_marker_tag: body inline hashtag detection
- has_marker_tag: edge cases (case insensitivity, hash prefix in input)
- remove_marker_tag: removal from frontmatter tags list
- remove_marker_tag: removal of #tag from body
- remove_marker_tag: no modification when tag not present
- remove_marker_tag: body content preserved byte-for-byte on unaffected lines
"""
from __future__ import annotations

import pytest

from obsassist.tag_scanner import has_marker_tag, remove_marker_tag
from obsassist.metadata_guard import split_frontmatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _note(fm_yaml: str = "", body: str = "# Title\n\nBody.\n") -> str:
    if fm_yaml:
        return f"---\n{fm_yaml.strip()}\n---\n{body}"
    return body


# ---------------------------------------------------------------------------
# has_marker_tag — frontmatter detection
# ---------------------------------------------------------------------------


class TestHasMarkerTagFrontmatter:
    def test_bare_tag_in_frontmatter_list(self):
        content = _note("tags:\n  - add-metadata\n  - topic1")
        assert has_marker_tag(content, "add-metadata") is True

    def test_hash_prefixed_input_tag(self):
        """Caller passes '#add-metadata'; function strips the leading hash."""
        content = _note("tags:\n  - add-metadata")
        assert has_marker_tag(content, "#add-metadata") is True

    def test_hash_prefixed_stored_tag(self):
        """Tag stored in frontmatter with '#' prefix should still match."""
        content = _note("tags:\n  - '#add-metadata'")
        assert has_marker_tag(content, "add-metadata") is True

    def test_case_insensitive_frontmatter(self):
        content = _note("tags:\n  - Add-Metadata")
        assert has_marker_tag(content, "add-metadata") is True

    def test_not_matching_partial_frontmatter(self):
        """'add-metadata-extra' should NOT match tag 'add-metadata'."""
        content = _note("tags:\n  - add-metadata-extra")
        assert has_marker_tag(content, "add-metadata") is False

    def test_no_tags_key(self):
        content = _note("status: draft")
        assert has_marker_tag(content, "add-metadata") is False

    def test_empty_tags_list(self):
        content = _note("tags: []")
        assert has_marker_tag(content, "add-metadata") is False

    def test_tag_among_several(self):
        content = _note("tags:\n  - topic1\n  - add-metadata\n  - topic2")
        assert has_marker_tag(content, "add-metadata") is True

    def test_different_tag_not_matched(self):
        content = _note("tags:\n  - topic1\n  - review")
        assert has_marker_tag(content, "add-metadata") is False


# ---------------------------------------------------------------------------
# has_marker_tag — body detection
# ---------------------------------------------------------------------------


class TestHasMarkerTagBody:
    def test_inline_hashtag_in_body(self):
        content = _note(body="# Title\n\nSome text #add-metadata here.\n")
        assert has_marker_tag(content, "add-metadata") is True

    def test_hashtag_alone_on_line(self):
        content = _note(body="# Title\n\n#add-metadata\n")
        assert has_marker_tag(content, "add-metadata") is True

    def test_hashtag_at_end_of_line(self):
        content = _note(body="# Title\n\nSee this note #add-metadata\n")
        assert has_marker_tag(content, "add-metadata") is True

    def test_hashtag_at_start_of_body(self):
        content = "#add-metadata\n\nNote without frontmatter.\n"
        assert has_marker_tag(content, "add-metadata") is True

    def test_body_case_insensitive(self):
        content = _note(body="# Title\n\n#Add-Metadata\n")
        assert has_marker_tag(content, "add-metadata") is True

    def test_partial_match_not_detected(self):
        """'#add-metadata-extra' should NOT match 'add-metadata'."""
        content = _note(body="# Title\n\n#add-metadata-extra\n")
        assert has_marker_tag(content, "add-metadata") is False

    def test_embedded_in_word_not_detected(self):
        """'pre#add-metadata' (no space before) should NOT match."""
        content = _note(body="# Title\n\npre#add-metadata\n")
        assert has_marker_tag(content, "add-metadata") is False

    def test_no_tag_anywhere(self):
        content = _note(body="# Title\n\nJust a normal note.\n")
        assert has_marker_tag(content, "add-metadata") is False

    def test_no_frontmatter_no_body_tag(self):
        content = "# Note\n\nNo markers here.\n"
        assert has_marker_tag(content, "add-metadata") is False

    def test_hash_input_matches_body_tag(self):
        """Input '#add-metadata' should match '#add-metadata' in body."""
        content = _note(body="# Title\n\n#add-metadata\n")
        assert has_marker_tag(content, "#add-metadata") is True


# ---------------------------------------------------------------------------
# has_marker_tag — combined frontmatter + body
# ---------------------------------------------------------------------------


class TestHasMarkerTagCombined:
    def test_only_in_frontmatter(self):
        content = _note("tags:\n  - add-metadata", body="# Title\n\nNo hashtag.\n")
        assert has_marker_tag(content, "add-metadata") is True

    def test_only_in_body(self):
        content = _note("tags:\n  - topic1", body="# Title\n\n#add-metadata\n")
        assert has_marker_tag(content, "add-metadata") is True

    def test_in_both(self):
        content = _note(
            "tags:\n  - add-metadata",
            body="# Title\n\n#add-metadata\n",
        )
        assert has_marker_tag(content, "add-metadata") is True


# ---------------------------------------------------------------------------
# remove_marker_tag — frontmatter
# ---------------------------------------------------------------------------


class TestRemoveMarkerTagFrontmatter:
    def test_removes_tag_from_list(self):
        content = _note("tags:\n  - add-metadata\n  - topic1")
        result = remove_marker_tag(content, "add-metadata")
        fm, _ = split_frontmatter(result)
        assert "add-metadata" not in [str(t).lstrip("#") for t in fm.get("tags", [])]
        assert "topic1" in fm.get("tags", [])

    def test_removes_only_tag_leaving_others(self):
        content = _note("tags:\n  - topic1\n  - add-metadata\n  - topic2")
        result = remove_marker_tag(content, "add-metadata")
        fm, _ = split_frontmatter(result)
        assert fm["tags"] == ["topic1", "topic2"]

    def test_removes_tags_key_when_last_tag(self):
        content = _note("tags:\n  - add-metadata")
        result = remove_marker_tag(content, "add-metadata")
        fm, _ = split_frontmatter(result)
        assert "tags" not in fm

    def test_case_insensitive_removal(self):
        content = _note("tags:\n  - Add-Metadata\n  - topic1")
        result = remove_marker_tag(content, "add-metadata")
        fm, _ = split_frontmatter(result)
        tag_values = [str(t).lstrip("#").lower() for t in fm.get("tags", [])]
        assert "add-metadata" not in tag_values

    def test_hash_prefixed_stored_tag_removed(self):
        """Frontmatter stores '#add-metadata' — should still be removed."""
        content = _note("tags:\n  - '#add-metadata'\n  - topic1")
        result = remove_marker_tag(content, "add-metadata")
        fm, _ = split_frontmatter(result)
        tag_values = [str(t).lstrip("#").lower() for t in fm.get("tags", [])]
        assert "add-metadata" not in tag_values

    def test_no_change_when_tag_absent(self):
        content = _note("tags:\n  - topic1")
        result = remove_marker_tag(content, "add-metadata")
        assert result == content

    def test_no_frontmatter_no_change(self):
        content = "# Note\n\nNo frontmatter.\n"
        result = remove_marker_tag(content, "add-metadata")
        assert result == content


# ---------------------------------------------------------------------------
# remove_marker_tag — body
# ---------------------------------------------------------------------------


class TestRemoveMarkerTagBody:
    def test_removes_inline_hashtag(self):
        content = _note(body="# Title\n\nSome text #add-metadata here.\n")
        result = remove_marker_tag(content, "add-metadata")
        _, body = split_frontmatter(result)
        assert "#add-metadata" not in body

    def test_removes_standalone_hashtag_line(self):
        content = _note(body="# Title\n\n#add-metadata\n\nOther content.\n")
        result = remove_marker_tag(content, "add-metadata")
        _, body = split_frontmatter(result)
        assert "#add-metadata" not in body

    def test_body_other_lines_unchanged(self):
        """Lines without the tag must be byte-for-byte identical."""
        content = _note(body="# Title\n\nLine 1.\nLine 2 #add-metadata\nLine 3.\n")
        result = remove_marker_tag(content, "add-metadata")
        _, body = split_frontmatter(result)
        assert "Line 1." in body
        assert "Line 3." in body

    def test_no_double_space_after_removal(self):
        content = _note(body="# Title\n\nword #add-metadata word\n")
        result = remove_marker_tag(content, "add-metadata")
        assert "  " not in result  # no double spaces

    def test_partial_match_in_body_not_removed(self):
        """'#add-metadata-extra' must NOT be removed when removing 'add-metadata'."""
        content = _note(body="# Title\n\n#add-metadata-extra\n")
        result = remove_marker_tag(content, "add-metadata")
        _, body = split_frontmatter(result)
        assert "#add-metadata-extra" in body

    def test_case_insensitive_removal_body(self):
        content = _note(body="# Title\n\n#Add-Metadata\n")
        result = remove_marker_tag(content, "add-metadata")
        _, body = split_frontmatter(result)
        assert "#Add-Metadata" not in body
        assert "#add-metadata" not in body


# ---------------------------------------------------------------------------
# remove_marker_tag — combined
# ---------------------------------------------------------------------------


class TestRemoveMarkerTagCombined:
    def test_removes_from_both(self):
        content = _note(
            "tags:\n  - add-metadata\n  - topic1",
            body="# Title\n\nSome text #add-metadata.\n",
        )
        result = remove_marker_tag(content, "add-metadata")
        fm, body = split_frontmatter(result)
        tag_values = [str(t).lstrip("#").lower() for t in fm.get("tags", [])]
        assert "add-metadata" not in tag_values
        assert "#add-metadata" not in body
        assert "topic1" in fm.get("tags", [])

    def test_noop_when_completely_absent(self):
        content = _note("tags:\n  - topic1", body="# Title\n\nPlain body.\n")
        result = remove_marker_tag(content, "add-metadata")
        assert result == content
