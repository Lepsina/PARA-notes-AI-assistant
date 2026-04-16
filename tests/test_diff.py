"""Unit tests for obsassist.diff."""
from __future__ import annotations

import pytest

from obsassist.diff import generate_diff


class TestGenerateDiff:
    def test_identical_content_returns_empty(self):
        content = "# My Note\n\nSome content.\n"
        assert generate_diff(content, content) == ""

    def test_modified_line_shows_minus_and_plus(self):
        original = "# My Note\n\nOld content.\n"
        updated = "# My Note\n\nNew content.\n"
        diff = generate_diff(original, updated)
        assert diff != ""
        assert "-Old content." in diff
        assert "+New content." in diff

    def test_addition_shows_plus(self):
        original = "# My Note\n\nContent.\n"
        updated = "# My Note\n\nContent.\n\n## Assistant\n\n### Summary\nTest\n"
        diff = generate_diff(original, updated)
        assert diff != ""
        assert "+## Assistant" in diff

    def test_deletion_shows_minus(self):
        original = "Line A\nLine B\nLine C\n"
        updated = "Line A\nLine C\n"
        diff = generate_diff(original, updated)
        assert "-Line B" in diff

    def test_filename_in_diff_header(self):
        diff = generate_diff("old\n", "new\n", filename="myfile.md")
        assert "myfile.md" in diff

    def test_default_filename_in_header(self):
        diff = generate_diff("old\n", "new\n")
        assert "note.md" in diff

    def test_unified_diff_header_format(self):
        diff = generate_diff("a\n", "b\n", filename="test.md")
        assert "--- a/test.md" in diff
        assert "+++ b/test.md" in diff

    def test_empty_to_content(self):
        diff = generate_diff("", "hello\n")
        assert "+hello" in diff

    def test_content_to_empty(self):
        diff = generate_diff("hello\n", "")
        assert "-hello" in diff

    def test_multiline_change(self):
        original = "line1\nline2\nline3\n"
        updated = "line1\nchanged\nline3\n"
        diff = generate_diff(original, updated)
        assert "-line2" in diff
        assert "+changed" in diff
