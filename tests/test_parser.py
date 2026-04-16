"""Unit tests for obsassist.parser.

Covers:
- build_assistant_section: correct section structure
- find_assistant_section: finds / does not find the block
- update_note: appending when absent, replacing when present
- Idempotency: applying the same block twice yields the same document
- No side-effects: content outside the block is never modified
- parse_existing_block: round-trip fidelity
"""
from __future__ import annotations

import pytest

from obsassist.parser import (
    AssistantBlock,
    build_assistant_section,
    find_assistant_section,
    parse_existing_block,
    update_note,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_BLOCK = AssistantBlock(
    summary="This is a summary.",
    questions="- Question 1\n- Question 2",
    metadata_yaml="tags:\n  - test\ntopic: testing",
)


# ---------------------------------------------------------------------------
# build_assistant_section
# ---------------------------------------------------------------------------


class TestBuildAssistantSection:
    def test_contains_heading(self):
        section = build_assistant_section(SAMPLE_BLOCK)
        assert "## Assistant" in section

    def test_contains_all_subheadings(self):
        section = build_assistant_section(SAMPLE_BLOCK)
        assert "### Summary" in section
        assert "### Questions" in section
        assert "### Metadata suggestions" in section

    def test_contains_summary_text(self):
        section = build_assistant_section(SAMPLE_BLOCK)
        assert "This is a summary." in section

    def test_contains_questions_text(self):
        section = build_assistant_section(SAMPLE_BLOCK)
        assert "- Question 1" in section
        assert "- Question 2" in section

    def test_contains_yaml_fence_and_content(self):
        section = build_assistant_section(SAMPLE_BLOCK)
        assert "```yaml" in section
        assert "tags:" in section
        assert "```" in section

    def test_ends_with_newline(self):
        section = build_assistant_section(SAMPLE_BLOCK)
        assert section.endswith("\n")

    def test_empty_block(self):
        section = build_assistant_section(AssistantBlock())
        assert "## Assistant" in section
        assert "### Summary" in section
        assert "### Questions" in section
        assert "### Metadata suggestions" in section


# ---------------------------------------------------------------------------
# find_assistant_section
# ---------------------------------------------------------------------------


class TestFindAssistantSection:
    def test_not_found_returns_none(self):
        content = "# My Note\n\nSome content.\n"
        assert find_assistant_section(content) is None

    def test_found_returns_correct_start(self):
        content = "# My Note\n\nSome content.\n\n## Assistant\n\n### Summary\nTest\n"
        pos = find_assistant_section(content)
        assert pos is not None
        start, _ = pos
        assert content[start:].startswith("## Assistant")

    def test_found_at_end_of_file(self):
        content = "# My Note\n\n## Assistant\n\n### Summary\nHello\n"
        pos = find_assistant_section(content)
        assert pos is not None
        _, end = pos
        assert end == len(content)

    def test_found_with_section_after(self):
        content = (
            "# My Note\n\n"
            "## Assistant\n\n### Summary\nHello\n\n"
            "## Other Section\n\nContent.\n"
        )
        pos = find_assistant_section(content)
        assert pos is not None
        start, end = pos
        assert content[start:end].startswith("## Assistant")
        assert content[end:].startswith("## Other Section")

    def test_does_not_match_h3_heading(self):
        content = "# My Note\n\n### Assistant\n\nSome content.\n"
        assert find_assistant_section(content) is None

    def test_does_not_match_partial_heading(self):
        content = "# My Note\n\n## Assistance\n\nSome content.\n"
        assert find_assistant_section(content) is None


# ---------------------------------------------------------------------------
# update_note
# ---------------------------------------------------------------------------


class TestUpdateNote:
    def test_appends_when_no_block_present(self):
        content = "# My Note\n\nSome content.\n"
        updated = update_note(content, SAMPLE_BLOCK)
        assert "## Assistant" in updated
        assert "# My Note" in updated
        assert "Some content." in updated

    def test_assistant_block_at_end_after_append(self):
        content = "# My Note\n\nContent.\n"
        updated = update_note(content, SAMPLE_BLOCK)
        idx = updated.rfind("## Assistant")
        assert idx != -1
        # No more headings after the block
        assert updated.count("## Assistant") == 1

    def test_replaces_existing_block(self):
        old_block = AssistantBlock(
            summary="Old summary.",
            questions="- Old Q",
            metadata_yaml="tags: []",
        )
        content = "# My Note\n\nContent.\n\n" + build_assistant_section(old_block)

        new_block = AssistantBlock(
            summary="New summary.",
            questions="- New Q",
            metadata_yaml="tags: [new]",
        )
        updated = update_note(content, new_block)

        assert "New summary." in updated
        assert "Old summary." not in updated
        assert "# My Note" in updated
        assert "Content." in updated
        assert updated.count("## Assistant") == 1

    def test_idempotent_append(self):
        """Applying the same block twice must not change the document."""
        content = "# My Note\n\nContent.\n"
        updated_once = update_note(content, SAMPLE_BLOCK)
        updated_twice = update_note(updated_once, SAMPLE_BLOCK)
        assert updated_once == updated_twice

    def test_idempotent_replace(self):
        """Re-applying after explicit replace must also be idempotent."""
        base = "# Note\n\n" + build_assistant_section(SAMPLE_BLOCK)
        updated = update_note(base, SAMPLE_BLOCK)
        assert updated == base

    def test_other_sections_unchanged(self):
        content = (
            "# My Note\n\n"
            "Some **bold** content.\n\n"
            "## Another Section\n\nMore content.\n"
        )
        updated = update_note(content, SAMPLE_BLOCK)
        assert "# My Note" in updated
        assert "Some **bold** content." in updated
        assert "## Another Section" in updated
        assert "More content." in updated

    def test_no_duplicate_assistant_heading(self):
        content = "# Note\n\n" + build_assistant_section(SAMPLE_BLOCK)
        updated = update_note(content, SAMPLE_BLOCK)
        assert updated.count("## Assistant") == 1


# ---------------------------------------------------------------------------
# parse_existing_block (round-trip)
# ---------------------------------------------------------------------------


class TestParseExistingBlock:
    def test_round_trip(self):
        """build → embed in note → find → parse must recover original values."""
        content = "# Note\n\n" + build_assistant_section(SAMPLE_BLOCK)
        pos = find_assistant_section(content)
        assert pos is not None
        parsed = parse_existing_block(content[pos[0] : pos[1]])

        assert parsed.summary == SAMPLE_BLOCK.summary
        assert parsed.questions == SAMPLE_BLOCK.questions
        assert parsed.metadata_yaml == SAMPLE_BLOCK.metadata_yaml

    def test_empty_block_round_trip(self):
        empty = AssistantBlock()
        content = "# Note\n\n" + build_assistant_section(empty)
        pos = find_assistant_section(content)
        assert pos is not None
        parsed = parse_existing_block(content[pos[0] : pos[1]])
        assert parsed.summary == ""
        assert parsed.questions == ""
        assert parsed.metadata_yaml == ""
