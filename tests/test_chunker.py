"""Tests for obsassist.chunker."""
from __future__ import annotations

from obsassist.chunker import Chunk, chunk_document, make_chunk_id


class TestMakeChunkId:
    def test_stable(self):
        assert make_chunk_id("a/b.md", 0) == make_chunk_id("a/b.md", 0)

    def test_different_index(self):
        assert make_chunk_id("a/b.md", 0) != make_chunk_id("a/b.md", 1)

    def test_different_path(self):
        assert make_chunk_id("a/b.md", 0) != make_chunk_id("x/y.md", 0)

    def test_length(self):
        assert len(make_chunk_id("note.md", 0)) == 16


class TestChunkDocument:
    def test_empty_returns_empty(self):
        assert chunk_document("note.md", "") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_document("note.md", "   \n\n  ") == []

    def test_single_small_section(self):
        content = "# Hello\n\nThis is a short note."
        chunks = chunk_document("note.md", content)
        assert len(chunks) == 1
        assert chunks[0].heading == "Hello"
        assert "short note" in chunks[0].content

    def test_chunk_ids_are_stable(self):
        content = "# Section\n\nSome text here."
        c1 = chunk_document("note.md", content)
        c2 = chunk_document("note.md", content)
        assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]

    def test_chunk_ids_change_with_path(self):
        content = "# Section\n\nText."
        c1 = chunk_document("note.md", content)
        c2 = chunk_document("other.md", content)
        assert c1[0].chunk_id != c2[0].chunk_id

    def test_multiple_headings(self):
        content = "# H1\n\nBody1.\n\n## H2\n\nBody2.\n\n### H3\n\nBody3."
        chunks = chunk_document("note.md", content)
        headings = [c.heading for c in chunks]
        assert "H1" in headings
        assert "H2" in headings
        assert "H3" in headings

    def test_pre_heading_content(self):
        content = "Intro text before any heading.\n\n# Title\n\nBody."
        chunks = chunk_document("note.md", content)
        # First chunk should have empty heading (pre-heading content)
        assert chunks[0].heading == ""
        assert "Intro" in chunks[0].content

    def test_chunk_index_sequential(self):
        content = "# A\n\nText A.\n\n## B\n\nText B.\n\n### C\n\nText C."
        chunks = chunk_document("note.md", content)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_large_section_split_by_size(self):
        # Create a section larger than chunk_size=50
        body = "word " * 20  # 100 chars
        content = f"# Big Section\n\n{body}"
        chunks = chunk_document("note.md", content, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert c.heading == "Big Section"

    def test_overlap_creates_shared_content(self):
        body = "A" * 100
        content = f"# Section\n\n{body}"
        chunks = chunk_document("note.md", content, chunk_size=50, chunk_overlap=20)
        # With overlap=20, the end of one chunk overlaps the start of the next
        assert len(chunks) >= 2
        # Second chunk should start within the overlap of first chunk's end
        first_end = chunks[0].char_end
        second_start = chunks[1].char_start
        assert second_start < first_end

    def test_char_offsets_within_document(self):
        content = "# Title\n\nSome text here."
        chunks = chunk_document("note.md", content)
        for c in chunks:
            assert 0 <= c.char_start < c.char_end <= len(content)

    def test_no_empty_chunks(self):
        content = "# H1\n\n\n\n## H2\n\nActual content."
        chunks = chunk_document("note.md", content)
        for c in chunks:
            assert c.content.strip() != ""

    def test_path_stored_in_chunk(self):
        content = "# Title\n\nText."
        chunks = chunk_document("projects/alpha.md", content)
        assert all(c.path == "projects/alpha.md" for c in chunks)

    def test_fallback_no_headings(self):
        content = "No headings here, just plain text content."
        chunks = chunk_document("note.md", content)
        assert len(chunks) == 1
        assert chunks[0].heading == ""
        assert "plain text" in chunks[0].content
