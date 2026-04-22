"""Markdown-aware text chunker for semantic retrieval.

Each document is split into :class:`Chunk` objects first by heading
boundaries, then by character size with overlap when a section is too large.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


@dataclass
class Chunk:
    """A contiguous text segment from a vault note."""

    chunk_id: str       # Stable ID: sha256(path:chunk_index)[:16]
    path: str           # Relative posix path in the vault
    chunk_index: int    # 0-based index within the document
    heading: str        # Nearest heading text (empty if before first heading)
    content: str        # Chunk text (stripped)
    char_start: int     # Start offset in the original document
    char_end: int       # End offset in the original document


def make_chunk_id(path: str, chunk_index: int) -> str:
    """Return a 16-char hex stable ID for (path, chunk_index)."""
    return hashlib.sha256(f"{path}:{chunk_index}".encode()).hexdigest()[:16]


def chunk_document(
    path: str,
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split *content* into overlapping chunks, respecting heading boundaries.

    Strategy:
    1. Split by headings into sections.
    2. Sections that fit in *chunk_size* chars become one chunk.
    3. Larger sections are further split with a sliding window of *chunk_overlap*.

    Args:
        path:          Relative vault path (used to generate stable chunk IDs).
        content:       Full document text.
        chunk_size:    Target maximum characters per chunk.
        chunk_overlap: Character overlap when splitting oversized sections.

    Returns:
        Ordered list of :class:`Chunk` objects.
    """
    sections = _split_by_headings(content)

    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, section_text, section_start in sections:
        if not section_text.strip():
            continue

        if len(section_text) <= chunk_size:
            chunk_id = make_chunk_id(path, chunk_index)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    path=path,
                    chunk_index=chunk_index,
                    heading=heading,
                    content=section_text.strip(),
                    char_start=section_start,
                    char_end=section_start + len(section_text),
                )
            )
            chunk_index += 1
        else:
            for sub_text, sub_offset in _split_by_size(section_text, chunk_size, chunk_overlap):
                if not sub_text.strip():
                    continue
                chunk_id = make_chunk_id(path, chunk_index)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        path=path,
                        chunk_index=chunk_index,
                        heading=heading,
                        content=sub_text.strip(),
                        char_start=section_start + sub_offset,
                        char_end=section_start + sub_offset + len(sub_text),
                    )
                )
                chunk_index += 1

    # Fallback: if nothing was produced for non-empty content
    if not chunks and content.strip():
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(path, 0),
                path=path,
                chunk_index=0,
                heading="",
                content=content.strip(),
                char_start=0,
                char_end=len(content),
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_by_headings(content: str) -> list[tuple[str, str, int]]:
    """Return ``[(heading, section_text, char_start), …]``.

    The heading line itself is included in *section_text*.
    Content before the first heading is yielded with an empty heading.
    """
    matches = list(_HEADING_RE.finditer(content))

    if not matches:
        return [("", content, 0)]

    sections: list[tuple[str, str, int]] = []

    # Text before the first heading
    pre = content[: matches[0].start()]
    if pre.strip():
        sections.append(("", pre, 0))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        section_start = match.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[section_start:section_end]
        sections.append((heading_text, section_text, section_start))

    return sections


def _split_by_size(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int]]:
    """Split *text* into overlapping windows.

    Returns a list of ``(chunk_text, offset_in_text)`` tuples.
    The stride between windows is ``chunk_size - chunk_overlap``.
    """
    if len(text) <= chunk_size:
        return [(text, 0)]

    stride = max(1, chunk_size - chunk_overlap)
    chunks: list[tuple[str, int]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((text[start:end], start))
        if end >= len(text):
            break
        start += stride

    return chunks
