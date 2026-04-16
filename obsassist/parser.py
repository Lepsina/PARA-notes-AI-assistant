"""Markdown parser for the ## Assistant block.

Responsibilities
----------------
* Build a canonical ``## Assistant`` section from an :class:`AssistantBlock`.
* Locate an existing ``## Assistant`` section in a note (returns character
  offsets so the caller can splice it).
* Update a note in-place: replace the existing block or append a new one.
* Parse the sub-sections of an existing ``## Assistant`` block back into an
  :class:`AssistantBlock` (used when only *metadata* should be refreshed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AssistantBlock:
    """Holds the content of the three sub-sections."""

    summary: str = ""
    questions: str = ""
    metadata_yaml: str = ""


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_assistant_section(block: AssistantBlock) -> str:
    """Return the full text of a ``## Assistant`` section.

    The returned string always starts with ``## Assistant`` and ends with a
    blank line so that subsequent sections in the document are separated
    correctly.
    """
    summary = block.summary.strip() or ""
    questions = block.questions.strip() or ""
    metadata_yaml = block.metadata_yaml.strip() or ""

    parts: list[str] = [
        "## Assistant\n",
        "\n",
        "### Summary\n",
        f"{summary}\n" if summary else "\n",
        "\n",
        "### Questions\n",
        f"{questions}\n" if questions else "\n",
        "\n",
        "### Metadata suggestions\n",
        "```yaml\n",
        f"{metadata_yaml}\n" if metadata_yaml else "",
        "```\n",
        "\n",  # trailing blank line for separation
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


def find_assistant_section(content: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` character offsets for the ``## Assistant`` section.

    *start* points to the ``#`` of ``## Assistant``.
    *end* points to the first character of the **next** ``## ``-level heading,
    or to ``len(content)`` if the section is the last one.

    Returns ``None`` when no ``## Assistant`` heading is present.
    """
    heading_re = re.compile(r"^## Assistant[ \t]*(?:\n|$)", re.MULTILINE)
    m = heading_re.search(content)
    if m is None:
        return None

    start = m.start()

    # End at next ## heading (same or higher level) or end of string.
    next_h2_re = re.compile(r"^## ", re.MULTILINE)
    next_m = next_h2_re.search(content, m.end())
    end = next_m.start() if next_m else len(content)

    return start, end


# ---------------------------------------------------------------------------
# Updating
# ---------------------------------------------------------------------------


def update_note(content: str, block: AssistantBlock) -> str:
    """Insert or replace the ``## Assistant`` block in *content*.

    * If the note already contains a ``## Assistant`` block, **only** that
      block is replaced; everything else stays unchanged.
    * Otherwise the block is appended after two newlines at the end of the
      note.
    """
    new_section = build_assistant_section(block)

    pos = find_assistant_section(content)
    if pos is not None:
        start, end = pos
        return content[:start] + new_section + content[end:]

    # Append at end, ensuring exactly one blank line separator.
    stripped = content.rstrip("\n")
    return stripped + "\n\n" + new_section


# ---------------------------------------------------------------------------
# Parsing an existing block
# ---------------------------------------------------------------------------


def parse_existing_block(section_content: str) -> AssistantBlock:
    """Parse the contents of an existing ``## Assistant`` section.

    *section_content* should be the raw text slice of the note that was
    returned by :func:`find_assistant_section`.
    """
    summary = _extract_subsection(section_content, "Summary")
    questions = _extract_subsection(section_content, "Questions")
    metadata_raw = _extract_subsection(section_content, "Metadata suggestions")

    # Strip YAML code fence if present
    metadata_yaml = ""
    if metadata_raw:
        yaml_m = re.search(r"```(?:yaml)?[ \t]*\n(.*?)```", metadata_raw, re.DOTALL)
        if yaml_m:
            metadata_yaml = yaml_m.group(1).strip()
        else:
            metadata_yaml = metadata_raw.strip()

    return AssistantBlock(
        summary=summary,
        questions=questions,
        metadata_yaml=metadata_yaml,
    )


def _extract_subsection(content: str, section_name: str) -> str:
    """Extract the text body of a ``### <section_name>`` sub-section.

    Works by splitting on ``^### `` boundaries (MULTILINE) and returning
    the body of the matching heading.  Returns ``""`` when not found.
    """
    # Split on ### headings; each element starts with the heading line.
    parts = re.split(r"^### ", content, flags=re.MULTILINE)
    for part in parts:
        newline_pos = part.find("\n")
        if newline_pos == -1:
            continue
        heading = part[:newline_pos].strip()
        if heading == section_name:
            return part[newline_pos + 1 :].strip()
    return ""
