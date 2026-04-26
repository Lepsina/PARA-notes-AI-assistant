"""Tag scanning utilities for the ``metadata apply`` batch workflow.

Provides helpers to:

* Detect whether a note contains a given marker tag (in frontmatter *tags*
  list or as an inline ``#tag`` in the body).
* Strip the marker tag from a note's content after a successful metadata
  update.
"""
from __future__ import annotations

import re

from .metadata_guard import build_content, split_frontmatter


def has_marker_tag(content: str, tag: str) -> bool:
    """Return *True* when *content* contains the marker *tag*.

    The marker is recognised in **two** places:

    * Frontmatter ``tags`` list — the bare tag name (``add-metadata``) is
      compared case-insensitively against each entry after stripping any
      leading ``#``.
    * Markdown body — the hashtag form (``#add-metadata``) appearing as a
      standalone token (not preceded by ``#`` or word/dash chars, not
      followed by word/dash chars).

    Args:
        content: Full note content (frontmatter + body).
        tag: Tag name with or without leading ``#`` (e.g. ``"add-metadata"``
             or ``"#add-metadata"``).
    """
    bare = tag.lstrip("#").lower()
    fm, body = split_frontmatter(content)

    # 1. Frontmatter tags list
    for t in fm.get("tags") or []:
        if str(t).lstrip("#").lower() == bare:
            return True

    # 2. Body inline hashtag (#add-metadata)
    if _body_has_tag(body, bare):
        return True

    return False


def remove_marker_tag(content: str, tag: str) -> str:
    """Remove the marker *tag* from *content*.

    Removes the tag from:

    * The frontmatter ``tags`` list (bare form, case-insensitive).
    * The markdown body (``#tag`` hashtag occurrences).

    The rest of the note is left byte-for-byte identical except for
    whitespace normalisation on lines where the tag was the only token.

    Args:
        content: Full note content.
        tag: Tag name with or without leading ``#``.
    """
    bare = tag.lstrip("#").lower()
    fm, body = split_frontmatter(content)

    # Short-circuit: nothing to remove
    if not has_marker_tag(content, tag):
        return content

    # --- Remove from frontmatter tags list ---
    existing_tags: list = list(fm.get("tags") or [])
    new_tags = [t for t in existing_tags if str(t).lstrip("#").lower() != bare]
    if len(new_tags) != len(existing_tags):
        fm = dict(fm)
        if new_tags:
            fm["tags"] = new_tags
        else:
            fm.pop("tags", None)

    # --- Remove from body ---
    new_body = _remove_body_tag(body, bare)

    return build_content(fm, new_body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_tag_pattern(bare: str) -> re.Pattern[str]:
    """Compile a regex that matches ``#bare`` as a standalone hashtag."""
    # Not preceded by #, word-char, or dash; not followed by word-char or dash.
    return re.compile(
        r"(?<![#\w-])#" + re.escape(bare) + r"(?![\w-])",
        re.IGNORECASE,
    )


def _body_has_tag(body: str, bare: str) -> bool:
    """Return True when *bare* tag appears as ``#bare`` in *body*."""
    return bool(_make_tag_pattern(bare).search(body))


def _remove_body_tag(body: str, bare: str) -> str:
    """Remove ``#bare`` hashtag occurrences from *body*.

    Only cleans up extra horizontal whitespace on lines where the tag was
    actually removed; all other lines are left byte-for-byte unchanged.
    """
    pattern = _make_tag_pattern(bare)
    lines = body.split("\n")
    new_lines: list[str] = []
    for line in lines:
        new_line, n_subs = pattern.subn("", line)
        if n_subs:
            # Collapse multiple spaces created by removal on this line only
            new_line = re.sub(r" {2,}", " ", new_line)
            new_line = new_line.rstrip()
        new_lines.append(new_line)
    return "\n".join(new_lines)
