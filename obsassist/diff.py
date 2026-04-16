"""Unified diff generation between original and updated note content."""
from __future__ import annotations

import difflib


def generate_diff(
    original: str,
    updated: str,
    filename: str = "note.md",
) -> str:
    """Return a unified diff string between *original* and *updated*.

    Returns an empty string when there are no differences.

    Args:
        original: The original file content.
        updated:  The updated file content.
        filename: Filename used in the diff header lines.
    """
    if original == updated:
        return ""

    original_lines = original.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )
    return "".join(diff_lines)
