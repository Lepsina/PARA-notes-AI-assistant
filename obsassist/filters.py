"""Path-exclusion logic for vault files."""
from __future__ import annotations

from pathlib import Path

#: Paths that are excluded by default (relative to vault root).
DEFAULT_EXCLUDE_PATHS: list[str] = [
    "Resources/",
    "Templates/",
    "Files/",
    "Excalidraw/",
    ".obsidian/",
]


def is_excluded(
    path: Path,
    vault_root: Path | None,
    exclude_paths: list[str],
) -> bool:
    """Return *True* when *path* should be skipped.

    A path is excluded when:

    * It cannot be made relative to *vault_root* (i.e. it lives outside the
      vault).  Only checked when *vault_root* is not ``None``.
    * Its POSIX-normalised relative path starts with one of the prefixes in
      *exclude_paths* (e.g. ``"Resources/"``).

    Args:
        path: Absolute (or relative) path to the file being checked.
        vault_root: Root directory of the Obsidian vault.  Pass ``None`` to
            skip the vault-membership check and only apply prefix rules
            against *path* itself.
        exclude_paths: List of directory prefix strings such as
            ``["Resources/", ".obsidian/"]``.
    """
    if vault_root is not None:
        try:
            rel = path.relative_to(vault_root)
        except ValueError:
            return True  # outside vault → always exclude
    else:
        rel = path

    rel_posix = rel.as_posix()

    for excl in exclude_paths:
        # Normalise the exclude entry: strip surrounding slashes / backslashes.
        excl_norm = excl.strip("/\\").replace("\\", "/")
        if rel_posix == excl_norm or rel_posix.startswith(excl_norm + "/"):
            return True

    return False
