"""SQLite FTS5-backed full-text index for Obsidian vault notes.

Public API
----------
build_index(vault_root, index_path, cfg)    -- full rebuild from scratch
update_index(vault_root, index_path, cfg)   -- incremental update
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import Config
from .filters import is_excluded

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS documents (
    path        TEXT    PRIMARY KEY,
    mtime       REAL    NOT NULL,
    size        INTEGER NOT NULL,
    hash        TEXT    NOT NULL,
    title       TEXT    NOT NULL DEFAULT '',
    headings    TEXT    NOT NULL DEFAULT '[]',
    tags        TEXT    NOT NULL DEFAULT '[]',
    links       TEXT    NOT NULL DEFAULT '[]',
    content     TEXT    NOT NULL DEFAULT '',
    updated_at  TEXT    NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path     UNINDEXED,
    title,
    headings,
    tags,
    links,
    content,
    content=documents,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS docs_ai
AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, path, title, headings, tags, links, content)
    VALUES (new.rowid, new.path, new.title, new.headings, new.tags, new.links, new.content);
END;

CREATE TRIGGER IF NOT EXISTS docs_au
AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, path, title, headings, tags, links, content)
    VALUES ('delete', old.rowid, old.path, old.title, old.headings, old.tags, old.links, old.content);
    INSERT INTO documents_fts(rowid, path, title, headings, tags, links, content)
    VALUES (new.rowid, new.path, new.title, new.headings, new.tags, new.links, new.content);
END;

CREATE TRIGGER IF NOT EXISTS docs_ad
AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, path, title, headings, tags, links, content)
    VALUES ('delete', old.rowid, old.path, old.title, old.headings, old.tags, old.links, old.content);
END;
"""

# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)", re.MULTILINE)
_H1_RE = re.compile(r"^#\s+(.*)", re.MULTILINE)
_INLINE_TAG_RE = re.compile(r"(?<!\S)#([A-Za-z][A-Za-z0-9_/-]*)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def _strip_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_text, body) pair."""
    m = _FRONTMATTER_RE.match(text)
    if m:
        return m.group(1), text[m.end():]
    return "", text


def _parse_yaml_tags(frontmatter: str) -> list[str]:
    """Extract tags from YAML frontmatter without importing pyyaml at module level."""
    try:
        import yaml  # already a project dependency

        data = yaml.safe_load(frontmatter) or {}
        raw = data.get("tags") or data.get("tag") or []
        if isinstance(raw, str):
            raw = [t.strip() for t in raw.split(",") if t.strip()]
        return [str(t) for t in raw]
    except Exception:
        return []


def extract_metadata(path: Path, content: str) -> dict:
    """Extract structured metadata from a markdown note.

    Returns a dict with keys:
        title, headings (list), tags (list), links (list), body (str).
    """
    frontmatter_text, body = _strip_frontmatter(content)

    # Title: first H1 or filename stem
    h1_match = _H1_RE.search(body)
    title = h1_match.group(1).strip() if h1_match else path.stem

    # Headings: all heading lines from body
    headings = [m.group(1).strip() for m in _HEADING_RE.finditer(body)]

    # Tags: YAML frontmatter tags + inline #tags from body
    yaml_tags = _parse_yaml_tags(frontmatter_text)
    inline_tags = _INLINE_TAG_RE.findall(body)
    tags = list(dict.fromkeys(yaml_tags + inline_tags))  # deduplicate, preserve order

    # Wikilinks
    links = list(dict.fromkeys(_WIKILINK_RE.findall(body)))

    return {
        "title": title,
        "headings": headings,
        "tags": tags,
        "links": links,
        "body": body,
    }


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _iter_vault_files(
    vault_root: Path, cfg: Config
) -> Iterator[Path]:
    """Yield all markdown files that should be indexed."""
    extensions = set(cfg.include_extensions)
    for file_path in vault_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue
        if is_excluded(file_path, vault_root, cfg.exclude_paths):
            continue
        yield file_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _open_db(index_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite database and apply schema."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _apply_schema(conn)
    return conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.commit()


def _upsert_document(conn: sqlite3.Connection, rel_path: str, meta: dict) -> None:
    now = _now_iso()
    conn.execute(
        """
        INSERT INTO documents(path, mtime, size, hash, title, headings, tags, links, content, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            mtime      = excluded.mtime,
            size       = excluded.size,
            hash       = excluded.hash,
            title      = excluded.title,
            headings   = excluded.headings,
            tags       = excluded.tags,
            links      = excluded.links,
            content    = excluded.content,
            updated_at = excluded.updated_at
        """,
        (
            rel_path,
            meta["mtime"],
            meta["size"],
            meta["hash"],
            meta["title"],
            json.dumps(meta["headings"], ensure_ascii=False),
            json.dumps(meta["tags"], ensure_ascii=False),
            json.dumps(meta["links"], ensure_ascii=False),
            meta["body"],
            now,
        ),
    )


def _delete_document(conn: sqlite3.Connection, rel_path: str) -> None:
    conn.execute("DELETE FROM documents WHERE path = ?", (rel_path,))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index(
    vault_root: Path,
    index_path: Path,
    cfg: Config,
) -> tuple[int, int]:
    """Full rebuild: drop all rows and re-index every vault file.

    Returns (indexed_count, skipped_count).
    """
    conn = _open_db(index_path)
    indexed = 0
    skipped = 0

    with conn:
        # Wipe existing data: delete from documents only — the docs_ad
        # trigger keeps the FTS index synchronized automatically.
        conn.execute("DELETE FROM documents")

        for file_path in _iter_vault_files(vault_root, cfg):
            try:
                stat = file_path.stat()
                content = file_path.read_text(encoding="utf-8", errors="replace")
                meta = extract_metadata(file_path, content)
                meta["mtime"] = stat.st_mtime
                meta["size"] = stat.st_size
                meta["hash"] = _file_hash(file_path)
                rel_path = file_path.relative_to(vault_root).as_posix()
                _upsert_document(conn, rel_path, meta)
                indexed += 1
            except OSError:
                skipped += 1

    conn.close()
    return indexed, skipped


def update_index(
    vault_root: Path,
    index_path: Path,
    cfg: Config,
) -> tuple[int, int, int]:
    """Incremental update: add/update changed files, remove deleted ones.

    Returns (added_or_updated, deleted, skipped).
    """
    conn = _open_db(index_path)
    added_or_updated = 0
    deleted = 0
    skipped = 0

    # Load existing records for change detection
    existing: dict[str, tuple[float, int, str]] = {}  # path -> (mtime, size, hash)
    for row in conn.execute("SELECT path, mtime, size, hash FROM documents"):
        existing[row[0]] = (row[1], row[2], row[3])

    current_paths: set[str] = set()

    with conn:
        for file_path in _iter_vault_files(vault_root, cfg):
            try:
                rel_path = file_path.relative_to(vault_root).as_posix()
                stat = file_path.stat()
                current_paths.add(rel_path)

                prev = existing.get(rel_path)
                if prev is not None:
                    prev_mtime, prev_size, prev_hash = prev
                    # Quick check: mtime+size unchanged → skip
                    if stat.st_mtime == prev_mtime and stat.st_size == prev_size:
                        continue
                    # Slower check: content hash
                    new_hash = _file_hash(file_path)
                    if new_hash == prev_hash:
                        continue
                else:
                    new_hash = _file_hash(file_path)

                content = file_path.read_text(encoding="utf-8", errors="replace")
                meta = extract_metadata(file_path, content)
                meta["mtime"] = stat.st_mtime
                meta["size"] = stat.st_size
                meta["hash"] = new_hash
                _upsert_document(conn, rel_path, meta)
                added_or_updated += 1
            except OSError:
                skipped += 1

        # Remove deleted files
        for rel_path in set(existing) - current_paths:
            _delete_document(conn, rel_path)
            deleted += 1

    conn.close()
    return added_or_updated, deleted, skipped
