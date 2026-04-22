"""Embeddings indexing pipeline.

Tables added to the shared SQLite DB
--------------------------------------
chunks      -- normalized text segments from every vault note
embeddings  -- vector representations of chunks

Public API
----------
build_embeddings(vault_root, index_path, cfg, embed_fn)   -- full rebuild
update_embeddings(vault_root, index_path, cfg, embed_fn)  -- incremental
"""
from __future__ import annotations

import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .chunker import chunk_document
from .config import Config
from .indexer import _file_hash, _iter_vault_files

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CHUNKS_DDL = """\
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    doc_hash    TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    heading     TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL,
    char_start  INTEGER NOT NULL DEFAULT 0,
    char_end    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(path, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _open_embeddings_db(index_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite DB and apply chunks/embeddings schema."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_CHUNKS_DDL)
    conn.commit()
    return conn


def _encode_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

EmbedFn = Callable[[list[str]], list[list[float]]]


def build_embeddings(
    vault_root: Path,
    index_path: Path,
    cfg: Config,
    embed_fn: EmbedFn,
) -> tuple[int, int]:
    """Full rebuild: drop all chunks/embeddings and re-index every vault file.

    Args:
        vault_root: Path to the Obsidian vault.
        index_path: Path to the shared SQLite DB file.
        cfg:        Tool configuration.
        embed_fn:   ``embed_fn(texts) -> list[vector]`` — called in batches.

    Returns:
        ``(embedded_count, skipped_count)``
    """
    conn = _open_embeddings_db(index_path)
    emb_cfg = cfg.embeddings
    embedded = 0
    skipped = 0

    with conn:
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM chunks")

        for file_path in _iter_vault_files(vault_root, cfg):
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = file_path.relative_to(vault_root).as_posix()
                doc_hash = _file_hash(file_path)

                chunks = chunk_document(
                    rel_path,
                    content,
                    chunk_size=emb_cfg.chunk_size,
                    chunk_overlap=emb_cfg.chunk_overlap,
                )

                if not chunks:
                    continue

                for c in chunks:
                    conn.execute(
                        "INSERT OR REPLACE INTO chunks"
                        "(id, path, doc_hash, chunk_index, heading, content, char_start, char_end)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (c.chunk_id, c.path, doc_hash, c.chunk_index,
                         c.heading, c.content, c.char_start, c.char_end),
                    )

                for batch_start in range(0, len(chunks), emb_cfg.batch_size):
                    batch = chunks[batch_start: batch_start + emb_cfg.batch_size]
                    vectors = embed_fn([c.content for c in batch])
                    now = _now_iso()
                    for chunk, vec in zip(batch, vectors):
                        conn.execute(
                            "INSERT OR REPLACE INTO embeddings"
                            "(chunk_id, model, vector, dim, updated_at)"
                            " VALUES (?, ?, ?, ?, ?)",
                            (chunk.chunk_id, emb_cfg.model,
                             _encode_vector(vec), len(vec), now),
                        )
                    embedded += len(batch)
            except Exception:
                skipped += 1

    conn.close()
    return embedded, skipped


def update_embeddings(
    vault_root: Path,
    index_path: Path,
    cfg: Config,
    embed_fn: EmbedFn,
) -> tuple[int, int, int]:
    """Incremental update: only rechunk+re-embed changed or new files.

    Changed detection is based on the file content hash stored in ``chunks.doc_hash``.

    Returns:
        ``(updated_count, deleted_count, skipped_count)``
    """
    conn = _open_embeddings_db(index_path)
    emb_cfg = cfg.embeddings
    updated = 0
    deleted = 0
    skipped = 0

    # Map path → known doc_hash
    existing_hashes: dict[str, str] = {
        row[0]: row[1]
        for row in conn.execute("SELECT DISTINCT path, doc_hash FROM chunks")
    }

    current_paths: set[str] = set()

    with conn:
        for file_path in _iter_vault_files(vault_root, cfg):
            try:
                rel_path = file_path.relative_to(vault_root).as_posix()
                current_paths.add(rel_path)
                new_hash = _file_hash(file_path)

                if existing_hashes.get(rel_path) == new_hash:
                    continue  # unchanged — skip

                content = file_path.read_text(encoding="utf-8", errors="replace")
                chunks = chunk_document(
                    rel_path,
                    content,
                    chunk_size=emb_cfg.chunk_size,
                    chunk_overlap=emb_cfg.chunk_overlap,
                )

                # Remove old data for this file
                old_ids = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM chunks WHERE path=?", (rel_path,)
                    )
                ]
                if old_ids:
                    conn.execute(
                        "DELETE FROM embeddings WHERE chunk_id IN ({})".format(
                            ",".join("?" * len(old_ids))
                        ),
                        old_ids,
                    )
                conn.execute("DELETE FROM chunks WHERE path=?", (rel_path,))

                if not chunks:
                    continue

                for c in chunks:
                    conn.execute(
                        "INSERT OR REPLACE INTO chunks"
                        "(id, path, doc_hash, chunk_index, heading, content, char_start, char_end)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (c.chunk_id, c.path, new_hash, c.chunk_index,
                         c.heading, c.content, c.char_start, c.char_end),
                    )

                for batch_start in range(0, len(chunks), emb_cfg.batch_size):
                    batch = chunks[batch_start: batch_start + emb_cfg.batch_size]
                    vectors = embed_fn([c.content for c in batch])
                    now = _now_iso()
                    for chunk, vec in zip(batch, vectors):
                        conn.execute(
                            "INSERT OR REPLACE INTO embeddings"
                            "(chunk_id, model, vector, dim, updated_at)"
                            " VALUES (?, ?, ?, ?, ?)",
                            (chunk.chunk_id, emb_cfg.model,
                             _encode_vector(vec), len(vec), now),
                        )

                updated += 1
            except Exception:
                skipped += 1

        # Remove deleted files
        for path in set(existing_hashes) - current_paths:
            old_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM chunks WHERE path=?", (path,)
                )
            ]
            if old_ids:
                conn.execute(
                    "DELETE FROM embeddings WHERE chunk_id IN ({})".format(
                        ",".join("?" * len(old_ids))
                    ),
                    old_ids,
                )
            conn.execute("DELETE FROM chunks WHERE path=?", (path,))
            deleted += 1

    conn.close()
    return updated, deleted, skipped
