"""Retrieval strategies for the ask command.

Three modes
-----------
fts     -- full-text search over the FTS5 ``documents_fts`` index.
vector  -- cosine similarity over all stored chunk embeddings.
hybrid  -- FTS to gather candidates, then rerank by cosine similarity.
"""
from __future__ import annotations

import math
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkResult:
    """A retrieved text chunk with relevance metadata."""

    path: str
    heading: str
    content: str
    score: float
    chunk_index: int


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------


def _decode_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _escape_fts_query(query: str) -> str:
    """Convert a natural-language *query* into a safe FTS5 MATCH expression.

    Individual terms are joined with OR so that results are found even when
    only some words appear in the document (natural-language style matching).
    Non-alphanumeric characters and bare FTS5 operators are stripped.
    """
    # Keep only alphanumeric characters and spaces; strip punctuation
    safe = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
    # Remove bare FTS5 operators to avoid syntax errors
    terms = [
        t for t in safe.split()
        if t.upper() not in ("OR", "AND", "NOT") and t
    ]
    if not terms:
        return '""'
    return " OR ".join(terms)


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


def retrieve_fts(
    index_path: Path,
    query: str,
    k: int = 12,
) -> list[ChunkResult]:
    """Retrieve top-*k* results via FTS5 full-text search.

    If the ``chunks`` table is populated the function returns representative
    chunks (first chunk per matching document).  When chunks are absent it
    falls back to the FTS snippet as the content proxy.
    """
    if not index_path.exists():
        return []

    conn = sqlite3.connect(str(index_path))
    conn.row_factory = sqlite3.Row

    try:
        safe_q = _escape_fts_query(query)
        doc_rows = conn.execute(
            """
            SELECT d.path,
                   snippet(documents_fts, 5, '[', ']', '…', 20) AS snippet,
                   documents_fts.rank AS rank
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY documents_fts.rank
            LIMIT ?
            """,
            (safe_q, k),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []

    results: list[ChunkResult] = []
    for doc in doc_rows:
        path = doc["path"]
        rank = float(doc["rank"])

        # Try to get chunks for this document
        try:
            chunk_row = conn.execute(
                "SELECT heading, content, chunk_index FROM chunks"
                " WHERE path=? ORDER BY chunk_index LIMIT 1",
                (path,),
            ).fetchone()
        except sqlite3.OperationalError:
            chunk_row = None

        if chunk_row:
            results.append(
                ChunkResult(
                    path=path,
                    heading=chunk_row["heading"],
                    content=chunk_row["content"],
                    score=rank,
                    chunk_index=chunk_row["chunk_index"],
                )
            )
        else:
            # Fallback: use FTS snippet
            results.append(
                ChunkResult(
                    path=path,
                    heading="",
                    content=str(doc["snippet"]),
                    score=rank,
                    chunk_index=0,
                )
            )

    conn.close()
    return results


def retrieve_vector(
    index_path: Path,
    query_vector: list[float],
    k: int = 12,
) -> list[ChunkResult]:
    """Retrieve top-*k* chunks by cosine similarity to *query_vector*."""
    if not index_path.exists():
        return []

    conn = sqlite3.connect(str(index_path))

    try:
        rows = conn.execute(
            """
            SELECT c.path, c.heading, c.content, c.chunk_index, e.vector
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id
            """
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []

    conn.close()

    if not rows:
        return []

    scored: list[tuple[float, ChunkResult]] = []
    for row in rows:
        vec = _decode_vector(row[4])
        sim = cosine_similarity(query_vector, vec)
        scored.append(
            (
                sim,
                ChunkResult(
                    path=row[0],
                    heading=row[1],
                    content=row[2],
                    score=sim,
                    chunk_index=row[3],
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def retrieve_hybrid(
    index_path: Path,
    query: str,
    query_vector: list[float],
    k: int = 12,
    candidates: int = 50,
) -> list[ChunkResult]:
    """Hybrid retrieval: FTS candidate pool → vector rerank.

    1. Run FTS to collect up to *candidates* document paths.
    2. Load all chunks + embeddings for those documents.
    3. Rank chunks by cosine similarity to *query_vector*.
    4. Return top-*k*.
    """
    if not index_path.exists():
        return []

    conn = sqlite3.connect(str(index_path))
    conn.row_factory = sqlite3.Row

    try:
        safe_q = _escape_fts_query(query)
        fts_rows = conn.execute(
            """
            SELECT d.path
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY documents_fts.rank
            LIMIT ?
            """,
            (safe_q, candidates),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []

    if not fts_rows:
        conn.close()
        return []

    candidate_paths = [row["path"] for row in fts_rows]
    placeholders = ",".join("?" * len(candidate_paths))

    try:
        rows = conn.execute(
            f"""
            SELECT c.path, c.heading, c.content, c.chunk_index, e.vector
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id
            WHERE c.path IN ({placeholders})
            """,
            candidate_paths,
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []

    conn.close()

    if not rows:
        return []

    scored: list[tuple[float, ChunkResult]] = []
    for row in rows:
        vec = _decode_vector(row["vector"])
        sim = cosine_similarity(query_vector, vec)
        scored.append(
            (
                sim,
                ChunkResult(
                    path=row["path"],
                    heading=row["heading"],
                    content=row["content"],
                    score=sim,
                    chunk_index=row["chunk_index"],
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]
