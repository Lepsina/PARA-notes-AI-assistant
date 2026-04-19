"""Full-text search over the SQLite FTS5 index."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchResult:
    path: str
    title: str
    snippet: str
    rank: float


def search(
    index_path: Path,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Search the FTS5 index and return ranked results.

    Args:
        index_path: Path to the SQLite database file.
        query: FTS5 query string (plain text or FTS5 syntax).
        limit: Maximum number of results to return.

    Returns:
        List of :class:`SearchResult` sorted by relevance (best first).
    """
    if not index_path.exists():
        return []

    conn = sqlite3.connect(str(index_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT
                d.path,
                d.title,
                snippet(documents_fts, 5, '[', ']', '…', 20) AS snippet,
                documents_fts.rank                             AS rank
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Table doesn't exist yet or malformed query
        conn.close()
        return []

    results = [
        SearchResult(
            path=row["path"],
            title=row["title"],
            snippet=row["snippet"] or "",
            rank=float(row["rank"]),
        )
        for row in rows
    ]
    conn.close()
    return results
