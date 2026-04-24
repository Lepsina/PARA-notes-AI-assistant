"""Strict metadata guardrails for YAML frontmatter updates.

Responsibilities
----------------
* Parse existing YAML frontmatter from a note without touching the body.
* Sanitize LLM-suggested metadata: schema allowlist, type coercion, key/value
  alias normalisation.
* Load and apply vocabulary normalisation from an external file.
* Conservative merge: fill missing fields only; never overwrite user values
  (unless *force=True*).
* Rebuild frontmatter and reconstruct the note with body byte-for-byte
  unchanged.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "title",
        "created",
        "updated",
        "type",
        "status",
        "lang",
        "tags",
        "topics",
        "entities",
        "summary",
        "priority",
        "source_type",
        "exclude_from_ai",
        "aliases",
    }
)

ARRAY_FIELDS: frozenset[str] = frozenset({"tags", "topics", "entities", "aliases"})
BOOL_FIELDS: frozenset[str] = frozenset({"exclude_from_ai"})

# Legacy key aliases: old_key (lower) → canonical_key
KEY_ALIASES: dict[str, str] = {
    "topic": "topics",
    "tag": "tags",
    "alias": "aliases",
    "entity": "entities",
}

# Status value aliases (lower → canonical)
STATUS_ALIASES: dict[str, str] = {
    "complete": "done",
    "completed": "done",
    "in_progress": "active",
    "in-progress": "active",
}

# ---------------------------------------------------------------------------
# Frontmatter parsing / writing
# ---------------------------------------------------------------------------

_FM_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)


def split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter_dict, body)``.

    *body* is the raw bytes after the closing ``---`` delimiter.  If no
    frontmatter is present the dict is empty and *body* equals *content*.
    """
    m = _FM_RE.match(content)
    if m:
        try:
            fm: dict[str, Any] = yaml.safe_load(m.group(1)) or {}
            if not isinstance(fm, dict):
                fm = {}
        except yaml.YAMLError:
            fm = {}
        body = content[m.end():]
        return fm, body
    return {}, content


def build_content(fm: dict[str, Any], body: str) -> str:
    """Reconstruct note content from *fm* dict and *body* text.

    If *fm* is empty, returns *body* unchanged (no frontmatter section).
    """
    if not fm:
        return body
    fm_text = yaml.dump(
        fm,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).rstrip("\n")
    return f"---\n{fm_text}\n---\n{body}"


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------


def load_vocab(vocab_path: Path | str | None) -> dict[str, Any]:
    """Load a vocabulary YAML file; return an empty dict on any error."""
    if not vocab_path:
        return {}
    path = Path(vocab_path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
    except yaml.YAMLError as exc:
        import warnings
        warnings.warn(f"obsassist: could not parse vocab file {path}: {exc}", stacklevel=2)
        return {}
    except OSError as exc:
        import warnings
        warnings.warn(f"obsassist: could not read vocab file {path}: {exc}", stacklevel=2)
        return {}


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------


def sanitize(
    raw: dict[str, Any],
    *,
    allowed_keys: frozenset[str] | set[str] = ALLOWED_KEYS,
    vocab: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sanitize and normalise LLM-suggested metadata.

    Steps applied in order:

    1. Rename legacy key aliases (``topic`` → ``topics``, etc.).
    2. Drop keys not in *allowed_keys*.
    3. Coerce array fields to ``list[str]``; coerce bool fields.
    4. Normalise status value aliases.
    5. Apply vocab normalisation (topics, priority_from_tags).
    """
    if not isinstance(raw, dict):
        return {}

    # Step 1: rename key aliases
    renamed: dict[str, Any] = {}
    for k, v in raw.items():
        canonical_key = KEY_ALIASES.get(str(k).lower(), str(k))
        renamed[canonical_key] = v

    # Step 2: drop unknown keys
    result = {k: v for k, v in renamed.items() if k in allowed_keys}

    # Step 3: coerce types
    for field in ARRAY_FIELDS:
        if field in result:
            result[field] = _to_list(result[field])
    for field in BOOL_FIELDS:
        if field in result:
            result[field] = bool(result[field])

    # Step 4: normalise status
    if "status" in result:
        s = str(result["status"]).strip().lower()
        result["status"] = STATUS_ALIASES.get(s, s)

    # Step 5: vocab normalisation
    if vocab:
        result = _apply_vocab(result, vocab)

    return result


def _to_list(value: Any) -> list[str]:
    """Coerce *value* to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        # Handle comma-separated string
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts if len(parts) > 1 else ([value.strip()] if value.strip() else [])
    if value is None:
        return []
    return [str(value)]


def _apply_vocab(fm: dict[str, Any], vocab: dict[str, Any]) -> dict[str, Any]:
    """Apply vocabulary normalisation from a vocab dict."""
    normalize_topics: dict[str, str] = {
        str(k).lower(): str(v)
        for k, v in (vocab.get("normalize_topics") or {}).items()
    }
    priority_from_tags: dict[str, str] = {
        str(k).lower(): str(v)
        for k, v in (vocab.get("priority_from_tags") or {}).items()
    }
    topics_allowed: list[str] = [
        str(t) for t in (vocab.get("topics_allowed") or [])
    ]

    # Normalise existing topics field values
    if "topics" in fm and normalize_topics:
        new_topics: list[str] = []
        for t in fm["topics"]:
            canonical = normalize_topics.get(str(t).lower(), t)
            new_topics.append(canonical)
        fm["topics"] = _deduplicate(new_topics)

    # Extract additional topics from tags via normalize_topics mapping
    if "tags" in fm and normalize_topics:
        existing_topics: set[str] = set(fm.get("topics", []))
        extra: list[str] = []
        for tag in fm["tags"]:
            canonical = normalize_topics.get(str(tag).lower().lstrip("#"))
            if canonical and canonical not in existing_topics:
                extra.append(canonical)
                existing_topics.add(canonical)
        if extra:
            fm["topics"] = list(fm.get("topics", [])) + extra

    # Filter topics to allowed list when configured
    if topics_allowed and "topics" in fm:
        allowed_set = set(topics_allowed)
        fm["topics"] = [t for t in fm["topics"] if t in allowed_set]

    # Extract priority from tags when not already set
    if "tags" in fm and priority_from_tags and "priority" not in fm:
        for tag in fm["tags"]:
            canonical_priority = priority_from_tags.get(
                str(tag).lower().lstrip("#")
            )
            if canonical_priority:
                fm["priority"] = canonical_priority
                break

    return fm


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Conservative merge
# ---------------------------------------------------------------------------


def merge(
    existing: dict[str, Any],
    suggested: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Merge *suggested* into *existing* frontmatter.

    Conservative mode (default, ``force=False``):
        Only fills in **missing** keys; never overwrites user-authored values.
        ``updated`` is set automatically if any key was actually added.

    Force mode (``force=True``):
        Overwrites all keys present in *suggested*.

    Returns ``(merged_dict, changed)`` where *changed* is ``True`` if at
    least one field was modified.
    """
    result = dict(existing)
    changed = False

    for key, value in suggested.items():
        if force:
            if result.get(key) != value:
                result[key] = value
                changed = True
        else:
            # Conservative: only fill absent / empty fields
            existing_val = result.get(key)
            is_empty = existing_val in (None, "", [], {})
            if key not in result or is_empty:
                if value not in (None, "", [], {}):
                    result[key] = value
                    changed = True

    if changed:
        result["updated"] = _now_iso()

    return result, changed


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# High-level: apply to note content
# ---------------------------------------------------------------------------


def apply_metadata_to_content(
    content: str,
    llm_yaml: str,
    *,
    allowed_keys: frozenset[str] | set[str] = ALLOWED_KEYS,
    vocab: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[str, bool]:
    """Parse LLM YAML, sanitise, merge into note frontmatter, return new content.

    The markdown body is **never modified** — only the YAML frontmatter block
    at the top of the file is touched.  If no frontmatter block exists one is
    created; the body remains byte-for-byte identical.

    Returns ``(new_content, changed)``.  When *changed* is ``False`` the
    caller can skip the file write.

    Raises ``ValueError`` with a human-readable message when *llm_yaml*
    cannot be parsed or contains no dict.
    """
    existing_fm, body = split_frontmatter(content)

    try:
        raw_suggested: Any = yaml.safe_load(llm_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"LLM returned invalid YAML: {exc}") from exc

    if raw_suggested is None:
        raw_suggested = {}
    if not isinstance(raw_suggested, dict):
        raise ValueError(
            f"LLM returned non-mapping YAML (got {type(raw_suggested).__name__}); "
            "expected a key: value mapping."
        )

    sanitized = sanitize(raw_suggested, allowed_keys=allowed_keys, vocab=vocab)
    merged, changed = merge(existing_fm, sanitized, force=force)

    if not changed:
        return content, False

    return build_content(merged, body), True
