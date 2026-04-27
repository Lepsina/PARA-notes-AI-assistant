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
* Deterministic type/status inference from file path and tags.
* YAML fence stripping for LLM output.
* Russian-language heuristic for summary validation.
* Confidence score computation.
"""
from __future__ import annotations

import re
import warnings
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
        "confidence",
    }
)

# Keys that the LLM is NOT allowed to set by default.
# Each can be unlocked via metadata.llm_allow.<key> = true in config.
LLM_RESTRICTED_KEYS: frozenset[str] = frozenset(
    {"title", "entities", "source_type", "priority"}
)

ARRAY_FIELDS: frozenset[str] = frozenset({"tags", "topics", "entities", "aliases"})
BOOL_FIELDS: frozenset[str] = frozenset({"exclude_from_ai"})

# Legacy key aliases: old_key (lower) → canonical_key
KEY_ALIASES: dict[str, str] = {
    "topic": "topics",
    "tag": "tags",
    "alias": "aliases",
    "entity": "entities",
    # Support the hyphenated alias for exclude_from_ai
    "exclude-from-ai": "exclude_from_ai",
}

# Status value aliases (lower → canonical)
STATUS_ALIASES: dict[str, str] = {
    "complete": "done",
    "completed": "done",
    "in_progress": "active",
    "in-progress": "active",
}

# ---------------------------------------------------------------------------
# Deterministic type inference
# ---------------------------------------------------------------------------

# Regex to recognise daily-note filenames (YYYY-MM-DD*.md)
_DAILY_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def infer_type(note_path: Path | str | None) -> str:
    """Infer the ``type`` frontmatter value deterministically from *note_path*.

    Rules (checked in order against the **resolved** path parts):
    - ``Daily/`` folder **or** filename matches ``YYYY-MM-DD*``  → ``daily``
    - ``Projects/`` folder                                        → ``project``
    - ``Areas/`` folder                                           → ``area``
    - ``Resources/`` folder                                       → ``resource``
    - ``0. Buffer/`` folder                                       → ``note``
    - anything else                                               → ``note``
    """
    if note_path is None:
        return "note"
    path = Path(note_path)
    parts = [p.lower() for p in path.parts]
    stem = path.stem

    # Daily by filename pattern takes priority over folder
    if _DAILY_FILENAME_RE.match(stem):
        return "daily"

    for part in parts:
        if part == "daily":
            return "daily"
        if part == "projects":
            return "project"
        if part == "areas":
            return "area"
        if part == "resources":
            return "resource"

    return "note"


# ---------------------------------------------------------------------------
# Deterministic status inference
# ---------------------------------------------------------------------------

# Tags that force status = draft
_DRAFT_TAGS: frozenset[str] = frozenset({"дописать", "#дописать"})
# Tags that force status = active
_ACTIVE_TAGS: frozenset[str] = frozenset({"просмотреть", "#просмотреть"})


def infer_status(
    note_path: Path | str | None,
    tags: list[str] | None = None,
    *,
    default_project_area_status: str = "active",
) -> str | None:
    """Infer ``status`` deterministically.

    Returns ``None`` when no status should be set (caller should omit the field).

    Rules:
    - In ``0. Buffer/``   → ``draft`` (regardless of tags)
    - tag ``#дописать``   → ``draft``
    - tag ``#просмотреть``→ ``active``
    - type project/area   → *default_project_area_status* (``active``)
    - otherwise           → ``None`` (do not set)
    """
    path = Path(note_path) if note_path else None

    # 0. Buffer → draft
    if path is not None:
        parts_lower = [p.lower() for p in path.parts]
        for part in parts_lower:
            if part.startswith("0.") and "buffer" in part:
                return "draft"

    # Tag-based overrides
    norm_tags = [str(t).lower().lstrip("#") for t in (tags or [])]
    if "дописать" in norm_tags:
        return "draft"
    if "просмотреть" in norm_tags:
        return "active"

    # Default by type
    note_type = infer_type(path)
    if note_type in ("project", "area"):
        return default_project_area_status

    return None


# ---------------------------------------------------------------------------
# exclude_from_ai deterministic rule
# ---------------------------------------------------------------------------


def infer_exclude_from_ai(
    note_path: Path | str | None,
    tags: list[str] | None = None,
    *,
    private_folders: list[str] | None = None,
) -> bool | None:
    """Return ``True`` if the note should be excluded from AI processing.

    Rules (only applied when *private_folders* is provided):
    - path is under any folder in *private_folders* (e.g. ``Private``, ``People``)
    - tag ``#private`` is present in *tags*

    Returns ``None`` when exclusion cannot be determined (feature disabled).
    """
    if not private_folders:
        return None

    path = Path(note_path) if note_path else None
    if path is not None:
        parts_lower = [p.lower() for p in path.parts]
        for folder in private_folders:
            if folder.lower().rstrip("/") in parts_lower:
                return True

    norm_tags = [str(t).lower().lstrip("#") for t in (tags or [])]
    if "private" in norm_tags:
        return True

    return None


# ---------------------------------------------------------------------------
# YAML fence stripper
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"^[ \t]*`{3,}(?:yaml)?[ \t]*\r?\n(.*?)\r?\n[ \t]*`{3,}[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def strip_yaml_fences(text: str) -> str:
    """Extract YAML content from a fenced code block, if present.

    Handles triple-backtick fences (yaml or plain), quadruple-backtick fences,
    and responses that start/end with a bare backtick fence line.

    If no fence is detected the text is returned stripped of leading/trailing
    whitespace.
    """
    stripped = text.strip()
    # Try to match a full fenced block first
    m = _FENCE_RE.search(stripped)
    if m:
        return m.group(1).strip()
    # Fallback: strip leading/trailing backtick lines
    lines = stripped.splitlines()
    # Remove first line if it looks like a fence opener
    if lines and re.match(r"^`{3,}(?:yaml)?[ \t]*$", lines[0]):
        lines = lines[1:]
    # Remove last line if it looks like a fence closer
    if lines and re.match(r"^`{3,}[ \t]*$", lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Russian language heuristic
# ---------------------------------------------------------------------------

# Cyrillic Unicode block: U+0400–U+04FF
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
# Any alphabetic character (broad)
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)

_RU_CYRILLIC_RATIO_THRESHOLD = 0.5  # ≥50 % of alpha chars must be Cyrillic


def is_russian(text: str) -> bool:
    """Return ``True`` when *text* contains predominantly Cyrillic characters.

    An empty string is considered *not* Russian (caller should treat the
    summary as absent).
    """
    if not text or not text.strip():
        return False
    alpha_chars = _ALPHA_RE.findall(text)
    if not alpha_chars:
        return False
    cyrillic_chars = _CYRILLIC_RE.findall(text)
    return len(cyrillic_chars) / len(alpha_chars) >= _RU_CYRILLIC_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------


def compute_confidence(
    *,
    yaml_parsed: bool,
    schema_valid: bool,
    body_unchanged: bool,
    type_inferred: bool,
    topics_valid: bool,
    summary_ru: bool | None,
) -> float:
    """Compute a deterministic confidence score in [0, 1].

    Each check contributes an equal share.  The *summary_ru* check is only
    included when a summary was actually produced (``None`` means "no summary
    → skip check").

    Returns a float rounded to 2 decimal places.
    """
    checks: list[bool] = [
        yaml_parsed,
        schema_valid,
        body_unchanged,
        type_inferred,
        topics_valid,
    ]
    if summary_ru is not None:
        checks.append(summary_ru)
    score = sum(checks) / len(checks) if checks else 0.0
    return round(score, 2)

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
        warnings.warn(f"obsassist: could not parse vocab file {path}: {exc}", stacklevel=2)
        return {}
    except OSError as exc:
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
    llm_allowed_keys: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Sanitize and normalise LLM-suggested metadata.

    Steps applied in order:

    1. Rename legacy key aliases (``topic`` → ``topics``, etc.).
    2. Drop keys not in *allowed_keys*.
    3. Drop keys in ``LLM_RESTRICTED_KEYS`` that are not in *llm_allowed_keys*
       (only when *llm_allowed_keys* is not ``None``).
    4. Coerce array fields to ``list[str]``; coerce bool fields.
    5. Normalise status value aliases.
    6. Apply vocab normalisation (topics, priority_from_tags).

    *llm_allowed_keys*: explicit set of restricted keys the LLM **is** allowed
    to set for this call.  When ``None`` (the default), LLM restrictions are
    **not** applied — this preserves backward compatibility for callers that
    do not go through the LLM pipeline.  Pass an explicit (possibly empty)
    set to enable the restriction check.
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

    # Step 3: drop restricted keys not explicitly permitted
    # Only enforced when llm_allowed_keys is provided (not None).
    if llm_allowed_keys is not None:
        permitted_restricted = set(llm_allowed_keys)
        for restricted_key in LLM_RESTRICTED_KEYS:
            if restricted_key in result and restricted_key not in permitted_restricted:
                del result[restricted_key]

    # Step 4: coerce types
    for field in ARRAY_FIELDS:
        if field in result:
            result[field] = _to_list(result[field])
    for field in BOOL_FIELDS:
        if field in result:
            result[field] = bool(result[field])

    # Step 5: normalise status
    if "status" in result:
        s = str(result["status"]).strip().lower()
        result["status"] = STATUS_ALIASES.get(s, s)

    # Step 6: vocab normalisation
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
    note_path: Path | str | None = None,
    llm_allowed_keys: frozenset[str] | set[str] | None = None,
    include_confidence: bool = False,
    private_folders: list[str] | None = None,
) -> tuple[str, bool]:
    """Parse LLM YAML, sanitise, merge into note frontmatter, return new content.

    The markdown body is **never modified** — only the YAML frontmatter block
    at the top of the file is touched.  If no frontmatter block exists one is
    created; the body remains byte-for-byte identical.

    Returns ``(new_content, changed)``.  When *changed* is ``False`` the
    caller can skip the file write.

    Raises ``ValueError`` with a human-readable message when *llm_yaml*
    cannot be parsed or contains no dict after fence stripping.

    Parameters
    ----------
    note_path:
        Path to the note on disk.  When provided, ``type`` and ``status`` are
        inferred deterministically from the path (overwriting any LLM value).
    llm_allowed_keys:
        Restricted keys the LLM is permitted to set (see ``sanitize``).
    include_confidence:
        When ``True``, compute and store a ``confidence`` field.
    private_folders:
        Folder names that trigger ``exclude_from_ai: true`` (e.g. ``Private``,
        ``People``).
    """
    existing_fm, body = split_frontmatter(content)

    # Strip markdown fences before parsing
    clean_yaml = strip_yaml_fences(llm_yaml)

    yaml_parsed = False
    try:
        raw_suggested: Any = yaml.safe_load(clean_yaml)
        yaml_parsed = True
    except yaml.YAMLError as exc:
        raise ValueError(f"LLM returned invalid YAML: {exc}") from exc

    if raw_suggested is None:
        raw_suggested = {}
    if not isinstance(raw_suggested, dict):
        raise ValueError(
            f"LLM returned non-mapping YAML (got {type(raw_suggested).__name__}); "
            "expected a key: value mapping."
        )

    sanitized = sanitize(
        raw_suggested,
        allowed_keys=allowed_keys,
        vocab=vocab,
        llm_allowed_keys=llm_allowed_keys,
    )

    # --------------------------------------------------------------------------
    # Deterministic overrides (not from LLM)
    # Only applied when note_path is provided; otherwise we stay backward-
    # compatible with tests and callers that don't supply a path.
    # --------------------------------------------------------------------------
    tags_for_inference: list[str] = list(
        sanitized.get("tags") or existing_fm.get("tags") or []
    )

    type_inferred = False
    if note_path is not None:
        # type: always set deterministically when path is known
        inferred_type = infer_type(note_path)
        sanitized["type"] = inferred_type
        type_inferred = True

        # status: set deterministically; only applied when absent or force
        inferred_status = infer_status(note_path, tags=tags_for_inference)
        if inferred_status is not None:
            if "status" not in sanitized:
                sanitized["status"] = inferred_status

        # exclude_from_ai: deterministic private-folder rule
        if private_folders:
            excl = infer_exclude_from_ai(note_path, tags=tags_for_inference,
                                         private_folders=private_folders)
            if excl is True and not existing_fm.get("exclude_from_ai"):
                sanitized["exclude_from_ai"] = True

    # --------------------------------------------------------------------------
    # Confidence score
    # --------------------------------------------------------------------------
    topics_valid = True
    summary_ru: bool | None = None
    if "summary" in sanitized:
        summary_ru = is_russian(str(sanitized["summary"]))
        if not summary_ru:
            # Drop non-Russian summary; caller should handle retry
            warnings.warn(
                "obsassist: LLM summary appears non-Russian; dropping field.",
                stacklevel=2,
            )
            del sanitized["summary"]
            summary_ru = None  # not counted in confidence (was omitted)

    if include_confidence:
        conf = compute_confidence(
            yaml_parsed=yaml_parsed,
            schema_valid=True,
            body_unchanged=True,
            type_inferred=type_inferred,
            topics_valid=topics_valid,
            summary_ru=summary_ru,
        )
        sanitized["confidence"] = conf

    merged, changed = merge(existing_fm, sanitized, force=force)

    if not changed:
        return content, False

    return build_content(merged, body), True
