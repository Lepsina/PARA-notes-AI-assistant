"""Prompt templates and Ollama-response parser.

The prompts are written in Russian to match the user's vault language.
The parser converts the structured model output into an :class:`AssistantBlock`.
"""
from __future__ import annotations

import re

from .parser import AssistantBlock

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_ANALYZE_TEMPLATE = """\
Ты — ассистент для анализа заметок в Obsidian. Проанализируй следующую заметку и предоставь три секции.

Отвечай **строго** в следующем формате (не добавляй ничего до или после):

### Summary
<краткое резюме заметки, 2–5 предложений>

### Questions
- <вопрос 1>
- <вопрос 2>
- <вопрос 3>

### Metadata suggestions
```yaml
<YAML с полями: tags, topic, status, category, related_topics>
```

Правила:
- Summary: главная идея, ключевые тезисы, возможные противоречия.
- Questions: открытые вопросы, несостыковки, что стоит исследовать дальше.
- Metadata: tags (список), topic, status (draft/in-progress/complete), \
category (Buffer/Projects/Areas/Resources/Archive/Daily), related_topics (список).
- Отвечай на том же языке, что и заметка. Если язык смешанный — используй русский.

---
Заметка:
{note_content}
"""

_METADATA_TEMPLATE = """\
Ты — ассистент для организации заметок в Obsidian. \
Предложи метаданные для следующей заметки.

Отвечай **строго** в следующем формате:

### Summary
(не заполняется)

### Questions
(не заполняется)

### Metadata suggestions
```yaml
<YAML с полями: tags, topic, status, category, related_topics>
```

Поля YAML:
- tags: список тегов (snake_case или kebab-case)
- topic: основная тема одной строкой
- status: draft | in-progress | complete
- category: Buffer | Projects | Areas | Resources | Archive | Daily
- related_topics: список связанных тем

Отвечай на том же языке, что и заметка. Если язык смешанный — используй русский.

---
Заметка:
{note_content}
"""


_FRONTMATTER_TEMPLATE = """\
Ты — ассистент по метаданным Obsidian. Проанализируй заметку и верни ТОЛЬКО YAML-поля для frontmatter.

Допустимые поля: title, status, lang, tags, topics, entities, summary, priority, source_type, aliases

Правила:
- Отвечай ТОЛЬКО валидным YAML (без markdown-обёрток, без пояснений, без дополнительного текста).
- tags: список тегов из заметки (snake_case, нижний регистр).
- topics: список канонических тем (английские ключевые слова, kebab-case).
- status: draft | active | done (только одно значение).
- priority: high | medium | low (только если очевидно из текста, иначе не включай поле).
- summary: краткое описание заметки одной строкой.
- Не добавляй поля, которые невозможно определить из текста.
- Не добавляй комментарии.

---
Заметка:
{note_content}
"""


def build_frontmatter_prompt(note_content: str) -> str:
    """Return a metadata-only prompt that asks for raw YAML frontmatter fields.

    Unlike :func:`build_metadata_prompt`, the response is expected to be a
    plain YAML mapping (no ``## Assistant`` section, no markdown fences).
    """
    return _FRONTMATTER_TEMPLATE.format(note_content=note_content)



_ASK_TEMPLATE = """\
Ты — ИИ-ассистент, отвечающий на вопросы по личной базе знаний (Obsidian vault).

Используй только приведённый ниже контекст. Если ответа нет в контексте — честно скажи об этом.
Отвечай на том же языке, что и вопрос пользователя. Если вопрос на русском — отвечай на русском.

Контекст:
{context}

---
Вопрос: {question}
"""


def build_analyze_prompt(note_content: str) -> str:
    """Return a full analysis prompt for *note_content*."""
    return _ANALYZE_TEMPLATE.format(note_content=note_content)


def build_metadata_prompt(note_content: str) -> str:
    """Return a metadata-only prompt for *note_content*."""
    return _METADATA_TEMPLATE.format(note_content=note_content)


def build_ask_prompt(context: str, question: str) -> str:
    """Return a RAG-style prompt with *context* chunks and a *question*."""
    return _ASK_TEMPLATE.format(context=context, question=question)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_ollama_response(response: str) -> AssistantBlock:
    """Parse a structured Ollama response into an :class:`AssistantBlock`.

    The parser is intentionally lenient: it splits the response on ``### ``
    headings and extracts the body of each known section.  Missing sections
    default to an empty string.
    """
    summary = _extract_response_section(response, "Summary")
    questions = _extract_response_section(response, "Questions")
    metadata_raw = _extract_response_section(response, "Metadata suggestions")

    metadata_yaml = _strip_code_fence(metadata_raw) if metadata_raw else ""

    return AssistantBlock(
        summary=summary,
        questions=questions,
        metadata_yaml=metadata_yaml,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_response_section(content: str, section_name: str) -> str:
    """Return the body text of a ``### <section_name>`` section."""
    parts = re.split(r"^### ", content, flags=re.MULTILINE)
    for part in parts:
        newline_pos = part.find("\n")
        if newline_pos == -1:
            continue
        heading = part[:newline_pos].strip()
        if heading == section_name:
            return part[newline_pos + 1 :].strip()
    return ""


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding ``` or ```yaml code fence from *text*."""
    m = re.search(r"```(?:yaml)?[ \t]*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()
