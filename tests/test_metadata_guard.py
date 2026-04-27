"""Tests for obsassist.metadata_guard.

Covers all acceptance criteria from PR4 and PR6:
- body unchanged after metadata update (byte-for-byte)
- unknown keys removed / ignored
- alias mapping (topic → topics, complete/in_progress → status canonical)
- vocab normalisation for sample tags/topics
- priority extraction from tags (High/Medium/Low)
- conservative merge preserves existing manual fields
- metadata model selection + fallback behaviour (via config)
- PR6: deterministic type/status inference by path and tags
- PR6: RU-only summary validation (non-RU dropped)
- PR6: YAML fence stripping
- PR6: confidence score computation
- PR6: LLM-restricted keys (title/entities/source_type/priority not written by default)
- PR6: exclude_from_ai auto-rule for private folders / #private tag
"""
from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import pytest
import yaml

from obsassist.config import LlmAllowConfig, MetadataConfig, _parse_config
from obsassist.metadata_guard import (
    ALLOWED_KEYS,
    LLM_RESTRICTED_KEYS,
    apply_metadata_to_content,
    build_content,
    compute_confidence,
    infer_exclude_from_ai,
    infer_status,
    infer_type,
    is_russian,
    load_vocab,
    merge,
    sanitize,
    split_frontmatter,
    strip_yaml_fences,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_VOCAB = {
    "normalize_topics": {
        "матан": "calculus",
        "линал": "linear-algebra",
        "дискра": "discrete-math",
        "bfs": "graph-algorithms",
        "dfs": "graph-algorithms",
        "физика": "physics",
    },
    "topics_allowed": [
        "calculus",
        "linear-algebra",
        "discrete-math",
        "graph-algorithms",
        "physics",
        "algorithms",
    ],
    "priority_from_tags": {
        "high": "high",
        "medium": "medium",
        "low": "low",
    },
}


def _note_with_frontmatter(fm_yaml: str, body: str = "# Title\n\nBody text.\n") -> str:
    """Return a note string with frontmatter + body."""
    return f"---\n{fm_yaml.strip()}\n---\n{body}"


# ---------------------------------------------------------------------------
# split_frontmatter / build_content
# ---------------------------------------------------------------------------


class TestSplitBuildRoundtrip:
    def test_no_frontmatter(self):
        body = "# Title\n\nContent here.\n"
        fm, extracted_body = split_frontmatter(body)
        assert fm == {}
        assert extracted_body == body

    def test_with_frontmatter(self):
        content = _note_with_frontmatter("tags:\n  - матан\nstatus: draft")
        fm, body = split_frontmatter(content)
        assert fm["tags"] == ["матан"]
        assert fm["status"] == "draft"
        assert body == "# Title\n\nBody text.\n"

    def test_build_content_preserves_body(self):
        body = "# Title\n\nSome **bold** text.\n\n[[WikiLink]]\n"
        fm = {"title": "Test", "status": "draft"}
        rebuilt = build_content(fm, body)
        assert rebuilt.endswith(body)
        # Body is byte-for-byte identical
        _, extracted_body = split_frontmatter(rebuilt)
        assert extracted_body == body

    def test_build_content_empty_fm(self):
        body = "# No frontmatter.\n"
        assert build_content({}, body) == body

    def test_invalid_yaml_frontmatter_returns_empty(self):
        content = "---\n: broken: yaml\n---\n# Body\n"
        fm, body = split_frontmatter(content)
        assert fm == {}
        assert "# Body" in body


# ---------------------------------------------------------------------------
# Body unchanged after metadata update
# ---------------------------------------------------------------------------


class TestBodyUnchanged:
    def test_body_byte_for_byte_with_existing_frontmatter(self):
        original_body = "# Дифференциал\n\n![[image.png]]\n\n[[Link1]]\n[[Link2]]\n"
        fm_yaml = "tags:\n  - матан\nstatus: draft\n"
        content = _note_with_frontmatter(fm_yaml, original_body)

        llm_yaml = "tags:\n  - матан\n  - дифференцирование\nstatus: active\n"
        new_content, _ = apply_metadata_to_content(content, llm_yaml)

        _, new_body = split_frontmatter(new_content)
        assert new_body == original_body

    def test_body_byte_for_byte_no_frontmatter(self):
        original_body = "# Fresh Note\n\nParagraph.\n"
        llm_yaml = "status: draft\ntags:\n  - test\n"
        new_content, changed = apply_metadata_to_content(original_body, llm_yaml)

        assert changed is True
        _, new_body = split_frontmatter(new_content)
        assert new_body == original_body

    def test_body_unchanged_no_actual_change(self):
        """When LLM suggests nothing new, content must be identical."""
        fm_yaml = "tags:\n  - матан\nstatus: draft\n"
        body = "# Note\n\nContent.\n"
        content = _note_with_frontmatter(fm_yaml, body)

        # Suggest the exact same values
        llm_yaml = "tags:\n  - матан\nstatus: draft\n"
        new_content, changed = apply_metadata_to_content(content, llm_yaml)

        assert changed is False
        assert new_content == content

    def test_assistant_section_not_injected(self):
        """metadata command must never add ## Assistant to the body."""
        body = "# Note\n\nSome content.\n"
        llm_yaml = "status: draft\ntags:\n  - test\n"
        new_content, _ = apply_metadata_to_content(body, llm_yaml)
        assert "## Assistant" not in new_content
        assert "### Summary" not in new_content
        assert "### Questions" not in new_content


# ---------------------------------------------------------------------------
# Unknown keys removed / ignored
# ---------------------------------------------------------------------------


class TestUnknownKeysRemoved:
    def test_unknown_key_dropped(self):
        llm_yaml = "status: draft\ncategory: Projects\nrelated_topics:\n  - X\n"
        result = sanitize(yaml.safe_load(llm_yaml))
        assert "category" not in result
        assert "related_topics" not in result
        assert result.get("status") == "draft"

    def test_only_allowed_keys_survive(self):
        raw = {k: "val" for k in ["title", "bogus_key", "another_bad", "status"]}
        result = sanitize(raw)
        for k in result:
            assert k in ALLOWED_KEYS, f"Unexpected key '{k}' passed through"

    def test_extra_allowed_keys_via_apply(self):
        """Keys in extra_allowed_keys config should survive sanitisation."""
        content = "# Note\n\nBody.\n"
        llm_yaml = "status: draft\ncustom_field: my_value\n"
        new_content, changed = apply_metadata_to_content(
            content, llm_yaml, allowed_keys=ALLOWED_KEYS | {"custom_field"}
        )
        assert changed
        fm, _ = split_frontmatter(new_content)
        assert fm.get("custom_field") == "my_value"


# ---------------------------------------------------------------------------
# Alias mapping
# ---------------------------------------------------------------------------


class TestAliasMapping:
    def test_topic_renamed_to_topics(self):
        raw = {"topic": "Calculus", "status": "draft"}
        result = sanitize(raw)
        assert "topics" in result
        assert "topic" not in result
        assert result["topics"] == ["Calculus"]

    def test_tag_renamed_to_tags(self):
        raw = {"tag": "матан", "status": "draft"}
        result = sanitize(raw)
        assert "tags" in result
        assert "tag" not in result

    def test_status_complete_to_done(self):
        raw = {"status": "complete"}
        result = sanitize(raw)
        assert result["status"] == "done"

    def test_status_completed_to_done(self):
        raw = {"status": "completed"}
        result = sanitize(raw)
        assert result["status"] == "done"

    def test_status_in_progress_to_active(self):
        raw = {"status": "in_progress"}
        result = sanitize(raw)
        assert result["status"] == "active"

    def test_status_in_progress_hyphen_to_active(self):
        raw = {"status": "in-progress"}
        result = sanitize(raw)
        assert result["status"] == "active"

    def test_alias_field_renamed_to_aliases(self):
        raw = {"alias": "My Note Alias"}
        result = sanitize(raw)
        assert "aliases" in result
        assert "alias" not in result

    def test_entity_field_renamed_to_entities(self):
        raw = {"entity": ["Person A"]}
        result = sanitize(raw)
        assert "entities" in result
        assert "entity" not in result


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


class TestTypeCoercion:
    def test_tags_string_to_list(self):
        raw = {"tags": "матан"}
        result = sanitize(raw)
        assert isinstance(result["tags"], list)
        assert result["tags"] == ["матан"]

    def test_tags_comma_separated(self):
        raw = {"tags": "матан, линал, дискра"}
        result = sanitize(raw)
        assert result["tags"] == ["матан", "линал", "дискра"]

    def test_exclude_from_ai_coerced_to_bool(self):
        raw = {"exclude_from_ai": 1}
        result = sanitize(raw)
        assert result["exclude_from_ai"] is True

    def test_entities_coerced_to_list(self):
        raw = {"entities": "Einstein"}
        result = sanitize(raw)
        assert isinstance(result["entities"], list)


# ---------------------------------------------------------------------------
# Vocab normalisation
# ---------------------------------------------------------------------------


class TestVocabNormalisation:
    def test_матан_normalises_to_calculus(self):
        raw = {"tags": ["матан", "линал"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        topics = result.get("topics", [])
        assert "calculus" in topics
        assert "linear-algebra" in topics

    def test_bfs_dfs_normalise_to_graph_algorithms(self):
        raw = {"tags": ["bfs", "dfs"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        topics = result.get("topics", [])
        assert topics.count("graph-algorithms") == 1  # deduplicated

    def test_topics_filtered_to_allowed(self):
        raw = {"tags": ["матан"], "topics": ["unknown-topic", "calculus"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        for t in result.get("topics", []):
            assert t in SAMPLE_VOCAB["topics_allowed"], f"'{t}' not in allowed list"

    def test_topics_deduplicated(self):
        # Both tags produce the same canonical topic
        raw = {"tags": ["bfs", "dfs"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        topics = result.get("topics", [])
        assert len(topics) == len(set(topics))

    def test_load_vocab_missing_file(self):
        vocab = load_vocab("/nonexistent/path/metadata.vocab.yml")
        assert vocab == {}

    def test_load_vocab_none(self):
        assert load_vocab(None) == {}

    def test_load_vocab_from_file(self, tmp_path: Path):
        vocab_file = tmp_path / "vocab.yml"
        vocab_file.write_text(
            "normalize_topics:\n  матан: calculus\npriority_from_tags:\n  high: high\n",
            encoding="utf-8",
        )
        vocab = load_vocab(vocab_file)
        assert vocab["normalize_topics"]["матан"] == "calculus"

    def test_existing_topics_normalised(self):
        """Topics already present in frontmatter are normalised via vocab."""
        raw = {"topics": ["матан", "линал"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        assert "calculus" in result["topics"]
        assert "linear-algebra" in result["topics"]
        assert "матан" not in result["topics"]


# ---------------------------------------------------------------------------
# Priority extraction from tags
# ---------------------------------------------------------------------------


class TestPriorityFromTags:
    def test_high_tag_sets_priority(self):
        raw = {"tags": ["High", "матан"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        assert result.get("priority") == "high"

    def test_medium_tag_sets_priority(self):
        raw = {"tags": ["Medium", "линал"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        assert result.get("priority") == "medium"

    def test_low_tag_sets_priority(self):
        raw = {"tags": ["Low"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        assert result.get("priority") == "low"

    def test_priority_not_overwritten_when_already_present(self):
        raw = {"tags": ["High"], "priority": "low"}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        # priority_from_tags only sets when not already present
        assert result.get("priority") == "low"

    def test_no_priority_tag_leaves_priority_absent(self):
        raw = {"tags": ["матан"]}
        result = sanitize(raw, vocab=SAMPLE_VOCAB)
        assert "priority" not in result


# ---------------------------------------------------------------------------
# Conservative merge
# ---------------------------------------------------------------------------


class TestConservativeMerge:
    def test_missing_field_filled(self):
        existing = {"title": "My Note", "status": "draft"}
        suggested = {"status": "active", "tags": ["test"]}
        merged, changed = merge(existing, suggested)
        # status already present → NOT overwritten
        assert merged["status"] == "draft"
        # tags was missing → filled
        assert merged["tags"] == ["test"]
        assert changed is True

    def test_existing_value_preserved(self):
        existing = {"title": "My Note", "summary": "User wrote this."}
        suggested = {"summary": "AI generated summary."}
        merged, changed = merge(existing, suggested)
        assert merged["summary"] == "User wrote this."
        assert changed is False

    def test_empty_field_filled(self):
        existing = {"tags": []}
        suggested = {"tags": ["новый"]}
        merged, changed = merge(existing, suggested)
        assert merged["tags"] == ["новый"]
        assert changed is True

    def test_updated_set_when_changed(self):
        existing = {}
        suggested = {"status": "draft"}
        merged, changed = merge(existing, suggested)
        assert changed is True
        assert "updated" in merged

    def test_updated_not_set_when_unchanged(self):
        existing = {"status": "draft"}
        suggested = {"status": "draft"}
        merged, changed = merge(existing, suggested)
        assert changed is False
        assert "updated" not in merged

    def test_force_overwrites_existing(self):
        existing = {"status": "draft", "title": "Old title"}
        suggested = {"status": "active", "title": "New title"}
        merged, changed = merge(existing, suggested, force=True)
        assert merged["status"] == "active"
        assert merged["title"] == "New title"
        assert changed is True

    def test_full_note_conservative_merge(self):
        body = "# My Math Note\n\nContent.\n"
        original_fm = "title: My Math Note\nstatus: draft\ntags:\n  - матан\n"
        content = _note_with_frontmatter(original_fm, body)

        # LLM suggests updating status and adding summary (summary in Russian)
        llm_yaml = "status: active\nsummary: Заметка о математическом анализе.\ntags:\n  - матан\n  - тейлор\n"
        new_content, changed = apply_metadata_to_content(content, llm_yaml)

        assert changed is True
        fm, new_body = split_frontmatter(new_content)
        assert new_body == body  # body unchanged
        assert fm["status"] == "draft"  # NOT overwritten
        assert fm["summary"] == "Заметка о математическом анализе."  # new field filled


# ---------------------------------------------------------------------------
# Validation robustness
# ---------------------------------------------------------------------------


class TestValidationRobustness:
    def test_invalid_yaml_raises_value_error(self):
        with pytest.raises(ValueError, match="invalid YAML"):
            apply_metadata_to_content("# Note\n", ": broken: {yaml")

    def test_non_dict_yaml_raises_value_error(self):
        with pytest.raises(ValueError, match="non-mapping"):
            apply_metadata_to_content("# Note\n", "- item1\n- item2\n")

    def test_empty_llm_response_no_change(self):
        content = "---\nstatus: draft\n---\n# Note\n"
        new_content, changed = apply_metadata_to_content(content, "")
        assert changed is False
        assert new_content == content

    def test_none_llm_response_no_change(self):
        """A YAML 'null' response (empty string) should not corrupt the note."""
        content = "---\nstatus: draft\n---\n# Note\n"
        new_content, changed = apply_metadata_to_content(content, "null")
        assert changed is False


# ---------------------------------------------------------------------------
# Metadata model config: selection + fallback
# ---------------------------------------------------------------------------


class TestMetadataModelConfig:
    def test_default_metadata_model_is_empty(self):
        cfg = _parse_config({})
        assert cfg.metadata.model == ""

    def test_metadata_model_parsed_from_config(self):
        data = {"metadata": {"model": "qwen2.5:3b-instruct"}}
        cfg = _parse_config(data)
        assert cfg.metadata.model == "qwen2.5:3b-instruct"

    def test_metadata_vocab_path_parsed(self):
        data = {"metadata": {"vocab_path": "/path/to/vocab.yml"}}
        cfg = _parse_config(data)
        assert cfg.metadata.vocab_path == "/path/to/vocab.yml"

    def test_metadata_force_parsed(self):
        data = {"metadata": {"force": True}}
        cfg = _parse_config(data)
        assert cfg.metadata.force is True

    def test_metadata_extra_allowed_keys_parsed(self):
        data = {"metadata": {"extra_allowed_keys": ["my_key"]}}
        cfg = _parse_config(data)
        assert "my_key" in cfg.metadata.extra_allowed_keys

    def test_fallback_uses_ollama_model(self):
        """When metadata.model is empty, the CLI should fall back to ollama.model."""
        data = {"ollama": {"model": "llama3:8b"}, "metadata": {}}
        cfg = _parse_config(data)
        # The effective model is: cfg.metadata.model or cfg.ollama.model
        effective = cfg.metadata.model or cfg.ollama.model
        assert effective == "llama3:8b"

    def test_metadata_model_takes_precedence(self):
        data = {
            "ollama": {"model": "llama3:8b"},
            "metadata": {"model": "qwen2.5:3b-instruct"},
        }
        cfg = _parse_config(data)
        effective = cfg.metadata.model or cfg.ollama.model
        assert effective == "qwen2.5:3b-instruct"

    def test_embeddings_model_unchanged(self):
        """Embeddings model must not be affected by metadata config."""
        data = {
            "metadata": {"model": "qwen2.5:3b-instruct"},
            "embeddings": {"model": "nomic-embed-text"},
        }
        cfg = _parse_config(data)
        assert cfg.embeddings.model == "nomic-embed-text"


# ---------------------------------------------------------------------------
# PR6: Deterministic type inference
# ---------------------------------------------------------------------------


class TestInferType:
    def test_daily_folder(self):
        assert infer_type(Path("vault/Daily/2024-01-15.md")) == "daily"

    def test_daily_by_filename_pattern(self):
        assert infer_type(Path("Notes/2024-05-20 meeting.md")) == "daily"

    def test_daily_filename_at_root(self):
        assert infer_type(Path("2023-12-31.md")) == "daily"

    def test_projects_folder(self):
        assert infer_type(Path("vault/Projects/my-project.md")) == "project"

    def test_areas_folder(self):
        assert infer_type(Path("vault/Areas/health.md")) == "area"

    def test_resources_folder(self):
        assert infer_type(Path("vault/Resources/book-notes.md")) == "resource"

    def test_buffer_folder(self):
        assert infer_type(Path("vault/0. Buffer/scratch.md")) == "note"

    def test_unknown_path_defaults_to_note(self):
        assert infer_type(Path("vault/RandomFolder/note.md")) == "note"

    def test_none_path_defaults_to_note(self):
        assert infer_type(None) == "note"

    def test_case_insensitive_folder(self):
        assert infer_type(Path("vault/PROJECTS/task.md")) == "project"
        assert infer_type(Path("vault/areas/focus.md")) == "area"

    def test_daily_filename_priority_over_projects_folder(self):
        # date-matching filename beats folder
        assert infer_type(Path("Projects/2024-01-01-review.md")) == "daily"


# ---------------------------------------------------------------------------
# PR6: Deterministic status inference
# ---------------------------------------------------------------------------


class TestInferStatus:
    def test_buffer_folder_returns_draft(self):
        assert infer_status(Path("vault/0. Buffer/idea.md")) == "draft"

    def test_дописать_tag_returns_draft(self):
        assert infer_status(Path("Notes/note.md"), tags=["дописать"]) == "draft"

    def test_дописать_with_hash_returns_draft(self):
        assert infer_status(None, tags=["#дописать"]) == "draft"

    def test_просмотреть_tag_returns_active(self):
        assert infer_status(None, tags=["просмотреть"]) == "active"

    def test_project_folder_returns_active_by_default(self):
        assert infer_status(Path("vault/Projects/task.md")) == "active"

    def test_area_folder_returns_active_by_default(self):
        assert infer_status(Path("vault/Areas/health.md")) == "active"

    def test_note_without_special_tags_returns_none(self):
        assert infer_status(Path("vault/Notes/generic.md")) is None

    def test_resource_folder_returns_none(self):
        assert infer_status(Path("vault/Resources/book.md")) is None

    def test_buffer_overrides_просмотреть(self):
        # Buffer folder takes priority; buffer → draft
        assert infer_status(Path("vault/0. Buffer/note.md"), tags=["просмотреть"]) == "draft"

    def test_none_path_with_no_special_tags_returns_none(self):
        assert infer_status(None) is None


# ---------------------------------------------------------------------------
# PR6: exclude_from_ai deterministic rule
# ---------------------------------------------------------------------------


class TestInferExcludeFromAi:
    def test_private_folder_excluded(self):
        result = infer_exclude_from_ai(
            Path("vault/Private/diary.md"), private_folders=["Private"]
        )
        assert result is True

    def test_people_folder_excluded(self):
        result = infer_exclude_from_ai(
            Path("vault/People/friend.md"), private_folders=["People", "Private"]
        )
        assert result is True

    def test_private_tag_excluded(self):
        result = infer_exclude_from_ai(
            Path("vault/Notes/note.md"),
            tags=["private"],
            private_folders=["Private"],
        )
        assert result is True

    def test_private_tag_with_hash_excluded(self):
        result = infer_exclude_from_ai(
            Path("vault/Notes/note.md"),
            tags=["#private"],
            private_folders=["Private"],
        )
        assert result is True

    def test_no_private_folder_configured_returns_none(self):
        # Feature disabled when private_folders is empty/None
        assert infer_exclude_from_ai(Path("vault/Private/note.md")) is None

    def test_non_private_path_not_excluded(self):
        result = infer_exclude_from_ai(
            Path("vault/Projects/task.md"),
            tags=["work"],
            private_folders=["Private"],
        )
        assert result is None


# ---------------------------------------------------------------------------
# PR6: YAML fence stripping
# ---------------------------------------------------------------------------


class TestStripYamlFences:
    def test_plain_yaml_unchanged(self):
        yaml_text = "status: draft\ntags:\n  - test\n"
        assert strip_yaml_fences(yaml_text) == yaml_text.strip()

    def test_triple_backtick_yaml_fence_stripped(self):
        fenced = "```yaml\nstatus: draft\ntags:\n  - test\n```"
        assert strip_yaml_fences(fenced) == "status: draft\ntags:\n  - test"

    def test_triple_backtick_no_lang_stripped(self):
        fenced = "```\nstatus: draft\n```"
        assert strip_yaml_fences(fenced) == "status: draft"

    def test_quadruple_backtick_fence_stripped(self):
        fenced = "````yaml\nstatus: active\n````"
        assert strip_yaml_fences(fenced) == "status: active"

    def test_content_within_fence_is_exact(self):
        inner = "tags:\n  - матан\nstatus: draft"
        fenced = f"```yaml\n{inner}\n```"
        assert strip_yaml_fences(fenced) == inner

    def test_apply_metadata_accepts_fenced_yaml(self):
        """apply_metadata_to_content must accept LLM output wrapped in fences."""
        content = "# Note\n\nBody.\n"
        fenced_yaml = "```yaml\nstatus: draft\ntags:\n  - test\n```"
        new_content, changed = apply_metadata_to_content(content, fenced_yaml)
        assert changed is True
        fm, _ = split_frontmatter(new_content)
        assert fm.get("status") == "draft"
        assert "test" in fm.get("tags", [])


# ---------------------------------------------------------------------------
# PR6: Russian summary heuristic
# ---------------------------------------------------------------------------


class TestIsRussian:
    def test_russian_text(self):
        assert is_russian("Заметка о математическом анализе.") is True

    def test_english_text(self):
        assert is_russian("A note about calculus.") is False

    def test_empty_string(self):
        assert is_russian("") is False

    def test_mixed_mostly_cyrillic(self):
        # "Привет мир" = 9 Cyrillic, " ok" = 2 Latin → 9/11 ≈ 82% Cyrillic
        assert is_russian("Привет мир ok") is True  # >50% Cyrillic

    def test_mixed_mostly_latin(self):
        # Long Latin sentence with one short Russian word → <50% Cyrillic
        assert is_russian("This is a long English note about analysis, да") is False

    def test_numbers_only(self):
        assert is_russian("12345") is False

    def test_non_russian_summary_dropped_from_apply(self):
        """Non-Russian summary must be dropped and a warning emitted."""
        content = "# Note\n\nBody.\n"
        llm_yaml = "summary: A note about calculus.\ntags:\n  - test\n"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new_content, changed = apply_metadata_to_content(content, llm_yaml)
        assert any("non-Russian" in str(warning.message) for warning in w)
        fm, _ = split_frontmatter(new_content)
        assert "summary" not in fm

    def test_russian_summary_preserved_in_apply(self):
        content = "# Note\n\nBody.\n"
        llm_yaml = "summary: Заметка о математическом анализе.\ntags:\n  - test\n"
        new_content, changed = apply_metadata_to_content(content, llm_yaml)
        fm, _ = split_frontmatter(new_content)
        assert fm.get("summary") == "Заметка о математическом анализе."


# ---------------------------------------------------------------------------
# PR6: Confidence score
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_all_checks_pass_score_1(self):
        score = compute_confidence(
            yaml_parsed=True,
            schema_valid=True,
            body_unchanged=True,
            type_inferred=True,
            topics_valid=True,
            summary_ru=None,
        )
        assert score == 1.0

    def test_all_checks_fail_score_0(self):
        score = compute_confidence(
            yaml_parsed=False,
            schema_valid=False,
            body_unchanged=False,
            type_inferred=False,
            topics_valid=False,
            summary_ru=None,
        )
        assert score == 0.0

    def test_with_summary_ru_true(self):
        score = compute_confidence(
            yaml_parsed=True,
            schema_valid=True,
            body_unchanged=True,
            type_inferred=True,
            topics_valid=True,
            summary_ru=True,
        )
        assert score == 1.0

    def test_with_summary_ru_false_lowers_score(self):
        score = compute_confidence(
            yaml_parsed=True,
            schema_valid=True,
            body_unchanged=True,
            type_inferred=True,
            topics_valid=True,
            summary_ru=False,
        )
        assert score < 1.0

    def test_score_has_two_decimal_precision(self):
        score = compute_confidence(
            yaml_parsed=True,
            schema_valid=True,
            body_unchanged=True,
            type_inferred=False,
            topics_valid=True,
            summary_ru=True,
        )
        # 5/6 ≈ 0.83
        assert score == round(5 / 6, 2)

    def test_include_confidence_in_apply(self):
        content = "# Note\n\nBody.\n"
        llm_yaml = "tags:\n  - тест\n"
        new_content, changed = apply_metadata_to_content(
            content, llm_yaml,
            note_path=Path("vault/Projects/note.md"),
            include_confidence=True,
        )
        assert changed is True
        fm, _ = split_frontmatter(new_content)
        assert "confidence" in fm
        conf = fm["confidence"]
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# PR6: LLM-restricted keys not written by default
# ---------------------------------------------------------------------------


class TestLlmRestrictedKeys:
    def test_title_dropped_by_default(self):
        raw = {"title": "My Note", "status": "draft"}
        result = sanitize(raw, llm_allowed_keys=frozenset())
        assert "title" not in result

    def test_entities_dropped_by_default(self):
        raw = {"entities": ["Person A"], "status": "draft"}
        result = sanitize(raw, llm_allowed_keys=frozenset())
        assert "entities" not in result

    def test_source_type_dropped_by_default(self):
        raw = {"source_type": "book", "status": "draft"}
        result = sanitize(raw, llm_allowed_keys=frozenset())
        assert "source_type" not in result

    def test_priority_dropped_by_default(self):
        raw = {"priority": "high", "status": "draft"}
        result = sanitize(raw, llm_allowed_keys=frozenset())
        assert "priority" not in result

    def test_title_allowed_when_permitted(self):
        raw = {"title": "My Note", "status": "draft"}
        result = sanitize(raw, llm_allowed_keys=frozenset({"title"}))
        assert result.get("title") == "My Note"

    def test_entities_allowed_when_permitted(self):
        raw = {"entities": ["Person A"]}
        result = sanitize(raw, llm_allowed_keys=frozenset({"entities"}))
        assert "entities" in result

    def test_no_restriction_when_llm_allowed_keys_is_none(self):
        """Backward compat: None means restrictions disabled."""
        raw = {"title": "My Note", "entities": ["X"], "priority": "high"}
        result = sanitize(raw, llm_allowed_keys=None)
        assert "title" in result
        assert "entities" in result
        assert "priority" in result

    def test_restricted_keys_not_set_via_apply_by_default(self):
        """apply_metadata_to_content with llm_allowed_keys=frozenset() drops restricted keys."""
        content = "# Note\n\nBody.\n"
        llm_yaml = (
            "title: AI Title\n"
            "entities:\n  - Person\n"
            "source_type: article\n"
            "priority: high\n"
            "status: draft\n"
        )
        new_content, changed = apply_metadata_to_content(
            content, llm_yaml,
            llm_allowed_keys=frozenset(),
        )
        assert changed is True
        fm, _ = split_frontmatter(new_content)
        assert "title" not in fm
        assert "entities" not in fm
        assert "source_type" not in fm
        assert "priority" not in fm
        assert fm.get("status") == "draft"


# ---------------------------------------------------------------------------
# PR6: Deterministic type/status set via apply_metadata_to_content
# ---------------------------------------------------------------------------


class TestDeterministicInferenceViaApply:
    def test_type_set_from_path_projects(self):
        content = "# Note\n\nBody.\n"
        new_content, changed = apply_metadata_to_content(
            content, "tags:\n  - work\n",
            note_path=Path("vault/Projects/task.md"),
        )
        assert changed is True
        fm, _ = split_frontmatter(new_content)
        assert fm["type"] == "project"

    def test_type_set_from_path_areas(self):
        content = "# Note\n\nBody.\n"
        new_content, _ = apply_metadata_to_content(
            content, "tags:\n  - health\n",
            note_path=Path("vault/Areas/wellness.md"),
        )
        fm, _ = split_frontmatter(new_content)
        assert fm["type"] == "area"

    def test_type_set_from_daily_filename(self):
        content = "# Note\n\nBody.\n"
        new_content, _ = apply_metadata_to_content(
            content, "tags:\n  - daily\n",
            note_path=Path("vault/2024-03-15.md"),
        )
        fm, _ = split_frontmatter(new_content)
        assert fm["type"] == "daily"

    def test_buffer_note_gets_status_draft(self):
        content = "# Note\n\nBody.\n"
        new_content, _ = apply_metadata_to_content(
            content, "tags:\n  - idea\n",
            note_path=Path("vault/0. Buffer/scratch.md"),
        )
        fm, _ = split_frontmatter(new_content)
        assert fm["type"] == "note"
        assert fm["status"] == "draft"

    def test_project_gets_status_active(self):
        content = "# Note\n\nBody.\n"
        new_content, _ = apply_metadata_to_content(
            content, "tags:\n  - work\n",
            note_path=Path("vault/Projects/task.md"),
        )
        fm, _ = split_frontmatter(new_content)
        assert fm["status"] == "active"

    def test_дописать_tag_forces_draft_even_in_projects(self):
        content = "# Note\n\nBody.\n"
        new_content, _ = apply_metadata_to_content(
            content, "tags:\n  - дописать\n",
            note_path=Path("vault/Projects/unfinished.md"),
        )
        fm, _ = split_frontmatter(new_content)
        # дописать tag → draft, overrides project default of active
        assert fm.get("status") == "draft"

    def test_exclude_from_ai_set_for_private_folder(self):
        content = "# Note\n\nBody.\n"
        new_content, _ = apply_metadata_to_content(
            content, "tags:\n  - personal\n",
            note_path=Path("vault/Private/diary.md"),
            private_folders=["Private"],
        )
        fm, _ = split_frontmatter(new_content)
        assert fm.get("exclude_from_ai") is True

    def test_type_not_set_when_no_path(self):
        content = "# Note\n\nBody.\n"
        llm_yaml = "status: draft\ntags:\n  - test\n"
        new_content, _ = apply_metadata_to_content(content, llm_yaml)
        fm, _ = split_frontmatter(new_content)
        assert "type" not in fm


# ---------------------------------------------------------------------------
# PR6: Config parsing for new fields
# ---------------------------------------------------------------------------


class TestMetadataConfigPR6:
    def test_include_confidence_default_false(self):
        cfg = _parse_config({})
        assert cfg.metadata.include_confidence is False

    def test_include_confidence_parsed(self):
        cfg = _parse_config({"metadata": {"include_confidence": True}})
        assert cfg.metadata.include_confidence is True

    def test_extract_priority_from_tags_default_false(self):
        cfg = _parse_config({})
        assert cfg.metadata.extract_priority_from_tags is False

    def test_extract_priority_from_tags_parsed(self):
        cfg = _parse_config({"metadata": {"extract_priority_from_tags": True}})
        assert cfg.metadata.extract_priority_from_tags is True

    def test_llm_allow_defaults_all_false(self):
        cfg = _parse_config({})
        la = cfg.metadata.llm_allow
        assert la.title is False
        assert la.entities is False
        assert la.source_type is False
        assert la.priority is False

    def test_llm_allow_title_parsed(self):
        cfg = _parse_config({"metadata": {"llm_allow": {"title": True}}})
        assert cfg.metadata.llm_allow.title is True

    def test_llm_allow_entities_parsed(self):
        cfg = _parse_config({"metadata": {"llm_allow": {"entities": True}}})
        assert cfg.metadata.llm_allow.entities is True

    def test_private_folders_default_empty(self):
        cfg = _parse_config({})
        assert cfg.metadata.private_folders == []

    def test_private_folders_parsed(self):
        cfg = _parse_config({"metadata": {"private_folders": ["Private", "People"]}})
        assert cfg.metadata.private_folders == ["Private", "People"]

    def test_exclude_from_ai_alias_in_sanitize(self):
        """exclude-from-ai hyphenated alias must be renamed to exclude_from_ai."""
        raw = {"exclude-from-ai": True}
        result = sanitize(raw)
        assert "exclude_from_ai" in result
        assert result["exclude_from_ai"] is True
        assert "exclude-from-ai" not in result
