"""Tests for obsassist.metadata_guard.

Covers all acceptance criteria from PR4:
- body unchanged after metadata update (byte-for-byte)
- unknown keys removed / ignored
- alias mapping (topic → topics, complete/in_progress → status canonical)
- vocab normalisation for sample tags/topics
- priority extraction from tags (High/Medium/Low)
- conservative merge preserves existing manual fields
- metadata model selection + fallback behaviour (via config)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from obsassist.config import MetadataConfig, _parse_config
from obsassist.metadata_guard import (
    ALLOWED_KEYS,
    apply_metadata_to_content,
    build_content,
    load_vocab,
    merge,
    sanitize,
    split_frontmatter,
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

        # LLM suggests updating status and adding summary
        llm_yaml = "status: active\nsummary: A note about calculus.\ntags:\n  - матан\n  - тейлор\n"
        new_content, changed = apply_metadata_to_content(content, llm_yaml)

        assert changed is True
        fm, new_body = split_frontmatter(new_content)
        assert new_body == body  # body unchanged
        assert fm["status"] == "draft"  # NOT overwritten
        assert fm["summary"] == "A note about calculus."  # new field filled


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
