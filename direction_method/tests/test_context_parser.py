import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from context_parser import (
    CONTEXT_PARSER_PROMPT,
    CONTEXT_PARSER_SCHEMA,
    ContextParserError,
    ParsedContext,
    build_prompt,
    parse_context,
)
from llm import LLMClient
from templates import CHECKLIST_KEYS


def fake_client(tmp_path) -> LLMClient:
    return LLMClient("fake-model", backend="fake", cache_dir=tmp_path)


# --- prompt construction ------------------------------------------------------------------


def test_build_prompt_contains_verbatim_system_prompt_and_description():
    prompt, schema = build_prompt("Navy blue kitchen lower cabinet with brass handles")
    assert prompt.startswith(CONTEXT_PARSER_PROMPT)
    assert prompt.endswith("Navy blue kitchen lower cabinet with brass handles")
    assert schema == CONTEXT_PARSER_SCHEMA


def test_build_prompt_is_text_only_no_image_placeholder():
    # sanity: the prompt is just text -- llm.call() is expected to be given image=None for this
    prompt, _ = build_prompt("cabinet")
    assert "[image]" not in prompt


# --- schema shape --------------------------------------------------------------------------


def test_schema_field_order_matches_output_fields():
    assert list(CONTEXT_PARSER_SCHEMA["properties"].keys()) == ["target_category", "target_phrase", "checklist"]


def test_schema_pins_checklist_keys_to_the_11_key_enum():
    checklist_schema = CONTEXT_PARSER_SCHEMA["properties"]["checklist"]
    assert checklist_schema["additionalProperties"] is False
    assert set(checklist_schema["properties"].keys()) == set(CHECKLIST_KEYS)
    assert len(CHECKLIST_KEYS) == 11


def test_schema_checklist_values_are_bounded_string_arrays():
    # maxItems bounds worst-case generation length -- confirmed against a live server that an
    # unbounded array lets the model loop-repeat the same assertion until max_tokens truncates
    # the response mid-string (never parses).
    checklist_schema = CONTEXT_PARSER_SCHEMA["properties"]["checklist"]
    for key_schema in checklist_schema["properties"].values():
        assert key_schema == {"type": "array", "items": {"type": "string"}, "maxItems": 8}


# --- parse_context: parsing, error handling -------------------------------------------------


def test_parse_context_category_only_description(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    payload = {"target_category": "kitchen lower cabinet", "target_phrase": "kitchen lower cabinet", "checklist": {}}
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps(payload))

    result = parse_context(client, "Kitchen lower cabinet")
    assert isinstance(result, ParsedContext)
    assert result.target_category == "kitchen lower cabinet"
    assert result.target_phrase == "kitchen lower cabinet"
    assert result.checklist == {}


def test_parse_context_with_target_and_relation_assertions(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    payload = {
        "target_category": "kitchen lower cabinet",
        "target_phrase": "navy blue kitchen lower cabinet",
        "checklist": {
            "Target": ["it is navy blue"],
            "above": ["a white farmhouse sink"],
        },
    }
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps(payload))

    result = parse_context(client, "Navy blue kitchen lower cabinet under a white farmhouse sink")
    assert result.target_category == "kitchen lower cabinet"
    assert result.target_phrase == "navy blue kitchen lower cabinet"
    assert result.checklist == {"Target": ["it is navy blue"], "above": ["a white farmhouse sink"]}


def test_parse_context_drops_empty_list_checklist_entries(tmp_path, monkeypatch):
    # if a real server fills every schema property instead of omitting unused ones, empty-list
    # keys must be normalized away so "category-only -> checklist == {}" still holds.
    client = fake_client(tmp_path)
    payload = {
        "target_category": "bed",
        "target_phrase": "white bed",
        "checklist": {k: [] for k in CHECKLIST_KEYS} | {"Target": ["it is white"]},
    }
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps(payload))

    result = parse_context(client, "White bed")
    assert result.checklist == {"Target": ["it is white"]}


def test_parse_context_raises_on_malformed_json(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: "not json")
    with pytest.raises(ContextParserError):
        parse_context(client, "cabinet")


def test_parse_context_raises_on_missing_required_field(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps({"target_category": "cabinet"}))
    with pytest.raises(ContextParserError):
        parse_context(client, "cabinet")


def test_parse_context_with_fake_backend_schema_filler_does_not_crash(tmp_path):
    # exercises the shared generic schema filler end to end (no monkeypatch)
    client = fake_client(tmp_path)
    result = parse_context(client, "Navy blue kitchen lower cabinet with brass handles")
    assert result.target_category == "fake"
    assert result.target_phrase == "fake"
    # FakeVLM's filler fills every checklist property with a 1-item list -- all 11 keys survive
    # the empty-list-drop normalization since none of them are actually empty here.
    assert set(result.checklist.keys()) == set(CHECKLIST_KEYS)
