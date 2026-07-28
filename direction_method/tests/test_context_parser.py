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
    _merge_other_objects_into_checklist,
    _target_assertions_with_spatial_words,
    _validate,
    build_prompt,
    parse_context,
)
from llm import LLMClient
from templates import CHECKLIST_KEYS


def fake_client(tmp_path) -> LLMClient:
    return LLMClient("fake-model", backend="fake", cache_dir=tmp_path)


def payload(target_category, target_phrase, other_objects, checklist):
    return {
        "target_category": target_category,
        "target_phrase": target_phrase,
        "other_objects": other_objects,
        "checklist": checklist,
    }


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


def test_prompt_contains_the_procedure_and_parts_vs_objects_sections():
    # these three sections exist specifically to stop the model generalizing "with X -> Target"
    # from the brass-handles example onto genuinely separate objects like a blanket.
    assert "=== PROCEDURE ===" in CONTEXT_PARSER_PROMPT
    assert "=== PARTS vs SEPARATE OBJECTS ===" in CONTEXT_PARSER_PROMPT
    assert "=== NEVER PUT A RELATION IN A TARGET ASSERTION ===" in CONTEXT_PARSER_PROMPT
    assert "could you carry it into another room" in CONTEXT_PARSER_PROMPT


def test_prompt_examples_include_the_beneath_to_above_inversion_cases():
    # real failures found against the live model -- kept as worked examples so the model sees the
    # inversion demonstrated, not just described.
    assert "Dark gray slatted heater beneath a round mirror" in CONTEXT_PARSER_PROMPT
    assert "Large beige carpet under a wooden coffee table" in CONTEXT_PARSER_PROMPT
    assert "Gray couch with pillows under three framed artworks" in CONTEXT_PARSER_PROMPT


def test_prompt_tells_the_model_checklist_is_target_only():
    # confirmed against a live server (6 types x 30 episodes) that the model's own checklist output
    # for non-Target keys is unreliable (35% overall, 90%+ on the richest description types) even
    # though other_objects is reliable -- the prompt now only asks the model for Target content.
    assert "checklist can be empty even when other_objects is not" in CONTEXT_PARSER_PROMPT.replace("\n", " ")


# --- schema shape --------------------------------------------------------------------------


def test_schema_field_order_matches_output_fields():
    # other_objects before checklist -- generation order is commitment order.
    assert list(CONTEXT_PARSER_SCHEMA["properties"].keys()) == [
        "target_category", "target_phrase", "other_objects", "checklist",
    ]


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


def test_schema_other_objects_field_order_is_object_then_cue_then_key():
    item_schema = CONTEXT_PARSER_SCHEMA["properties"]["other_objects"]["items"]
    assert list(item_schema["properties"].keys()) == ["object", "cue", "key"]


def test_schema_other_objects_key_enum_excludes_target():
    item_schema = CONTEXT_PARSER_SCHEMA["properties"]["other_objects"]["items"]
    key_enum = item_schema["properties"]["key"]["enum"]
    assert "Target" not in key_enum
    assert set(key_enum) == set(CHECKLIST_KEYS) - {"Target"}


def test_schema_other_objects_array_is_bounded():
    assert CONTEXT_PARSER_SCHEMA["properties"]["other_objects"]["maxItems"] == 8


# --- validator: spatial words in Target -------------------------------------------------------


def test_target_assertion_with_next_to_is_flagged():
    hits = _target_assertions_with_spatial_words({"Target": ["the bed is next to a nightstand"]})
    assert hits == [("the bed is next to a nightstand", "next to")]


def test_target_assertion_with_beneath_is_flagged():
    hits = _target_assertions_with_spatial_words({"Target": ["it is beneath a round mirror"]})
    assert hits and hits[0][1] == "beneath"


def test_target_assertion_without_spatial_word_is_not_flagged():
    hits = _target_assertions_with_spatial_words({"Target": ["it is navy blue", "it has brass handles"]})
    assert hits == []


def test_spatial_word_detection_does_not_false_positive_on_substrings():
    # "on" must not match inside "wooden"; word-boundary matching is required.
    hits = _target_assertions_with_spatial_words({"Target": ["it is a wooden cabinet"]})
    assert hits == []


def test_no_target_key_at_all_is_not_flagged():
    assert _target_assertions_with_spatial_words({"above": ["a mirror"]}) == []


def test_validate_flags_spatial_words_in_target():
    checklist = {"Target": ["it is white", "it is next to a nightstand"]}
    problems = _validate(checklist)
    assert len(problems) == 1


def test_validate_clean_checklist_has_no_problems():
    assert _validate({"Target": ["it is white"], "above": ["a mirror"]}) == []


# --- merge: checklist's non-Target entries are built from other_objects, not the model ---------


def test_merge_builds_non_target_keys_purely_from_other_objects():
    other_objects = [{"object": "a round mirror", "cue": "beneath", "key": "above"}]
    merged = _merge_other_objects_into_checklist(other_objects, {"Target": ["it is dark gray"]})
    assert merged == {"Target": ["it is dark gray"], "above": ["a round mirror"]}


def test_merge_discards_non_target_keys_the_model_wrote_directly():
    # the model's own non-Target checklist content is never trusted, even if present -- confirmed
    # against a live server that this content is unreliable (35% overall, 90%+ on rich descriptions).
    other_objects = []
    model_checklist = {"Target": ["it is white"], "above": ["something the model invented"]}
    merged = _merge_other_objects_into_checklist(other_objects, model_checklist)
    assert merged == {"Target": ["it is white"]}


def test_merge_omits_target_key_when_model_checklist_has_no_target():
    merged = _merge_other_objects_into_checklist([], {})
    assert merged == {}


def test_merge_omits_target_key_when_model_target_is_an_empty_list():
    merged = _merge_other_objects_into_checklist([], {"Target": []})
    assert merged == {}


def test_merge_multiple_other_objects_under_the_same_key():
    other_objects = [
        {"object": "a nightstand", "cue": "next to", "key": "next to"},
        {"object": "a lamp", "cue": "beside", "key": "next to"},
    ]
    merged = _merge_other_objects_into_checklist(other_objects, {})
    assert merged == {"next to": ["a nightstand", "a lamp"]}


def test_merge_dedups_exact_match_after_normalization():
    other_objects = [
        {"object": "a nightstand", "cue": "next to", "key": "next to"},
        {"object": "  A   Nightstand  ", "cue": "beside", "key": "next to"},
    ]
    merged = _merge_other_objects_into_checklist(other_objects, {})
    assert merged == {"next to": ["a nightstand"]}


def test_merge_does_not_mutate_model_checklist_target_list():
    model_checklist = {"Target": ["it is white"]}
    _merge_other_objects_into_checklist([], model_checklist)
    assert model_checklist == {"Target": ["it is white"]}


# --- parse_context: parsing, error handling -------------------------------------------------


def test_parse_context_category_only_description(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(
        client._backend, "generate",
        lambda *a, **k: json.dumps(payload("kitchen lower cabinet", "kitchen lower cabinet", [], {})),
    )

    result = parse_context(client, "Kitchen lower cabinet")
    assert isinstance(result, ParsedContext)
    assert result.target_category == "kitchen lower cabinet"
    assert result.target_phrase == "kitchen lower cabinet"
    assert result.other_objects == []
    assert result.checklist == {}
    assert result.validation_problems == []


def test_parse_context_merges_other_objects_into_checklist(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    other_objects = [{"object": "a white farmhouse sink", "cue": "under", "key": "above"}]
    # model's own checklist entry for "above" is present here but should make no difference --
    # the merge always rebuilds non-Target keys from other_objects regardless.
    model_checklist = {"Target": ["it is navy blue"], "above": ["a white farmhouse sink"]}
    monkeypatch.setattr(
        client._backend, "generate",
        lambda *a, **k: json.dumps(payload("kitchen lower cabinet", "navy blue kitchen lower cabinet", other_objects, model_checklist)),
    )

    result = parse_context(client, "Navy blue kitchen lower cabinet under a white farmhouse sink")
    assert result.target_category == "kitchen lower cabinet"
    assert result.target_phrase == "navy blue kitchen lower cabinet"
    assert result.other_objects == other_objects
    assert result.checklist == {"Target": ["it is navy blue"], "above": ["a white farmhouse sink"]}
    assert result.validation_problems == []


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
    # exercises the shared generic schema filler end to end (no monkeypatch). FakeVLM fills
    # other_objects with one item (key="left", the first non-Target enum value) and checklist with
    # all 11 keys -- only "Target" (from the model) and "left" (from other_objects) survive the merge.
    client = fake_client(tmp_path)
    result = parse_context(client, "Navy blue kitchen lower cabinet with brass handles")
    assert result.target_category == "fake"
    assert result.target_phrase == "fake"
    assert set(result.checklist.keys()) == {"Target", "left"}


# --- parse_context: validation-triggered retry --------------------------------------------


def test_parse_context_retries_once_on_validation_failure_and_succeeds(tmp_path, monkeypatch):
    bad = payload(
        "bed", "white bed",
        [{"object": "a nightstand", "cue": "next to", "key": "next to"}],
        {"Target": ["it is white", "it is next to a nightstand"]},  # relation folded into Target
    )
    good = payload(
        "bed", "white bed",
        [{"object": "a nightstand", "cue": "next to", "key": "next to"}],
        {"Target": ["it is white"]},
    )
    responses = [json.dumps(bad), json.dumps(good)]
    calls = []

    def fake_generate(prompt, image, response_schema, timeout_s):
        calls.append(prompt)
        return responses[len(calls) - 1]

    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", fake_generate)

    result = parse_context(client, "White bed next to a nightstand")
    assert len(calls) == 2  # first call flagged, one retry made
    assert result.checklist == {"Target": ["it is white"], "next to": ["a nightstand"]}
    assert result.validation_problems == []
    assert result.retried is True


def test_parse_context_retried_is_false_when_first_attempt_is_clean(tmp_path, monkeypatch):
    good = payload("bed", "white bed", [], {"Target": ["it is white"]})
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps(good))

    result = parse_context(client, "White bed")
    assert result.retried is False


def test_parse_context_retry_bypasses_cache_not_a_replay(tmp_path, monkeypatch):
    # if the retry read from cache it would just replay the same flagged response -- assert the
    # backend is actually invoked a second time, not served from the first call's cache entry.
    bad = payload("bed", "white bed", [], {"Target": ["it is next to a nightstand"]})
    calls = []

    def fake_generate(prompt, image, response_schema, timeout_s):
        calls.append(1)
        return json.dumps(bad)

    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", fake_generate)

    parse_context(client, "White bed next to a nightstand")
    assert len(calls) == 2


def test_parse_context_logs_and_proceeds_when_still_flagged_after_retry(tmp_path, monkeypatch, capsys):
    bad = payload("bed", "white bed", [], {"Target": ["it is next to a nightstand"]})
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps(bad))

    result = parse_context(client, "White bed next to a nightstand")

    assert result.validation_problems != []  # never raises -- proceeds with the flagged result
    captured = capsys.readouterr()
    assert "[WARN] context_parser" in captured.out
