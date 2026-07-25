import json

import numpy as np

from agent import schema
from agent.extract import (
    ExtractionParseError,
    _build_extraction_prompt,
    _build_frame,
    _majority_vote,
    _slot_value_from_field,
    extract,
    response_schema,
)
from agent.llm import LLMCallFailed, LLMClient, LLMResult
from agent.state import ObservationFrame

MAX_ADJACENT = 3
IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def _scripted_frame_json(**overrides) -> str:
    payload = {k: {"value": "unclear", "visibility": "not_visible"} for k in schema.queryable_slot_keys(MAX_ADJACENT)}
    payload.update(overrides)
    return json.dumps(payload)


def test_response_schema_excludes_category_and_tier_c():
    s = response_schema(MAX_ADJACENT)
    assert "obj.category" not in s["properties"]
    tier_c = [k for k in schema.all_slot_keys(MAX_ADJACENT) if schema.spec_for(k).tier == schema.TIER_C]
    assert not (set(s["properties"]) & set(tier_c))


def test_clear_visibility_meets_default_tau_obs():
    """Regression test: _slot_value_from_field used to multiply in a fake neutral logprob (0.5),
    which meant even a "clear" observation computed confidence=0.5 — below the tau_obs=0.80
    default — silently failing every slot's confidence check with no visible error.
    """
    value = _slot_value_from_field("obj.material", {"value": "oak", "visibility": "clear"})
    assert value.confidence == 1.0
    assert value.confidence >= 0.80  # config.yaml's default tau_obs
    assert value.certainty == "resolved"
    assert value.canon == "oak"


def test_partial_visibility_below_default_tau_obs():
    value = _slot_value_from_field("obj.material", {"value": "oak", "visibility": "partial"})
    assert value.confidence == 0.5
    assert value.confidence < 0.80


def test_not_visible_is_unknown_regardless_of_value():
    value = _slot_value_from_field("obj.material", {"value": "oak", "visibility": "not_visible"})
    assert value.canon is None
    assert value.confidence == 0.0
    assert value.certainty == "unknown"


def test_unclear_value_is_unknown_even_if_marked_clear():
    value = _slot_value_from_field("obj.material", {"value": "unclear", "visibility": "clear"})
    assert value.canon is None
    assert value.certainty == "unknown"


def test_build_extraction_prompt_contains_schema_keys():
    prompt = _build_extraction_prompt(MAX_ADJACENT)
    for key in schema.queryable_slot_keys(MAX_ADJACENT):
        assert key in prompt
    assert "obj.category" not in prompt.split("Return ONLY strict JSON")[1]


def test_build_frame_from_scripted_json():
    text = _scripted_frame_json(**{
        "obj.color_primary": {"value": "white", "visibility": "clear"},
        "ctx.support.object": {"value": "wall", "visibility": "partial"},
    })
    frame = _build_frame("hash123", json.loads(text), MAX_ADJACENT)
    assert frame.get("obj.color_primary").canon == "white"
    assert frame.get("obj.color_primary").confidence == 1.0
    assert frame.get("ctx.support.object").canon == "wall"
    assert frame.get("ctx.support.object").confidence == 0.5


def test_extract_end_to_end_with_scripted_llm(monkeypatch):
    scripted_text = _scripted_frame_json(**{"obj.color_primary": {"value": "navy", "visibility": "clear"}})
    monkeypatch.setattr(
        LLMClient, "call",
        lambda *a, **k: LLMResult(text=scripted_text, logprobs=None, latency_s=0.0, cached=False),
    )
    client = LLMClient("fake-model")
    frame = extract(IMG, client, max_adjacent=MAX_ADJACENT)
    assert frame.get("obj.color_primary").canon == "navy"
    assert frame.get("obj.color_primary").confidence == 1.0


def test_extract_degrades_on_llm_failure(monkeypatch):
    def _fail(*a, **k):
        raise LLMCallFailed("no server")

    monkeypatch.setattr(LLMClient, "call", _fail)
    client = LLMClient("fake-model")
    try:
        extract(IMG, client, max_adjacent=MAX_ADJACENT)
        raise AssertionError("expected LLMCallFailed to propagate when no samples succeeded")
    except LLMCallFailed:
        pass


def test_extract_malformed_json_raises_extraction_parse_error(monkeypatch):
    monkeypatch.setattr(
        LLMClient, "call",
        lambda *a, **k: LLMResult(text="not json", logprobs=None, latency_s=0.0, cached=False),
    )
    client = LLMClient("fake-model")
    try:
        extract(IMG, client, max_adjacent=MAX_ADJACENT)
        raise AssertionError("expected ExtractionParseError")
    except ExtractionParseError:
        pass


def test_majority_vote_agreement_fraction():
    frames = [
        ObservationFrame(image_hash="h"),
        ObservationFrame(image_hash="h"),
        ObservationFrame(image_hash="h"),
    ]
    from agent.state import SlotValue

    values = ["white", "white", "navy"]
    for f, v in zip(frames, values):
        for slot_key in schema.queryable_slot_keys(MAX_ADJACENT):
            f.slots[slot_key] = SlotValue(certainty="unknown", confidence=0.0)
        f.slots["obj.color_primary"] = SlotValue(raw=v, canon=v, confidence=1.0, certainty="resolved")

    merged = _majority_vote(frames, MAX_ADJACENT)
    assert merged.get("obj.color_primary").canon == "white"
    assert merged.get("obj.color_primary").confidence == 2 / 3
