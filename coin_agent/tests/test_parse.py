import json

from agent import schema
from agent.parse import NOT_MENTIONED, parse_description, parse_description_llm_response
from agent.llm import LLMCallFailed, LLMClient, LLMResult

MAX_ADJACENT = 3


def _scripted_response(**overrides) -> str:
    payload = {k: NOT_MENTIONED for k in schema.queryable_slot_keys(MAX_ADJACENT)}
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_description_llm_response_ignores_not_mentioned():
    text = _scripted_response(**{"obj.color_primary": "white", "ctx.support.object": "wall"})
    slots = parse_description_llm_response(text, MAX_ADJACENT)
    assert set(slots) == {"obj.color_primary", "ctx.support.object"}
    assert slots["obj.color_primary"].canon == "white"
    assert slots["ctx.support.object"].canon == "wall"


def test_parse_description_llm_response_all_slots_authoritative():
    text = _scripted_response(**{"obj.color_primary": "navy", "room.type": "kitchen"})
    slots = parse_description_llm_response(text, MAX_ADJACENT)
    for value in slots.values():
        assert value.certainty == "resolved"
        assert value.confidence == 1.0
        assert value.provenance == "description"


def test_parse_description_llm_response_excludes_category_and_tier_c():
    assert "obj.category" not in schema.queryable_slot_keys(MAX_ADJACENT)
    tier_c_keys = [k for k in schema.all_slot_keys(MAX_ADJACENT) if schema.spec_for(k).tier == schema.TIER_C]
    assert not (set(schema.queryable_slot_keys(MAX_ADJACENT)) & set(tier_c_keys))


def test_bare_noun_adjacent_object_with_color_split():
    # Real example: "White faceted clock next to a black framed picture on a wall"
    text = _scripted_response(**{
        "obj.color_primary": "white",
        "ctx.support.object": "wall",
        "ctx.adjacent[0].object": "picture",
        "ctx.adjacent[0].color": "black",
    })
    slots = parse_description_llm_response(text, MAX_ADJACENT)
    assert slots["ctx.adjacent[0].object"].canon == "picture"
    assert slots["ctx.adjacent[0].color"].canon == "black"
    assert slots["ctx.support.object"].canon == "wall"


def test_accent_material_and_color_mapping():
    # Real example: "White shower with red tile accents"
    text = _scripted_response(**{
        "obj.color_primary": "white", "obj.color_secondary": "red", "obj.material": "tile",
    })
    slots = parse_description_llm_response(text, MAX_ADJACENT)
    assert slots["obj.color_primary"].canon == "white"
    assert slots["obj.color_secondary"].canon == "red"
    assert slots["obj.material"].canon == "tile"


def test_malformed_json_raises_extraction_parse_error():
    from agent.extract import ExtractionParseError

    try:
        parse_description_llm_response("not json at all", MAX_ADJACENT)
        raise AssertionError("expected ExtractionParseError")
    except ExtractionParseError:
        pass


def test_parse_description_seeds_category_and_llm_slots(monkeypatch):
    scripted_text = _scripted_response(**{"obj.color_primary": "white", "ctx.support.object": "wall"})
    monkeypatch.setattr(LLMClient, "call", lambda *a, **k: LLMResult(text=scripted_text, logprobs=None, latency_s=0.0, cached=False))
    client = LLMClient("fake-model")

    belief = parse_description("White clock hanging on a wall", info_category="Clock", llm_client=client)
    assert belief.get("obj.category").canon == "clock"
    assert belief.get("obj.color_primary").canon == "white"
    assert belief.get("ctx.support.object").canon == "wall"
    assert belief.noun_phrase == "the clock"


def test_parse_description_degrades_gracefully_on_llm_failure(monkeypatch):
    def _fail(*a, **k):
        raise LLMCallFailed("no server")

    monkeypatch.setattr(LLMClient, "call", _fail)
    client = LLMClient("fake-model")

    belief = parse_description("White clock hanging on a wall", info_category="Clock", llm_client=client)
    assert belief.get("obj.category").canon == "clock"
    assert belief.get("ctx.support.object").canon is None  # nothing beyond category was seeded


def test_parse_description_without_llm_client_only_seeds_category():
    belief = parse_description("White clock hanging on a wall", info_category="Clock", llm_client=None)
    assert belief.get("obj.category").canon == "clock"
    assert belief.get("obj.color_primary").canon is None
