import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from llm import LLMClient
from zone_gen import (
    LOCATE_SCHEMA,
    MAX_REGIONS,
    REGION_KEYS,
    ZONES_SCHEMA,
    ZoneGenError,
    ZonesResult,
    build_zones_prompt,
    draw_red_box,
    locate,
    resolve_relations,
    zones,
)
import zone_gen as zg


@pytest.fixture(autouse=True)
def _clear_module_caches():
    # locate()/zones() memoize per (image_hash, target) at module scope -- distinct tests must
    # not see each other's cached results just because they reuse a plain zero image.
    zg._locate_cache.clear()
    zg._zones_cache.clear()
    yield
    zg._locate_cache.clear()
    zg._zones_cache.clear()


def make_image(fill: int = 0, size: int = 40) -> np.ndarray:
    img = np.full((size, size, 3), fill, dtype=np.uint8)
    return img


def fake_client(tmp_path) -> LLMClient:
    return LLMClient("fake-model", backend="fake", cache_dir=tmp_path)


# --- 5-1 locate ----------------------------------------------------------------------------


def test_locate_selects_largest_box_by_area(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    boxes = [
        {"bbox_2d": [10, 10, 100, 100], "label": "cabinet", "note": "small one"},
        {"bbox_2d": [0, 0, 900, 900], "label": "cabinet", "note": "whole run"},
    ]
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps({"boxes": boxes}))

    result = locate(client, make_image(), "kitchen lower cabinet")
    assert result.bbox_2d == (0, 0, 900, 900)
    assert result.n_boxes == 2


def test_locate_raises_on_empty_boxes(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps({"boxes": []}))
    with pytest.raises(ZoneGenError):
        locate(client, make_image(), "cabinet")


def test_locate_raises_on_malformed_json(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: "not json")
    with pytest.raises(ZoneGenError):
        locate(client, make_image(), "cabinet")


def test_locate_memoizes_per_image_and_target(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    calls = []

    def fake_generate(prompt, image, response_schema, timeout_s):
        calls.append(prompt)
        return json.dumps({"boxes": [{"bbox_2d": [0, 0, 500, 500], "label": "x", "note": "n"}]})

    monkeypatch.setattr(client._backend, "generate", fake_generate)
    img = make_image()
    locate(client, img, "cabinet")
    locate(client, img, "cabinet")
    assert len(calls) == 1  # second call served from the in-process memo, not even the disk cache


def test_locate_prompt_substitutes_target_category_only(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    captured = {}

    def fake_generate(prompt, image, response_schema, timeout_s):
        captured["prompt"] = prompt
        return json.dumps({"boxes": [{"bbox_2d": [0, 0, 10, 10], "label": "x", "note": "n"}]})

    monkeypatch.setattr(client._backend, "generate", fake_generate)
    locate(client, make_image(), "kitchen lower cabinet")

    assert "Find the kitchen lower cabinet in this image." in captured["prompt"]
    # the OUTPUT section's literal JSON braces must survive the {target_category} substitution
    assert '{"boxes": [{"bbox_2d": [x0, y0, x1, y1], "label": "...", "note": "..."}]}' in captured["prompt"]


def test_locate_schema_requires_four_element_bbox():
    bbox_schema = LOCATE_SCHEMA["properties"]["boxes"]["items"]["properties"]["bbox_2d"]
    assert bbox_schema["minItems"] == 4 and bbox_schema["maxItems"] == 4


# --- box drawing (deterministic, no VLM) ----------------------------------------------------


def test_draw_red_box_outlines_without_fill():
    img = make_image(fill=255, size=40)  # all-white
    boxed = draw_red_box(img, (5, 5, 35, 35))
    # a pixel on the box edge is red
    assert tuple(boxed[5, 20]) == (255, 0, 0)
    # a pixel well inside the box is untouched (no fill)
    assert tuple(boxed[20, 20]) == (255, 255, 255)


def test_draw_red_box_minimum_line_width_2px():
    # short side = 40 -> 0.5% = 0.2px, must clamp up to the 2px minimum
    img = make_image(fill=255, size=40)
    boxed = draw_red_box(img, (5, 5, 35, 35))
    # both the boundary row and the row just inside it should be red if width clamped to >=2
    assert tuple(boxed[5, 20]) == (255, 0, 0)
    assert tuple(boxed[6, 20]) == (255, 0, 0)


def test_draw_red_box_preserves_original_image_untouched():
    img = make_image(fill=255, size=40)
    original_copy = img.copy()
    draw_red_box(img, (5, 5, 35, 35))
    assert np.array_equal(img, original_copy)  # PIL.Image.fromarray + draw must not mutate input


# --- 5-2 zones -------------------------------------------------------------------------------


def test_build_zones_prompt_substitutes_max_regions_and_appends_target_field():
    prompt, schema = build_zones_prompt("kitchen lower cabinet")
    assert f"Return AT MOST {MAX_REGIONS} regions" in prompt
    assert "target: kitchen lower cabinet" in prompt
    assert schema == ZONES_SCHEMA
    # literal JSON braces in the OUTPUT section must survive the {MAX_REGIONS} substitution
    assert '{"scene": "...", "regions": [{"note": "...", "key": "left"}]}' in prompt


def test_zones_schema_field_order_scene_before_regions_note_before_key():
    assert list(ZONES_SCHEMA["properties"].keys()) == ["scene", "regions"]
    item_props = ZONES_SCHEMA["properties"]["regions"]["items"]["properties"]
    assert list(item_props.keys()) == ["note", "key"]


def test_zones_schema_max_items_matches_max_regions_config():
    assert ZONES_SCHEMA["properties"]["regions"]["maxItems"] == MAX_REGIONS


def test_zones_parses_scene_and_regions(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    payload = {"scene": "cabinet centered, touches bottom edge", "regions": [{"note": "tiled wall", "key": "left"}]}
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: json.dumps(payload))

    result = zones(client, make_image(), "cabinet")
    assert isinstance(result, ZonesResult)
    assert result.scene == "cabinet centered, touches bottom edge"
    assert result.raw_regions == [{"note": "tiled wall", "key": "left"}]


def test_zones_raises_on_malformed_json(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    monkeypatch.setattr(client._backend, "generate", lambda *a, **k: "garbage")
    with pytest.raises(ZoneGenError):
        zones(client, make_image(), "cabinet")


def test_zones_memoizes_per_boxed_image_and_target(tmp_path, monkeypatch):
    client = fake_client(tmp_path)
    calls = []

    def fake_generate(prompt, image, response_schema, timeout_s):
        calls.append(1)
        return json.dumps({"scene": "s", "regions": []})

    monkeypatch.setattr(client._backend, "generate", fake_generate)
    img = make_image()
    zones(client, img, "cabinet")
    zones(client, img, "cabinet")
    assert len(calls) == 1


def test_zones_with_fake_backend_schema_filler_does_not_crash(tmp_path):
    # exercises the shared generic schema filler end to end (no monkeypatch) for zone_gen's shape
    client = fake_client(tmp_path)
    result = zones(client, make_image(), "cabinet")
    assert result.scene == "fake"
    assert result.raw_regions == [{"note": "fake", "key": "left"}]


# --- post-processing ---------------------------------------------------------------------------


def _zr(keys):
    return ZonesResult(scene="s", raw_regions=[{"note": "n", "key": k} for k in keys])


def test_resolve_relations_dedups_keeping_first_occurrence():
    zr = _zr(["left", "right", "left", "on"])
    out = resolve_relations(zr, bbox_2d=(200, 200, 800, 800), existing_parent_keys=set())
    assert out.relations == ["left", "right", "on"]
    assert out.used_fallback is False


def test_resolve_relations_removes_existing_checklist_parent_keys():
    zr = _zr(["left", "right", "on"])
    out = resolve_relations(zr, bbox_2d=(200, 200, 800, 800), existing_parent_keys={"left"})
    assert out.relations == ["right", "on"]


def test_resolve_relations_empty_after_checklist_dedup_is_legitimate_no_fallback():
    zr = _zr(["left"])
    out = resolve_relations(zr, bbox_2d=(200, 200, 800, 800), existing_parent_keys={"left"})
    assert out.relations == []
    assert out.used_fallback is False  # empty because of checklist dedup, not a failed VLM call


def test_resolve_relations_applies_geometry_fallback_when_vlm_returns_nothing():
    zr = _zr([])
    # box floats in the middle -- all four directions have room
    out = resolve_relations(zr, bbox_2d=(200, 200, 800, 800), existing_parent_keys=set())
    assert out.used_fallback is True
    assert set(out.relations) == {"left", "right", "above", "below"}


def test_resolve_relations_fallback_excludes_touched_edges():
    zr = _zr([])
    # box touches the left edge (x0=0) and the bottom edge (y1=1000) -- no room there
    out = resolve_relations(zr, bbox_2d=(0, 200, 800, 1000), existing_parent_keys=set())
    assert out.used_fallback is True
    assert set(out.relations) == {"right", "above"}


def test_resolve_relations_fallback_then_checklist_dedup_still_applies():
    zr = _zr([])
    out = resolve_relations(zr, bbox_2d=(0, 200, 800, 1000), existing_parent_keys={"right"})
    assert out.relations == ["above"]
    assert out.used_fallback is True


def test_edge_touch_log_flags_touched_direction_and_now_filters_it():
    # box touches the left edge; the VLM still returned "left" -- logged AND dropped
    # (hard filter added 2026-08-05: no image content past a touched edge, regardless of what
    # the VLM said).
    zr = _zr(["left", "above"])
    out = resolve_relations(zr, bbox_2d=(0, 200, 800, 700), existing_parent_keys=set())
    assert "left" not in out.relations  # filtered
    assert out.relations == ["above"]
    assert out.edge_touch_log == ["left"]  # still logged, since "above" doesn't touch an edge here


def test_resolve_relations_filters_all_edge_touching_keys_to_empty():
    # every VLM-returned key touches a bbox edge -- filtered to empty, same "legitimate no
    # fallback" shape as the checklist-dedup-to-empty case (VLM DID answer, just all unusable).
    zr = _zr(["left", "below"])
    out = resolve_relations(zr, bbox_2d=(0, 200, 800, 1000), existing_parent_keys=set())
    assert out.relations == []
    assert out.used_fallback is False
    assert set(out.edge_touch_log) == {"left", "below"}


@pytest.mark.parametrize(
    "episode,bbox,too_thin",
    [
        # real locate() output from the 2026-08-11 25-image verification pass; each of these had a
        # direction selected with only a few frame-units of room on that side.
        ("7c754d7d", (203, 0, 835, 997), "below"),   # 3 units below
        ("c0622d58", (0, 646, 967, 997), "below"),   # 3 units below
        ("162a896c", (68, 753, 997, 998), "below"),  # 2 units below
        ("f9140bf5", (362, 273, 580, 963), "below"), # 37 units below
        ("861d2297", (172, 1, 998, 835), "right"),   # 2 units right
    ],
)
def test_directions_with_no_usable_room_are_filtered(episode, bbox, too_thin):
    """A strip a few frame-units wide holds nothing describable, but the old eps=1 threshold only
    caught boxes flush against the frame. `below` was the worst relation in the same run's official
    logs (19 false-"no"s on the true match vs 1 distractor rejection).
    """
    zr = _zr([too_thin])
    out = resolve_relations(zr, bbox_2d=bbox, existing_parent_keys=set())
    assert too_thin not in out.relations, f"{episode}: {too_thin} should be filtered for {bbox}"
    assert too_thin in out.edge_touch_log


def test_direction_with_ample_room_is_kept():
    # the other side of the threshold -- a real box with plenty of room must survive
    zr = _zr(["below", "left", "right"])
    out = resolve_relations(zr, bbox_2d=(335, 538, 672, 753), existing_parent_keys=set())
    assert set(out.relations) == {"below", "left", "right"}
    assert out.edge_touch_log == []


def test_edge_threshold_is_a_usable_strip_not_pixel_contact():
    # guards the constant itself: a 1-unit threshold is what let the above cases through.
    from zone_gen import _EDGE_EPS, _EDGE_FRAME
    assert _EDGE_EPS >= 0.02 * _EDGE_FRAME


def test_edge_touch_log_ignores_on_key():
    # "on" isn't a directional/edge-relative key -- must never appear in the edge-touch log
    zr = _zr(["on"])
    out = resolve_relations(zr, bbox_2d=(0, 0, 1000, 1000), existing_parent_keys=set())
    assert out.edge_touch_log == []


def test_region_keys_match_spec_vocabulary_and_order():
    assert REGION_KEYS == [
        "left", "right", "above", "below",
        "left-top", "right-top", "left-bottom", "right-bottom",
        "on",
    ]
