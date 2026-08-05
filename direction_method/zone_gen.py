"""zone_gen (SPEC.md §5): image + target -> bbox (5-1, locate), then a relation set (5-2, zones),
with a deterministic, non-VLM box-drawing step in between.

Two VLM calls, never one merged call: 5-1 grounds the target with the category noun only (SPEC.md
§5 -- attributes make grounding fail on exactly the distractors that differ in that attribute),
5-2 needs the box already drawn onto the image so it can reason about "left of the box" etc.

Prompt text below is used verbatim -- do not paraphrase, shorten, or "clean up" the rule lists,
same policy as self_check.py.
"""

import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from llm import LLMClient, image_hash

MAX_REGIONS = 4

REGION_KEYS = [
    "left", "right", "above", "below",
    "left-top", "right-top", "left-bottom", "right-bottom",
    "on",
]

# The frame the model's bbox_2d coordinates are assumed to live in (SPEC.md §5: "relative
# [0, 1000]"). NOT verified against real model output yet -- smart_resize can shift the reference
# frame the model actually used. Verify on the first 20-30 images tomorrow (scripts/verify_zone_gen.py)
# and correct _bbox_to_pixels / _EDGE_FRAME below if the real convention differs.
_EDGE_FRAME = 1000
_EDGE_EPS = 1  # touching an edge means within this many frame-units of 0 or _EDGE_FRAME


class ZoneGenError(Exception):
    """Raised when the locate call's response can't be turned into a usable box -- malformed
    JSON, or an empty box list (the locate prompt asserts the object IS present, so an empty
    list is a real anomaly, not an expected "not found" case)."""


LOCATE_PROMPT = """Find the {target_category} in this image.

The object is present. Your task is to box it correctly, not to decide whether
it exists.

=== WHAT TO BOX ===

Box the complete object, including parts that belong to it (doors, drawers,
handles, legs, frame). Include parts that are partially occluded, as long as you
can tell where the object extends to.

Do NOT include objects merely resting on it, leaning against it, or hanging above
it. A cabinet's box covers the cabinet, not the items on its counter.

=== BUILT-IN RUNS ===

Some categories form continuous runs rather than separate units: kitchen cabinets,
countertops, open shelving, wall units.

For these, box the ENTIRE visible run as ONE object, not each door or segment.

  "kitchen lower cabinet" + a row of five cabinet doors under one counter
  -> ONE box covering the whole row

Return separate boxes only for units that are clearly physically detached — a
freestanding island apart from the wall run, a cabinet on the opposite wall.

=== SEVERAL INSTANCES ===

If genuinely separate instances are present, return all of them, largest and most
central first.

=== OUTPUT ===

{"boxes": [{"bbox_2d": [x0, y0, x1, y1], "label": "...", "note": "..."}]}

note — under 10 words, what you boxed."""

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            # maxItems: same repetition-loop guard as context_parser/checklist_update -- this
            # array had no bound at all before, the original unbounded-array failure mode.
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "label": {"type": "string", "maxLength": 200},
                    "note": {"type": "string", "maxLength": 200, "description": "Under 10 words. What was boxed."},
                },
                "required": ["bbox_2d", "label", "note"],
            },
        },
    },
    "required": ["boxes"],
}

ZONES_SYSTEM_PROMPT = """An indoor scene. A red box marks the TARGET object.

Select which regions around the target are worth asking about.

=== REGION KEYS ===

left, right, above, below,
left-top, right-top, left-bottom, right-bottom,
on

Directional keys refer to the area of the image on that side of the red box, as
the image is viewed on screen — screen-left is "left", screen-right is "right".
Never mirror this as if the target object itself were a person facing the camera;
the red box has no facing direction, only a position in the frame.
"on" means the top surface of the target itself — objects resting on it.

=== INCLUDE A REGION WHEN ===

- It lies inside the image frame, and
- It contains at least one describable object or surface.

A plain wall, a bare floor, or a blank surface DOES count as describable.

For "on": include it when one or more objects rest on the target's top surface.
If the top is bare, or the target has no usable top surface (a wall-mounted
object, a rug, a window), leave it out.

=== EXCLUDE A REGION WHEN ===

- It falls outside the image frame. If the red box touches the bottom edge of the
  image, there is no "below". If it touches the left edge, no "left".
- It is a sliver too narrow to hold anything.
- It is entirely occluded by the target or by something in front of it.

=== CORNER KEYS ===

Use "left-top" instead of "left" only when the content sits clearly in the upper
part of the left side and the lower part is empty or out of frame. When content
spans the whole left side, use "left". Prefer the simple keys.

=== ORDER AND LIMIT ===

Return AT MOST {MAX_REGIONS} regions, most informative first.

Informative: distinctive furniture, appliances, fixtures, distinctive wall or
floor treatment. Less informative: plain empty surface, generic background.

=== OUTPUT ===

First describe where the target sits in the frame and which edges it touches.
Then list the regions. For each, note what you see there before choosing its key.

{"scene": "...", "regions": [{"note": "...", "key": "left"}]}

scene — under 20 words. Target's position in the frame and which image edges it
        touches or comes close to.
note  — under 10 words. What is actually in that region."""

# Field order matches generation order (scene before regions, note before key within each item) --
# both act as short CoT before the enum-pinned key, same pattern as self_check's evidence/verdict.
ZONES_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "string",
            "maxLength": 200,
            "description": "Under 20 words. Target's position in the frame and which edges it touches or comes close to.",
        },
        "regions": {
            "type": "array",
            "maxItems": MAX_REGIONS,
            "items": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "maxLength": 200, "description": "Under 10 words. What is actually in that region."},
                    "key": {"type": "string", "enum": REGION_KEYS},
                },
                "required": ["note", "key"],
            },
        },
    },
    "required": ["scene", "regions"],
}


# --- 5-1 locate ----------------------------------------------------------------------------


@dataclass
class LocateResult:
    bbox_2d: tuple  # raw, as returned by the model -- native coordinate frame, unconverted
    n_boxes: int  # total boxes returned; >1 is an ambiguity signal to log (SPEC.md §5)
    boxed_image: np.ndarray
    raw_boxes: list = field(default_factory=list)


def _bbox_area(bbox_2d) -> float:
    x0, y0, x1, y1 = bbox_2d
    return abs(x1 - x0) * abs(y1 - y0)


def _bbox_to_pixels(bbox_2d, img_width: int, img_height: int) -> tuple:
    # Assumes the _EDGE_FRAME (0-1000) relative convention -- unverified, see the module docstring
    # note above _EDGE_FRAME. TODO(tomorrow, needs real model output): confirm or correct.
    x0, y0, x1, y1 = bbox_2d
    scale_x = img_width / _EDGE_FRAME
    scale_y = img_height / _EDGE_FRAME
    return (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)


def draw_red_box(image: np.ndarray, bbox_px: tuple) -> np.ndarray:
    """Deterministic, no VLM. Red outline, no fill, no label text. Line width is 0.5% of the
    image's short side, minimum 2px (SPEC.md §5-1 box drawing)."""
    pil_image = Image.fromarray(image).convert("RGB")
    short_side = min(pil_image.width, pil_image.height)
    line_width = max(2, round(short_side * 0.005))
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle(list(bbox_px), outline=(255, 0, 0), width=line_width)
    return np.array(pil_image)


def _image_size(image) -> tuple:
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        return w, h
    return image.width, image.height


_locate_cache: dict = {}


def locate(llm_client: LLMClient, image: np.ndarray, target_category: str, *, use_cache: bool = True) -> LocateResult:
    # Memoized per (original image hash, target_category), on top of llm.py's own disk cache, so
    # repeated ask_or_conclude calls for the same candidate don't re-run box drawing (SPEC.md §5:
    # "Cache the 5-1 and 5-2 results per candidate against the original image hash").
    cache_key = (image_hash(image), target_category)
    if cache_key in _locate_cache:
        return _locate_cache[cache_key]

    prompt = LOCATE_PROMPT.replace("{target_category}", target_category)
    result = llm_client.call(prompt, image, response_schema=LOCATE_SCHEMA, use_cache=use_cache)

    try:
        boxes = json.loads(result.text)["boxes"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ZoneGenError(f"locate: malformed response for target_category={target_category!r}: {result.text!r}") from e
    if not boxes:
        raise ZoneGenError(f"locate: no boxes returned for target_category={target_category!r} (object is asserted present)")

    largest = max(boxes, key=lambda b: _bbox_area(b["bbox_2d"]))
    img_w, img_h = _image_size(image)
    bbox_px = _bbox_to_pixels(largest["bbox_2d"], img_w, img_h)
    boxed_image = draw_red_box(image, bbox_px)

    out = LocateResult(bbox_2d=tuple(largest["bbox_2d"]), n_boxes=len(boxes), boxed_image=boxed_image, raw_boxes=boxes)
    _locate_cache[cache_key] = out
    return out


# --- 5-2 zones -------------------------------------------------------------------------------


@dataclass
class ZonesResult:
    scene: str
    raw_regions: list  # [{"note": ..., "key": ...}], as returned, before any post-processing


_zones_cache: dict = {}


def build_zones_prompt(target_category: str) -> tuple:
    system_prompt = ZONES_SYSTEM_PROMPT.replace("{MAX_REGIONS}", str(MAX_REGIONS))
    variable_part = f"target: {target_category}"
    prompt = f"{system_prompt}\n\n{variable_part}"
    return prompt, ZONES_SCHEMA


def zones(llm_client: LLMClient, boxed_image: np.ndarray, target_category: str, *, use_cache: bool = True) -> ZonesResult:
    # boxed_image and the original are different images with different hashes and separate
    # prefix-cache entries (SPEC.md §5) -- but zone_gen itself is a pure function of the original
    # image + target (the box is drawn deterministically), so memoizing here on the boxed image's
    # hash still satisfies "cache against the original image hash" one level up, in locate()'s
    # own memoization: a given original+target always produces the same boxed_image.
    cache_key = (image_hash(boxed_image), target_category)
    if cache_key in _zones_cache:
        return _zones_cache[cache_key]

    prompt, schema = build_zones_prompt(target_category)
    result = llm_client.call(prompt, boxed_image, response_schema=schema, use_cache=use_cache)

    try:
        parsed = json.loads(result.text)
        out = ZonesResult(scene=parsed["scene"], raw_regions=parsed["regions"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ZoneGenError(f"zones: malformed response for target_category={target_category!r}: {result.text!r}") from e

    _zones_cache[cache_key] = out
    return out


# --- post-processing (outside the VLM call) ---------------------------------------------------


_EDGE_OF_KEY = {
    "left": "left", "left-top": "left", "left-bottom": "left",
    "right": "right", "right-top": "right", "right-bottom": "right",
    "above": "above",
    "below": "below",
    "on": None,
}


def _touched_edges(bbox_2d) -> dict:
    x0, y0, x1, y1 = bbox_2d
    return {
        "left": x0 <= _EDGE_EPS,
        "right": x1 >= _EDGE_FRAME - _EDGE_EPS,
        "above": y0 <= _EDGE_EPS,
        "below": y1 >= _EDGE_FRAME - _EDGE_EPS,
    }


def _dedup_preserve_order(keys: list) -> list:
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _fallback_from_bbox_margins(bbox_2d) -> list:
    """SPEC.md §5: any direction with room inside the frame, at least one guaranteed. Only used
    when the VLM returns an empty region set -- never when dedup against the checklist empties it
    (that's a legitimate "conclude match")."""
    touched = _touched_edges(bbox_2d)
    return [k for k in ("left", "right", "above", "below") if not touched[k]]


@dataclass
class ResolvedZones:
    relations: list  # final relation keys, after fallback + edge filter + checklist dedup; empty is legitimate
    used_fallback: bool
    edge_touch_log: list  # VLM-returned keys whose edge is touched by the box -- now filtered, not just logged


def resolve_relations(zones_result: ZonesResult, bbox_2d, existing_parent_keys) -> ResolvedZones:
    vlm_keys = _dedup_preserve_order([r["key"] for r in zones_result.raw_regions])

    touched = _touched_edges(bbox_2d)
    edge_touch_log = [k for k in vlm_keys if _EDGE_OF_KEY.get(k) and touched[_EDGE_OF_KEY[k]]]

    used_fallback = False
    keys = vlm_keys
    if not keys:
        # _fallback_from_bbox_margins already excludes touched edges by construction, so it
        # needs no separate filtering below.
        keys = _fallback_from_bbox_margins(bbox_2d)
        used_fallback = True
    elif edge_touch_log:
        # [HARD FILTER -- 2026-08-05] Was logging-only ("decision for later, from the logged
        # frequency" -- SPEC.md §5-2/§11). Real frequency came in at 6/25 (24%) on the first
        # verification pass and kept recurring in production runs -- a direction whose edge the
        # box touches has no image content there, geometrically, no matter what the VLM said
        # (e.g. "left" on a box already flush against the frame's left edge). Drop it rather than
        # asking an unanswerable question. "on" is exempt: _EDGE_OF_KEY["on"] is None, since "on"
        # is the target's own top surface, not a frame-edge direction.
        keys = [k for k in keys if k not in edge_touch_log]

    relations = [k for k in keys if k not in existing_parent_keys]
    return ResolvedZones(relations=relations, used_fallback=used_fallback, edge_touch_log=edge_touch_log)
